from typing import Optional, List
import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

import hydra
import pydantic
from omegaconf import DictConfig

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str
    loss: LossConfig


class TestConfig(pydantic.BaseModel):
    # Architecture config
    arch: ArchConfig
    
    # Model checkpoint
    load_checkpoint: str
    
    # Data paths
    data_paths: List[str]
    data_paths_test: List[str] = []

    num_samples: int = 10  # Number of random samples to test
    show_intermediate_steps: bool = False  # Whether to show each reasoning step
    
    # Test parameters
    batch_size: int = 1

    # Extras
    seed: int = 0
    eval_save_outputs: List[str] = []


@dataclass
class TrainState:
    model: nn.Module
    carry: Any

    step: int
    total_steps: int

def init_train_state(config: TestConfig, train_metadata: PuzzleDatasetMetadata) -> TrainState:

    total_steps = config.num_samples
    # Load model
    model = load_model_from_checkpoint(config, train_metadata)
    
    return TrainState(
        model=model,
        carry=None,
        step=0,
        total_steps=total_steps
    )

def load_model_from_checkpoint(config: TestConfig, metadata: PuzzleDatasetMetadata) -> nn.Module:
    print("="*80)
    print("LOADING MODEL")
    print("="*80)
    print(f"Checkpoint: {config.load_checkpoint}")
    print(f"Architecture: {config.arch.name}")
    
    # Create model config
    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore
        batch_size=config.batch_size,
        vocab_size=metadata.vocab_size,
        seq_len=metadata.seq_len,
        num_puzzle_identifiers=metadata.num_puzzle_identifiers,
        causal=False
    )
    
    # Load model class
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)
    
    # Instantiate model
    with torch.device("cuda"):
        model: nn.Module = model_cls(model_cfg)
        print(model)
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)  # type: ignore
        if "DISABLE_COMPILE" not in os.environ:
            model = torch.compile(model)  # type: ignore
        
        # Load checkpoint
        if not os.path.exists(config.load_checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {config.load_checkpoint}")
        
        state_dict = torch.load(config.load_checkpoint, map_location="cuda")

        # Resize and reset puzzle emb if needed
        puzzle_emb_name = "_orig_mod.model.inner.puzzle_emb.weights"
        expected_shape: torch.Size = model.model.puzzle_emb.weights.shape  # type: ignore
        if puzzle_emb_name in state_dict:
            puzzle_emb = state_dict[puzzle_emb_name]
            if puzzle_emb.shape != expected_shape:
                print(f"Resetting puzzle embedding as shape is different. Found {puzzle_emb.shape}, Expected {expected_shape}")
                # Re-initialize using mean
                state_dict[puzzle_emb_name] = (
                    torch.mean(puzzle_emb, dim=0, keepdim=True).expand(expected_shape).contiguous()
                )
        
        # Load state dict
        missing, unexpected = model.load_state_dict(state_dict, assign=True)
        
        if len(missing) > 0:
            print(f"Missing keys: {len(missing)}")
            for k in missing[:5]:
                print(f"  - {k}")
        if len(unexpected) > 0:
            print(f"Unexpected keys: {len(unexpected)}")
            for k in unexpected[:5]:
                print(f"  - {k}")
        
        model.eval()
    return model

def create_evaluators(config: TestConfig, metadata: PuzzleDatasetMetadata) -> List[Any]:
    data_paths =config.data_paths_test if len(config.data_paths_test)>0 else config.data_paths
    # Initialize evaluators
    evaluators = []
    for cfg in config.evaluators:
        for data_path in data_paths:
            cls = load_model_class(cfg.name, "evaluators.")(
                data_path=data_path, eval_metadata=metadata, **cfg.__pydantic_extra__
            )  # type: ignore
            evaluators.append(cls)

    return evaluators

def test_model(
    config: TestConfig,
    train_state: TrainState,
    dataloader: DataLoader,
    metadata: PuzzleDatasetMetadata,
    evaluators: List[Any],
    rank: int,
    world_size: int,
    cpu_group: Optional[dist.ProcessGroup],
):
    reduced_metrics = None

    with torch.inference_mode():
        return_keys = set(config.eval_save_outputs)
        return_keys.add('preds')  # Always include preds for displaying sample
        for evaluator in evaluators:
            evaluator.begin_eval()
            return_keys.update(evaluator.required_outputs)

        set_ids = {k: idx for idx, k in enumerate(metadata.sets)}

        save_preds = {}

        metric_keys = []
        metric_values = None

        carry = None
        processed_batches = 0
        max_batches = config.num_samples  # Limit number of batches to process
        sample_data = None  # For displaying one sample
        
        for set_name, batch, global_batch_size in dataloader:
            # Check if reached maximum batches
            if processed_batches >= max_batches:
                if rank == 0:
                    print(f"\nReached maximum {max_batches} batches, stopping...")
                break

            processed_batches += 1
            if rank == 0:
                print(f"Processing batch {processed_batches}: {set_name}")
            
            # To device
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.device("cuda"):
                carry = train_state.model.initial_carry(batch)  # type: ignore

            # Forward
            inference_steps = 0
            while True:
                carry, loss, metrics, preds, all_finish = train_state.model(
                    carry=carry, batch=batch, return_keys=return_keys
                )
                inference_steps += 1

                if all_finish:
                    break

            if rank == 0:
                print(f"  Completed inference in {inference_steps} steps")
            
            # Capture first sample for display
            if sample_data is None and rank == 0:
                sample_data = {
                    'inputs': batch['inputs'][0].cpu(),
                    'labels': batch['labels'][0].cpu(),
                    'preds': preds.get('preds', None),
                    'set_name': set_name
                }
                if sample_data['preds'] is not None:
                    sample_data['preds'] = sample_data['preds'][0].cpu()

            for collection in (batch, preds):
                for k, v in collection.items():
                    if k in config.eval_save_outputs:
                        save_preds.setdefault(k, [])
                        save_preds[k].append(v.cpu())  # Move to CPU for saving GPU memory

            del carry, loss, preds, batch, all_finish

            # Aggregate metrics
            set_id = set_ids[set_name]

            if metric_values is None:
                metric_keys = list(
                    sorted(metrics.keys())
                )  # Sort keys to guarantee all processes use the same order.
                metric_values = torch.zeros(
                    (len(set_ids), len(metrics.values())), dtype=torch.float32, device="cuda"
                )

            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])

            del metrics

        # concatenate save preds
        # save_preds = {k: torch.cat(v, dim=0) for k, v in save_preds.items()}
        # # Save preds
        # if config.checkpoint_path is not None and len(save_preds):
        #     # Each rank save predictions independently
        #     os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)
        #     torch.save(
        #         save_preds, os.path.join(config.checkpoint_path, f"step_{train_state.step}_all_preds.{rank}")
        #     )
        # del save_preds

        # Reduce to rank 0
        if metric_values is not None:
            if world_size > 1:
                dist.reduce(metric_values, dst=0)

            if rank == 0:
                reduced_metrics = metric_values.cpu().numpy()
                reduced_metrics = {
                    set_name: {
                        metric_name: reduced_metrics[set_id, metric_id]
                        for metric_id, metric_name in enumerate(metric_keys)
                    }
                    for set_id, set_name in enumerate(set_ids)
                }

                # Postprocess
                for set_name, m in reduced_metrics.items():
                    count = m.pop("count")
                    reduced_metrics[set_name] = {k: v / count for k, v in m.items()}

        # Run evaluators
        if rank == 0:
            print(f"\nRunning {len(evaluators)} evaluator(s)...")

        for i, evaluator in enumerate(evaluators):
            if rank == 0:
                print(f"Running evaluator {i+1}/{len(evaluators)}: {evaluator.__class__.__name__}")
                
            # Path for saving
            evaluator_save_path = None
            if config.checkpoint_path is not None:
                evaluator_save_path = os.path.join(
                    config.checkpoint_path,
                    f"evaluator_{evaluator.__class__.__name__}_step_{train_state.step}",
                )
                os.makedirs(evaluator_save_path, exist_ok=True)

            # Run and log
            metrics = evaluator.result(evaluator_save_path, rank=rank, world_size=world_size, group=cpu_group)
            if rank == 0 and metrics is not None:
                if reduced_metrics is None:
                    reduced_metrics = {}

                reduced_metrics.update(metrics)
                print(f"  Completed {evaluator.__class__.__name__}")
                
        if rank == 0:
            print("All evaluators completed!")

    return reduced_metrics, sample_data


def create_dataloader(config: TestConfig, split: str, rank: int, world_size: int, **kwargs):
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,
        dataset_paths=config.data_paths_test if len(config.data_paths_test)>0 and split=="test" else config.data_paths,
        rank=rank,
        num_replicas=world_size,
        **kwargs
    ), split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=8,
        pin_memory=True,
        persistent_workers=True
    )

    print(f"Dataset: {dataset.config.dataset_paths}")
    print(f"Split: {split}")
    print(f"Vocab size: {dataset.metadata.vocab_size}")
    print(f"Sequence length: {dataset.metadata.seq_len}")
    print(f"Total puzzles: {dataset.metadata.total_puzzles}")
    print()

    return dataloader, dataset.metadata


@hydra.main(config_path="config", config_name="cfg_test", version_base=None)
def main(hydra_config: DictConfig):
    RANK = 0
    WORLD_SIZE = 1
    CPU_PROCESS_GROUP = None

    # Initialize distributed training if in distributed environment (e.g. torchrun)
    if "LOCAL_RANK" in os.environ:
        # Initialize distributed, default device and dtype
        dist.init_process_group(backend="nccl")

        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        
        # CPU GLOO process group
        CPU_PROCESS_GROUP = dist.new_group(backend="gloo")
        assert (
            dist.get_rank(CPU_PROCESS_GROUP) == RANK and dist.get_world_size(CPU_PROCESS_GROUP) == WORLD_SIZE
        )
    
    config = TestConfig(**hydra_config)  # type: ignore
    
    # Set random seed
    torch.random.manual_seed(config.seed)
    
    # Load dataset
    dataloader, metadata = create_dataloader(config, split="test", test_set_mode=True, epochs_per_iter=1, global_batch_size=config.batch_size, rank=RANK, world_size=WORLD_SIZE)

    try:
        evaluators = create_evaluators(config, metadata)
    except:
        print("No evaluator found")
        evaluators = []
    
    # Load model
    train_state = init_train_state(config, metadata)
    train_state.model.eval()

    print("Model and dataset loaded successfully.")

    metrics, sample_data = test_model(
        config, 
        train_state, 
        dataloader, 
        metadata, 
        evaluators,
        rank=RANK, 
        world_size=WORLD_SIZE, 
        cpu_group=CPU_PROCESS_GROUP
    )

    if RANK == 0:
        # Display one sample
        if sample_data is not None:
            print("\n")
            print("Sample Data (First Batch)")
            print(f"Set: {sample_data['set_name']}")
            print("0 : batch padding token")
            print("1 : empty token (need to be filled by model)")
            print("2~10 : 1~9")
            print()
            
            input_data = sample_data['inputs'].numpy()
            label_data = sample_data['labels'].numpy()
            
            print(f"Input:\n  {input_data} \n")
            print(f"Label:\n  {label_data} \n")
            
            if sample_data['preds'] is not None:
                pred_data = sample_data['preds'].numpy()
                print(f"Preds:\n  {pred_data} \n")
                
                # Calculate accuracy for this sample
                mask = label_data != -100
                if mask.sum() > 0:
                    correct = (pred_data == label_data) & mask
                    sample_acc = correct.sum() / mask.sum()
                    print(f"\nSample Accuracy: {sample_acc*100:.2f}%")
        
        print("\n")
        print("Test Results")

        if metrics:
            for set_name, set_metrics in metrics.items():
                print(f"\nDataset: {set_name}")
                
                # Main accuracy metrics
                if 'accuracy' in set_metrics:
                    print(f"Token Accuracy:    {set_metrics['accuracy']*100:6.2f}%  (per-token correctness)")
                if 'exact_accuracy' in set_metrics:
                    print(f"Exact Accuracy: {set_metrics['exact_accuracy']*100:6.2f}%  (entire sequence correct)")
                
                print()
                
                # Inference metrics
                if 'steps' in set_metrics:
                    print(f"Avg Inference Steps: {set_metrics['steps']:.1f}")
                if 'q_halt_accuracy' in set_metrics:
                    print(f"Q-Halt Accuracy:     {set_metrics['q_halt_accuracy']*100:6.2f}%")
                
                print()
                
                # Loss metrics
                if 'lm_loss' in set_metrics:
                    print(f"Language Model Loss: {set_metrics['lm_loss']:.4f}")
                if 'q_halt_loss' in set_metrics:
                    print(f"Q-Halt Loss:         {set_metrics['q_halt_loss']:.6f}")

    # finalize
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
