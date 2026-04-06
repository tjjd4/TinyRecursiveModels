"""
Split test set into correct/incorrect puzzles, save in PuzzleDataset-compatible format.
Outputs are loadable by puzzle_dataset.py.

Output structure:
    {save_dir}/
        correct/test/   dataset.json  all__inputs.npy  all__labels.npy  ...
        incorrect/test/ dataset.json  all__inputs.npy  all__labels.npy  ...

Usage:
    python split_dataset.py arch=trm_same_input_trace \\
        load_checkpoint=checkpoints/.../model.pt \\
        data_paths_test='[data/sudoku-extreme-1k-aug-1000]' \\
        save_dir=data/split/my_run
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import math
from typing import List
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import tqdm
import hydra
import pydantic
from omegaconf import DictConfig

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class, load_checkpoint_from_path
from models.losses.loss_fn import IGNORE_LABEL_ID


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str
    loss: LossConfig


class SplitConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    arch: ArchConfig
    eval_loss: LossConfig
    load_checkpoint: str
    data_paths: List[str]
    data_paths_test: List[str] = []
    global_batch_size: int
    seed: int
    save_dir: str


def _save_split(out_dir: str, inputs: list, labels: list, puzzle_identifiers: list, metadata_dict: dict) -> None:
    N = len(inputs)
    os.makedirs(out_dir, exist_ok=True)

    arrays = {
        "inputs":             np.stack(inputs).astype(np.int32),
        "labels":             np.stack(labels).astype(np.int32),
        "puzzle_identifiers": np.array(puzzle_identifiers, dtype=np.int32),
        "puzzle_indices":     np.arange(N + 1, dtype=np.int32),
        "group_indices":      np.arange(N + 1, dtype=np.int32),
    }

    for k, v in arrays.items():
        np.save(os.path.join(out_dir, f"all__{k}.npy"), v)

    with open(os.path.join(out_dir, "dataset.json"), "w") as f:
        json.dump(metadata_dict, f, indent=2)

    print(f"[split] Saved {N} samples → {out_dir}")


@dataclass
class SplitCollector:
    ignore_label_id: int

    inputs_correct:              List[np.ndarray] = field(default_factory=list)
    labels_correct:              List[np.ndarray] = field(default_factory=list)
    puzzle_identifiers_correct:  List[int]        = field(default_factory=list)
    inputs_incorrect:            List[np.ndarray] = field(default_factory=list)
    labels_incorrect:            List[np.ndarray] = field(default_factory=list)
    puzzle_identifiers_incorrect: List[int]       = field(default_factory=list)
    n_skipped: int = 0

    def process_batch(self, batch_inputs_np: np.ndarray, preds_np: np.ndarray,
                      labels_np: np.ndarray, puzzle_identifiers_np: np.ndarray) -> None:
        # Mirror act_loss.py: mask = empty cells, loss_counts = # cells to evaluate
        mask        = (labels_np != IGNORE_LABEL_ID)
        loss_counts = mask.sum(-1)
        is_correct  = mask & (preds_np == labels_np)        # per-position
        seq_is_correct = is_correct.sum(-1) == loss_counts  # full-board correctness

        labels_raw = labels_np.copy()
        labels_raw[labels_raw == IGNORE_LABEL_ID] = self.ignore_label_id

        for b in range(batch_inputs_np.shape[0]):
            if loss_counts[b] == 0:
                self.n_skipped += 1
                continue
            puzzle_id = int(puzzle_identifiers_np[b])
            if seq_is_correct[b]:
                self.inputs_correct.append(batch_inputs_np[b])
                self.labels_correct.append(labels_raw[b])
                self.puzzle_identifiers_correct.append(puzzle_id)
            else:
                self.inputs_incorrect.append(batch_inputs_np[b])
                self.labels_incorrect.append(labels_raw[b])
                self.puzzle_identifiers_incorrect.append(puzzle_id)

    @property
    def n_correct(self): return len(self.inputs_correct)

    @property
    def n_incorrect(self): return len(self.inputs_incorrect)

    def _make_metadata_dict(self, dataset_metadata: PuzzleDatasetMetadata, N: int) -> dict:
        return dict(
            seq_len=dataset_metadata.seq_len,
            vocab_size=dataset_metadata.vocab_size,
            pad_id=dataset_metadata.pad_id,
            ignore_label_id=dataset_metadata.ignore_label_id,
            blank_identifier_id=dataset_metadata.blank_identifier_id,
            num_puzzle_identifiers=1,
            total_groups=N,
            mean_puzzle_examples=1.0,
            total_puzzles=N,
            sets=["all"],
        )

    def save(self, save_dir: str, dataset_metadata: PuzzleDatasetMetadata) -> None:
        for split_name, inputs, labels, puzzle_identifiers in [
            ("correct",   self.inputs_correct,   self.labels_correct,   self.puzzle_identifiers_correct),
            ("incorrect", self.inputs_incorrect, self.labels_incorrect, self.puzzle_identifiers_incorrect),
        ]:
            _save_split(
                out_dir=os.path.join(save_dir, split_name, "test"),
                inputs=inputs,
                labels=labels,
                puzzle_identifiers=puzzle_identifiers,
                metadata_dict=self._make_metadata_dict(dataset_metadata, len(inputs)),
            )
        print(f"[split] correct={self.n_correct}  incorrect={self.n_incorrect}  skipped={self.n_skipped}")


def create_dataloader(config: SplitConfig, metadata_out: list):
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,
        dataset_paths=config.data_paths_test if config.data_paths_test else config.data_paths,
        rank=0,
        num_replicas=1,
        test_set_mode=True,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
    ), split="test")
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=8,
        pin_memory=True,
        persistent_workers=True,
    )
    metadata_out.append(dataset.metadata)
    return dataloader


def load_model_from_checkpoint(config: SplitConfig, metadata) -> nn.Module:
    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore
        batch_size=config.global_batch_size,
        vocab_size=metadata.vocab_size,
        seq_len=metadata.seq_len,
        num_puzzle_identifiers=metadata.num_puzzle_identifiers,
        causal=False,
    )
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.eval_loss.name)

    with torch.device("cuda"):
        model: nn.Module = model_cls(model_cfg)
        model = loss_head_cls(model, **config.eval_loss.__pydantic_extra__)  # type: ignore
        if "DISABLE_COMPILE" not in os.environ:
            print("Compiling model")
            model = torch.compile(model)  # type: ignore
        else:
            print("Skipping torch.compile")
        if not os.path.exists(config.load_checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {config.load_checkpoint}")
        load_checkpoint_from_path(model, config.load_checkpoint)

    return model


def run(config: SplitConfig, collector,
        dataloader: DataLoader, metadata) -> None:
    model = load_model_from_checkpoint(config, metadata)
    model.eval()

    total_steps = math.ceil(
        metadata.total_puzzles * metadata.mean_puzzle_examples / config.global_batch_size
    )
    progress_bar = tqdm.tqdm(total=total_steps, desc="Splitting", unit="batch")

    samples_seen = 0
    with torch.inference_mode():
        for _set_name, batch, global_batch_size in dataloader:
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.device("cuda"):
                carry = model.initial_carry(batch)

            while True:
                carry, loss, metrics, preds, all_finish = model(
                    carry=carry, batch=batch, return_keys={"preds"})
                if all_finish:
                    break

            collector.process_batch(
                batch_inputs_np=batch["inputs"].cpu().numpy(),
                preds_np=preds["preds"].cpu().numpy(),
                labels_np=carry.current_data["labels"].cpu().numpy(),
                puzzle_identifiers_np=batch["puzzle_identifiers"].cpu().numpy(),
            )

            samples_seen += global_batch_size
            progress_bar.update(1)
            progress_bar.set_postfix(correct=collector.n_correct, incorrect=collector.n_incorrect)
            del carry, loss, metrics, preds, batch, all_finish

    progress_bar.close()
    print(f"\n[split] Done. Processed {samples_seen} samples total.")

    # data/split/{dataset_name}/correct|incorrect/test/
    data_paths = config.data_paths_test if len(config.data_paths_test) > 0 else config.data_paths
    dataset_name = os.path.basename(data_paths[0].rstrip("/"))
    save_dir = os.path.join(config.save_dir, dataset_name)
    collector.save(save_dir, metadata)


@hydra.main(config_path="config", config_name="cfg_split_dataset", version_base=None)
def main(hydra_config: DictConfig):
    config = SplitConfig(**hydra_config)
    metadata_out: list = []
    dataloader = create_dataloader(config, metadata_out)
    collector = SplitCollector(ignore_label_id=metadata_out[0].ignore_label_id or 0)
    run(config, collector, dataloader, metadata_out[0])


if __name__ == "__main__":
    main()
