"""
Multi-run consistency check: evaluate the same dataset N times and compare
which puzzle IDs are correct / incorrect across runs.

Output (saved to save_dir/):
  consistency_results.json  —  per-puzzle run-level breakdown + aggregate stats
  consistency_summary.txt   —  human-readable summary

The script is intentionally single-GPU and single-process (no torchrun needed).
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from typing import List, Dict, Set
from collections import defaultdict

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


class ConsistencyConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    arch: ArchConfig
    eval_loss: LossConfig
    load_checkpoint: str
    data_paths: List[str]
    data_paths_test: List[str] = []
    global_batch_size: int
    seed: int = 0
    num_runs: int = 5
    save_dir: str = "logs/consistency_check"


def load_model(config: ConsistencyConfig, metadata: PuzzleDatasetMetadata) -> nn.Module:
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
            print("Compiling model...")
            model = torch.compile(model)  # type: ignore
        else:
            print("Skipping torch.compile")

        if not os.path.exists(config.load_checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {config.load_checkpoint}")
        load_checkpoint_from_path(model, config.load_checkpoint)

    return model


def create_dataloader(config: ConsistencyConfig) -> tuple:
    data_paths = config.data_paths_test if config.data_paths_test else config.data_paths
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,
        dataset_paths=data_paths,
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
    print(f"Dataset: {data_paths}")
    print(f"Total puzzles: {dataset.metadata.total_puzzles}")
    print()
    return dataloader, dataset.metadata


def run_single_eval(
    model: nn.Module,
    dataloader: DataLoader,
    run_idx: int,
) -> tuple:
    """Return (set of correct sample indices, set of incorrect sample indices) for one pass.

    NOTE: puzzle_identifiers in the sudoku dataset are all 0, so they cannot be
    used as unique puzzle IDs.  Instead we use a sequential counter (sample_idx)
    that matches the deterministic iteration order produced by test_set_mode=True.
    As long as the dataloader seed and config are identical across runs the same
    index always refers to the same puzzle.
    """
    correct_ids: Set[int] = set()
    incorrect_ids: Set[int] = set()
    n_skipped = 0
    sample_idx = 0  # sequential puzzle index within this run

    progress = tqdm.tqdm(dataloader, desc=f"Run {run_idx + 1}", unit="batch")

    with torch.inference_mode():
        for _set_name, batch, _global_bs in progress:
            batch_gpu = {k: v.cuda() for k, v in batch.items()}
            with torch.device("cuda"):
                carry = model.initial_carry(batch_gpu)  # type: ignore

            while True:
                carry, _loss, _metrics, preds, all_finish = model(
                    carry=carry, batch=batch_gpu, return_keys={"preds"}
                )
                if all_finish:
                    break

            # Evaluate per-puzzle correctness
            preds_np = preds["preds"].cpu().numpy()                       # (B, seq_len)
            labels_np = carry.current_data["labels"].cpu().numpy()        # (B, seq_len)

            mask = labels_np != IGNORE_LABEL_ID        # cells to predict
            loss_counts = mask.sum(-1)                 # (B,)
            is_correct = mask & (preds_np == labels_np)
            seq_correct = is_correct.sum(-1) == loss_counts  # (B,) bool

            for b in range(preds_np.shape[0]):
                if loss_counts[b] == 0:
                    n_skipped += 1
                    sample_idx += 1
                    continue
                if seq_correct[b]:
                    correct_ids.add(sample_idx)
                else:
                    incorrect_ids.add(sample_idx)
                sample_idx += 1

            del carry, _loss, _metrics, preds, batch_gpu, all_finish
            progress.set_postfix(
                correct=len(correct_ids),
                incorrect=len(incorrect_ids),
                skipped=n_skipped,
            )

    progress.close()
    print(
        f"  Run {run_idx + 1}: correct={len(correct_ids)}  "
        f"incorrect={len(incorrect_ids)}  skipped={n_skipped}"
    )
    return correct_ids, incorrect_ids


def analyse(
    all_correct: List[Set[int]],
    all_incorrect: List[Set[int]],
    num_runs: int,
) -> dict:
    """Compute consistency statistics across runs."""
    all_puzzle_ids: Set[int] = set()
    for s in all_correct + all_incorrect:
        all_puzzle_ids.update(s)

    # Per-puzzle: how many runs was it correct?
    puzzle_correct_count: Dict[int, int] = defaultdict(int)
    for correct_set in all_correct:
        for pid in correct_set:
            puzzle_correct_count[pid] += 1

    always_correct   = {pid for pid, cnt in puzzle_correct_count.items() if cnt == num_runs}
    always_incorrect = {pid for pid in all_puzzle_ids if puzzle_correct_count[pid] == 0}
    inconsistent     = all_puzzle_ids - always_correct - always_incorrect

    # Jaccard similarity between run pairs (on incorrect sets)
    jaccard_scores = []
    for i in range(num_runs):
        for j in range(i + 1, num_runs):
            a, b = all_incorrect[i], all_incorrect[j]
            union = a | b
            if union:
                jaccard_scores.append(len(a & b) / len(union))

    mean_jaccard = float(np.mean(jaccard_scores)) if jaccard_scores else 1.0

    # Per-puzzle breakdown (sorted by puzzle ID)
    per_puzzle = {
        int(pid): int(puzzle_correct_count.get(pid, 0))
        for pid in sorted(all_puzzle_ids)
    }

    return dict(
        num_runs=num_runs,
        total_puzzles=len(all_puzzle_ids),
        always_correct_count=len(always_correct),
        always_incorrect_count=len(always_incorrect),
        inconsistent_count=len(inconsistent),
        mean_jaccard_incorrect=round(mean_jaccard, 6),
        always_correct_ids=sorted(always_correct),
        always_incorrect_ids=sorted(always_incorrect),
        inconsistent_ids=sorted(inconsistent),
        per_puzzle_correct_count=per_puzzle,   # pid -> number of runs it was correct
        run_correct_counts=[len(s) for s in all_correct],
        run_incorrect_counts=[len(s) for s in all_incorrect],
    )


def print_summary(results: dict) -> str:
    n = results["num_runs"]
    total = results["total_puzzles"]
    lines = [
        "=" * 60,
        "CONSISTENCY CHECK SUMMARY",
        "=" * 60,
        f"Runs:                {n}",
        f"Total puzzles seen:  {total}",
        "",
        f"Always correct    ({n}/{n} runs):  {results['always_correct_count']:>6}  "
        f"({100*results['always_correct_count']/total:.1f}%)",
        f"Always incorrect  (0/{n} runs):  {results['always_incorrect_count']:>6}  "
        f"({100*results['always_incorrect_count']/total:.1f}%)",
        f"Inconsistent               :  {results['inconsistent_count']:>6}  "
        f"({100*results['inconsistent_count']/total:.1f}%)",
        "",
        f"Mean Jaccard (incorrect sets):  {results['mean_jaccard_incorrect']:.4f}",
        "  (1.0 = perfectly consistent, 0.0 = completely random)",
        "",
        "Per-run accuracy:",
    ]
    for i, (nc, ni) in enumerate(zip(results["run_correct_counts"], results["run_incorrect_counts"])):
        acc = 100 * nc / (nc + ni) if (nc + ni) > 0 else 0.0
        lines.append(f"  Run {i+1}: {nc} correct / {ni} incorrect  ({acc:.2f}%)")
    lines.append("=" * 60)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(config_path="../config", config_name="cfg_eval", version_base=None)
def main(hydra_config: DictConfig):
    config = ConsistencyConfig(**hydra_config)

    os.makedirs(config.save_dir, exist_ok=True)

    print(f"Loading dataloader (seed={config.seed})...")
    dataloader, metadata = create_dataloader(config)

    print("Loading model...")
    model = load_model(config, metadata)
    model.eval()

    all_correct: List[Set[int]] = []
    all_incorrect: List[Set[int]] = []

    for run_idx in range(config.num_runs):
        correct_ids, incorrect_ids = run_single_eval(model, dataloader, run_idx)
        all_correct.append(correct_ids)
        all_incorrect.append(incorrect_ids)

    print("\nAnalysing consistency across runs...")
    results = analyse(all_correct, all_incorrect, config.num_runs)

    summary = print_summary(results)
    print("\n" + summary)

    # Save
    out_json = os.path.join(config.save_dir, "consistency_results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    out_txt = os.path.join(config.save_dir, "consistency_summary.txt")
    with open(out_txt, "w") as f:
        f.write(summary + "\n")

    print(f"\nSaved results to {config.save_dir}/")


if __name__ == "__main__":
    main()
