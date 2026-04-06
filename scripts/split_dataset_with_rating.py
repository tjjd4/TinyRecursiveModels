import json
import os
import sys
# add root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Optional
from dataclasses import dataclass, field

import numpy as np
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig
import math
import torch
import tqdm

from puzzle_dataset_with_rating import PuzzleDataset, PuzzleDatasetConfig
from dataset.build_sudoku_dataset_with_rating import PuzzleDatasetMetadataWithRating
from models.losses.loss_fn import IGNORE_LABEL_ID

from scripts.split_dataset import SplitConfig, load_model_from_checkpoint


def _save_split_with_rating(out_dir: str, inputs: list, labels: list,
                             puzzle_identifiers: list, ratings: list,
                             metadata_dict: dict) -> None:
    N = len(inputs)
    os.makedirs(out_dir, exist_ok=True)

    arrays = {
        "inputs":             np.stack(inputs).astype(np.int32),
        "labels":             np.stack(labels).astype(np.int32),
        "puzzle_identifiers": np.array(puzzle_identifiers, dtype=np.int32),
        "puzzle_indices":     np.arange(N + 1, dtype=np.int32),
        "group_indices":      np.arange(N + 1, dtype=np.int32),
        "ratings":            np.array(ratings, dtype=np.int32),
    }

    for k, v in arrays.items():
        np.save(os.path.join(out_dir, f"all__{k}.npy"), v)

    with open(os.path.join(out_dir, "dataset.json"), "w") as f:
        json.dump(metadata_dict, f, indent=2)

    print(f"[split] Saved {N} samples → {out_dir}")


@dataclass
class SplitCollectorWithRating:
    ignore_label_id: int

    inputs_correct:              List[np.ndarray] = field(default_factory=list)
    labels_correct:              List[np.ndarray] = field(default_factory=list)
    ratings_correct:             List[int]        = field(default_factory=list)
    puzzle_identifiers_correct:  List[int]        = field(default_factory=list)
    inputs_incorrect:            List[np.ndarray] = field(default_factory=list)
    labels_incorrect:            List[np.ndarray] = field(default_factory=list)
    ratings_incorrect:           List[int]        = field(default_factory=list)
    puzzle_identifiers_incorrect: List[int]       = field(default_factory=list)
    n_skipped: int = 0

    @property
    def n_correct(self): return len(self.inputs_correct)

    @property
    def n_incorrect(self): return len(self.inputs_incorrect)

    def process_batch(self, batch_inputs_np: np.ndarray, preds_np: np.ndarray,
                      labels_np: np.ndarray, ratings_np: Optional[np.ndarray] = None,
                      puzzle_identifiers_np: Optional[np.ndarray] = None) -> None:

        mask = (labels_np != IGNORE_LABEL_ID)
        loss_counts = mask.sum(-1)
        is_correct = mask & (preds_np == labels_np)
        seq_is_correct = is_correct.sum(-1) == loss_counts

        labels_raw = labels_np.copy()
        labels_raw[labels_raw == IGNORE_LABEL_ID] = self.ignore_label_id

        for b in range(batch_inputs_np.shape[0]):
            if not loss_counts[b] > 0:
                self.n_skipped += 1
                continue
            rating = int(ratings_np[b]) if ratings_np is not None else -1
            puzzle_id = int(puzzle_identifiers_np[b]) if puzzle_identifiers_np is not None else 0
            if seq_is_correct[b]:
                self.inputs_correct.append(batch_inputs_np[b])
                self.labels_correct.append(labels_raw[b])
                self.ratings_correct.append(rating)
                self.puzzle_identifiers_correct.append(puzzle_id)
            else:
                self.inputs_incorrect.append(batch_inputs_np[b])
                self.labels_incorrect.append(labels_raw[b])
                self.ratings_incorrect.append(rating)
                self.puzzle_identifiers_incorrect.append(puzzle_id)

    def save(self, save_dir: str, dataset_metadata: PuzzleDatasetMetadataWithRating) -> None:
        for split_name, inputs, labels, ratings, puzzle_identifiers in [
            ("correct",   self.inputs_correct,   self.labels_correct,   self.ratings_correct,   self.puzzle_identifiers_correct),
            ("incorrect", self.inputs_incorrect, self.labels_incorrect, self.ratings_incorrect, self.puzzle_identifiers_incorrect),
        ]:
            N = len(inputs)
            metadata_dict = {
                **dataset_metadata.model_dump(),
                "num_puzzle_identifiers": 1,
                "total_groups": N,
                "mean_puzzle_examples": 1.0,
                "total_puzzles": N,
                "sets": ["all"],
            }
            _save_split_with_rating(
                out_dir=os.path.join(save_dir, split_name, "test"),
                inputs=inputs,
                labels=labels,
                puzzle_identifiers=puzzle_identifiers,
                ratings=ratings,
                metadata_dict=metadata_dict,
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


def run_with_ratings(config: SplitConfig, collector: SplitCollectorWithRating,
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

            ratings_np = batch["ratings"].cpu().numpy() if "ratings" in batch else None
            puzzle_identifiers_np = batch["puzzle_identifiers"].cpu().numpy() if "puzzle_identifiers" in batch else None
            collector.process_batch(
                batch_inputs_np=batch["inputs"].cpu().numpy(),
                preds_np=preds["preds"].cpu().numpy(),
                labels_np=carry.current_data["labels"].cpu().numpy(),
                ratings_np=ratings_np,
                puzzle_identifiers_np=puzzle_identifiers_np,
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


@hydra.main(config_path="../config", config_name="cfg_split_dataset", version_base=None)
def main(hydra_config: DictConfig):
    config = SplitConfig(**hydra_config)
    metadata_out: list = []
    dataloader = create_dataloader(config, metadata_out)
    collector = SplitCollectorWithRating(ignore_label_id=metadata_out[0].ignore_label_id or 0)
    run_with_ratings(config, collector, dataloader, metadata_out[0])


if __name__ == "__main__":
    main()
