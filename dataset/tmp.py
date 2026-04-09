from typing import Optional
import os
import json
import numpy as np

from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm


cli = ArgParser()

SEQ_LEN = 4
DIM = 1408


class DataProcessConfig(BaseModel):
    output_dir: str = "data/tmp_dataset"

    train_size: int = 10
    test_size: int = 10
    num_aug: int = 0

    seed: int = 42


class PuzzleDatasetMetadata(BaseModel):
    seq_len: int
    dim: int

    pad_id: int
    ignore_label_id: int

    blank_identifier_id: int
    num_puzzle_identifiers: int

    total_groups: int
    mean_puzzle_examples: float
    sets: list[str]


def generate_sequence() -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a single example: a sequence of 4 float32 vectors of shape (4, DIM).
    Returns:
        inputs:  (4, DIM) — first and last tokens filled, middle two zeroed out
        labels:  (4, DIM) — full sequence (all 4 tokens)
    """
    seq = np.random.randn(SEQ_LEN, DIM).astype(np.float32)

    inputs = seq.copy()
    inputs[1] = 0.0
    inputs[2] = 0.0

    return inputs, seq


def convert_subset(set_name: str, config: DataProcessConfig):
    size = config.train_size if set_name == "train" else config.test_size
    num_augments = config.num_aug if set_name == "train" else 0

    all_inputs = []
    all_labels = []
    puzzle_indices = [0]
    group_indices = [0]
    puzzle_identifiers = []

    example_id = 0
    puzzle_id = 0

    for _ in tqdm(range(size), desc=set_name):
        orig_inp, orig_out = generate_sequence()

        for aug_idx in range(1 + num_augments):
            if aug_idx == 0:
                inp, out = orig_inp, orig_out
            else:
                inp, out = generate_sequence()

            all_inputs.append(inp)
            all_labels.append(out)
            example_id += 1
            puzzle_id += 1

            puzzle_indices.append(example_id)
            puzzle_identifiers.append(0)

        group_indices.append(puzzle_id)

    inputs_arr = np.stack(all_inputs, axis=0)
    labels_arr = np.stack(all_labels, axis=0)

    results = {
        "inputs": inputs_arr,
        "labels": labels_arr,
        "group_indices": np.array(group_indices, dtype=np.int32),
        "puzzle_indices": np.array(puzzle_indices, dtype=np.int32),
        "puzzle_identifiers": np.array(puzzle_identifiers, dtype=np.int32),
    }

    metadata = PuzzleDatasetMetadata(
        seq_len=SEQ_LEN,
        dim=DIM,

        pad_id=0,
        ignore_label_id=0,

        blank_identifier_id=0,
        num_puzzle_identifiers=1,

        total_groups=len(group_indices) - 1,
        mean_puzzle_examples=1.0 + num_augments,
        sets=["all"],
    )

    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f, indent=2)

    for k, v in results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)
        print(f"  saved {k}: {v.shape} {getattr(v, 'dtype', '')}")

    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    np.random.seed(config.seed)
    convert_subset("train", config)
    convert_subset("test", config)


if __name__ == "__main__":
    cli()