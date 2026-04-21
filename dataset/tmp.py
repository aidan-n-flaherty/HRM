from typing import Optional
import os
import json
import numpy as np
from argdantic import ArgParser
from pydantic import BaseModel
from pathlib import Path
from tqdm import tqdm

cli = ArgParser()

SEQ_LEN          = 1024   # 4 frames × 256 patch tokens
DIM              = 1024   # V-JEPA 2 ViT-L hidden dim
TOKENS_PER_FRAME = 256
NUM_FRAMES       = 4


class DataProcessConfig(BaseModel):
    encoded_dir: str = "../encoded_dataset"
    output_dir:  str = "data/encoded_dataset"
    train_size:  int = 64
    test_size:   int = 10
    num_aug:     int = 0
    seed:        int = 42


class PuzzleDatasetMetadata(BaseModel):
    seq_len:                int
    dim:                    int
    pad_id:                 int
    ignore_label_id:        int
    blank_identifier_id:    int
    num_puzzle_identifiers: int
    total_groups:           int
    mean_puzzle_examples:   float
    sets:                   list[str]
    tokens_per_frame:       int
    num_frames:             int


def load_example(path: Path) -> Optional[np.ndarray]:
    """
    Load one .npz file and return a (4, 256, 1024) float32 array
    by stacking the four per-image token grids.

    Expected keys: start_encoding, frame_1_encoding,
                   frame_2_encoding, end_encoding
    Each value shape: (256, 1024)
    Returns None if any key is missing or shapes are wrong.
    """
    data = np.load(path, allow_pickle=True)
    keys = ['start_encoding', 'frame_1_encoding',
            'frame_2_encoding', 'end_encoding']
    try:
        frames = [data[k] for k in keys]
    except KeyError as e:
        print(f"  WARNING: {path.name} missing key {e}, skipping.")
        return None

    stacked = np.stack(frames, axis=0).astype(np.float32)  # (4, 256, 1024)
    if stacked.shape != (NUM_FRAMES, TOKENS_PER_FRAME, DIM):
        print(f"  WARNING: {path.name} unexpected shape {stacked.shape}, skipping.")
        return None

    return stacked


def make_input_label(seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Args:
        seq: (4, 256, 1024)  — full token grids for all 4 frames

    Returns:
        inputs: (1024, 1024) — frames 1 & 2 zeroed out, then concatenated
        labels: (1024, 1024) — full sequence concatenated
    """
    inp = seq.copy()
    inp[1] = 0.0   # zero all 256 tokens for frame_1
    inp[2] = 0.0   # zero all 256 tokens for frame_2

    # Flatten frame dimension: (4, 256, 1024) -> (1024, 1024)
    inputs = inp.reshape(SEQ_LEN, DIM)
    labels = seq.reshape(SEQ_LEN, DIM)
    return inputs, labels


def convert_subset(
    set_name:  str,
    npz_files: list[Path],
    config:    DataProcessConfig,
):
    num_augments = config.num_aug if set_name == "train" else 0

    all_inputs         = []
    all_labels         = []
    puzzle_indices     = [0]
    group_indices      = [0]
    puzzle_identifiers = []
    example_id = 0
    puzzle_id  = 0

    for fpath in tqdm(npz_files, desc=set_name):
        seq = load_example(fpath)
        if seq is None:
            continue

        orig_inp, orig_out = make_input_label(seq)

        for aug_idx in range(1 + num_augments):
            if aug_idx == 0:
                inp, out = orig_inp, orig_out
            else:
                inp, out = make_input_label(seq)

            all_inputs.append(inp)
            all_labels.append(out)
            example_id += 1

        puzzle_id += 1
        puzzle_indices.append(example_id)
        puzzle_identifiers.append(0)
        group_indices.append(puzzle_id)

    inputs_arr = np.stack(all_inputs, axis=0)   # (N, 1024, 1024)
    labels_arr = np.stack(all_labels, axis=0)   # (N, 1024, 1024)

    results = {
        "inputs":             inputs_arr,
        "labels":             labels_arr,
        "group_indices":      np.array(group_indices,      dtype=np.int32),
        "puzzle_indices":     np.array(puzzle_indices,     dtype=np.int32),
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
        tokens_per_frame=TOKENS_PER_FRAME,
        num_frames=NUM_FRAMES,
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

    encoded_dir = Path(config.encoded_dir)
    all_files   = sorted(encoded_dir.glob("example_*.npz"))
    if not all_files:
        raise FileNotFoundError(f"No example_*.npz found in {encoded_dir}")

    print(f"Found {len(all_files)} encoded examples.")

    rng      = np.random.default_rng(config.seed)
    shuffled = rng.permutation(len(all_files))

    train_end = config.train_size
    test_end  = config.train_size + config.test_size
    if test_end > len(all_files):
        raise ValueError(
            f"Requested {test_end} examples but only {len(all_files)} available."
        )

    train_files = [all_files[i] for i in shuffled[:train_end]]
    test_files  = [all_files[i] for i in shuffled[train_end:test_end]]

    convert_subset("train", train_files, config)
    convert_subset("test",  test_files,  config)


if __name__ == "__main__":
    cli()