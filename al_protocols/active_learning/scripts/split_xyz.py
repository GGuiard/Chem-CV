"""
Train/test dataset split
"""

import argparse
import random
from pathlib import Path

import ase.io


def split_dataset(path_input="dft.xyz", outdir=".", train_split=0.8):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dataset = ase.io.read(str(path_input), ":")
    random.shuffle(dataset)
    n_train = int(len(dataset) * train_split)

    ase.io.write(str(outdir / "train.xyz"), dataset[:n_train])
    ase.io.write(str(outdir / "test.xyz"),  dataset[n_train:])


def _parser():
    p = argparse.ArgumentParser(description="Split DFT dataset into train/test",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    p.add_argument("--input", default="dft.xyz", help="Input xyz")
    p.add_argument("--outdir", default=".", help="Output directory")
    p.add_argument("--train-split", type=float, default=0.8, help="Train fraction")

    return p


if __name__ == "__main__":
    args = _parser().parse_args()

    split_dataset(args.input, args.outdir, args.train_split)