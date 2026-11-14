"""Download and preprocess the NCT-CRC-HE histology dataset once, before launching parallel jobs.

Downloads ``CRC-VAL-HE-7K.zip`` (test, 0.8 GB) and ``NCT-CRC-HE-100K.zip`` (train, 11.7 GB) from
https://zenodo.org/records/1214456 into ``<data_root>/NCT_CRC_HE/raw`` (md5-checked; skipped when already
present) and writes the decoded, optionally resized, uint8 arrays to ``<data_root>/NCT_CRC_HE/processed``
(~15 GB for the train split at the native 224 px).  See :mod:`ebnn.histology`.

Example:
    python scripts/prepare_nct_crc_he.py                     # native 224 px
    python scripts/prepare_nct_crc_he.py --image_size 64     # one array set per image size
"""

import argparse

import numpy as np

from ebnn.histology import DEFAULT_IMAGE_SIZE, NCTCRCHE


def main():
    p = argparse.ArgumentParser(description="Download + preprocess NCT-CRC-HE (zenodo.org/records/1214456)")
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument(
        "--image_size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help=f"Patch side length after resizing (default {DEFAULT_IMAGE_SIZE} = native, no resizing).",
    )
    args = p.parse_args()

    for train in (False, True):  # small test split first, so problems surface early
        ds = NCTCRCHE(root=args.data_root, train=train, download=True, image_size=args.image_size)
        counts = np.bincount(np.asarray(ds.targets), minlength=len(ds.classes))
        print(
            f"{ds.split}: {len(ds)} patches, array {tuple(ds.data.shape)} {ds.data.dtype} at {ds.images_path}\n"
            "  per class: " + ", ".join(f"{c}={n}" for c, n in zip(ds.classes, counts))
        )


if __name__ == "__main__":
    main()
