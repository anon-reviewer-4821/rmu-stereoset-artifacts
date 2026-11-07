"""Reduce the original StereoSet files to the subset used in the paper.

Keeps only the intrasentence items with ``bias_type == "gender"`` (771 in test, 255 in dev,
1,026 in total), in their original order, so the seeded shuffle in ``rmu.data`` produces exactly
the same batches as with the full files.

    python scripts/prepare_stereoset.py --src /path/to/stereoset --dst data/stereoset
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--src", type=Path, required=True, help="Folder with test.json and dev.json")
    parser.add_argument("--dst", type=Path, default=Path("data/stereoset"))
    args = parser.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)
    for split in ("test", "dev"):
        with open(args.src / f"{split}.json", encoding="utf-8") as f:
            full = json.load(f)
        items = [x for x in full["data"]["intrasentence"] if x["bias_type"] == "gender"]
        subset = {"version": full["version"], "data": {"intrasentence": items}}
        with open(args.dst / f"{split}.json", "w", encoding="utf-8") as f:
            json.dump(subset, f, ensure_ascii=False)
        print(f"{split}: {len(items)} gender intrasentence items -> {args.dst / f'{split}.json'}")


if __name__ == "__main__":
    main()
