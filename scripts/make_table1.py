"""Table 1: forget (F) and retain (R) activations of adaptive RMU, last 30 of 129 steps.

Reads the losses.csv of the three runs (default: the logs shipped in results/runs/) and prints
k, cos(a, u) for F and R, and the forget-minus-retain cosine to the frozen model. The paired
(no-split) run has 257 steps; it is cut to the first 129 so that all rows cover the same steps.
Also prints the predicted optimum of Section 4.2 (Eq. 5) and the rescale-only value.

    python scripts/make_table1.py                    # from the shipped logs
    python scripts/make_table1.py --runs_dir runs    # from new runs (runs/<name>/losses.csv)
"""

import argparse
import math
from pathlib import Path

import pandas as pd

RUNS = {"L7, paired": "L7_paired", "L7, split": "L7_split", "L32, split": "L32_split"}
STEPS, LAST = 129, 30


def find_log(runs_dir: Path, name: str) -> Path:
    for path in (runs_dir / f"{name}.csv", runs_dir / name / "losses.csv"):
        if path.exists():
            return path
    raise FileNotFoundError(f"no {name}.csv or {name}/losses.csv in {runs_dir}")


def row(log: pd.DataFrame) -> dict:
    late = log.iloc[:STEPS].iloc[-LAST:]
    return {
        "k F": late.forget_norm_ratio.mean(),
        "k R": late.retain_norm_ratio.mean(),
        "cos(a,u) F": late.forget_cos_u.mean(),
        "cos(a,u) R": late.retain_cos_u.mean(),
        "dcos(a,a0)": (late.unlearn_cosine_sim - late.retain_cosine_sim).mean(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs_dir", type=Path, default=Path("results/runs"))
    parser.add_argument("--alpha", type=float, default=5.0)
    args = parser.parse_args()

    table = pd.DataFrame(
        {run: row(pd.read_csv(find_log(args.runs_dir, name))) for run, name in RUNS.items()}
    ).T
    print(table.to_string(float_format=lambda v: f"{v:.4f}"))

    a = args.alpha
    print(
        f"\nPredicted optimum (Eq. 5, alpha={a:g}): "
        f"k = {math.sqrt(a**2 + 1) / (1 + a):.3f}, cos(a,u) = {1 / math.sqrt(a**2 + 1):.3f}"
    )
    print(f"Rescale only: k = {a / (1 + a):.3f}, cos(a,u) = 0")


if __name__ == "__main__":
    main()
