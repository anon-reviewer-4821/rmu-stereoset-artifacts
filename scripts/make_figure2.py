"""Figure 2: norm ratio k and cos(a, u) over training, adaptive RMU, split data, layers 7 and 32.

    python scripts/make_figure2.py                                 # from the shipped logs
    python scripts/make_figure2.py --runs_dir runs --out figures/figure2.pdf
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from rmu.plots import FORGET, RETAIN

PANELS = {"Layer 7": "L7_split", "Layer 32": "L32_split"}
ROWS = [("norm_ratio", "norm ratio k"), ("cos_u", "cos(a, u)")]
SMOOTH = 5  # rolling-mean window, in steps


def find_log(runs_dir: Path, name: str) -> Path:
    for path in (runs_dir / f"{name}.csv", runs_dir / name / "losses.csv"):
        if path.exists():
            return path
    raise FileNotFoundError(f"no {name}.csv or {name}/losses.csv in {runs_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs_dir", type=Path, default=Path("results/runs"))
    parser.add_argument("--out", type=Path, default=Path("figures/figure2.pdf"))
    args = parser.parse_args()

    fig, axes = plt.subplots(2, 2, figsize=(5.2, 4.2), sharex=True, sharey="row")
    for col, (title, name) in enumerate(PANELS.items()):
        log = pd.read_csv(find_log(args.runs_dir, name))
        axes[0, col].set_title(title, fontsize=9)
        for r, (column, label) in enumerate(ROWS):
            ax = axes[r, col]
            for prefix, colour, style, legend in [
                ("forget", FORGET, "-", "forget (stereotype)"),
                ("retain", RETAIN, "--", "retain (anti-stereotype)"),
            ]:
                series = log[f"{prefix}_{column}"].rolling(SMOOTH, min_periods=1).mean()
                ax.plot(log.step, series, style, color=colour, lw=1.4, label=legend)
            ax.grid(color="#e4e3df", lw=0.6)
            ax.tick_params(labelsize=7)
            if col == 0:
                ax.set_ylabel(label, fontsize=8)
            if r == 1:
                ax.set_xlabel("step", fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=2, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
