"""Figure 3 and the Section 4.3 numbers: shrinkage vs rotation of the forget-loss drop per layer.

Reads sweep.csv written by ``python -m rmu.layer_sweep`` (default: the shipped results) and
prints, per layer, the shrinkage term (1 - k^2)/2, the rotation term k cos(a, u), the rotation
share and the forget-retain gaps.

    python scripts/make_figure3.py                                       # shipped sweep.csv
    python scripts/make_figure3.py --sweep runs/layer_sweep/sweep.csv --out figures/figure3.pdf
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from rmu.plots import plot_layer_sweep


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sweep", type=Path, default=Path("results/layer_sweep/sweep.csv"))
    parser.add_argument("--out", type=Path, default=Path("figures/figure3.pdf"))
    args = parser.parse_args()

    sweep = pd.read_csv(args.sweep).sort_values("layer").reset_index(drop=True)
    for prefix in ("forget", "retain"):
        terms = sweep[f"{prefix}_shrinkage"] + sweep[f"{prefix}_rotation"]
        sweep[f"{prefix}_rot_share"] = sweep[f"{prefix}_rotation"] / terms
    sweep["dk (R-F)"] = sweep.retain_k - sweep.forget_k
    sweep["dcos_u (F-R)"] = sweep.forget_cos_u - sweep.retain_cos_u
    sweep["drot_share (F-R)"] = sweep.forget_rot_share - sweep.retain_rot_share

    columns = [
        "layer", "forget_k", "forget_cos_u", "forget_drop", "forget_shrinkage",
        "forget_rotation", "forget_rot_share", "dk (R-F)", "dcos_u (F-R)", "drot_share (F-R)",
    ]  # fmt: skip
    print(sweep[columns].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_layer_sweep(sweep, save_to=args.out)
    plt.close(fig)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
