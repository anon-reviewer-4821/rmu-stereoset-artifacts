# One Word Apart: Can Representation Unlearning Separate Stereotypes from Anti-Stereotypes?

Code, data and training logs for the paper. Adaptive RMU is applied to StereoSet gender
stereotype (forget) / anti-stereotype (retain) pairs in Lapa-12B-Instruct, and the change of the
activations is logged per step.

```
src/rmu/        RMU training (unlearn.py), layer sweep (layer_sweep.py), data, plotting
scripts/        Table 1, Figure 2, Figure 3 from the logs; StereoSet preprocessing
config/         paths and model short names (override in config/local.yaml)
data/stereoset/ StereoSet files go here (not included, see "Data" below)
results/        per-step logs of the paper's runs (runs/*.csv) and the layer sweep (sweep.csv)
tests/          unit tests (CPU, no model download)
```

## Environment and installation

Python 3.13. The paper's runs used one 96 GB GH200 GPU in bf16 (two copies of the 12B model,
~23 GB each, are kept in memory: frozen and updated).

Results were produced with PyTorch 2.9.1 (GH200 build); `requirements.txt` pins torch 2.14.1
for local development, with all other packages at the versions used for the paper.

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .          # editable install: paths are resolved from the repository root
pytest                    # 44 tests, a few seconds on CPU
```

**Model.** Lapa-12B-Instruct v0.1.2 (Paniv et al., 2026), Hub id
`lapa-llm/lapa-v0.1.2-instruct`, revision `2969fd997f07b33727c6f10795d78ef23cc5c037`:

```bash
hf download lapa-llm/lapa-v0.1.2-instruct --revision 2969fd997f07b33727c6f10795d78ef23cc5c037 \
    --local-dir models/lapa-v0.1.2-instruct
```

`--model_name_or_path lapa-12b-instruct` then resolves to `models/lapa-v0.1.2-instruct`
(change `paths.models_root` in `config/local.yaml` to keep models elsewhere); any local path or
Hub id also works.

**Data.** StereoSet (Nadeem et al., 2021; CC BY-SA 4.0) is not redistributed here. The runs
use the intrasentence items with `bias_type = gender` from both original JSON files
(`{"version": ..., "data": {"intrasentence": [...], "intersentence": [...]}}`):

| File | Version field | Gender intrasentence items | SHA-256 of the original file |
|---|---|---|---|
| `dev.json` | `1.0-dev` | 255 | `73a0f31b711688112602e4c3ac6ab1e1a7cadcdd67df6c6fd55501c889676c90` |
| `test.json` | `1.0-test` | 771 | `f3f70d1880b443388744958d192c25364d68c2b8ae7aeb9174903bcf415c394e` |

`dev.json` is part of the official StereoSet release by the dataset authors (the Hugging Face
copy contains the same validation split, but in a different format). The test split is kept
by the StereoSet authors for their leaderboard and is not part of the public Hugging Face
release; obtain `test.json` from the dataset authors. With both files in `/path/to/stereoset`:

```bash
python scripts/prepare_stereoset.py --src /path/to/stereoset --dst data/stereoset
```

This keeps the 1,026 gender items in their original order, so the seeded shuffle gives exactly
the batches of the paper's runs (the `forget_ids` column of `results/runs/*.csv` lists them).
Without `test.json` the training runs give different batches and numbers; Table 1 and Figures
2–3 can still be reproduced from the shipped logs, which need no data.

## Reproducing the paper

All runs use `--seed 42` (control vector u, torch) and a fixed item shuffle (seed 42 in
`rmu.data`). Each run writes `losses.csv` (one row per step) and `losses.png` to `--output_dir`.
GPU kernels are not bit-wise deterministic, so re-runs can differ in the third decimal.

The `scripts/make_*.py` commands below reproduce the tables and figures from the logs in
`results/` without a GPU. To use your own runs instead, pass `--runs_dir runs` or
`--sweep runs/layer_sweep/sweep.csv`.

### Table 1 and Figure 2 (Sections 4.1–4.2)

```bash
M=lapa-12b-instruct
COMMON="--model_name_or_path $M --loss_positions words --beta 1 --alpha 5 --lr 5e-4"
python -m rmu.unlearn $COMMON --no-split --layer_id 7  --layer_ids 5,6,7    --output_dir runs/L7_paired   # 257 steps
python -m rmu.unlearn $COMMON --split    --layer_id 7  --layer_ids 5,6,7    --output_dir runs/L7_split    # 129 steps
python -m rmu.unlearn $COMMON --split    --layer_id 32 --layer_ids 30,31,32 --output_dir runs/L32_split   # 129 steps

python scripts/make_table1.py      # Table 1
python scripts/make_figure2.py     # figures/figure2.pdf
```

Expected output of `make_table1.py` (last 30 of 129 steps; the paired run is cut to its first
129 steps):

```
              k F    k R  cos(a,u) F  cos(a,u) R  dcos(a,a0)
L7, paired 0.8303 0.8413      0.1083      0.1071     -0.0003
L7, split  0.8391 0.8451      0.0960      0.0947     -0.0001
L32, split 0.8582 0.8636      0.1926      0.1879     -0.0008

Predicted optimum (Eq. 5, alpha=5): k = 0.850, cos(a,u) = 0.196
Rescale only: k = 0.833, cos(a,u) = 0
```

Figure 2 plots `forget_/retain_norm_ratio` and `forget_/retain_cos_u` of the two split runs
(five-step rolling mean).

### Figure 3 and Section 4.3 (layer sweep)

```bash
python -m rmu.layer_sweep --model_name_or_path lapa-12b-instruct --output_dir runs/layer_sweep
python scripts/make_figure3.py     # per-layer table + figures/figure3.pdf (as in the paper)
```

The sweep trains layers L−2..L for every hooked layer L in {0, 4, …, 44, 47} (split, words,
β = 1, α = 5, lr 5e-4), restores the original weights between layers and reuses the same u.
It appends one row per layer to `sweep.csv`. Expected values (`results/layer_sweep/sweep.csv`):
rotation share of the forget-loss drop 0.24–0.39 for layers 0–20, 0.33 at layer 8 and 0.56 at
layer 32; cos(a, u) 0.193–0.197 and k 0.853–0.858 from layer 32 on; total drop 0.22 (layer 8)
to 0.29–0.30 (layers ≥ 32); forget and retain cos(a, u) within 0.006 at every layer.

### Pilot runs with standard RMU (Section 3)

The pilot runs motivate the setup and no table or figure depends on them. They can be
reproduced with the commands below; their logs are not included. Standard RMU (no `--beta`,
target s·u), layer 7, `--layer_ids 5,6,7`:

| s (`--steering_coeffs`) | `--lr` | `--alpha` | data | `--loss_positions` |
|---|---|---|---|---|
| 20 | 5e-5 | 100 | `--no-split` | `all` and `words` |
| 7000, 2500 | 5e-5 | 100 | `--split` | `words` |
| 2500 | 5e-2 | 100 | `--no-split`, `--split` | `words` |
| 2500 | 5e-2 | 10 | `--split` | `words` |

```bash
python -m rmu.unlearn --model_name_or_path lapa-12b-instruct --no-split --loss_positions words \
    --steering_coeffs 20 --lr 5e-5 --alpha 100 --output_dir runs/pilot_s20_lr5e-5_a100
```

At step 0 every run prints the activation norm of each position of the first forget sentence.
At layer 7 the BOS token has norm ≈ 65,000 vs ≈ 2,400 for word tokens (≈ 27×, Section 3), which
is why the main runs use `--loss_positions words`.

## Logged columns (`losses.csv`)

`forget_/retain_norm_ratio` = k = ‖a‖/‖a⁰‖ and `forget_/retain_cos_u` = cos(a, u), both averaged
over word tokens; `unlearn_/retain_cosine_sim` = cos(a, a⁰); `unlearn_loss_frozen` = L⁰_F, the
forget loss of the frozen model on the same batch; `forget_ids` = StereoSet sentence ids of the
forget batch.

## License

MIT (see `LICENSE`). `forward_with_cache` and the RMU training loop are adapted from the
reference RMU implementation of Li et al. (2024), released under the MIT License.
