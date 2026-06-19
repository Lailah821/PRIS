# PRIS — Path-Ranked In-context Specialists

Training-free **LLM-based contextual composition** of specialist forecasters for
**no-peek shadowed traffic forecasting**: predicting a target road's future speed
when the target sensor's own history is unavailable (sensor failure, outage, or a
road that was never instrumented), using only neighboring sensors and graph context.

Rather than using the LLM as a numerical forecaster, PRIS uses it as a **composition
agent**: at inference time it reads the per-sample spatial context (A0 graph reference,
A1 K-score support paths) and the predictions of **6 pretrained specialists**, then
outputs blend weights. Numerical prediction stays inside the specialists; the LLM only
decides *who to trust*. **No composition parameters are trained** — porting PRIS to a
new sensor network requires only specialists trained on that network; the composition
layer transfers as-is.

This repository accompanies the SIGSPATIAL 2026 paper *"PRIS: LLM-Based Composition
of Specialist Forecasters for Shadowed Traffic Prediction."*

## The 6 Specialists
`pgtft_c0`, `pgtft_cstd`, `gru_c0`, `graph_wavenet_cstd`, `agcrn_cstd`, `dcrnn_cstd`
— a heterogeneous pool spanning graph-transformer, recurrent, and graph-convolutional
inductive biases.

## Repository layout
```
PRIS/
├── README.md
├── requirements.txt
├── prompts/
│   └── PRIS_prompt.md        # verbatim Action-LLM prompt (Open- / Explicit-policy)
└── code/
    ├── core6/
    │   ├── eval_pems_pris_shadowed_core6_open_no_knn.py      # Open-policy entry point (paper main): A0–A4 pipeline, no effectiveness hints
    │   ├── eval_pems_pris_shadowed_core6_explicit_no_knn.py  # Explicit-policy entry point (6-Specialist wrapper)
    │   └── eval_pems_pris_shadowed_core6_explicit.py         # Explicit-policy launcher (frozen; legacy plan_c/core7 ids inside)
    ├── pems_metr_dataloader_v2.py                    # shadowed loader (self-mask, Gaussian top-8 neighbors)
    ├── phase_6_llm_client.py                         # Action-LLM client (Qwen3 / GPT-4o / Gemini)
    ├── phase_6_config.py
    ├── pems_metr_metrics.py                          # MAE / RMSE / MAPE>5
    ├── run_shadowed_baselines.py                     # specialist builder/trainer (graph models)
    ├── run_shadowed_pgtft.py                         # PGTFT specialist trainer
    ├── pems_metr_train_shadowed.py
    ├── pems_metr_{dcrnn,agcrn,gwn,stgcn,dlinear,gru,lstm,pgtft}_*.py  # specialist architectures
    ├── PGTFT_arch.py
```

## How PRIS works (A0–A4, single pass)
- **A0 — Spatial reference:** neighbor metadata, Gaussian adjacency, current neighbor speeds.
- **A1 — K-score support paths:** candidate target→neighbor→neighbor sequences ranked by
  `K = 0.4·(1−U) + 0.3·C + 0.3·P`.
- **A2 — Composer:** the LLM outputs `blend_weights` over the 6 specialists (sum to 1).
  The final forecast is the weighted average of the specialist predictions.
- **A3 — Guard / A4 — Residual:** conservative checks against the neighbor anchor
  (inactive on most samples).

## Setup
```bash
pip install -r requirements.txt
```
The Action LLM defaults to **Qwen3-14B** served locally (e.g., via Ollama; the client
also supports GPT-4o and a Gemini advisor). Set the relevant API keys / endpoints as
environment variables for those backends.

## Running the composer
```bash
python code/core6/eval_pems_pris_shadowed_core6_open_no_knn.py --dataset pems_bay --device cuda
```
- This script reproduces the **Open-policy** variant (paper main; keep the default
  `PEMS_NO_WP=1`). Open-policy *withholds* effectiveness hints, which most cleanly
  isolates the framework's own A0/A1 graph-context contribution. *(Note: the
  `PEMS_NO_WP` flag toggles a legacy weighting-policy branch and is **not** the
  Explicit/Open distinction reported in the paper.)*
- The **Explicit-policy** variant additionally supplies aggregate
  specialist-effectiveness hints (a `tool_prior` field, dataset-anchored expertise
  lines, and spread few-shot examples — see
  [`prompts/PRIS_prompt.md`](prompts/PRIS_prompt.md)). It is reproduced by a separate,
  frozen launcher:
  ```bash
  python code/core6/eval_pems_pris_shadowed_core6_explicit_no_knn.py \
      --dataset pems_bay --eval-cell Cstd --agent-model qwen3:14b
  ```
  (This launcher is frozen for reproducibility; its internal identifiers and output-JSON
  keys retain the historical `plan_c`/`core7` names, but the logic is unmodified.)
- Specialist checkpoints and the benchmark tensors are required to run end-to-end and
  are **available from the authors on request** (too large for GitHub).

## Deploying on a new sensor network
1. Train the specialists on the new network under the shadowed setup
   (`run_shadowed_baselines.py`, `run_shadowed_pgtft.py`).
2. Run the PRIS composer — **no composer training / target-domain labels needed.**

> **Extensibility — the specialist pool is not fixed.** This study uses six
> specialists, but PRIS is **tool-agnostic**: for deployment and real-world use you
> can plug in any set of specialists trained on data from your *target location or
> network* (e.g., region-specific or sensor-specific models). The composition layer
> — the prompts and the A0–A4 pipeline — operates unchanged on whatever pool you
> provide, so the six tools here are a starting point, not a constraint.

## Notes
- The released prompt (`prompts/PRIS_prompt.md`) is kept **verbatim** for exact
  reproducibility; see the note there about legacy "7 tools / KNN" wording in the
  no-KNN 6-Specialist configuration.
- Code/result identifiers historically used the name `plan_c`/`core7`; the public
  package is renamed to **PRIS** / **core6** (6-Specialists).

## Citation
```bibtex
@inproceedings{lee2026pris,
  title     = {PRIS: LLM-Based Composition of Specialist Forecasters for Shadowed Traffic Prediction},
  author    = {Lee, Jiyoon and Kang, Youngok},
  booktitle = {Proceedings of the 34th ACM International Conference on Advances in Geographic Information Systems (SIGSPATIAL '26)},
  year      = {2026}
}
```

## License
Released under the MIT License (see `LICENSE`).
