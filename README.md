<div align="center">

# Machine Learning for Battery Cathode Materials

**A three-task toolkit that also reports where its own numbers fail**

![voltage](https://img.shields.io/badge/voltage%20MAE-0.352%20V%20(two--tower%20CGCNN)-brightgreen)
![formation energy](https://img.shields.io/badge/formation%20energy-0.0401%20eV%2Fcell-brightgreen)
![structure coverage](https://img.shields.io/badge/structure%20coverage-40%2F40-blue)
![tests](https://img.shields.io/badge/tests-205%20(204%20pass%20%2B%201%20skip)-orange)
![license](https://img.shields.io/badge/license-MIT-green)

<p align="center">
  <b>English</b> · <a href="./README.zh-CN.md">简体中文</a>
</p>

</div>

---

> Predict band gap, formation energy, average voltage and specific capacity for Li-ion cathode materials —
> then decide which candidates are actually worth making.

This repository is a complete pipeline: **data → structure models → uncertainty calibration → multi-objective decision**.
It builds on three public data sources (matbench, Materials Project, and 39 curated literature performance records) and
can be reproduced **offline** — every artifact needed for the headline numbers is committed.

**The main documentation is written in Chinese: 23 appendices plus a full report.** This page is the English summary
and the entry points; the Chinese version is [README.zh-CN.md](./README.zh-CN.md).

## Results at a glance

| Task | Data | Model | Metric |
|---|---|---|---|
| **Band gap** | matbench `expt_gap` (4,604) | RF + HistGradientBoosting | **0.4415 eV** MAE |
| **Formation energy (composition)** | matbench `perovskites` (18,928) | RF + HGB | 0.4837 eV/cell |
| **Formation energy (structure)** | same + 21 structural descriptors | RF + HGB | 0.1929 eV/cell |
| **Formation energy (graph)** | same, end-to-end | self-contained **CGCNN** | **0.0401 eV/cell** (1.5× off the matbench leader coGN 0.0269) |
| **Formation energy (MP domain)** | 8,000 MP Li materials with structures | CGCNN | **0.0962 eV/atom** (3k subset: 0.1921) |
| **Band gap (structure)** | same, DFT `band_gap` | CGCNN | 0.5275 eV — only **1.06×** better than composition (see below) |
| **Average voltage** | MP insertion electrodes (2,672 rows) | RF + HGB | 0.4187 V (grouped 0.4733 V) |
| **Average voltage (two-phase)** | 2,532 structure pairs | **two-tower CGCNN** | **0.3525 V** (grouped 0.4007 V) |
| **Specific capacity** | same electrode set | RF + HGB | 25.16 mAh/g |
| **Uncertainty** | OOF residuals | split conformal + locally-adaptive | 90% interval, measured coverage 90.0%; locally-adaptive calibrated at 88.7% with 9.4× differentiation |

**External validation**: 12 literature cathodes → MAE **0.160 V**, Spearman ρ 0.842.
Cross-schema MP validation → offset-corrected MAE 0.735 eV/atom against a 0.805 baseline MAD.

### What the numbers do not show

Several results here are **negative**, and they are documented as carefully as the positive ones:

| Finding | Evidence |
|---|---|
| **Rocksalt prototypes are not usable** | CGCNN predictions differ from MP ground states by **+1.25 eV/atom** — not a relaxation artefact (unchanged after CHGNet relaxation and across lattice-constant sweeps), but a polymorph difference |
| **Isostructural substitution + MLIP selection is usable** | residual drops to **+0.09 eV/atom**, ρ **0.968** → those candidates join the main ranking (structure coverage 40/40) |
| **A structure band-gap model does not help** | ρ between PBE and experimental-caliber gap rankings is **−0.56**; PBE systematically underestimates gaps → conductivity keeps the experimentally-trained model |
| **A voltage model cannot rank working voltage** | the voltage dimension correlates ≈0 with measured specific energy → used as a **risk item** (>4.5 V / >5.0 V / <2.5 V), not a linear scoring term |
| **Constructed two-phase pairs double the error** | "remove Li + CHGNet relax" residual **0.3468 V**, the same size as the model's own 0.3525 V → reported, but **never used for scoring** (test-enforced) |

Two further structural findings:

- **The prior scoring weights were negatively correlated with measured performance** (ρ = −0.1434);
  multi-objective calibration lifts it to **+0.5444**, and a bootstrap gives weight intervals so wide
  (stability 0.20–0.75) that any two-decimal claim over-reads the data.
- **Six-objective Pareto fronts inflate to 34/40** — when 85% of candidates are non-dominated, the frontier stops
  discriminating. The answer is **hard feasibility constraints first** (Co ≤ 0.10, toxicity ≤ 2, ≤ 4.5 V,
  ≥ 120 mAh/g → 40 → 20 candidates), then preferences.

## What you can do with this toolkit

Seven concrete jobs, each with a runnable entry point. The pipeline is offline; only the optional
structure / MLIP track needs a GPU and an API key.

| Job | Entry point |
|---|---|
| **Screen a candidate list into a ranked, risk-flagged shortlist** | [src/screening_struct.py](src/screening_struct.py) (structure-driven, GPU track) · [src/predictor.py](src/predictor.py) `--top 10 --min-voltage 3.0 --max-voltage 5.0` |
| **Predict band gap, formation energy and voltage for any formula** | [src/predictor.py](src/predictor.py) `--formulas "LiFePO4,LiCoO2,LiMn2O4"` · [src/predict_voltage_cgcnn.py](src/predict_voltage_cgcnn.py) |
| **Predict from a crystal structure instead of a formula** | [src/predict_cgcnn.py](src/predict_cgcnn.py) `--structures data/raw/mp_structures_cathodes.json.gz` (10 committed cathodes, offline) |
| **Put an error bar on any prediction** | [src/conformal.py](src/conformal.py) · [src/local_conformal.py](src/local_conformal.py) · [src/drx_ensemble.py](src/drx_ensemble.py) |
| **Decide under hard feasibility constraints, not just weights** | [src/pareto_interactive.py](src/pareto_interactive.py) `--constraints default` (40 → 20 candidates) · `--scenario eu2031 / eu2036 / strict` |
| **Audit where every number comes from** | [data/reference_cathodes.csv](data/reference_cathodes.csv) (per-row citation + uncertainty; 24/39 Crossref-verified) · [src/annotate_reference.py](src/annotate_reference.py) |
| **Re-derive the published metrics end to end** | `python run_pipeline.py` · [src/validate_known_cathodes.py](src/validate_known_cathodes.py) (12 literature cathodes, MAE 0.160 V) |

Typical uses: (1) a **screening baseline** for a new cathode candidate list; (2) a **reusable
uncertainty stack** (split conformal + locally-adaptive intervals) you can lift into another
regression task; (3) a **worked example of documenting negative results** — six of them, each with
its evidence; (4) **teaching material** for multi-objective decisions, where a single score is
provably arbitrary.

Reproducibility: `pytest tests -q` (**205 tests: 204 pass + 1 conditional skip**) needs no API key and
no raw dataset download — the cleaned MP data, the MP Li structures and the demo structure/pair files
are committed. CI runs 8 jobs (ubuntu + windows × Python 3.9 / 3.11 / 3.12), green on every release.

## How to use this repository

| If you are… | Route | Entry point |
|---|---|---|
| **Evaluating the method** | Results → negative-results table → report appendices A–W | This page, [docs/项目报告.md](docs/项目报告.md) |
| **Screening candidates** | Install → run the structure-driven screening pipeline | [src/screening_struct.py](src/screening_struct.py) |
| **Using the models directly** | Composition CLI, or CGCNN / two-tower inference CLIs | [src/predictor.py](src/predictor.py), [src/predict_cgcnn.py](src/predict_cgcnn.py), [src/predict_voltage_cgcnn.py](src/predict_voltage_cgcnn.py) |
| **Reviewing provenance** | Citation table with Crossref-verified DOIs, plus the entries explicitly *not* verified | [data/reference_cathodes.csv](data/reference_cathodes.csv) |
| **Reusing the uncertainty stack** | Split conformal, locally-adaptive intervals, ensemble noise — each validated | [src/conformal.py](src/conformal.py), [src/local_conformal.py](src/local_conformal.py), [src/drx_ensemble.py](src/drx_ensemble.py) |
| **Deciding with constraints** | Hard constraints + 6-objective Pareto + preference profiles | [src/pareto_interactive.py](src/pareto_interactive.py) |

## Repository map

```
├── README.md                    ← you are here (English)
├── README.zh-CN.md              ← full Chinese documentation
├── src/                         # 49 modules: features → models → uncertainty → decision
│   ├── predictor.py             #   composition-level prediction CLI
│   ├── train_gap.py / train_ef.py / train_voltage.py / train_capacity.py
│   ├── cgcnn_data.py            #   pure-NumPy crystal graph builder (torch-free, CI-tested)
│   ├── cgcnn_model.py           #   gated-conv CGCNN shared by training, MP domain and inference
│   ├── train_cgcnn.py           #   matbench perovskites (eV/cell)
│   ├── train_cgcnn_mp.py        #   MP Li materials (eV/atom: e_form or band_gap)
│   ├── train_cgcnn_voltage.py   #   two-tower CGCNN: structure pair → voltage
│   ├── predict_cgcnn.py         #   structure → formation energy CLI
│   ├── predict_voltage_cgcnn.py #   formula → voltage CLI
│   ├── fetch_mp_structures.py   #   MP retrieval (exact / reduced formula / tolerant matching)
│   ├── template_substitution.py #   polymorph enumeration + isostructural substitution
│   ├── generate_prototype_structures.py  # rocksalt prototypes + CHGNet relaxation
│   ├── drx_ensemble.py          #   disorder ensemble (decoration noise)
│   ├── build_voltage_pairs.py   #   constructed two-phase pairs (reported, not scored)
│   ├── conformal.py / local_conformal.py  # uncertainty (global + locally adaptive)
│   ├── calibrate_weights.py / annotate_reference.py  # weights + provenance
│   ├── resource_risk.py         #   supply-risk / toxicity ordinal scores
│   └── screening_struct.py      #   structure-driven screening (single entry point)
├── data/                        # commits only what offline reproduction needs
│   ├── processed/               #   cleaned MP voltage data, MP Li structures, matbench subsets
│   ├── reference_cathodes.csv   #   39 literature records with citations + per-row uncertainty
│   └── cathode_candidates.csv   #   40 candidate cathodes
├── docs/项目报告.md              # full report with appendices A–W
├── results/                     # metrics JSON / figures / model weights (pkl files stay out of git)
└── tests/                       # 205 tests (204 pass + 1 conditional skip)
```

## Quick start

```bash
# 0) environment — no API key and no raw datasets required
pip install -r requirements.txt
pytest tests -q                        # 205 tests

# 1) offline pipeline (exactly what CI runs)
python src/train_voltage.py --data data/processed/mp_voltage.csv --tag voltage
python src/train_capacity.py
python src/train_gap.py
python src/train_ef.py
python src/conformal.py --tag voltage
python src/prepare_voltage_struct.py
python src/train_voltage_struct.py
python src/validate_known_cathodes.py    # 12 literature cathodes
python src/validate_mp_external.py

# 2) prediction CLI (composition level)
python src/predictor.py --formulas "LiFePO4,LiCoO2,LiMn2O4"

# 3) optional GPU track (structure models; an RTX 4050 with 6 GB is sufficient)
python -m venv D:\venvs\bcm-cgcnn
D:\venvs\bcm-cgcnn\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu124
D:\venvs\bcm-cgcnn\Scripts\pip install numpy pandas matplotlib scikit-learn mp-api chgnet
D:\venvs\bcm-cgcnn\Scripts\python src/train_cgcnn_mp.py --epochs 150
D:\venvs\bcm-cgcnn\Scripts\python src/build_voltage_pairs.py --variants 4
D:\venvs\bcm-cgcnn\Scripts\python src/screening_struct.py
```

## Decision layer

| Mechanism | Role | Implementation |
|---|---|---|
| **Hard constraints** | remove infeasible candidates, no weighting | [src/pareto_interactive.py](src/pareto_interactive.py) `--constraints default` (40 → 20) |
| **Regulation scenarios** | supply-pressure scenarios (EU 2023/1542 recycled-content targets used as a Co-cap proxy) | `--scenario eu2031 / eu2036 / strict` |
| **Calibrated weights** | rank inside the feasible set | multi-objective calibration (ρ −0.14 → +0.54) |
| **Pareto frontier** | keep non-dominated alternatives instead of one "winner" | `pareto_rank`, 6 objectives |
| **Preference profiles** | "the answer depends on what you prioritise" | 5 profiles + robust picks |
| **Risk items** | flag but do not delete: voltage window, resources, toxicity, structure confidence | `risk_level`, `voltage_risk`, `resource_flags` |
| **Error bars** | ensemble noise + local intervals | `ef_ens_std_eV_atom`, `voltage_pi90_halfwidth_V` |

## Known limitations and scope boundary

- **Domain extrapolation is the main risk.** The band gap and formation energy composition models are trained on
  experimental / perovskite data; for Li cathodes this is extrapolation (a negative predicted gap is the signal).
- **Conductivity has no trustworthy structure model.** Structure band gap correlates *negatively* with the
  experimental-caliber ranking, so it is kept as a cross-check column only.
- **Voltage coverage is 20/40 for real MP electrode pairs.** Constructed pairs raise structural coverage to 92.5%
  but carry a 0.35 V residual, so they never drive scoring.
- **Disorder is a first-order approximation.** DRX / Li-rich candidates use an 8-decoration ensemble; short-range
  order is not modelled.
- **Screening is at material level.** No Li-content variation, voltage curves, kinetics, electrolyte compatibility,
  cost curves or manufacturability.
- **The reference performance table is literature-typical.** 24/39 entries carry a Crossref-verified DOI; the rest are
  review-level and explicitly marked as unverified.

## Verification and provenance

- **205 tests** (204 pass + 1 conditional skip); CI runs **8 jobs** (ubuntu × windows, Python 3.9 / 3.11 / 3.12),
  green on every release.
- **Two guard rails enforced by source-level assertions** rather than review: a partially-run script must rename its
  artifacts (`*_smoke` / `*_<profile>` / `*_validation`) so it cannot overwrite full results; and CLI scripts must keep
  heavy imports (`torch`, `pymatgen`, `chgnet`) inside functions so `--help` works in CI without them.
- **Reproducible without an API key**: cleaned MP voltage data, 3,000 MP Li structures, structure-feature tables and
  demo structure/pair files are committed.
- **Citations verified against the Crossref API**, with the table recording which entries are *not* DOI-verified.
- Full narrative including the negative results: [docs/项目报告.md](docs/项目报告.md).

## Sibling repositories

| Repository | What it does |
|---|---|
| [battery-lifetime-guide](https://github.com/ilo-love7/battery-lifetime-guide) | From-scratch reproduction of Severson et al. 2019, crash scenes left in |
| [battery-soh-tsfm-benchmark](https://github.com/ilo-love7/battery-soh-tsfm-benchmark) | Do zero-shot time-series foundation models beat linear extrapolation? |
| [battery-ecm-rul](https://github.com/ilo-love7/battery-ecm-rul) | Thevenin ECM identification (NASA PCoE) + early cycle-life prediction + PyBaMM |
| [drift-diffusion-pn-junction](https://github.com/ilo-love7/drift-diffusion-pn-junction) | 1D drift–diffusion–Poisson solver with three-layer verification |
| [car-following](https://github.com/ilo-love7/car-following) | Car-following traffic modelling toolkit (7 models) |
| [project-packer](https://github.com/ilo-love7/project-packer) | Turning real project work into interview-defensible experience (fact dossier + question tree) |

## License and data credits

MIT — see [LICENSE](LICENSE).

Data: Materials Project (CC BY 4.0, attribution required), matbench (`matbench_expt_gap`, `matbench_perovskites`), and a
curated table of literature-typical cathode performance ([data/reference_cathodes.csv](data/reference_cathodes.csv)) with
per-row citations and uncertainty. The MLIP relaxation path uses the `chgnet` package (BSD-3-Clause) and its pretrained
weights.

```bibtex
@software{battery_cathode_ml,
  title  = {Machine Learning for Battery Cathode Materials},
  author = {ilo-love7},
  year   = {2026},
  note   = {version = {1.24.0}},
  url    = {https://github.com/ilo-love7/battery-cathode-ml}
}
```