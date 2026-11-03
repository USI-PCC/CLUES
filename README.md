# CLUES: Cognitive Load Understanding through Experimental Sensing dataset

Code for the CLUES paper on multimodal cognitive load classification. 55 participants (25 at JSI, Ljubljana: `s_001`–`s_025`; 30 at USI, Lugano: `s_026`–`s_055`) performed cognitive tasks at varying load levels (none, low, high) while wearing OCOSense smart glasses, an Empatica wristband, an eye tracker, and being recorded by RGB and thermal cameras.

The pipeline synchronizes multi-stream sensor data, segments it into time windows, extracts statistical features, and trains classifiers (LightGBM, XGBoost, RandomForest) using leave-one-subject-out (LOSO) and leave-one-group-out (LOGO) cross-validation. Late-fusion approaches combine per-modality predictions.

## Paper and dataset

- **Paper:** A. Krstevska et al. *CLUES: Cognitive Load Understanding through Experimental Sensing Dataset.* Proc. ACM Interact. Mob. Wearable Ubiquitous Technol. 10(3), Article 116, 2026. https://doi.org/10.1145/3831656
- **Dataset:** Zenodo, https://doi.org/10.5281/zenodo.21410787. Access is restricted to non-commercial academic use and granted on request under a Dataset Sharing Agreement; the record explains how to apply.

Please cite both when using this code or the data (see [Citation](#citation)).

## Setup

Tested with Python 3.11.

```bash
pip install -r requirements.txt
```

## Repository layout

Two packages: `clues/` (the pipeline) and `lib/` (configuration and path
helpers it imports). Every module runs as `python -m clues.<module>` with the
repo root as the working directory; running a file directly
(`python clues/pipeline/sync.py`) will not resolve its imports.

```
lib/
  config.py                     # Shared constants (paths, labels, sensor configs, subject offsets)
  paths.py                      # Path helpers, incl. the CLUES nested-layout resolver
clues/
  pipeline/
    sync.py                     # Sync XDF + OCO into one CSV per subject (stage 1)
    segmentation.py             # 30 s / 7.5 s sliding-window segmentation (stage 3)
    modeling.py                 # OCOSense feature extraction + LGBM/XGB/RF (stage 4)
    process_shared.py           # Label RGB frame clock + eye-tracker samples from the raw recordings
  reporting/
    export_lgbm_predictions.py  # Export per-window OCOSense predictions for fusion
    harmonize_features.py       # Standardise per-modality feature CSV schemas
    merge_results.py            # Aggregate per-modality + fusion metrics into one summary CSV
    plot_results.py             # Bar chart + heatmap of merged metrics
    plot_coverage.py            # Per-modality window-coverage bars vs empatica reference
  data_loaders.py               # Generic per-modality prediction loaders for fusion
  features.py                   # Statistical feature extraction (shared by modeling + export)
  prep/                         # Shared-dataset import/export, cohort column harmonization
  fusion/                       # Late-fusion package — one method per file (plugin pattern)
  analysis/                     # QA, alignment, sensor-interference, dataset-stats scripts
visualize.ipynb                 # Label coverage + cross-modality sync QA on the shared dataset
data/                           # Data root (gitignored — see "Data" section below)
```

## Visualization notebook

`visualize.ipynb` is the place to start: it runs directly on the shared CLUES dataset (see [Data](#data)) and needs no preprocessing. For OCOSense, eye tracker and Empatica it:

- plots the label distribution per 1 s window and per sample,
- lists missing files,
- measures cross-modality label agreement (overall, per subject, pairwise, most common disagreement patterns),
- plots the label sequence of each modality over time for one subject.

```bash
CLUES_DIR=/path/to/CLUES jupyter notebook visualize.ipynb
```

Without `CLUES_DIR`, edit `BASE_DATA_DIR` at the top of the notebook. Jupyter is not in `requirements.txt`.

## Pipeline

The stages below reproduce the headline results and are meant to be run in order.

**Stages 1–4 process OCOSense only.** This repository does not extract features from, or train classifiers on, Empatica, eye-tracker or RGB data. The fusion experiments (stage 5) need those modalities already preprocessed, as per-window predictions and features placed under `data/results/`:

| Modality | Predictions (late fusion) | Features (early fusion) |
|---|---|---|
| Empatica | `data/results/empatica/{loso_2class,loso_3class,logo_3class}/lgbm_predictions.csv` | `data/results/empatica/artifacts/empatica_features.csv` |
| Eye tracker | `data/results/eyetracker/{loso_2class,loso_3class,logo_3class}/lgbm_predictions.csv` | `data/results/eyetracker/artifacts/eyetracker_features.csv` |
| RGB | `data/results/rgb/{loso_2class,loso_3class,logo_3class}/lgbm_predictions.csv` | `data/results/rgb/artifacts/rgb_features.csv` |

The columns each loader expects are documented in `clues/data_loaders.py` and `clues/reporting/harmonize_features.py`. Thermal is not used in fusion.

There are two ways into the OCOSense stages:

- **From the shared dataset** (all 55 participants). `clues_to_segmented` builds the windows that stage 4 reads, replacing stages 1–3:

  ```bash
  python -m clues.prep.clues_to_segmented --clues-dir /path/to/CLUES --out-dir /path/to/segmented
  python -m clues.pipeline.modeling --oco-dir /path/to/segmented
  ```

- **From the raw USI recordings** (XDF and OCOSense files, not distributed): stages 1–3 below.

### 1. Synchronization (OCOSense, raw recordings)

```bash
python -m clues.pipeline.sync
```

Covers the 30 USI raw recordings (`s_026`–`s_055`). Loads XDF files (Lab Streaming Layer) and OCOSense text files, aligns OCO timestamps to PsychoPy event markers from the LSL stream. Outputs synced CSVs to `data/usi-synced/`.

Options:
- `--base-path PATH` — use an alternate data directory (default: `data/` relative to repo root)
- `--output-suffix SUFFIX` — control output filename suffix (default: `_oco`)
- `--zero-timestamps` — zero-base timestamps and rename `experimental_phase` to `label`

### 2. Data preparation (OCOSense, raw recordings)

```bash
for f in data/usi-synced/s_*/oco/s_*_oco.csv; do s=$(basename "$f" _oco.csv); mkdir -p data/preprocessed/oco/$s && cp "$f" data/preprocessed/oco/$s/; done
python -m clues.prep.harmonize_oco_columns
```

Copies each subject's synced file into `data/preprocessed/oco/<subject>/`, the layout segmentation reads. Stage 1 does not produce the JSI participants (`s_001`–`s_025`); their files go into the same layout. `harmonize_oco_columns` then drops the columns only that cohort has (`ml_arousal`, `ml_expression_Intensity`, `ml_valence`).

### 3. Segmentation (OCOSense, raw recordings)

```bash
python -m clues.pipeline.segmentation                           # All subjects
python -m clues.pipeline.segmentation --from-subject s_026 --skip-existing   # Process only missing
```

Sliding window segmentation: 30-second windows with 7.5-second overlap. Outputs `*_oco_segmented.csv` per subject.

### 4. Feature extraction & classification (OCOSense)

```bash
python -m clues.pipeline.modeling             # LGBM/XGB/RF, LOSO + LOGO
python -m clues.pipeline.modeling --xai       # With SHAP / feature-importance plots
```

Extracts statistical features per window (mean, std, quartiles, SMA, energy, skew/kurtosis, axis correlations, ML probability stats, trend slopes). Per-window predictions are written to `data/results/ocosense/{experiment}/{model}_predictions.csv`. Subject `s_046` is excluded due to data-quality issues; rows with >50 % NaN are dropped; questionnaire phases are skipped.

### 5. Fusion (all modalities)

Requires the Empatica, eye-tracker and RGB inputs listed at the top of this section.

```bash
python -m clues.reporting.export_lgbm_predictions            # Per-window OCOSense predictions
python -m clues.reporting.harmonize_features                 # Common feature schema; needed by early fusion
python -m clues.fusion.runner                                 # All fusion methods × all experiments
python -m clues.fusion.runner --methods majority_voting,stacking     # Run a subset of methods
python -m clues.fusion.runner --experiments loso_three_class         # Restrict to one experiment
```

Combines OCOSense, Empatica, eye tracker and RGB. Available methods: majority voting, weighted probability, confidence-based weighting, DST (Dempster–Shafer) voting, stacking meta-classifier, and early (feature-level) fusion. Handles the window-size mismatch between 3 s and 30 s modalities. Outputs go to `rebuttal/fusion/`.

### 6. Aggregation & plots

```bash
python -m clues.reporting.merge_results        # All metrics → data/results/all_results.csv
python -m clues.reporting.plot_results         # Bar charts + heatmap → data/results/plots/
python -m clues.reporting.plot_coverage        # Modality coverage bars  → data/results/plots/
```

### Building the shared-dataset files from the raw recordings (USI)

```bash
python -m clues.prep.export_ocosense --out-dir /path/to/CLUES   # <subject>/ocosense/<subject>.csv
python -m clues.pipeline.process_shared                         # <subject>/rgb/<subject>.csv, <subject>/eyetracker/<subject>.csv
```

`export_ocosense` turns the stage-1 output into the dataset's OCOSense files: `experimental_phase` becomes `label`, timestamps start at 0, `software_timestamp` is dropped. `process_shared` reads the XDF recordings together with each subject's raw camera frame clock (`rgb/<subject>.txt`) and Tobii export (`eyetracker/<subject>.tsv`) under `data/shared/<subject>/`, and writes the timestamped, labelled versions next to them as `rgb/<subject>.csv` and `eyetracker/<subject>.csv`.

## Analysis & QA

Standalone scripts under `clues/analysis/`, runnable individually (`python -m clues.analysis.<name>`):

- `check_timestamp_gaps.py` — timestamp gaps in the OCOSense streams.
- `measure_rgb_eyetracker_alignment.py` — temporal coverage / offset between RGB and eye-tracker windows.
- `oco_dataset_stats.py` — per-subject and aggregate OCOSense recording stats, written as JSON for reviewer-facing tables.
- `sensor_interference.py` — mixed-effects regression of eye-tracker gaze-loss on OCOSense motion; addresses whether glasses-induced motion biases eye-tracking.

## Adding a new fusion method

The fusion package uses a plugin pattern: each method is a `FusionMethod` subclass in its own file under `clues/fusion/`. To add one:

1. Create `clues/fusion/my_method.py` with a class inheriting `FusionMethod` (see `majority_voting.py` for the simplest example).
2. Register it in `clues/fusion/runner.py` so it can be discovered by name.
3. Run `python -m clues.fusion.runner --methods my_method`.

`clues/fusion/utils.py` collects the shared helpers for window alignment, confidence scoring, and metric computation. All late-fusion methods emit a `fused_uncertainty` column (normalized entropy of the fused distribution).

## Data

Data files are not included in this repository; request access through the [Zenodo record](https://doi.org/10.5281/zenodo.21410787). The shared CLUES dataset is organized as:

```
CLUES/
├── self_reports_and_performance.csv   # One row per participant: questionnaires and task performance
└── <subject_id>/               # One folder per participant
    ├── empatica/               # Empatica E4 wristband
    ├── ocosense/               # Emteq OCOsense smartglasses
    ├── eyetracker/             # Tobii Pro Spark
    ├── thermal/                # FLIR Lepton thermal camera
    └── rgb/                    # Logitech BRIO 4K RGB camera
```

`self_reports_and_performance.csv` holds demographics, NASA-TLX and ISA ratings, and task scores per participant. Its per-condition columns (`nasa_tlx_<condition>`, `isa_<condition>`, `score_<condition>`, `score_pct_<condition>`) use the same condition names as the `label` column of the sensor files, so they join directly on `subject_id` + `label`. The full column description is in the dataset's README.

`visualize.ipynb` runs directly on this folder (see [Visualization notebook](#visualization-notebook)). To run the OCOSense pipeline on it, point `clues_to_segmented` at this folder (see [Pipeline](#pipeline)); it reads `<subject_id>/ocosense/<subject_id>.csv`.

The raw-recording route (stage 1 onwards) uses this layout under `data/`:

```
data/
  usi-unsynced/
    lsl/          # XDF files per subject
    oco/          # OCOSense raw text files
  usi-synced/     # Output from sync.py
  preprocessed/
    oco/          # Per-subject OCOSense data for segmentation/modeling (override with CLUES_DIR)
  shared/         # Per-subject folders in the dataset layout; process_shared reads rgb/ and eyetracker/
  results/        # OCOSense results (stage 4), the other modalities' inputs (stage 5), fusion metrics
```

Subject numbering: `s_001`–`s_025` are the JSI participants; the USI raw recordings are numbered 1–30 and map to `s_026`–`s_055` (offset +25). The index into `PSYCHOPY_START_TIMES` / `PSYCHOPY_END_TIMES` in `lib/config.py` is `subject_num - 26`.

## Experimental design

- **12 cognitive tasks**: 6 easy/difficult pairs (n-back, Stroop, memory, image questions, difference images, visual pursuit)
- **3 rest conditions**: rest, listen to music, passive viewing
- **Target labels**:
  - Binary: 0 = no load, 1 = cognitive load
  - Three-class: 0 = no load, 1 = low load, 2 = high load
- **LOGO folds**: each fold holds out one task pair and its matched rest condition.
- **Window sizes**: OCOSense and Empatica use 30 s windows; RGB, thermal, and eye tracker use 3 s windows. Fusion scripts handle this mismatch.

## Citation

If you use this code or the dataset, please cite both the paper and the dataset:

```bibtex
@article{krstevska2026clues,
  author    = {Krstevska, Ana and Goyal, Shivalika and Fiorini, Linda and Bombassei De Bona, Francesco and Kirilenko, Daniil and Li, Mohan and Trojer, Sebastijan and Božak, Tomi and Anžur, Zoja and Luštrek, Mitja and Gjoreski, Hristijan and Langheinrich, Marc and Slapničar, Gašper and Gjoreski, Martin},
  title     = {{CLUES}: Cognitive Load Understanding through Experimental Sensing Dataset},
  journal   = {Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies},
  volume    = {10},
  number    = {3},
  articleno = {116},
  pages     = {1--33},
  year      = {2026},
  month     = sep,
  publisher = {Association for Computing Machinery},
  doi       = {10.1145/3831656}
}

@misc{krstevska2026cluesdata,
  author    = {Krstevska, Ana and Goyal, Shivalika and Fiorini, Linda and Bombassei De Bona, Francesco and Kirilenko, Daniil and Li, Mohan and Trojer, Sebastijan and Božak, Tomi and Anžur, Zoja and Luštrek, Mitja and Gjoreski, Hristijan and Langheinrich, Marc and Slapničar, Gašper and Gjoreski, Martin},
  title     = {Data from ``{CLUES}: Cognitive Load Understanding through Experimental Sensing Dataset''},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21410787}
}
```

## License

The code in this repository is released under the [Creative Commons Attribution 4.0 International](LICENSE) license (CC BY 4.0). The dataset is distributed separately, under its own Dataset Sharing Agreement (see [Paper and dataset](#paper-and-dataset)).
