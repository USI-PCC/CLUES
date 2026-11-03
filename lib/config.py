"""
Shared configuration constants (paths, labels, sensor settings, subject offsets) for the clues package.
"""

import os
from pathlib import Path

# ==============================================================================
# Paths
# ==============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SYNCED_DIR = DATA_DIR / "usi-synced"
UNSYNCED_DIR = DATA_DIR / "usi-unsynced"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"
OCO_DIR = Path(os.environ.get("CLUES_DIR", str(PREPROCESSED_DIR / "oco")))
SHARED_DIR = DATA_DIR / "shared"
REBUTTAL_DIR = PROJECT_ROOT / "rebuttal"
RESULTS_DIR = DATA_DIR / "results"

# ==============================================================================
# Subject Management
# ==============================================================================

EXCLUDED_SUBJECTS = {"s_046"}

# Raw subject numbers (1–30) map to folder names s_026–s_055.
# Index into PSYCHOPY_START_TIMES / PSYCHOPY_END_TIMES is subject_num - 26.
SUBJECT_OFFSET = 25
SUBJECTS = range(1, 31)

# ==============================================================================
# PsychoPy Timestamps
# ==============================================================================

PSYCHOPY_START_TIMES = [
    "2024-12-10 13h58.41.347977+0100",
    "2024-12-10 15h56.55.282097+0100",
    "2025-01-13 14h23.33.866581+0100",
    "2025-01-13 16h26.57.587980+0100",
    "2025-01-13 18h09.13.700817+0100",
    "2025-01-13 19h28.14.380292+0100",
    "2025-01-14 10h47.50.542712+0100",
    "2025-01-14 12h01.01.873832+0100",
    "2025-01-14 14h55.29.914682+0100",
    "2025-01-14 16h04.04.997130+0100",
    "2025-01-14 17h48.45.212817+0100",
    "2025-01-14 19h09.49.745539+0100",
    "2025-01-15 13h47.10.645120+0100",
    "2025-01-15 15h09.40.816212+0100",
    "2025-01-15 16h45.27.631804+0100",
    "2025-01-15 18h01.54.982483+0100",
    "2025-01-16 10h19.51.802178+0100",
    "2025-01-16 12h08.22.498669+0100",
    "2025-01-16 15h24.12.837388+0100",
    "2025-01-16 16h53.20.945250+0100",
    "2025-01-27 10h38.18.172875+0100",
    "2025-01-27 13h26.36.008151+0100",
    "2025-01-27 14h30.08.754424+0100",
    "2025-01-27 15h56.02.053832+0100",
    "2025-01-27 17h03.22.778137+0100",
    "2025-01-28 10h59.18.700332+0100",
    "2025-01-28 13h23.13.181180+0100",
    "2025-01-28 15h36.01.325378+0100",
    "2025-01-28 16h44.15.832924+0100",
    "2025-01-28 17h54.41.074493+0100",
]

PSYCHOPY_END_TIMES = [
    "2024-12-10 14:44:18",
    "2024-12-10 16:48:22",
    "2025-01-13 15:13:50",
    "2025-01-13 17:19:02",
    "2025-01-13 18:54:07",
    "2025-01-13 20:15:34",
    "2025-01-14 11:32:28",
    "2025-01-14 12:44:36",
    "2025-01-14 15:40:00",
    "2025-01-14 16:48:24",
    "2025-01-14 18:37:30",
    "2025-01-14 19:59:30",
    "2025-01-15 14:37:42",
    "2025-01-15 15:53:53",
    "2025-01-15 17:40:27",
    "2025-01-15 18:54:13",
    "2025-01-16 11:03:14",
    "2025-01-16 12:52:36",
    "2025-01-16 16:03:41",
    "2025-01-16 17:41:45",
    "2025-01-27 11:22:51",
    "2025-01-27 14:13:27",
    "2025-01-27 15:21:37",
    "2025-01-27 16:44:33",
    "2025-01-27 17:53:47",
    "2025-01-28 11:44:53",
    "2025-01-28 14:23:51",
    "2025-01-28 16:21:39",
    "2025-01-28 17:42:46",
    "2025-01-28 18:37:54",
]

# ==============================================================================
# Experimental Phase Markers
# ==============================================================================

QUESTIONNAIRE_MARKERS = ["now you will be asked", "how", "please rate"]

RELAX_MARKERS = [
    "time for a break. sit still and relax",
    "let's relax and listen to some music",
    "sit still, relax",
]
RELAX_MARKERS_LABELS = ["rest", "listen_music", "passive_viewing"]

TASK_MARKERS = [
    "2-back",
    "3-back",
    "the stroop",
    "the difficult stroop",
    "7 words",
    "15 words",
    "image for 1 minute",
    "challenging image",
    "two images next",
    "images with more",
    "visually track",
    "move from one textbox",
]
TASK_MARKERS_LABELS = [
    "2_back",
    "3_back",
    "stroop_easy",
    "stroop_difficult",
    "memory_easy",
    "memory_difficult",
    "images_questions_easy",
    "images_questions_difficult",
    "difference_images_easy",
    "difference_images_difficult",
    "pursuit_easy",
    "pursuit_difficult",
]

# ==============================================================================
# Cognitive Load Labels
# ==============================================================================

NO_COG_LOAD = ["rest", "listen_music", "passive_viewing"]
LOW_COG_LOAD = [
    "2_back",
    "stroop_easy",
    "memory_easy",
    "images_questions_easy",
    "difference_images_easy",
    "pursuit_easy",
]
HIGH_COG_LOAD = [
    "stroop_difficult",
    "3_back",
    "memory_difficult",
    "images_questions_difficult",
    "difference_images_difficult",
    "pursuit_difficult",
]

# Each LOGO fold holds out one task pair and its matched rest condition.
LOGO_GROUPS = [
    ["2_back", "3_back", "rest"],
    ["stroop_easy", "stroop_difficult", "rest"],
    ["memory_easy", "memory_difficult", "listen_music"],
    ["images_questions_easy", "images_questions_difficult", "listen_music"],
    ["difference_images_easy", "difference_images_difficult", "passive_viewing"],
    ["pursuit_easy", "pursuit_difficult", "passive_viewing"],
]

# ==============================================================================
# Label Mappings (for fusion scripts)
# ==============================================================================

# Three-class: 0=high_load, 1=low_load, 2=rest/no_load
LABEL_MAP_3CLASS = {
    "high_load": 0,
    "low_load": 1,
    "rest": 2,
    "no_load": 2,
    "passive_viewing": 2,
    "listen_music": 2,
}

# Two-class: 0=load, 1=rest/no_load
LABEL_MAP_2CLASS = {
    "load": 0,
    "rest": 1,
    "no_load": 1,
}

# ==============================================================================
# Sensor Definitions
# ==============================================================================

SENSOR_AXES = {
    "accelerometer": ["accelerometer_x", "accelerometer_y", "accelerometer_z"],
    "gyroscope": ["gyroscope_x", "gyroscope_y", "gyroscope_z"],
    "magnetometer": ["magnetometer_x", "magnetometer_y", "magnetometer_z"],
}
SENSOR_AXES_ALL = sum(SENSOR_AXES.values(), [])

ML_COLS = [
    "ml_attention",
    "ml_engagement",
    "ml_expressivity",
    "ml_imu_intensity",
]
PROBA_COLS = [
    "ml_pred_proba_0_0",
    "ml_pred_proba_1_0",
    "ml_pred_proba_2_0",
    "ml_pred_proba_3_0",
    "ml_pred_proba_4_0",
    "ml_pred_proba_5_0",
]
EXPR_COL = "ml_expressions"
EXPR_VALUES = ["neutral", "smile", "frown", "eyebrow_raise", "squeezed_eyes", "other"]

# ==============================================================================
# Windowing
# ==============================================================================

WINDOW_SIZE = 30        # seconds — OCOSense and Empatica
WINDOW_OVERLAP = 7.5    # seconds
WINDOW_SIZE_SHORT = 3   # seconds — RGB, thermal, eyetracker

# ==============================================================================
# Thresholds
# ==============================================================================

NAN_DROP_THRESHOLD = 0.5          # drop rows where > 50 % of columns are NaN
TIMESTAMP_GAP_THRESHOLD = 0.04    # flag gaps larger than this (seconds) in OCO data

# ==============================================================================
# Feature Configurations
# ==============================================================================

# Minimum set used by default in modeling.py / export_lgbm_predictions.py
MIN_FEATURE_CONFIG = {
    "basic_stats": True,
    "expression_counts": True,
    "sma": False,
    "energy": False,
    "range": False,
    "skew_kurtosis": False,
    "zero_crossing": False,
    "axis_correlations": False,
    "ml_proba_stats": True,
    "trend": False,
    "first_last_diff": False,
}

# Full feature set used for XAI / feature-importance analysis
FULL_FEATURE_CONFIG = {
    "basic_stats": True,
    "expression_counts": True,
    "sma": True,
    "energy": True,
    "range": False,
    "skew_kurtosis": True,
    "zero_crossing": False,
    "axis_correlations": True,
    "ml_proba_stats": True,
    "trend": True,
    "first_last_diff": False,
}

# ==============================================================================
# Modality Performance (macro F1 from individual experiments)
# ==============================================================================

MODALITY_F1_SCORES = {
    "ocosense": 0.80,
    "eyetracker": 0.65,
    "empatica": 0.62,
    "rgb": 0.51,
    "thermal": 0.49,
}

# ==============================================================================
# Lists
# ==============================================================================

MODALITIES = ["ocosense", "empatica", "eyetracker", "rgb"]
EXPERIMENTS = ["loso_two_class", "loso_three_class", "logo_three_class"]

# ==============================================================================
# Column Harmonization
# ==============================================================================

# Columns present only in the older cohort (s_001–s_025) that must be dropped
# to produce a consistent column structure for modeling.
COLUMNS_TO_DROP_OLD_SUBJECTS = [
    "ml_arousal",
    "ml_expression_Intensity",
    "ml_valence",
]
OLD_SUBJECTS = [f"s_{i:03d}" for i in range(1, 26)]
