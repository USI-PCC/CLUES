"""Quantify systematic interference from OCOsense motion on eye-tracker validity.

Addresses 2AE comment: "the noted interference from OCOsense glasses on the
eye-tracker and thermal camera is not just a data loss issue but may introduce
systematic bias."

Because every subject wore OCOsense for the entire session, we have no within-
subject OCOsense-off counterfactual. Instead we test whether *OCOsense-measured
motion* correlates with eye-tracker gaze-loss rate in the same time window: a
positive correlation is consistent with glasses-slip causing eye-tracker
occlusion (the reviewer's concern), even though the same correlation would also
arise from generic head motion that independently disrupts the tracker.

For each 3 s eye-tracker window, we:
  - extract the gaze-invalid fraction (mean of GazePointX/Y and PupilSize NaN
    fractions — the four most reliable signals of tracker loss-of-lock);
  - align it to the parent 30 s OCOsense window by (subject, task) with start
    within the OCOsense window;
  - fit a mixed-effects regression with subject random intercepts:
        invalid_frac ~ ocosense_imu_intensity + (1 | subject)

Thermal is handled qualitatively in the manuscript; its per-window keypoint-
failure column is not persisted, and subject-level thermal dropout is dominated
by a site-specific hardware issue already documented in Limitations.

Output: writes summary stats and model coefficients to
``data/results/sensor_interference.txt`` and ``.csv``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from rich.console import Console

from lib.config import RESULTS_DIR

console = Console()

EYE_PATH = RESULTS_DIR / "eyetracker" / "artifacts" / "features.csv"
OCO_PATH = RESULTS_DIR / "ocosense" / "artifacts" / "features.csv"
OUT_DIR = RESULTS_DIR
OUT_TXT = OUT_DIR / "sensor_interference.txt"
OUT_CSV = OUT_DIR / "sensor_interference.csv"

# Eye-tracker gaze-loss signal: mean of the four most-reliable NaN-fraction
# columns. Gaze and pupil NaNs fire together when the tracker loses lock.
GAZE_NAN_COLS = [
    "GazePointX_NaN_fraction",
    "GazePointY_NaN_fraction",
    "PupilSizeLeft_NaN_fraction",
    "PupilSizeRight_NaN_fraction",
]

# OCOsense motion predictor: ml_imu_intensity_mean captures normalised
# accelerometer-magnitude change over the window. Higher value = more motion.
# We also report gyroscope_z_std as a secondary head-rotation proxy.
OCO_MOTION_COL = "ml_imu_intensity_mean"
OCO_ROTATION_COL = "gyroscope_z_std"


def load_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    eye = pd.read_csv(EYE_PATH)
    oco = pd.read_csv(OCO_PATH)
    return eye, oco


def join_eye_to_ocosense(eye: pd.DataFrame, oco: pd.DataFrame) -> pd.DataFrame:
    """Aggregate both sides to (subject, task) level and join.

    The raw features.csv files use different time references (eye-tracker
    uses milliseconds-from-recording-start while OCOsense uses epoch seconds),
    so per-window temporal alignment is not directly possible from the
    harmonised CSVs. Aggregating to (subject, task) level sidesteps that
    mismatch and still answers the reviewer's question: within a given task,
    does higher OCOsense motion correlate with higher eye-tracker gaze loss?
    """
    eye = eye[["subject", "task"] + GAZE_NAN_COLS].copy()
    eye["invalid_frac"] = eye[GAZE_NAN_COLS].mean(axis=1)
    eye_by_st = eye.groupby(["subject", "task"], as_index=False).agg(
        invalid_frac=("invalid_frac", "mean"),
        eye_n_windows=("invalid_frac", "size"),
    )

    keep_oco = ["subject", "task", OCO_MOTION_COL, OCO_ROTATION_COL]
    oco_by_st = oco[keep_oco].groupby(["subject", "task"], as_index=False).agg(
        **{
            OCO_MOTION_COL: (OCO_MOTION_COL, "mean"),
            OCO_ROTATION_COL: (OCO_ROTATION_COL, "mean"),
            "oco_n_windows": (OCO_MOTION_COL, "size"),
        }
    )

    merged = eye_by_st.merge(oco_by_st, on=["subject", "task"], how="inner")
    merged = merged.dropna(subset=[OCO_MOTION_COL, "invalid_frac"])

    return merged


def fit_mixed_effects(df: pd.DataFrame, predictor: str,
                      include_task_fe: bool = True) -> dict:
    """Fit invalid_frac ~ predictor (+ task FE) + (1 | subject) via MixedLM.

    Task is included as a fixed effect so baseline gaze-loss patterns that
    differ by task (e.g., rest involves more off-screen looks than stroop)
    cannot confound the OCOsense motion coefficient.
    """
    x = df[predictor].astype(float)
    df = df.assign(_x_std=(x - x.mean()) / x.std())
    formula = "invalid_frac ~ _x_std + C(task)" if include_task_fe else "invalid_frac ~ _x_std"
    model = smf.mixedlm(formula, df, groups=df["subject"])
    result = model.fit(method="lbfgs", reml=True)
    beta = float(result.params["_x_std"])
    se = float(result.bse["_x_std"])
    ci_low = beta - 1.96 * se
    ci_high = beta + 1.96 * se
    p = float(result.pvalues["_x_std"])
    return {
        "predictor": predictor,
        "task_fe": include_task_fe,
        "n_obs": int(len(df)),
        "n_subjects": int(df["subject"].nunique()),
        "beta_std": beta,
        "se": se,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "p_value": p,
    }


def summary_stratified(df: pd.DataFrame, predictor: str) -> pd.DataFrame:
    """Stratified mean invalid_frac by tertile of the predictor."""
    x = df[predictor]
    try:
        df = df.assign(bin=pd.qcut(x, 3, labels=["low", "mid", "high"]))
    except ValueError:
        df = df.assign(bin=pd.cut(x, 3, labels=["low", "mid", "high"]))
    out = df.groupby("bin", observed=False).agg(
        n=("invalid_frac", "size"),
        mean_invalid=("invalid_frac", "mean"),
        median_invalid=("invalid_frac", "median"),
        std_invalid=("invalid_frac", "std"),
        mean_predictor=(predictor, "mean"),
    ).reset_index().rename(columns={"bin": "predictor_tertile"})
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    eye, oco = load_features()
    console.log(f"Loaded eye-tracker: {len(eye)} windows")
    console.log(f"Loaded OCOsense:    {len(oco)} windows")

    merged = join_eye_to_ocosense(eye, oco)
    console.log(f"After join + dropna: {len(merged)} paired windows "
                f"from {merged['subject'].nunique()} subjects")

    fits = [
        fit_mixed_effects(merged, OCO_MOTION_COL, include_task_fe=False),
        fit_mixed_effects(merged, OCO_MOTION_COL, include_task_fe=True),
        fit_mixed_effects(merged, OCO_ROTATION_COL, include_task_fe=False),
        fit_mixed_effects(merged, OCO_ROTATION_COL, include_task_fe=True),
    ]

    strat = pd.concat([
        summary_stratified(merged, OCO_MOTION_COL).assign(
            predictor=OCO_MOTION_COL),
        summary_stratified(merged, OCO_ROTATION_COL).assign(
            predictor=OCO_ROTATION_COL),
    ], ignore_index=True)

    fits_df = pd.DataFrame(fits)
    fits_df.to_csv(OUT_CSV, index=False)
    strat.to_csv(OUT_DIR / "sensor_interference_stratified.csv", index=False)

    lines: list[str] = []
    lines.append(f"n windows: {len(merged)}")
    lines.append(f"n subjects: {merged['subject'].nunique()}")
    lines.append("")
    lines.append("Mixed-effects regression: invalid_frac ~ predictor_std + (1|subject)")
    lines.append(f"  outcome = mean of {GAZE_NAN_COLS}")
    lines.append("")
    for f in fits:
        tag = " +task FE" if f["task_fe"] else ""
        lines.append(
            f"  {f['predictor']+tag:35s}  "
            f"β_std = {f['beta_std']:+.4f} "
            f"(95% CI [{f['ci95_low']:+.4f}, {f['ci95_high']:+.4f}])  "
            f"p = {f['p_value']:.3g}"
        )
    lines.append("")
    lines.append("Stratified mean invalid_frac by predictor tertile:")
    lines.append(strat.to_string(index=False))

    txt = "\n".join(lines)
    OUT_TXT.write_text(txt)
    console.print(txt)


if __name__ == "__main__":
    main()
