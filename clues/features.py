"""Shared feature extraction functions for cognitive load classification."""

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, linregress
from tqdm import tqdm


def compute_basic_stats(df_window, columns):
    features = {}
    for col in columns:
        if col in df_window.columns:
            features[f"{col}_mean"] = df_window[col].mean()
            features[f"{col}_std"] = df_window[col].std()
            features[f"{col}_median"] = df_window[col].median()
    return features


def compute_quartiles(df_window, columns):
    features = {}
    for col in columns:
        q1 = df_window[col].quantile(0.25)
        q2 = df_window[col].quantile(0.5)
        q3 = df_window[col].quantile(0.75)
        features[f"{col}_q1"] = q1
        features[f"{col}_q2"] = q2
        features[f"{col}_q3"] = q3
    return features


def compute_expression_counts(df_window, col, values):
    features = {}
    if col not in df_window.columns:
        return features
    for v in values:
        features[f"{col}_count_{v}"] = (
            df_window[col]
            .fillna("")
            .apply(lambda x: x.lower().replace(" ", "_") if isinstance(x, str) else "")
            == v
        ).sum()
    return features


def compute_signal_magnitude_area(df_window, axes):
    features = {}
    for sensor in axes:
        x, y, z = axes[sensor]
        sma = (df_window[x].abs() + df_window[y].abs() + df_window[z].abs()).mean()
        features[f"{sensor}_sma"] = sma
    return features


def compute_energy(df_window, columns):
    features = {}
    for col in columns:
        features[f"{col}_energy"] = (df_window[col] ** 2).sum() / len(df_window)
    return features


def compute_range(df_window, columns):
    features = {}
    for col in columns:
        features[f"{col}_range"] = df_window[col].max() - df_window[col].min()
    return features


def compute_skew_kurtosis(df_window, columns):
    features = {}
    for col in columns:
        features[f"{col}_skew"] = skew(df_window[col])
        features[f"{col}_kurtosis"] = kurtosis(df_window[col])
    return features


def compute_zero_crossing(df_window, columns):
    features = {}
    for col in columns:
        data = df_window[col].values
        features[f"{col}_zero_crossings"] = ((data[:-1] * data[1:]) < 0).sum()
    return features


def compute_axis_correlations(df_window, axes):
    features = {}
    for sensor in axes:
        x, y, z = axes[sensor]
        features[f"{sensor}_corr_xy"] = df_window[[x, y]].corr().iloc[0, 1]
        features[f"{sensor}_corr_xz"] = df_window[[x, z]].corr().iloc[0, 1]
        features[f"{sensor}_corr_yz"] = df_window[[y, z]].corr().iloc[0, 1]
    return features


def compute_ml_proba_stats(df_window, proba_cols):
    features = {}
    for col in proba_cols:
        features[f"{col}_mean"] = df_window[col].mean()
        features[f"{col}_std"] = df_window[col].std()
        features[f"{col}_max"] = df_window[col].max()
        features[f"{col}_min"] = df_window[col].min()
    proba_arr = df_window[proba_cols].values
    argmaxes = np.argmax(proba_arr, axis=1)
    for i in range(proba_arr.shape[1]):
        features[f"ml_pred_proba_argmax_count_{i}"] = (argmaxes == i).sum()
    return features


def compute_trend(df_window, columns):
    features = {}
    for col in columns:
        y = df_window[col].values
        x = np.arange(len(y))
        if len(y) > 1:
            slope = linregress(x, y).slope
        else:
            slope = 0.0
        features[f"{col}_trend"] = slope
    return features


def compute_first_last_diff(df_window, columns):
    features = {}
    for col in columns:
        features[f"{col}_first"] = df_window[col].iloc[0]
        features[f"{col}_last"] = df_window[col].iloc[-1]
        features[f"{col}_diff"] = df_window[col].iloc[-1] - df_window[col].iloc[0]
    return features


def extract_features(df, sensor_axes, ml_cols, proba_cols, expr_col, expr_values, cont_ml_cols, feature_config):
    """Extract statistical features per (subject, window_index, experimental_phase) group."""
    features_list = []
    group_cols = ["subject", "window_index", "experimental_phase"]
    for (subject, window_index, phase), df_window in tqdm(df.groupby(group_cols), desc="Extracting features"):
        feats = {"subject": subject, "window_index": window_index, "experimental_phase": phase}
        if feature_config.get("basic_stats"):
            feats.update(compute_basic_stats(df_window, sensor_axes["all"] + ml_cols))
            feats.update(compute_quartiles(df_window, sensor_axes["all"] + ml_cols))
        if feature_config.get("expression_counts"):
            feats.update(compute_expression_counts(df_window, expr_col, expr_values))
        if feature_config.get("sma"):
            feats.update(compute_signal_magnitude_area(df_window, {k: v for k, v in sensor_axes.items() if k != "all"}))
        if feature_config.get("energy"):
            feats.update(compute_energy(df_window, sensor_axes["all"]))
        if feature_config.get("range"):
            feats.update(compute_range(df_window, sensor_axes["all"]))
        if feature_config.get("skew_kurtosis"):
            feats.update(compute_skew_kurtosis(df_window, sensor_axes["all"]))
        if feature_config.get("zero_crossing"):
            feats.update(compute_zero_crossing(df_window, sensor_axes["all"]))
        if feature_config.get("axis_correlations"):
            feats.update(compute_axis_correlations(df_window, {k: v for k, v in sensor_axes.items() if k != "all"}))
        if feature_config.get("ml_proba_stats"):
            feats.update(compute_ml_proba_stats(df_window, proba_cols))
        if feature_config.get("trend"):
            feats.update(compute_trend(df_window, cont_ml_cols))
        if feature_config.get("first_last_diff"):
            feats.update(compute_first_last_diff(df_window, cont_ml_cols))
        features_list.append(feats)
    return pd.DataFrame(features_list)
