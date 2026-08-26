"""Anomaly detection and alert-event grouping utilities."""

from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

try:
    from backend.config import load_profile
except ModuleNotFoundError:
    from config import load_profile


DEFAULT_CONFIG = {
    "contamination": 0.05,
    "random_state": 42,
    "n_estimators": 200,
    "n_jobs": -1,
    "feature_cols": [
        "supply_temp",
        "return_temp",
        "heat_load",
        "outdoor_temp",
    ],
}

ROOT_CAUSE_DEFAULT_CONFIG = {
    "temperature_zscore_threshold": 3.5,
    "delta_t_abs_threshold": 0.5,
    "heat_load_high_quantile": 0.75,
    "outdoor_stability_window": 24,
    "outdoor_stability_std_threshold": 1.0,
    "heat_load_baseline_window": 24,
    "heat_load_deviation_threshold": 3.0,
}


def detect_anomalies(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Detect anomalous rows with IsolationForest.

    Only rows with complete values in the selected numeric feature columns are
    used for fitting and prediction. All original rows and columns are kept;
    rows that cannot be detected receive a NaN prediction.
    """
    result = df.copy()
    profile_config = load_profile().get("anomaly", {})
    settings = {**DEFAULT_CONFIG, **profile_config, **(config or {})}
    requested_features = settings["feature_cols"]
    if not isinstance(requested_features, (list, tuple)):
        raise TypeError("config['feature_cols'] must be a list or tuple.")

    feature_cols = [
        column
        for column in requested_features
        if column in result.columns and pd.api.types.is_numeric_dtype(result[column])
    ]
    if not feature_cols:
        raise ValueError("没有可用的数值型检测特征列。")

    valid_mask = result[feature_cols].notna().all(axis=1).to_numpy()
    detection_count = int(valid_mask.sum())
    if detection_count == 0:
        raise ValueError("检测特征列存在，但没有包含完整数值的行可供检测。")

    detector = IsolationForest(
        contamination=settings["contamination"],
        random_state=settings["random_state"],
        n_estimators=settings["n_estimators"],
        n_jobs=settings["n_jobs"],
    )
    detection_data = result.loc[valid_mask, feature_cols]
    detector.fit(detection_data)
    predictions = detector.predict(detection_data)

    result["iforest_prediction"] = np.nan
    result.loc[valid_mask, "iforest_prediction"] = predictions

    anomaly_count = int((predictions == -1).sum())
    anomaly_ratio = anomaly_count / detection_count
    print(f"实际使用的检测特征列: {feature_cols}")
    print(f"总数据行数: {len(result)}")
    print(f"实际参与检测的行数: {detection_count}")
    print(f"异常点数量: {anomaly_count}")
    print(f"异常比例: {anomaly_ratio:.4%}")
    return result


def merge_alerts(df: pd.DataFrame, gap_hours: float = 2) -> list[dict[str, Any]]:
    """Merge nearby anomalous timestamps into alert events."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df.index must be a pandas DatetimeIndex.")
    if "iforest_prediction" not in df.columns:
        raise ValueError("df must contain an iforest_prediction column.")
    if gap_hours < 0:
        raise ValueError("gap_hours must be non-negative.")

    anomaly_rows = df.loc[df["iforest_prediction"] == -1].sort_index()
    if anomaly_rows.empty:
        return []

    timestamps = anomaly_rows.index
    events: list[dict[str, Any]] = []
    start_time = timestamps[0]
    previous_time = timestamps[0]
    anomaly_count = 1
    max_gap = pd.Timedelta(hours=gap_hours)

    for timestamp in timestamps[1:]:
        if timestamp - previous_time <= max_gap:
            anomaly_count += 1
        else:
            events.append(
                {
                    "start_time": start_time,
                    "end_time": previous_time,
                    "duration_hours": (previous_time - start_time).total_seconds() / 3600,
                    "anomaly_count": anomaly_count,
                }
            )
            start_time = timestamp
            anomaly_count = 1
        previous_time = timestamp

    events.append(
        {
            "start_time": start_time,
            "end_time": previous_time,
            "duration_hours": (previous_time - start_time).total_seconds() / 3600,
            "anomaly_count": anomaly_count,
        }
    )
    return events


def root_cause_analysis(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Assign an interpretable suspected cause to anomaly rows.

    Rules are evaluated only for rows where iforest_prediction equals -1.
    Missing or unavailable fields disable only the related rule. The input
    DataFrame is copied and returned with a suspected_cause column.
    """
    if "iforest_prediction" not in df.columns:
        raise ValueError("df must contain an iforest_prediction column.")

    result = df.copy()
    profile_config = load_profile().get("anomaly", {})
    yaml_root_cause = profile_config.get("root_cause", {})
    settings = {
        **ROOT_CAUSE_DEFAULT_CONFIG,
        "delta_t_abs_threshold": yaml_root_cause.get(
            "small_delta_t", ROOT_CAUSE_DEFAULT_CONFIG["delta_t_abs_threshold"]
        ),
        "high_heat_load": yaml_root_cause.get("high_heat_load", 20.0),
        "outdoor_stability_std_threshold": yaml_root_cause.get(
            "outdoor_temp_stable",
            ROOT_CAUSE_DEFAULT_CONFIG["outdoor_stability_std_threshold"],
        ),
        "heat_load_deviation_threshold": yaml_root_cause.get(
            "local_heat_load_deviation",
            ROOT_CAUSE_DEFAULT_CONFIG["heat_load_deviation_threshold"],
        ),
        **(config or {}),
    }
    anomaly_mask = result["iforest_prediction"] == -1
    cause = pd.Series("", index=result.index, dtype="object")

    def robust_outlier_mask(column: str) -> pd.Series:
        """Return robust outliers for one available numeric column."""
        if column not in result.columns or not pd.api.types.is_numeric_dtype(result[column]):
            return pd.Series(False, index=result.index)
        series = result[column]
        if series.notna().sum() < 2:
            return pd.Series(False, index=result.index)
        median = series.median()
        mad = (series - median).abs().median()
        if mad > 0:
            score = 0.6745 * (series - median).abs() / mad
        else:
            std = series.std()
            if pd.isna(std) or std == 0:
                return pd.Series(False, index=result.index)
            score = (series - series.mean()).abs() / std
        return score > settings["temperature_zscore_threshold"]

    temperature_rule = pd.Series(False, index=result.index)
    for column in ("supply_temp", "return_temp"):
        temperature_rule |= robust_outlier_mask(column)
    cause.loc[anomaly_mask & temperature_rule] = "疑似温度传感器异常"

    remaining = anomaly_mask & cause.eq("")
    if {"delta_T", "heat_load"}.issubset(result.columns):
        delta_t = result["delta_T"]
        heat_load = result["heat_load"]
        if (
            pd.api.types.is_numeric_dtype(delta_t)
            and pd.api.types.is_numeric_dtype(heat_load)
            and heat_load.notna().any()
        ):
            high_load = heat_load >= settings["high_heat_load"]
            efficiency_rule = delta_t.abs() <= settings["delta_t_abs_threshold"]
            cause.loc[remaining & efficiency_rule & high_load] = "疑似换热效率异常"

    remaining = anomaly_mask & cause.eq("")
    if {"outdoor_temp", "heat_load"}.issubset(result.columns):
        outdoor_temp = result["outdoor_temp"]
        heat_load = result["heat_load"]
        if (
            pd.api.types.is_numeric_dtype(outdoor_temp)
            and pd.api.types.is_numeric_dtype(heat_load)
        ):
            window = int(settings["outdoor_stability_window"])
            baseline_window = int(settings["heat_load_baseline_window"])
            outdoor_stable = (
                outdoor_temp.rolling(window=window, min_periods=1).std().fillna(0)
                <= settings["outdoor_stability_std_threshold"]
            )
            local_baseline = heat_load.rolling(
                window=baseline_window,
                center=True,
                min_periods=1,
            ).median()
            load_jump = (
                heat_load - local_baseline
            ).abs() >= settings["heat_load_deviation_threshold"]
            cause.loc[remaining & outdoor_stable & load_jump] = "疑似热负荷突变"

    cause.loc[anomaly_mask & cause.eq("")] = "需要人工复核"
    result["suspected_cause"] = cause
    return result


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from backend.data_pipeline import clean_data, load_data
    from backend.feature_engineer import build_features

    data = build_features(clean_data(load_data(root / "data" / "unified_data.csv")))
    result = detect_anomalies(data)
    result = root_cause_analysis(result)
    alerts = merge_alerts(result)

    detection_count = int(result["iforest_prediction"].notna().sum())
    anomaly_count = int((result["iforest_prediction"] == -1).sum())
    print(f"总行数: {len(result)}")
    print(f"实际参与检测的行数: {detection_count}")
    print(f"异常点数: {anomaly_count}")
    print("各 suspected_cause 数量:")
    print(result.loc[result["iforest_prediction"] == -1, "suspected_cause"].value_counts())
    print(f"异常事件数: {len(alerts)}")
    for event in alerts:
        print(
            f"起始时间: {event['start_time']}, "
            f"结束时间: {event['end_time']}, "
            f"持续时间: {event['duration_hours']:.2f} 小时, "
            f"异常点数量: {event['anomaly_count']}"
        )
