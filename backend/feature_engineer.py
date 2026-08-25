"""Feature engineering for cleaned hourly equipment data."""

from pathlib import Path

import pandas as pd

from .data_pipeline import clean_data, load_data


def build_features(df):
    """Add time-series features without using values from the current time."""
    result = df.copy()
    timestamp = result.index

    result["hour"] = timestamp.hour
    result["day_of_week"] = timestamp.dayofweek
    result["month"] = timestamp.month
    result["is_weekend"] = (timestamp.dayofweek >= 5).astype(int)

    heat_load = result["heat_load"]
    supply_temp = result["supply_temp"]
    result["heat_load_lag1"] = heat_load.shift(1)
    result["heat_load_lag24"] = heat_load.shift(24)
    result["heat_load_rolling_mean_24h"] = heat_load.shift(1).rolling(window=24).mean()
    result["supply_temp_rolling_mean_24h"] = supply_temp.shift(1).rolling(window=24).mean()

    result["delta_T"] = result["supply_temp"] - result["return_temp"]

    # 暂不可用：flow_rate 全为空，因此 Q_theory 保持为 NaN。
    flow_rate = result["flow_rate"] if "flow_rate" in result else pd.NA
    result["Q_theory"] = 4.18 * flow_rate * result["delta_T"] / 3600

    feature_columns = [
        "heat_load_lag1",
        "heat_load_lag24",
        "heat_load_rolling_mean_24h",
        "supply_temp_rolling_mean_24h",
        "delta_T",
    ]
    result[feature_columns] = result[feature_columns].fillna(0)
    return result


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    data = clean_data(load_data(root / "data" / "unified_data.csv"))
    features = build_features(data)

    print("前几行:")
    print(features.head())
    print("各列统计信息:")
    print(features.describe(include="all"))
