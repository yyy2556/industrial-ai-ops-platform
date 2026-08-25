"""Basic data loading, cleaning, and time-ordered splitting utilities."""

from pathlib import Path

import pandas as pd


INTERPOLATION_COLUMNS = [
    "outdoor_temp",
    "supply_temp",
    "return_temp",
    "heat_load",
]


def load_data(path):
    """Read a CSV and use its parsed timestamp column as the index."""
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.set_index("timestamp")


def clean_data(df):
    """Sort by time and interpolate only the four usable numeric columns."""
    df = df.sort_index().copy()

    # Keep auxiliary columns unchanged, including their all-null values.
    columns = [column for column in INTERPOLATION_COLUMNS if column in df.columns]
    df[columns] = df[columns].interpolate(method="linear", limit_direction="both")
    return df


def split_train_test(df, test_ratio=0.2):
    """Split a time series in order without shuffling."""
    split_index = int(len(df) * (1 - test_ratio))
    return df.iloc[:split_index], df.iloc[split_index:]


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    data = clean_data(load_data(root / "data" / "unified_data.csv"))
    train_data, test_data = split_train_test(data)

    print(f"总行数: {len(data)}")
    print(f"训练集行数: {len(train_data)}")
    print(f"测试集行数: {len(test_data)}")
    print("各列缺失值数量:")
    print(data.isna().sum())
