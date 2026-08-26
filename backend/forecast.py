"""Heat-load forecasting model utilities."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

try:
    from backend.config import load_profile
except ModuleNotFoundError:
    from config import load_profile


DEFAULT_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": 4,
}


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict[str, Any] | None = None,
) -> XGBRegressor:
    """Train an XGBoost regression model using the supplied features."""
    if not isinstance(X_train, pd.DataFrame):
        raise TypeError("X_train must be a pandas DataFrame.")
    if not isinstance(y_train, pd.Series):
        raise TypeError("y_train must be a pandas Series.")

    model_params = {**DEFAULT_PARAMS, **(params or {})}
    model = XGBRegressor(**model_params)
    model.fit(X_train, y_train)
    return model


def predict_model(model: XGBRegressor, X_test: pd.DataFrame) -> np.ndarray:
    """Validate feature names and predict with a trained model."""
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError("X_test must be a pandas DataFrame.")

    expected = getattr(model, "feature_names_in_", None)
    if expected is None:
        expected = getattr(model.get_booster(), "feature_names", None)
    if expected is None:
        raise ValueError("The trained model does not contain feature names for validation.")

    expected = list(expected)
    actual = list(X_test.columns)
    if actual != expected:
        raise ValueError(
            "X_test feature columns do not match the training columns. "
            f"Expected {expected}, got {actual}."
        )

    return np.asarray(model.predict(X_test))


def evaluate_model(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate MAPE, MAE, RMSE, and R2 without modifying the inputs."""
    true_values = np.asarray(y_true)
    predicted_values = np.asarray(y_pred)
    if len(true_values) != len(predicted_values):
        raise ValueError("y_true and y_pred must have the same length.")

    non_zero = true_values != 0
    if np.any(non_zero):
        mape = float(
            np.mean(
                np.abs(
                    (true_values[non_zero] - predicted_values[non_zero])
                    / true_values[non_zero]
                )
            )
            * 100
        )
    else:
        mape = float(np.nan)

    return {
        "mape": mape,
        "mae": float(mean_absolute_error(true_values, predicted_values)),
        "rmse": float(np.sqrt(mean_squared_error(true_values, predicted_values))),
        "r2": float(r2_score(true_values, predicted_values)),
    }


def evaluate_by_load_thresholds(
    y_true: pd.Series,
    y_pred: np.ndarray,
    thresholds: tuple[float, ...] = (10, 20, 30),
) -> list[dict[str, float | int]]:
    """Evaluate predictions for samples above each real-load threshold."""
    true_values = np.asarray(y_true)
    predicted_values = np.asarray(y_pred)
    if len(true_values) != len(predicted_values):
        raise ValueError("y_true and y_pred must have the same length.")

    results = []
    for threshold in thresholds:
        mask = true_values >= threshold
        sample_count = int(mask.sum())
        if sample_count == 0:
            print(f"提示: heat_load >= {threshold} 没有样本，已跳过。")
            continue

        metrics = evaluate_model(true_values[mask], predicted_values[mask])
        results.append(
            {
                "threshold": threshold,
                "sample_count": sample_count,
                **metrics,
            }
        )
    return results


if __name__ == "__main__":
    try:
        from backend.data_pipeline import clean_data, load_data, split_train_test
        from backend.feature_engineer import build_features
    except ModuleNotFoundError:
        from data_pipeline import clean_data, load_data, split_train_test
        from feature_engineer import build_features

    project_root = Path(__file__).resolve().parents[1]
    profile = load_profile("heat_exchange_station")
    forecast_config = profile.get("forecast", {})
    target_column = forecast_config.get("target")
    feature_columns = forecast_config.get("feature_cols")
    yaml_model_params = forecast_config.get("model_params", {})

    if not isinstance(target_column, str) or not target_column:
        raise ValueError("YAML forecast.target must be a non-empty string.")
    if not isinstance(feature_columns, list) or not feature_columns:
        raise ValueError("YAML forecast.feature_cols must be a non-empty list.")
    if not isinstance(yaml_model_params, dict):
        raise ValueError("YAML forecast.model_params must be a dictionary.")

    data_path = project_root / "data" / "unified_data.csv"
    data = build_features(clean_data(load_data(data_path)))

    missing_features = [column for column in feature_columns if column not in data.columns]
    if missing_features:
        raise ValueError(f"YAML forecast feature columns are missing from data: {missing_features}")
    if target_column not in data.columns:
        raise ValueError(f"YAML forecast target column is missing from data: {target_column}")

    model_params = {**DEFAULT_PARAMS, **yaml_model_params}
    print(f"实际使用的目标列: {target_column}")
    print(f"实际使用的特征列: {feature_columns}")
    print(f"实际使用的模型参数: {model_params}")

    train_data, test_data = split_train_test(data)
    X_train = train_data[feature_columns]
    y_train = train_data[target_column]
    X_test = test_data[feature_columns]
    y_test = test_data[target_column]

    if X_train.isna().any().any() or X_test.isna().any().any():
        raise ValueError("Feature data contains NaN values; clean it before training.")
    if y_train.isna().any() or y_test.isna().any():
        raise ValueError("Target data contains NaN values; clean it before training.")

    model = train_model(X_train, y_train, params=model_params)
    predictions = predict_model(model, X_test)
    metrics = evaluate_model(y_test, predictions)
    print(f"MAE: {metrics['mae']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"R2: {metrics['r2']:.6f}")
    print(f"MAPE: {metrics['mape']:.6f}%")

    threshold_metrics = evaluate_by_load_thresholds(y_test, predictions)
    print("\n按真实热负荷阈值分层评估:")
    print("threshold\tsample_count\tMAE\tRMSE\tR2\tMAPE")
    for item in threshold_metrics:
        print(
            f"{item['threshold']}\t{item['sample_count']}\t"
            f"{item['mae']:.6f}\t{item['rmse']:.6f}\t"
            f"{item['r2']:.6f}\t{item['mape']:.6f}%"
        )
