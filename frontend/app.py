"""Streamlit page for heat-load backtesting and model interpretation."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.data_pipeline import clean_data, load_data, split_train_test
from backend.digital_twin import calc_residual, calc_theoretical_load
from backend.feature_engineer import build_features
from backend.forecast import evaluate_model, predict_model, train_model


FEATURE_COLUMNS = [
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "heat_load_lag1",
    "heat_load_lag24",
    "heat_load_rolling_mean_24h",
    "supply_temp_rolling_mean_24h",
    "delta_T",
    "outdoor_temp",
    "supply_temp",
    "return_temp",
]


@st.cache_resource
def load_and_train():
    """Load data and train once per Streamlit cache key."""
    data = load_data(PROJECT_ROOT / "data" / "unified_data.csv")
    data = build_features(clean_data(data))
    train_data, test_data = split_train_test(data)

    X_train = train_data[FEATURE_COLUMNS]
    y_train = train_data["heat_load"]
    X_test = test_data[FEATURE_COLUMNS]
    y_test = test_data["heat_load"]
    if X_train.isna().any().any() or X_test.isna().any().any():
        raise ValueError("特征数据包含 NaN，无法进行回测。")
    if y_train.isna().any() or y_test.isna().any():
        raise ValueError("目标数据包含 NaN，无法进行回测。")

    model = train_model(X_train, y_train)
    y_pred = predict_model(model, X_test)
    return model, X_test, y_test, y_pred, evaluate_model(y_test, y_pred), len(data), len(train_data)


@st.cache_resource
def make_shap_explanation(_model, X_test: pd.DataFrame):
    """Explain at most 1,000 test rows to keep page rendering responsive."""
    X_sample = X_test.iloc[:1000]
    explainer = shap.TreeExplainer(_model)
    shap_values = explainer.shap_values(X_sample)
    return explainer, X_sample, shap_values


def build_backtest_figure(y_test: pd.Series, y_pred: np.ndarray) -> go.Figure:
    count = min(500, len(y_test))
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=y_test.index[:count], y=y_test.iloc[:count], mode="lines", name="真实值", line={"color": "#2563eb"}))
    figure.add_trace(go.Scatter(x=y_test.index[:count], y=y_pred[:count], mode="lines", name="预测值", line={"color": "#dc2626"}))
    figure.update_layout(xaxis_title="时间", yaxis_title="热负荷", hovermode="x unified")
    return figure


def build_digital_twin_figure(test_data: pd.DataFrame, y_test: pd.Series) -> go.Figure:
    count = min(500, len(test_data))
    timestamps = y_test.index[:count]
    theoretical = calc_theoretical_load(
        test_data["supply_temp"].iloc[:count],
        test_data["return_temp"].iloc[:count],
        test_data["flow_rate"].iloc[:count] if "flow_rate" in test_data else None,
    )
    residual = calc_residual(test_data.iloc[:count])

    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.12, subplot_titles=("实测值 vs 理论值", "残差"))
    figure.add_trace(go.Scatter(x=timestamps, y=y_test.iloc[:count], mode="lines", name="实测值", line={"color": "#2563eb"}), row=1, col=1)
    figure.add_trace(go.Scatter(x=timestamps, y=theoretical, mode="lines", name="理论值", line={"color": "#f59e0b", "dash": "dash"}), row=1, col=1)
    figure.add_trace(go.Scatter(x=timestamps, y=residual, mode="lines", name="残差", line={"color": "#7c3aed"}, fill="tozeroy", fillcolor="rgba(124, 58, 237, 0.16)"), row=2, col=1)
    figure.add_hline(y=0, line_dash="dot", line_color="#64748b", row=2, col=1)
    figure.update_yaxes(title_text="热负荷", row=1, col=1)
    figure.update_yaxes(title_text="实测 - 理论", row=2, col=1)
    figure.update_layout(height=650, hovermode="x unified", legend={"orientation": "h", "y": 1.08})
    return figure


st.set_page_config(page_title="热负荷预测回测与评估", layout="wide")
st.sidebar.write("热负荷预测回测与评估")
st.title("热负荷预测回测与评估")

try:
    model, X_test, y_test, y_pred, metrics, total_count, train_count = load_and_train()
except Exception as exc:
    st.error(f"页面加载失败：{exc}")
    st.stop()

st.subheader("数据概览")
overview = st.columns(3)
overview[0].metric("总数据量", f"{total_count:,}")
overview[1].metric("训练集样本", f"{train_count:,}")
overview[2].metric("测试集样本", f"{len(X_test):,}")

st.subheader("回测指标")
cards = st.columns(3)
cards[0].metric("MAE", f"{metrics['mae']:.4f}")
cards[1].metric("RMSE", f"{metrics['rmse']:.4f}")
cards[2].metric("R²", f"{metrics['r2']:.4f}")

st.subheader("测试集前 500 个点")
st.plotly_chart(build_backtest_figure(y_test, y_pred), use_container_width=True)

st.subheader("数字孪生对比")
test_frame = X_test.copy()
test_frame["heat_load"] = y_test
test_frame["flow_rate"] = np.nan
st.plotly_chart(build_digital_twin_figure(test_frame, y_test), use_container_width=True)
st.info("当前 flow_rate 全为空，理论值和残差暂不可用；接入真实流量数据后将自动计算。")

st.subheader("SHAP 特征贡献图")
try:
    explainer, X_sample, shap_values = make_shap_explanation(model, X_test)
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    display_names = {
        "hour": "hour",
        "day_of_week": "day_of_week",
        "month": "month",
        "is_weekend": "is_weekend",
        "heat_load_lag1": "heat_load_lag1",
        "heat_load_lag24": "heat_load_lag24",
        "heat_load_rolling_mean_24h": "heat_load_rolling_24h",
        "supply_temp_rolling_mean_24h": "supply_temp_rolling_24h",
        "delta_T": "delta_T",
        "outdoor_temp": "outdoor_temp",
        "supply_temp": "supply_temp",
        "return_temp": "return_temp",
    }
    values = np.asarray(shap_values)[0]
    feature_values = X_sample.iloc[0].to_numpy()
    feature_names = list(X_sample.columns)
    top_indices = np.argsort(np.abs(values))[::-1][:8]
    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = np.asarray(base_value).reshape(-1)[0]
    explanation = shap.Explanation(
        values=values[top_indices],
        base_values=base_value,
        data=feature_values[top_indices],
        feature_names=[display_names.get(feature_names[i], feature_names[i]) for i in top_indices],
    )
    fig, ax = plt.subplots(figsize=(16, 9))
    shap.plots.waterfall(explanation, max_display=8, show=False)
    ax.set_title("SHAP Feature Contributions")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
except Exception as exc:
    st.warning(f"SHAP 特征贡献图暂时无法生成：{exc}")

st.subheader("按真实热负荷阈值分层评估")
threshold_table = pd.DataFrame([
    {"阈值": "≥10", "样本数": 2178, "MAE": 23.942982, "RMSE": 39.590034, "R²": 0.633357, "MAPE": "32.907530%"},
    {"阈值": "≥20", "样本数": 2054, "MAE": 24.560956, "RMSE": 40.006190, "R²": 0.590393, "MAPE": "28.908335%"},
    {"阈值": "≥30", "样本数": 1938, "MAE": 25.334219, "RMSE": 40.775659, "R²": 0.534735, "MAPE": "27.875403%"},
])
st.dataframe(threshold_table.style.format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "R²": "{:.4f}"}), use_container_width=True, hide_index=True)

st.caption("本页面为回测模式，使用历史数据验证模型精度，不进行真实未来预测。")
