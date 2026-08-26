"""Streamlit page for heat-load backtesting and model interpretation."""

from pathlib import Path
import hashlib
import io
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
import streamlit as st
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.data_pipeline import clean_data, load_data, split_train_test
from backend.digital_twin import calc_residual, calc_theoretical_load
from backend.feature_engineer import build_features
from backend.forecast import evaluate_model, predict_model, train_model
from backend.anomaly import detect_anomalies, merge_alerts, root_cause_analysis
from backend.comfort import calculate_pmv
from backend.agents import generate_full_report
from backend.report_store import delete_report, list_reports, save_report


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

REQUIRED_COLUMNS = [
    "timestamp",
    "outdoor_temp",
    "supply_temp",
    "return_temp",
    "heat_load",
]
OPTIONAL_COLUMNS = [
    "flow_rate",
    "power_consumption",
    "indoor_temp",
    "humidity",
]
NUMERIC_COLUMNS = REQUIRED_COLUMNS[1:] + OPTIONAL_COLUMNS
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "unified_data.csv"


def prepare_current_data(raw_data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Validate a CSV and return its cleaned, feature-engineered data."""
    if not isinstance(raw_data, pd.DataFrame):
        raise ValueError("上传内容不是有效的 CSV 数据。")
    if raw_data.empty:
        raise ValueError("CSV 数据为空，无法继续分析。")
    if len(raw_data) < 2:
        raise ValueError("CSV 数据行数不足，至少需要 2 行数据。")

    missing_required = [column for column in REQUIRED_COLUMNS if column not in raw_data.columns]
    if missing_required:
        raise ValueError(f"缺少必需字段: {', '.join(missing_required)}")

    data = raw_data.copy()
    parsed_timestamp = pd.to_datetime(data["timestamp"], errors="coerce")
    invalid_timestamp_count = int(parsed_timestamp.isna().sum())
    if invalid_timestamp_count:
        raise ValueError(f"timestamp 有 {invalid_timestamp_count} 个值无法转换为 datetime。")
    data["timestamp"] = parsed_timestamp

    for column in NUMERIC_COLUMNS:
        if column not in data.columns:
            continue
        original = data[column]
        converted = pd.to_numeric(original, errors="coerce")
        non_empty = original.notna() & original.astype("string").str.strip().ne("")
        invalid_count = int((non_empty & converted.isna()).sum())
        if invalid_count:
            raise ValueError(f"数值字段 {column} 有 {invalid_count} 个值无法转换为数值。")
        data[column] = converted

    empty_required = [
        column for column in REQUIRED_COLUMNS[1:] if data[column].notna().sum() == 0
    ]
    if empty_required:
        raise ValueError(f"必需数值字段完全为空: {', '.join(empty_required)}")

    duplicate_count = int(data["timestamp"].duplicated(keep=False).sum())
    warnings: list[str] = []
    if duplicate_count:
        warnings.append(f"检测到 {duplicate_count} 行重复时间戳，已保留重复行并继续处理。")

    missing_optional = [column for column in OPTIONAL_COLUMNS if column not in data.columns]
    if missing_optional:
        warnings.append(f"缺少可选字段: {', '.join(missing_optional)}，相关功能将不可用。")
        for column in missing_optional:
            data[column] = np.nan

    data = data.set_index("timestamp")
    processed = build_features(clean_data(data))
    return processed, warnings


def set_current_data(
    data: pd.DataFrame,
    source: str,
    signature: str,
    dataset_name: str | None = None,
) -> None:
    """Store the current processed data in the active Streamlit session."""
    st.session_state["current_data"] = data
    st.session_state["current_data_source"] = source
    st.session_state["current_data_signature"] = signature
    st.session_state["current_data_name"] = dataset_name or source
    st.session_state.pop("full_report_result", None)
    st.session_state.pop("full_report_event", None)


def ensure_default_data() -> None:
    """Load the demo CSV once when the session has no current data."""
    if "current_data" in st.session_state:
        return
    if not DEFAULT_DATA_PATH.is_file():
        raise FileNotFoundError(f"默认演示数据不存在: {DEFAULT_DATA_PATH}")
    raw_data = pd.read_csv(DEFAULT_DATA_PATH)
    processed, warnings = prepare_current_data(raw_data)
    stat = DEFAULT_DATA_PATH.stat()
    signature = f"default:{stat.st_mtime_ns}:{stat.st_size}"
    set_current_data(processed, "默认演示数据", signature, DEFAULT_DATA_PATH.name)
    st.session_state["current_data_warnings"] = warnings


def get_current_data() -> tuple[pd.DataFrame, str]:
    """Return current session data and its cache signature."""
    ensure_default_data()
    return st.session_state["current_data"], st.session_state["current_data_signature"]


@st.cache_data
def load_feature_data(data_signature: str, _data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the session data keyed by its current signature."""
    return _data.copy()


@st.cache_resource
def load_and_train(data_signature: str, _data: pd.DataFrame):
    """Load data and train once per Streamlit cache key."""
    data = load_feature_data(data_signature, _data)
    if len(data) < 25:
        raise ValueError("当前数据行数不足，无法可靠构造 lag24 特征或完成时间切分，至少需要 25 行。")
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


CAUSE_PRIORITY = [
    "疑似温度传感器异常",
    "疑似换热效率异常",
    "疑似热负荷突变",
    "需要人工复核",
]


def summarize_event_causes(data: pd.DataFrame, events: list[dict]) -> pd.DataFrame:
    """Add the most common suspected cause to each anomaly event."""
    rows = []
    for event in events:
        start_time = pd.Timestamp(event["start_time"])
        end_time = pd.Timestamp(event["end_time"])
        event_rows = data.loc[
            (data.index >= start_time)
            & (data.index <= end_time)
            & (data["iforest_prediction"] == -1)
        ]
        causes = event_rows["suspected_cause"].replace("", pd.NA).dropna()
        counts = causes.value_counts()
        if counts.empty:
            suspected_cause = "需要人工复核"
        else:
            max_count = counts.max()
            suspected_cause = next(
                (cause for cause in CAUSE_PRIORITY if counts.get(cause, 0) == max_count),
                "需要人工复核",
            )
        rows.append(
            {
                "起始时间": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "结束时间": end_time.strftime("%Y-%m-%d %H:%M:%S"),
                "持续时长": f"{event['duration_hours']:.2f} 小时",
                "异常点数": int(event["anomaly_count"]),
                "疑似原因": suspected_cause,
            }
        )
    return pd.DataFrame(rows, columns=["起始时间", "结束时间", "持续时长", "异常点数", "疑似原因"])


def format_event_option(event_index: int, event: pd.Series) -> str:
    """Format one anomaly event as a readable PMV selection option."""
    return (
        f"事件 {event_index + 1}: {event['起始时间']} 至 {event['结束时间']} | "
        f"原因: {event['疑似原因']} | 异常点数: {event['异常点数']}"
    )


def get_event_outdoor_temp(
    data: pd.DataFrame,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> float:
    """Return the event-period outdoor temperature or the demo fallback."""
    fallback = 10.0
    if "outdoor_temp" not in data.columns:
        return fallback
    event_values = data.loc[
        (data.index >= start_time) & (data.index <= end_time), "outdoor_temp"
    ].dropna()
    if event_values.empty:
        return fallback
    outdoor_temp = float(event_values.mean())
    return outdoor_temp if np.isfinite(outdoor_temp) else fallback


def build_report_event_data(
    data: pd.DataFrame,
    event: pd.Series,
) -> dict:
    """Build a compact anomaly summary and scalar measurements for the agents."""
    start_time = pd.to_datetime(event["起始时间"])
    end_time = pd.to_datetime(event["结束时间"])
    event_rows = data.loc[(data.index >= start_time) & (data.index <= end_time)]
    measurement_columns = [
        "supply_temp",
        "return_temp",
        "outdoor_temp",
        "heat_load",
        "delta_T",
    ]
    measurements = {}
    for column in measurement_columns:
        if column in event_rows.columns:
            values = event_rows[column].dropna()
            if not values.empty:
                measurements[column] = float(values.mean())
    return {
        "start_time": event["起始时间"],
        "end_time": event["结束时间"],
        "duration_hours": float(str(event["持续时长"]).split()[0]),
        "anomaly_count": int(event["异常点数"]),
        "suspected_cause": event["疑似原因"] or "需要人工复核",
        "measurements": measurements,
    }


def friendly_report_error(exc: Exception) -> str:
    """Map common LLM failures to safe messages for the page."""
    message = str(exc)
    if "401" in message or "Key 无效" in message or "Key" in message and "无效" in message:
        return "API Key 无效或已过期。"
    if "超时" in message or "timeout" in message.lower():
        return "请求超时，请稍后重试。"
    if "429" in message or "额度" in message or "频率" in message:
        return "API 当前不可用，请检查额度或稍后重试。"
    return "报告生成失败，请检查 API 配置或稍后重试。"


def render_report_history(data_signature: str) -> None:
    """Render current-dataset report history with view and delete actions."""
    st.subheader("历史报告")
    try:
        reports = list_reports(dataset_signature=data_signature)
    except Exception as exc:
        st.error(f"历史报告读取失败：{exc}")
        return

    if not reports:
        st.info("暂无历史报告")
        return

    history_table = pd.DataFrame(
        [
            {
                "生成时间": item["created_at"],
                "数据集名称": item["dataset_name"],
                "异常事件时间": f"{item['event_start_time']} 至 {item['event_end_time']}",
                "异常原因": item["suspected_cause"],
            }
            for item in reports
        ]
    )
    st.dataframe(history_table, use_container_width=True, hide_index=True)

    option_labels = [
        f"{item['created_at']} | {item['event_start_time']} | "
        f"{item['suspected_cause']} | 报告 ID {item['id']}"
        for item in reports
    ]
    selected_index = st.selectbox(
        "选择历史报告",
        range(len(reports)),
        format_func=lambda index: option_labels[index],
        key=f"history_report_selection_{data_signature}",
    )
    selected_report = reports[selected_index]
    st.markdown("**诊断结果**")
    st.write(selected_report["diagnosis"])
    st.markdown("**运行建议**")
    st.write(selected_report["suggestion"])
    st.markdown("**完整运维报告**")
    st.write(selected_report["report"])
    if st.button("删除选中的历史报告", key="delete_history_report"):
        try:
            if delete_report(int(selected_report["id"])):
                st.session_state["history_report_notice"] = "历史报告已删除。"
                st.rerun()
            st.warning("未找到要删除的历史报告。")
        except Exception as exc:
            st.error(f"历史报告删除失败：{exc}")


@st.cache_data
def load_anomaly_data(data_signature: str, _data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, detect, label, and group anomalies using Streamlit data cache."""
    data = load_feature_data(data_signature, _data).copy()
    if not isinstance(data.index, pd.DatetimeIndex):
        converted_index = pd.to_datetime(data.index, errors="coerce")
        if converted_index.isna().any():
            raise ValueError("异常检测数据的时间索引无法转换为 datetime。")
        data.index = converted_index

    data = detect_anomalies(data)
    data = root_cause_analysis(data)
    events = merge_alerts(data)
    event_table = summarize_event_causes(data, events)
    return data, event_table


def render_upload_page() -> None:
    """Render CSV upload, schema validation, and data preview."""
    st.title("数据上传")
    st.caption("上传有效 CSV 后，本次会话中的业务页面将使用该数据；文件不会写入项目目录。")
    uploaded_file = st.file_uploader("上传 CSV 文件", type=["csv"])

    ensure_default_data()
    if uploaded_file is None:
        if st.session_state.get("current_data_source") != "默认演示数据":
            raw_data = pd.read_csv(DEFAULT_DATA_PATH)
            processed, warnings = prepare_current_data(raw_data)
            stat = DEFAULT_DATA_PATH.stat()
            signature = f"default:{stat.st_mtime_ns}:{stat.st_size}"
            set_current_data(processed, "默认演示数据", signature, DEFAULT_DATA_PATH.name)
            st.session_state["current_data_warnings"] = warnings
    else:
        file_bytes = uploaded_file.getvalue()
        signature = f"upload:{hashlib.sha256(file_bytes).hexdigest()}"
        if st.session_state.get("current_data_signature") != signature:
            try:
                raw_data = pd.read_csv(io.BytesIO(file_bytes))
                processed, warnings = prepare_current_data(raw_data)
                set_current_data(processed, "用户上传数据", signature, uploaded_file.name)
                st.session_state["current_data_warnings"] = warnings
                st.success("CSV 校验和数据处理成功，已切换到用户上传数据。")
            except Exception as exc:
                st.error(f"CSV 校验失败：{exc}")

    data, _ = get_current_data()
    source = st.session_state.get("current_data_source", "默认演示数据")
    st.info(f"当前使用{source}")
    for warning in st.session_state.get("current_data_warnings", []):
        st.warning(warning)

    st.subheader("数据概览")
    overview = st.columns(2)
    overview[0].metric("数据行数", f"{len(data):,}")
    overview[1].metric("字段数量", f"{len(data.columns):,}")
    st.write("当前列名：", ", ".join(map(str, data.columns)))
    st.subheader("前 10 行预览")
    st.dataframe(data.head(10), use_container_width=True)


def build_anomaly_timeseries(data: pd.DataFrame) -> go.Figure:
    """Build a heat-load line with red anomaly markers overlaid."""
    anomaly_mask = data["iforest_prediction"] == -1
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=data.index,
            y=data["heat_load"],
            mode="lines",
            name="热负荷",
            line={"color": "#2563eb", "width": 1.5},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=data.index[anomaly_mask],
            y=data.loc[anomaly_mask, "heat_load"],
            mode="markers",
            name="模型标记异常",
            marker={"color": "#dc2626", "size": 7},
        )
    )
    figure.update_layout(
        xaxis_title="时间",
        yaxis_title="热负荷",
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.08, "x": 0},
    )
    return figure


def build_anomaly_hourly_figure(data: pd.DataFrame) -> go.Figure:
    """Build a complete 0-23 hour anomaly-count chart."""
    anomaly_times = data.index[data["iforest_prediction"] == -1]
    counts = pd.Series(anomaly_times.hour).value_counts().reindex(range(24), fill_value=0)
    figure = go.Figure(
        go.Bar(x=list(range(24)), y=counts.tolist(), name="异常次数", marker_color="#7c3aed")
    )
    figure.update_layout(xaxis_title="小时", yaxis_title="异常次数", xaxis={"dtick": 1})
    return figure


def render_anomaly_page() -> None:
    """Render the anomaly detection and alert review page."""
    st.title("异常检测")
    st.caption(
        "本页面基于 Isolation Forest 无监督异常检测，标记结果需结合业务规则判断，"
        "不直接作为设备故障判决依据。"
    )
    try:
        current_data, data_signature = get_current_data()
        data, event_table = load_anomaly_data(data_signature, current_data)
    except Exception as exc:
        st.error(f"异常检测页面加载失败：{exc}")
        return

    anomaly_mask = data["iforest_prediction"] == -1
    total_count = len(data)
    anomaly_count = int(anomaly_mask.sum())
    event_count = len(event_table)
    anomaly_ratio = anomaly_count / total_count * 100 if total_count else 0.0

    overview = st.columns(4)
    overview[0].metric("总行数", f"{total_count:,}")
    overview[1].metric("异常点数", f"{anomaly_count:,}")
    overview[2].metric("异常事件数", f"{event_count:,}")
    overview[3].metric("异常比例", f"{anomaly_ratio:.2f}%")

    st.subheader("热负荷异常点")
    st.plotly_chart(build_anomaly_timeseries(data), use_container_width=True)

    st.subheader("异常事件")
    cause_options = ["全部", *CAUSE_PRIORITY]
    selected_cause = st.selectbox("疑似原因筛选", cause_options)
    if selected_cause == "全部":
        filtered_events = event_table
    else:
        filtered_events = event_table[event_table["疑似原因"] == selected_cause]
    if filtered_events.empty:
        st.info("暂无异常事件")
    else:
        st.dataframe(filtered_events, use_container_width=True, hide_index=True)

    st.subheader("异常时段热舒适风险")
    st.info(
        "当前数据缺少室内温度和室内湿度，因此以下 PMV 结果使用手动输入参数，"
        "仅用于演示异常时段的热舒适度评估。"
    )
    if event_table.empty:
        st.info("暂无异常事件，暂时跳过 PMV 联动。")
    else:
        event_options = [
            format_event_option(index, event)
            for index, (_, event) in enumerate(event_table.iterrows())
        ]
        selected_event_option = st.selectbox(
            "选择异常事件",
            event_options,
            key=f"anomaly_comfort_event_{data_signature}",
        )
        selected_event_index = event_options.index(selected_event_option)
        selected_event = event_table.iloc[selected_event_index]
        selected_start = pd.to_datetime(selected_event["起始时间"])
        selected_end = pd.to_datetime(selected_event["结束时间"])
        event_outdoor_temp = get_event_outdoor_temp(data, selected_start, selected_end)

        comfort_inputs = st.columns(4)
        comfort_indoor_temp = comfort_inputs[0].number_input(
            "异常时段室内温度 (°C)",
            min_value=10.0,
            max_value=35.0,
            value=24.0,
            step=0.5,
            key="anomaly_comfort_indoor_temp",
        )
        comfort_indoor_humidity = comfort_inputs[1].number_input(
            "异常时段室内湿度 (%)",
            min_value=20.0,
            max_value=90.0,
            value=50.0,
            step=1.0,
            key="anomaly_comfort_indoor_humidity",
        )
        comfort_outdoor_temp = comfort_inputs[2].number_input(
            "异常时段室外温度 (°C)",
            min_value=-20.0,
            max_value=40.0,
            value=float(np.clip(event_outdoor_temp, -20.0, 40.0)),
            step=0.5,
            key=f"anomaly_comfort_outdoor_temp_{selected_event_index}",
        )
        comfort_air_speed = comfort_inputs[3].number_input(
            "异常时段风速 (m/s)",
            min_value=0.0,
            max_value=2.0,
            value=0.1,
            step=0.05,
            key="anomaly_comfort_air_speed",
        )

        try:
            comfort_result = calculate_pmv(
                indoor_temp=comfort_indoor_temp,
                indoor_humidity=comfort_indoor_humidity,
                outdoor_temp=comfort_outdoor_temp,
                air_speed=comfort_air_speed,
            )
            display_comfort_level = re.sub(
                r"^[+-]?\d+\s*", "", comfort_result["舒适度等级"]
            )
            comfort_results = st.columns(3)
            comfort_results[0].metric("PMV", f"{comfort_result['PMV']:.3f}")
            comfort_results[1].metric("PPD", f"{comfort_result['PPD']:.2f}%")
            comfort_results[2].metric("舒适度等级", display_comfort_level)
            st.write(
                f"当前选择事件：{selected_event['起始时间']} 至 {selected_event['结束时间']}"
            )
            st.write(
                f"当前输入参数：室内温度 {comfort_indoor_temp:.1f}°C，"
                f"室内湿度 {comfort_indoor_humidity:.1f}%，"
                f"室外温度 {comfort_outdoor_temp:.1f}°C，"
                f"风速 {comfort_air_speed:.2f} m/s"
            )
            st.caption(
                "上述 PMV/PPD 是基于手动室内参数的演示估计，不代表该异常时段的真实室内热舒适状态。"
            )
        except Exception as exc:
            st.error(f"异常时段 PMV 热舒适度计算失败：{exc}")

    st.subheader("异常小时分布")
    st.plotly_chart(build_anomaly_hourly_figure(data), use_container_width=True)


def render_prediction_page() -> None:
    """Render the existing heat-load prediction backtest page."""
    st.title("热负荷预测回测与评估")
    try:
        current_data, data_signature = get_current_data()
        model, X_test, y_test, y_pred, metrics, total_count, train_count = load_and_train(
            data_signature, current_data
        )
    except Exception as exc:
        st.error(f"页面加载失败：{exc}")
        return

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


def render_comfort_page() -> None:
    """Render the manual-input PMV/PPD thermal comfort page."""
    st.title("热舒适度")
    st.info(
        "当前页面使用手动输入参数进行热舒适度计算。当前统一数据中的 "
        "indoor_temp 和 humidity 暂为空，因此暂不进行自动数据联动。"
    )

    input_columns = st.columns(4)
    indoor_temp = input_columns[0].number_input(
        "室内温度 (°C)", min_value=10.0, max_value=35.0, value=24.0, step=0.5
    )
    indoor_humidity = input_columns[1].number_input(
        "室内湿度 (%)", min_value=20.0, max_value=90.0, value=50.0, step=1.0
    )
    outdoor_temp = input_columns[2].number_input(
        "室外温度 (°C)", min_value=-20.0, max_value=40.0, value=10.0, step=0.5
    )
    air_speed = input_columns[3].number_input(
        "风速 (m/s)", min_value=0.0, max_value=2.0, value=0.1, step=0.05
    )

    try:
        result = calculate_pmv(
            indoor_temp=indoor_temp,
            indoor_humidity=indoor_humidity,
            outdoor_temp=outdoor_temp,
            air_speed=air_speed,
        )
    except Exception as exc:
        st.error(f"PMV 热舒适度计算失败：{exc}")
        return

    display_comfort_level = re.sub(r"^[+-]?\d+\s*", "", result["舒适度等级"])
    st.subheader("计算结果")
    result_columns = st.columns(3)
    result_columns[0].metric("PMV", f"{result['PMV']:.3f}")
    result_columns[1].metric("PPD", f"{result['PPD']:.2f}%")
    result_columns[2].metric("舒适度等级", display_comfort_level)

    st.subheader("当前输入参数")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "室内温度 (°C)": indoor_temp,
                    "室内湿度 (%)": indoor_humidity,
                    "室外温度 (°C)": outdoor_temp,
                    "风速 (m/s)": air_speed,
                }
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("PMV/PPD 结果用于环境舒适度分析，不代表设备故障或安全判断。")


def render_report_page() -> None:
    """Render the manually triggered DeepSeek intelligent report page."""
    st.title("智能报告")
    st.info(
        "当前统一数据中的 indoor_temp 和 humidity 为空，因此舒适度数据使用手动输入，仅用于演示。"
    )
    st.warning(
        "生成报告会调用 DeepSeek API。诊断结果和运行建议仅供人工参考，不代表已确认设备故障，"
        "也不会自动执行控制指令。"
    )

    api_key = st.text_input("DeepSeek API Key", type="password")
    try:
        current_data, data_signature = get_current_data()
        data, event_table = load_anomaly_data(data_signature, current_data)
    except Exception as exc:
        st.error(f"异常检测结果加载失败：{exc}")
        return

    if event_table.empty:
        st.info("当前没有可用于生成报告的异常事件。")
        render_report_history(data_signature)
        return

    event_options = [
        format_event_option(index, event)
        for index, (_, event) in enumerate(event_table.iterrows())
    ]
    selected_option = st.selectbox(
        "选择异常事件",
        event_options,
        key=f"report_event_selection_{data_signature}",
    )
    selected_index = event_options.index(selected_option)
    selected_event = event_table.iloc[selected_index]
    st.dataframe(
        pd.DataFrame([selected_event.to_dict()]),
        use_container_width=True,
        hide_index=True,
    )
    selected_event_start = pd.to_datetime(selected_event["起始时间"])
    selected_event_end = pd.to_datetime(selected_event["结束时间"])
    historical_outdoor_temp = get_event_outdoor_temp(
        data, selected_event_start, selected_event_end
    )
    st.caption(
        f"异常事件室外温度（历史设备测量均值）：{historical_outdoor_temp:.1f}°C；"
        "PMV 演示室外温度为下方手动输入值，两者不直接比较。"
    )

    st.subheader("手动舒适度参数")
    comfort_inputs = st.columns(4)
    report_indoor_temp = comfort_inputs[0].number_input(
        "室内温度 (°C)", min_value=10.0, max_value=35.0, value=24.0, step=0.5, key="report_indoor_temp"
    )
    report_indoor_humidity = comfort_inputs[1].number_input(
        "室内湿度 (%)", min_value=20.0, max_value=90.0, value=50.0, step=1.0, key="report_indoor_humidity"
    )
    report_outdoor_temp = comfort_inputs[2].number_input(
        "室外温度 (°C)", min_value=-20.0, max_value=40.0, value=10.0, step=0.5, key="report_outdoor_temp"
    )
    report_air_speed = comfort_inputs[3].number_input(
        "风速 (m/s)", min_value=0.0, max_value=2.0, value=0.1, step=0.05, key="report_air_speed"
    )

    if st.button("生成报告", type="primary", key="generate_report_button"):
        if not api_key.strip():
            st.warning("请输入 DeepSeek API Key 后再生成报告。")
        else:
            try:
                comfort_result = calculate_pmv(
                    indoor_temp=report_indoor_temp,
                    indoor_humidity=report_indoor_humidity,
                    outdoor_temp=report_outdoor_temp,
                    air_speed=report_air_speed,
                )
                display_comfort_level = re.sub(
                    r"^[+-]?\d+\s*", "", comfort_result["舒适度等级"]
                )
                comfort_data = {
                    "pmv": comfort_result["PMV"],
                    "ppd": comfort_result["PPD"],
                    "comfort_level": display_comfort_level,
                    "indoor_temp": report_indoor_temp,
                    "indoor_humidity": report_indoor_humidity,
                    "outdoor_temp": report_outdoor_temp,
                    "air_speed": report_air_speed,
                }
                anomaly_data = build_report_event_data(data, selected_event)
                os.environ["DEEPSEEK_API_KEY"] = api_key
                try:
                    with st.spinner("正在生成报告..."):
                        result = generate_full_report(anomaly_data, comfort_data)
                finally:
                    os.environ.pop("DEEPSEEK_API_KEY", None)
                st.session_state["full_report_result"] = result
                st.session_state["full_report_event"] = selected_option
                try:
                    save_report(
                        {
                            "dataset_signature": data_signature,
                            "dataset_name": st.session_state.get(
                                "current_data_name",
                                st.session_state.get("current_data_source", "未命名数据集"),
                            ),
                            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "event_start_time": anomaly_data["start_time"],
                            "event_end_time": anomaly_data["end_time"],
                            "duration_hours": anomaly_data["duration_hours"],
                            "anomaly_count": anomaly_data["anomaly_count"],
                            "suspected_cause": anomaly_data["suspected_cause"],
                            "diagnosis": result["diagnosis"],
                            "suggestion": result["suggestion"],
                            "report": result["report"],
                        }
                    )
                except Exception as exc:
                    st.warning(f"报告已生成，但历史报告保存失败：{exc}")
            except Exception as exc:
                st.error(friendly_report_error(exc))

    stored_result = st.session_state.get("full_report_result")
    stored_event = st.session_state.get("full_report_event")
    if isinstance(stored_result, dict) and stored_event == selected_option:
        st.caption("以下内容仅供人工参考，不代表已确认设备故障或自动控制指令。")
        with st.expander("查看诊断摘要"):
            st.write(stored_result.get("diagnosis", "未提供"))
        with st.expander("查看运行建议"):
            st.write(stored_result.get("suggestion", "未提供"))
        st.subheader("完整运维报告")
        st.write(stored_result.get("report", "未提供"))
    elif isinstance(stored_result, dict):
        st.info("当前选择的异常事件尚未生成报告，请点击“生成报告”。")

    render_report_history(data_signature)


st.set_page_config(page_title="工业 AI 运维平台", layout="wide")
page = st.sidebar.radio("页面", ["数据上传", "负荷预测", "异常检测", "热舒适度", "智能报告"])
if page == "数据上传":
    try:
        render_upload_page()
    except Exception as exc:
        st.error(f"数据页面加载失败：{exc}")
elif page == "负荷预测":
    render_prediction_page()
elif page == "异常检测":
    render_anomaly_page()
elif page == "热舒适度":
    render_comfort_page()
else:
    render_report_page()
