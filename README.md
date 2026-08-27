# 工业 AI 运维平台

[![Version](https://img.shields.io/badge/version-v1.0.0-blue)](https://github.com/yyy2556/industrial-ai-ops-platform)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 项目简介

面向换热站场景的工业 AI 运维与能效分析平台，提供热负荷预测、异常检测、PMV/PPD 热舒适度评估、SHAP 分析和 DeepSeek 智能报告。当前为公开演示版，主要用于历史回测和辅助分析。

在线演示：

- [Streamlit Community Cloud](https://industrial-ai-ops-platform.streamlit.app/)
- [魔搭社区 ModelScope](https://modelscope.cn/studios/yyy2556/industrial-ai-ops-platform)

访问提示：Streamlit Community Cloud 初次加载可能较慢，请耐心等待。

## 系统架构

数据按照以下流程处理：

默认数据或用户上传 CSV -> 数据校验、清洗与特征工程 -> XGBoost 热负荷预测 / Isolation Forest 异常检测 -> 原因分析、事件合并、评估与可解释性分析 -> Streamlit 多页面应用

异常事件 + 手动舒适度参数 -> 诊断 Agent -> 建议 Agent -> 报告 Agent -> 智能运维报告

## 核心功能模块

- **数据管道**：CSV 上传与校验、数据清洗、时间序列切分、YAML 配置化
- **负荷预测**：XGBoost 回测、分层评估（MAE 9.90 / R² 0.86）、SHAP 可解释性
- **异常检测**：Isolation Forest 标记（34,368 行 / 737 事件）、事件合并、规则根因分析
- **热舒适**：ISO 7730 PMV/PPD 计算（典型工况 -0.213 / 5.94%）
- **智能报告**：DeepSeek 三 Agent 编排、八章节报告、Markdown 下载
- **用户与存储**：访客模式、演示登录、按用户隔离报告、SQLite 历史报告、缓存与上传限制
- **界面体验**：工业运维控制台风格、统一指标卡与图表、响应式布局和键盘焦点支持

## 快速开始

### 1. 创建并进入虚拟环境（可选但推荐）

    python -m venv .venv
    .venv\Scripts\Activate.ps1

### 2. 安装依赖

    python -m pip install -r requirements.txt

如果在国内下载速度较慢，可使用清华 PyPI 镜像：

    python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

### 3. 启动 Streamlit 页面

在项目根目录运行：

    python -m streamlit run frontend/app.py

启动后，在浏览器打开：http://localhost:8501

## 项目结构

    industrial-ai-ops-platform/
    ├── backend/
    │   ├── __init__.py            # backend Python 包标识
    │   ├── auth.py                # 演示登录和账号校验
    │   ├── data_pipeline.py       # 数据加载、清洗和时间顺序切分
    │   ├── feature_engineer.py    # 时间序列特征工程
    │   ├── forecast.py            # XGBoost 训练、预测和评估
    │   ├── digital_twin.py        # 理论热负荷和残差计算接口
    │   ├── anomaly.py             # Isolation Forest、原因分析和事件合并
    │   ├── comfort.py             # PMV/PPD 热舒适度计算
    │   ├── agents.py              # DeepSeek API 和 Agent
    │   └── config.py              # YAML 设备配置加载
    ├── config/
    │   └── device_profiles.yaml   # 换热站预测与异常检测配置
    ├── frontend/
    │   └── app.py                 # 五页 Streamlit 应用和上传入口
    ├── .streamlit/
    │   └── config.toml            # Streamlit 上传大小配置
    ├── data/
    │   └── unified_data.csv       # 默认换热站演示数据；report_history.db 运行时生成且被忽略
    ├── notebooks/                 # 探索性分析目录
    ├── requirements.txt           # Python 依赖
    └── README.md                  # 项目说明

## 核心指标

### 全量测试集

| 指标 | 结果 |
| --- | ---: |
| MAE | 9.8974 |
| RMSE | 24.0854 |
| R² | 0.8598 |

### PMV 典型工况验证

| 场景 | PMV | PPD | 舒适等级 |
| --- | ---: | ---: | --- |
| 办公室典型工况（24°C, 50%, 10°C, 0.1m/s） | -0.213 | 5.94% | 中性 |

### 按真实热负荷阈值分层

以下结果使用同一个模型和测试集，仅按照真实 heat_load 值进行筛选，不重新训练模型。

| 真实热负荷条件 | 样本数 | MAE | RMSE | R² | MAPE |
| --- | ---: | ---: | ---: | ---: | ---: |
| heat_load >= 10 | 2,178 | 23.9430 | 39.5900 | 0.6334 | 32.9075% |
| heat_load >= 20 | 2,054 | 24.5610 | 40.0062 | 0.5904 | 28.9083% |
| heat_load >= 30 | 1,938 | 25.3342 | 40.7757 | 0.5347 | 27.8754% |

由于数据中包含较多零负荷和低负荷样本，MAPE 对接近零的真实值非常敏感。因此，当前版本将 MAE、RMSE、R² 和分层结果结合起来解读，不使用单一 MAPE 判断模型效果。

## 异常检测结果

以下结果基于当前统一数据和默认 Isolation Forest 配置：

| 数据行数 | 异常点数 | 异常事件数 | 异常比例 |
| ---: | ---: | ---: | ---: |
| 34,368 | 1,719 | 737 | 5.00% |

## 智能报告与 API 安全

智能报告页面需要用户手动输入 DeepSeek API Key。API Key：

- 不写入代码、文件或 URL。
- 不进入缓存或页面状态存储。
- 只在用户点击“生成报告”时临时设置并使用。
- 报告生成结束后清理临时环境变量。

DeepSeek API 调用可能产生费用并受账户额度、请求频率和模型可用性限制。

## 数据说明

当前演示数据包含以下主要字段：

- timestamp：时间戳
- outdoor_temp：室外温度
- supply_temp：供水温度
- return_temp：回水温度
- heat_load：热负荷目标变量
- flow_rate、power_consumption、indoor_temp、humidity：当前数据中暂为空，暂不参与模型训练

当前特征中的 Q_theory 因为缺少有效 flow_rate 数据，暂不参与训练。

## 版本历史

| 版本 | 主要更新 |
|---|---|
| v0.2.0 | Isolation Forest 异常检测、事件合并、原因分析、PMV/PPD 热舒适、YAML 配置化 |
| v0.3.0 | DeepSeek 三 Agent 编排（诊断/建议/报告）、八章节智能运维报告页面 |
| v0.4.0 | CSV 上传校验、业务页面接入、SQLite 历史报告存储与查询 |
| v0.4.1 | 演示登录系统（按用户隔离报告）、缓存与上传限制（20MB/200k 行）、Markdown 报告下载、访客模式 |
| v1.0.0 | 工业运维控制台视觉升级、统一页面样式、响应式布局、访客直接使用和按需登录 |

## 当前限制

- 演示数据中的 flow_rate、power_consumption、indoor_temp、humidity 为空，数字孪生和 PMV 自动联动暂不可用，PMV 使用手动输入。
- 异常检测结果来自无监督模型，不等于已确认故障；智能报告仅供人工参考，所有建议需人工确认后执行。
- 智能报告依赖外部 DeepSeek API，不保证持续可用，API 调用可能产生费用。
- 当前登录系统为演示级（demo/demo123、admin/admin123），不适合正式公网环境；未登录可正常使用分析和 API，但报告不会保存到历史记录。
- SQLite 历史报告存储在本地，公网平台（如 Streamlit Community Cloud）不保证长期保留；CSV 上传限制为 20MB、200,000 行，缓存最多保留 3 天和最近 5 个数据集。

由于数据中包含较多零负荷和低负荷样本，百分比误差指标对接近零的真实值非常敏感。因此，当前版本的主要模型指标使用 MAE、RMSE 和 R²，并结合不同热负荷区间的分层结果进行解读，不使用单一百分比误差指标判断模型效果。

## 免责声明

本项目用于学习、研究和展示。模型结果不能直接替代现场人员的供热调度、设备控制或安全判断。
