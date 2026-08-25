# 工业 AI 运维平台

[![Version](https://img.shields.io/badge/version-v0.1.0-blue)](https://github.com/yyy2556/industrial-ai-ops-platform/releases/tag/v0.1.0)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 项目简介

基于 XGBoost 的换热站热负荷预测回测平台，包含数据管道、特征工程、模型训练、误差评估和 SHAP 可解释性分析。当前版本面向历史数据回测，用于验证模型在不同热负荷工况下的预测效果。

## 系统架构

数据按照以下流程处理：

数据文件 unified_data.csv -> 数据加载、清洗与时间顺序切分 -> 特征工程 -> XGBoost 热负荷回归模型 -> 预测结果、误差指标、分层评估、SHAP 分析 -> Streamlit 回测页面

## 已完成功能

- 数据自动加载与清洗：读取统一 CSV，解析时间戳，按时间排序并进行缺失值插值。
- 时间序列切分：按照时间顺序划分训练集和测试集，不随机打乱数据。
- 特征工程：构造小时、星期、月份、周末、负荷滞后、24 小时滚动均值、供水温度滚动均值和供回水温差等特征。
- XGBoost 模型训练：支持默认参数训练和自定义参数覆盖。
- 模型评估：输出 MAE、RMSE、R² 和 MAPE，并按真实热负荷阈值进行分层评估。
- Streamlit 回测看板：展示数据概览、指标卡片、真实值与预测值曲线和分层评估表。
- SHAP 特征贡献分析：展示单条测试样本中主要特征对预测结果的影响。
- 当前页面明确标注为历史回测，不将回测结果表述为真实未来预测效果。

## 快速开始

### 1. 创建并进入虚拟环境（可选但推荐）

    python -m venv .venv
    .venv\Scripts\Activate.ps1

### 2. 安装依赖

    python -m pip install -r requirements.txt

### 3. 启动 Streamlit 页面

在项目根目录运行：

    python -m streamlit run frontend/app.py

启动后，在浏览器打开：http://localhost:8501

## 项目结构

    industrial-ai-ops-platform/
    ├── backend/
    │   ├── __init__.py            # backend Python 包标识
    │   ├── data_pipeline.py       # 数据加载、清洗和时间顺序切分
    │   ├── feature_engineer.py    # 时间序列特征工程
    │   ├── forecast.py            # XGBoost 训练、预测和评估
    │   ├── digital_twin.py        # 理论热负荷和残差计算接口
    │   ├── anomaly.py             # 异常检测模块预留
    │   ├── comfort.py             # 热舒适度模块预留
    │   ├── agents.py              # 大模型 Agent 模块预留
    │   └── config.py              # 配置模块预留
    ├── data/
    │   └── unified_data.csv       # 统一换热站演示数据
    ├── frontend/
    │   └── app.py                 # Streamlit 回测页面
    ├── notebooks/                 # 探索性分析目录
    ├── requirements.txt           # Python 依赖
    └── README.md                  # 项目说明

old project/ 为本地历史项目资料，仅作为迁移参考，不纳入最终仓库。

## 核心指标

### 全量测试集

| 指标 | 结果 |
| --- | ---: |
| MAE | 9.8974 |
| RMSE | 24.0854 |
| R² | 0.8598 |

### 按真实热负荷阈值分层

以下结果使用同一个模型和测试集，仅按照真实 heat_load 值进行筛选，不重新训练模型。

| 真实热负荷条件 | 样本数 | MAE | RMSE | R² | MAPE |
| --- | ---: | ---: | ---: | ---: | ---: |
| heat_load >= 10 | 2,178 | 23.9430 | 39.5900 | 0.6334 | 32.9075% |
| heat_load >= 20 | 2,054 | 24.5610 | 40.0062 | 0.5904 | 28.9083% |
| heat_load >= 30 | 1,938 | 25.3342 | 40.7757 | 0.5347 | 27.8754% |

由于数据中包含较多零负荷和低负荷样本，MAPE 对接近零的真实值非常敏感。因此，当前版本将 MAE、RMSE、R² 和分层结果结合起来解读，不使用单一 MAPE 判断模型效果。

## 数据说明

当前演示数据包含以下主要字段：

- timestamp：时间戳
- outdoor_temp：室外温度
- supply_temp：供水温度
- return_temp：回水温度
- heat_load：热负荷目标变量
- flow_rate、power_consumption、indoor_temp、humidity：当前数据中暂为空，暂不参与模型训练

当前特征中的 Q_theory 因为缺少有效 flow_rate 数据，暂不参与训练。

## 当前限制

- 当前页面是历史回测页面，不是接入实时设备数据的在线预测系统。
- flow_rate 当前为空，因此数字孪生理论热负荷和残差分析暂不可用于有效业务判断。
- 当前模型使用固定的一组 XGBoost 参数，尚未进行系统超参数搜索。
- 分层评估用于观察不同负荷工况下的表现，不代表模型在所有运行条件下都达到相同效果。

## 版本路线

- v0.2.0：整合换热站异常检测、异常告警合并、根因分析和 PMV 热舒适度模块。
- v1.0.0：接入大模型 Agent，生成异常诊断、运行建议和结构化运维报告。

## 免责声明

本项目用于学习、研究和面试展示。模型结果不能直接替代现场人员的供热调度、设备控制或安全判断。
