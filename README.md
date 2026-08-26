# 工业 AI 运维平台

[![Version](https://img.shields.io/badge/version-v0.3.0--dev-blue)](https://github.com/yyy2556/industrial-ai-ops-platform)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 项目简介

面向换热站场景的工业 AI 运维与能效优化平台，包含热负荷预测、无监督异常检测、异常原因分析、异常事件合并、PMV/PPD 热舒适度评估、SHAP 可解释性分析和 DeepSeek 智能报告生成。当前版本仍是历史回测、手动参数演示和辅助分析系统，不是生产系统。

## 系统架构

数据按照以下流程处理：

数据文件 unified_data.csv -> 数据加载与清洗 -> 特征工程 -> XGBoost 热负荷预测 / Isolation Forest 异常检测 -> 原因分析、事件合并、评估与可解释性分析 -> Streamlit 多页面应用

异常事件 + 手动舒适度参数 -> 诊断 Agent -> 建议 Agent -> 报告 Agent -> 智能运维报告

## 已完成功能

- 数据自动加载与清洗：读取统一 CSV，解析时间戳，按时间排序并进行缺失值插值。
- 时间序列切分：按照时间顺序划分训练集和测试集，不随机打乱数据。
- 特征工程：构造小时、星期、月份、周末、负荷滞后、24 小时滚动均值、供水温度滚动均值和供回水温差等特征。
- XGBoost 模型训练：支持默认参数训练和自定义参数覆盖。
- 模型评估：输出 MAE、RMSE、R² 和 MAPE，并按真实热负荷阈值进行分层评估。
- Streamlit 回测看板：展示数据概览、指标卡片、真实值与预测值曲线和分层评估表。
- SHAP 特征贡献分析：展示单条测试样本中主要特征对预测结果的影响。
- Isolation Forest 异常检测：使用无监督模型标记异常点，并保留检测结果供后续分析。
- 异常事件合并：按时间间隔将相邻异常点合并为异常事件。
- 异常原因分析：根据温度、温差、室外温度和热负荷等字段生成可解释的疑似原因。
- PMV/PPD 热舒适度评估：基于 ISO 7730 算法计算 PMV、PPD 和舒适度等级。
- Streamlit 多页面应用：包含负荷预测、异常检测和热舒适度页面。
- YAML 配置化：设备类型、预测特征、异常检测特征、模型参数和部分异常规则集中维护在 config/device_profiles.yaml。
- 预测与异常模块从 YAML 读取配置，减少算法参数和业务配置的重复维护。
- DeepSeek API 底层调用：通过 HTTPS Chat Completions 接口调用指定模型。
- API Key 密码输入和安全处理：页面使用密码框，Key 仅在用户点击生成报告时临时使用。
- 诊断 Agent：基于结构化异常事件摘要生成有限长度的诊断摘要。
- 建议 Agent：根据诊断结果和可选舒适度数据生成需要人工确认的运维建议。
- 完整报告 Agent：整合诊断和建议，生成八章节智能运维报告。
- 八章节智能运维报告：覆盖异常概况、判断、优先级、检查对象、观察时间、人工确认、禁止自动执行操作和限制说明。
- Streamlit 智能报告页面：选择异常事件、填写手动舒适度参数并按需生成报告。
- 当前页面明确标注为历史回测或手动演示，不将分析结果表述为真实未来预测或已确认故障。

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
    ├── data/
    │   └── unified_data.csv       # 统一换热站演示数据
    ├── frontend/
    │   └── app.py                 # 四页 Streamlit 应用
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

## v0.2.0 已完成

- Isolation Forest 异常检测
- 异常事件合并
- 异常原因分析
- PMV/PPD 热舒适度评估
- 异常检测与热舒适度页面
- YAML 配置化
- 预测和异常模块从 YAML 读取配置

## v0.3.0 开发中

- 完成 DeepSeek API 底层调用和 API Key 环境变量接入。
- 完成诊断 Agent、建议 Agent、报告 Agent 和完整报告编排。
- 完成智能报告页面，并与异常事件及手动舒适度参数联动。

## 当前限制与使用说明

- 当前统一数据中的 flow_rate、indoor_temp、humidity 为空。
- flow_rate 为空时，数字孪生理论热负荷和残差无法形成有效业务结论。
- PMV 页面不从空的 indoor_temp 和 humidity 字段读取数据，使用手动输入参数进行演示。
- 异常检测是 Isolation Forest 无监督模型的标记结果，需要结合业务规则和现场信息判断，不等于已经确认的设备故障。
- 异常时段热舒适度联动使用手动输入的室内温度和室内湿度，不宣称为异常时段的真实 PMV。
- 当前页面是历史回测和演示页面，不是接入实时设备数据的在线预测系统。
- 当前模型使用固定的一组 XGBoost 参数，尚未进行系统超参数搜索。
- 分层评估用于观察不同负荷工况下的表现，不代表模型在所有运行条件下都达到相同效果。
- 本项目迁移自三个独立的旧项目（异常检测、负荷预测、PMV 热舒适），旧项目代码仅作为迁移参考，不包含在本仓库中。
- PMV 页面使用手动输入，因为当前统一数据中的 indoor_temp 和 humidity 为空；异常时段的 PMV 结果仅用于演示，不代表真实室内状态。
- 智能报告内容仅供人工参考；Isolation Forest 异常标记不等于已确认故障。
- 报告不会自动执行设备控制，所有建议必须由现场人员确认后执行。
- 报告生成依赖外部 DeepSeek API，不保证服务持续可用，也不代表生产环境效果。

由于数据中包含较多零负荷和低负荷样本，百分比误差指标对接近零的真实值非常敏感。因此，当前版本的主要模型指标使用 MAE、RMSE 和 R²，并结合不同热负荷区间的分层结果进行解读，不使用单一百分比误差指标判断模型效果。

## 版本路线

- v1.0.0：完整智能运维报告和平台整合。

## 免责声明

本项目用于学习、研究和展示。模型结果不能直接替代现场人员的供热调度、设备控制或安全判断。
