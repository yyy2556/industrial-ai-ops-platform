"""Minimal DeepSeek Chat Completions API client."""

import os
from typing import Any

import requests


DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-v4-flash"
REQUEST_TIMEOUT_SECONDS = 30


def _error_from_status(status_code: int) -> RuntimeError:
    """Create a safe, actionable error without exposing request credentials."""
    if status_code == 401:
        return RuntimeError("DeepSeek API Key 无效或已过期（HTTP 401）。")
    if status_code == 403:
        return RuntimeError("DeepSeek API 权限不足（HTTP 403）。")
    if status_code == 429:
        return RuntimeError("DeepSeek API 请求频率或额度受限（HTTP 429）。")
    if status_code in (400, 404):
        return RuntimeError(
            f"DeepSeek API 模型不存在或请求参数错误（HTTP {status_code}，当前模型: {DEEPSEEK_MODEL}）。"
        )
    if 500 <= status_code <= 599:
        return RuntimeError(f"DeepSeek API 服务暂时不可用（HTTP {status_code}）。")
    return RuntimeError(f"DeepSeek API 请求失败（HTTP {status_code}）。")


def call_llm(
    prompt: str,
    system_prompt: str | None = None,
    temperature: float = 0.3,
) -> str:
    """Call the configured DeepSeek model and return only its reply text.

    The API key is read from DEEPSEEK_API_KEY and is never included in errors,
    logs, or return values. The model name is deliberately kept in one
    constant so unsupported model errors remain explicit and easy to update.
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("未检测到 DEEPSEEK_API_KEY。")
    if not isinstance(prompt, str) or not prompt.strip():
        raise RuntimeError("prompt 必须是非空字符串。")
    if not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2:
        raise RuntimeError("temperature 必须是 0 到 2 之间的数字。")

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    payload: dict[str, Any] = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            DEEPSEEK_API_URL,
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.Timeout as exc:
        raise RuntimeError("DeepSeek API 请求超时（timeout=30）。") from exc
    except requests.ConnectionError as exc:
        raise RuntimeError("无法连接 DeepSeek API，请检查网络连接。") from exc
    except requests.RequestException as exc:
        raise RuntimeError("DeepSeek API 网络请求失败。") from exc

    if response.status_code != 200:
        raise _error_from_status(response.status_code)

    try:
        response_data = response.json()
    except ValueError as exc:
        raise RuntimeError("DeepSeek API 返回内容不是有效 JSON。") from exc

    if not isinstance(response_data, dict):
        raise RuntimeError("DeepSeek API 返回 JSON 结构错误。")
    choices = response_data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("DeepSeek API 返回内容缺少 choices。")
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise RuntimeError("DeepSeek API 返回的 choices 结构错误。")
    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("DeepSeek API 返回内容缺少 message。")
    content = message.get("content")
    if not isinstance(content, str):
        raise RuntimeError("DeepSeek API 返回内容缺少 message.content。")
    return content


DIAGNOSTIC_SYSTEM_PROMPT = (
    "你是工业换热站运维诊断助手。只能根据输入数据进行分析。"
    "Isolation Forest 是无监督异常标记，不等于确认故障。"
    "信息不足时必须明确说明需要人工复核。"
    "不得编造不存在的传感器数据。"
    "不得直接给出危险的自动控制指令。"
)


def _format_measurements(measurements: object) -> str:
    """Format a bounded set of scalar measurements for the user prompt."""
    if not isinstance(measurements, dict) or not measurements:
        return "未提供"

    formatted: list[str] = []
    for key, value in list(measurements.items())[:20]:
        if not isinstance(key, str):
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            value_text = str(value)
            if len(value_text) <= 200:
                formatted.append(f"- {key}: {value_text}")
    return chr(10).join(formatted) if formatted else "未提供"


def diagnostic_agent(anomaly_data: dict) -> str:
    """Analyze one structured anomaly event through the existing LLM client.

    The function accepts an event summary rather than a DataFrame and sends
    only event metadata plus a bounded set of scalar measurements. Missing
    fields are represented as 未提供. API keys, network errors, and model
    errors remain handled centrally by call_llm().
    """
    if not isinstance(anomaly_data, dict):
        raise TypeError("anomaly_data 必须是 dict。")
    if not anomaly_data:
        raise ValueError("anomaly_data 不能为空。")

    prompt_lines = [
        "请分析以下工业换热站异常事件，并给出不超过300字的简洁诊断摘要。",
        "只说明当前观察、判断依据和信息缺口，最多列出3条可能原因。",
        "不要重复完整异常原始数据，不要扩展输入中没有的事实。",
        "",
        f"异常时间范围: {anomaly_data.get('start_time', '未提供')} 至 {anomaly_data.get('end_time', '未提供')}",
        f"持续时长（小时）: {anomaly_data.get('duration_hours', '未提供')}",
        f"异常点数: {anomaly_data.get('anomaly_count', '未提供')}",
        f"规则分析疑似原因: {anomaly_data.get('suspected_cause', '未提供')}",
        "可用测量值（其中 outdoor_temp 表示异常事件的历史设备测量）:",
        _format_measurements(anomaly_data.get("measurements")),
        "",
        "信息不足时必须写：需要人工复核。",
    ]
    user_prompt = chr(10).join(prompt_lines)
    return call_llm(user_prompt, system_prompt=DIAGNOSTIC_SYSTEM_PROMPT, temperature=0.3)


SUGGESTION_SYSTEM_PROMPT = (
    "你是工业换热站运维建议助手。只能根据输入的诊断结果和舒适度数据提出建议。"
    "信息不足时必须明确说明需要人工确认。"
    "不得编造不存在的设备数据。"
    "不得给出未经确认的确定性故障结论。"
    "不得直接下达危险的自动控制指令。"
    "所有建议必须由现场人员确认后执行。"
)


def _format_diagnosis_result(diagnosis_result: str | dict) -> str:
    """Format a bounded diagnosis text or dictionary for the user prompt."""
    if isinstance(diagnosis_result, str):
        return diagnosis_result[:4000]

    lines: list[str] = []
    for key, value in list(diagnosis_result.items())[:20]:
        if not isinstance(key, str):
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            value_text = str(value)
        elif isinstance(value, dict):
            value_text = _format_measurements(value)
        else:
            continue
        lines.append(f"- {key}: {value_text[:300]}")
    return chr(10).join(lines) if lines else "未提供"


def suggestion_agent(
    diagnosis_result: str | dict,
    comfort_data: dict | None = None,
) -> str:
    """Generate cautious, human-reviewed operation suggestions.

    The function accepts only a diagnosis summary and optional comfort values;
    it does not accept or send a complete DataFrame. API key, network, and
    model errors are handled by the existing call_llm() function.
    """
    if isinstance(diagnosis_result, str):
        if not diagnosis_result.strip():
            raise ValueError("diagnosis_result 不能为空。")
    elif isinstance(diagnosis_result, dict):
        if not diagnosis_result:
            raise ValueError("diagnosis_result 不能为空。")
    else:
        raise TypeError("diagnosis_result 必须是非空字符串或字典。")

    if comfort_data is not None and not isinstance(comfort_data, dict):
        raise TypeError("comfort_data 必须是字典或 None。")

    comfort = comfort_data or {}
    comfort_lines = [
        f"PMV: {comfort.get('pmv', '未提供')}",
        f"PPD: {comfort.get('ppd', '未提供')}",
        f"舒适度等级: {comfort.get('comfort_level', '未提供')}",
        f"室内温度: {comfort.get('indoor_temp', '未提供')}",
        f"室内湿度: {comfort.get('indoor_humidity', '未提供')}",
        f"PMV 演示室外温度: {comfort.get('outdoor_temp', '未提供')}",
    ]
    prompt_lines = [
        "请基于以下诊断结果和舒适度数据，输出不超过400字的精简运维建议。",
        "只输出建议优先级、检查对象、观察时间和人工确认事项，最多列出3条优先级建议。",
        "不要重复诊断结果的完整内容，不要生成自动控制指令。",
        "",
        "诊断结果:",
        _format_diagnosis_result(diagnosis_result),
        "",
        "舒适度数据:",
        chr(10).join(comfort_lines) if comfort_data is not None else "舒适度数据未提供。",
        "其中 comfort_data 的 outdoor_temp 是用户手动输入的 PMV 演示参数，不是异常事件历史测量。",
        "",
        "请按以下结构回答:",
        "1. 当前判断",
        "2. 建议优先级",
        "3. 建议检查的设备或传感器",
        "4. 建议观察的时间范围",
        "5. 需要人工确认的事项",
        "6. 不建议自动执行的操作",
        "信息不足时必须写：需要人工确认。不要把建议表述为已确认的故障结论。",
    ]
    user_prompt = chr(10).join(prompt_lines)
    return call_llm(user_prompt, system_prompt=SUGGESTION_SYSTEM_PROMPT, temperature=0.3)


REPORT_SYSTEM_PROMPT = (
    "你是工业换热站值班运维工程师，只能根据输入内容生成报告。"
    "不得编造不存在的传感器数据；信息不足时在对应章节写出需要人工复核。"
    "统一安全表述：Isolation Forest 异常标记不等于已确认故障，所有建议须经现场人员确认，禁止自动控制指令。"
)


def report_agent(
    diagnosis_result: str | dict,
    suggestion_result: str | dict,
    comfort_data: dict | None = None,
) -> str:
    """Generate a structured operation report from diagnosis and suggestions."""
    if isinstance(diagnosis_result, str):
        if not diagnosis_result.strip():
            raise ValueError("diagnosis_result 不能为空。")
    elif isinstance(diagnosis_result, dict):
        if not diagnosis_result:
            raise ValueError("diagnosis_result 不能为空。")
    else:
        raise TypeError("diagnosis_result 必须是非空字符串或字典。")

    if isinstance(suggestion_result, str):
        if not suggestion_result.strip():
            raise ValueError("suggestion_result 不能为空。")
    elif isinstance(suggestion_result, dict):
        if not suggestion_result:
            raise ValueError("suggestion_result 不能为空。")
    else:
        raise TypeError("suggestion_result 必须是非空字符串或字典。")

    if comfort_data is not None and not isinstance(comfort_data, dict):
        raise TypeError("comfort_data 必须是字典或 None。")

    comfort = comfort_data or {}
    comfort_lines = [
        f"PMV: {comfort.get('pmv', '未提供')}",
        f"PPD: {comfort.get('ppd', '未提供')}",
        f"舒适度等级: {comfort.get('comfort_level', '未提供')}",
        f"室内温度: {comfort.get('indoor_temp', '未提供')}",
        f"室内湿度: {comfort.get('indoor_humidity', '未提供')}",
        f"PMV 演示室外温度: {comfort.get('outdoor_temp', '未提供')}",
    ]
    prompt_lines = [
        "请根据以下诊断结果、运维建议和舒适度数据生成600到1000个中文字符的精简值班运维报告。",
        "报告负责整合诊断和建议，不要逐字重复 diagnosis_result 或 suggestion_result，也不要重复完整输入数据。",
        "必须保留八个章节；每个章节写1到3条要点。第四章建议检查对象最多5条，第七章不建议自动执行的操作最多4条，第八章数据和模型限制最多4条。",
        "安全与人工复核要求只集中表述一次，不要在多个章节重复相同警示。",
        "",
        "诊断结果:",
        _format_diagnosis_result(diagnosis_result),
        "",
        "建议结果:",
        _format_diagnosis_result(suggestion_result),
        "",
        "舒适度数据（PMV 演示参数）:",
        chr(10).join(comfort_lines) if comfort_data is not None else "舒适度数据未提供。",
        "温度口径必须分开：异常事件室外温度来自 anomaly_data 的历史设备测量，仅用于异常工况分析；PMV 演示室外温度来自 comfort_data 的用户手动输入，仅用于 PMV/PPD 计算。两者用途不同，不得表述为数据冲突、传感器不一致或相互矛盾。",
        "",
        "必须严格包含以下八个章节，章节标题不得省略:",
        "一、异常概况",
        "二、当前判断",
        "三、建议优先级",
        "四、建议检查的设备或传感器",
        "五、建议观察的时间范围",
        "六、需要人工确认的事项",
        "七、不建议自动执行的操作",
        "八、数据和模型限制",
        "若信息不足，在第六章集中写出需要人工复核；不要把异常标记写成已确认故障。",
    ]
    user_prompt = chr(10).join(prompt_lines)
    return call_llm(user_prompt, system_prompt=REPORT_SYSTEM_PROMPT, temperature=0.3)


def generate_full_report(
    anomaly_data: dict,
    comfort_data: dict | None = None,
) -> dict[str, str]:
    """Run diagnosis, suggestion, and report agents for one anomaly summary."""
    if not isinstance(anomaly_data, dict):
        raise TypeError("anomaly_data 必须是 dict。")
    if not anomaly_data:
        raise ValueError("anomaly_data 不能为空。")
    if comfort_data is not None and not isinstance(comfort_data, dict):
        raise TypeError("comfort_data 必须是字典或 None。")

    diagnosis_result = diagnostic_agent(anomaly_data)
    suggestion_result = suggestion_agent(diagnosis_result, comfort_data)
    report_result = report_agent(diagnosis_result, suggestion_result, comfort_data)
    return {
        "diagnosis": diagnosis_result,
        "suggestion": suggestion_result,
        "report": report_result,
    }


if __name__ == "__main__":
    test_anomaly_data = {
        "start_time": "2020-01-01 08:00:00",
        "end_time": "2020-01-01 10:00:00",
        "duration_hours": 2.0,
        "anomaly_count": 3,
        "suspected_cause": "需要人工复核",
        "measurements": {
            "supply_temp": 55.0,
            "return_temp": 30.0,
            "outdoor_temp": 8.0,
            "heat_load": 110.0,
            "delta_T": 25.0,
        },
    }

    from unittest.mock import patch

    with patch(__name__ + ".call_llm", return_value="本地 Mock 诊断结果"):
        mock_result = diagnostic_agent(test_anomaly_data)
    print(f"diagnostic_agent 本地结构测试通过: {mock_result}")

    test_comfort_data = {
        "pmv": -0.213,
        "ppd": 5.94,
        "comfort_level": "中性",
        "indoor_temp": 24.0,
        "indoor_humidity": 50.0,
        "outdoor_temp": 10.0,
    }
    with patch(__name__ + ".call_llm", return_value="本地 Mock 运维建议"):
        suggestion_result = suggestion_agent(
            "疑似温度传感器异常，需要人工复核。",
            test_comfort_data,
        )
    print(f"suggestion_agent 本地结构测试通过: {suggestion_result}")

    with patch(
        __name__ + ".call_llm",
        side_effect=["本地 Mock 诊断结果", "本地 Mock 运维建议", "本地 Mock 值班报告"],
    ):
        full_report = generate_full_report(test_anomaly_data, test_comfort_data)
    required_keys = {"diagnosis", "suggestion", "report"}
    if set(full_report) != required_keys or not all(full_report.values()):
        raise AssertionError("generate_full_report 本地 Mock 测试结果不完整。")
    print(f"generate_full_report 本地结构测试通过: {list(full_report)}")

    if not os.getenv("DEEPSEEK_API_KEY"):
        print("未检测到 DEEPSEEK_API_KEY，跳过真实 API 测试。")
    elif os.getenv("RUN_LLM_TEST") != "1":
        print("已检测到 API Key，但未启用真实 API 测试；设置 RUN_LLM_TEST=1 后再测试。")
    else:
        try:
            reply = call_llm(
                "你好，请用一句话介绍你自己。",
                system_prompt="你是一个简洁、专业的工业运维助手。",
                temperature=0.3,
            )
            print(reply)
        except RuntimeError as exc:
            print(f"真实 API 测试失败: {exc}")
