"""YAML-backed device profile loading utilities."""

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_profile(device_type: str = "heat_exchange_station") -> dict[str, Any]:
    """Load one device profile from the project YAML configuration.

    Args:
        device_type: Top-level device profile key to load.

    Returns:
        An independent dictionary containing the selected device profile.

    Raises:
        FileNotFoundError: If config/device_profiles.yaml is missing.
        ValueError: If the requested device type is not defined.
    """
    project_root = Path(__file__).resolve().parents[1]
    profile_path = project_root / "config" / "device_profiles.yaml"
    if not profile_path.is_file():
        raise FileNotFoundError(f"设备配置文件不存在: {profile_path}")

    with profile_path.open("r", encoding="utf-8") as file:
        profiles = yaml.safe_load(file) or {}

    if device_type not in profiles:
        available = ", ".join(sorted(profiles)) or "无"
        raise ValueError(
            f"未找到设备类型 '{device_type}'。可用设备类型: {available}"
        )

    return deepcopy(profiles[device_type])


if __name__ == "__main__":
    profile = load_profile()
    print(f"display_name: {profile['display_name']}")
    print(f"forecast.target: {profile['forecast']['target']}")
    print(f"forecast.feature_cols: {profile['forecast']['feature_cols']}")
    print(f"anomaly.feature_cols: {profile['anomaly']['feature_cols']}")

    try:
        load_profile("unknown_device")
    except ValueError as exc:
        print(f"不存在设备类型测试: {exc}")
