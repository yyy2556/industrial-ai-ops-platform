"""PMV/PPD thermal comfort calculation based on ISO 7730."""

from __future__ import annotations

import math
from typing import Any


def _validate_inputs(indoor_temp: float, indoor_humidity: float, air_speed: float) -> None:
    """Validate the user-facing thermal comfort inputs."""
    values = {
        "indoor_temp": indoor_temp,
        "indoor_humidity": indoor_humidity,
        "air_speed": air_speed,
    }
    for name, value in values.items():
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
    if not 0 <= indoor_humidity <= 100:
        raise ValueError("indoor_humidity must be between 0 and 100 percent")
    if air_speed < 0:
        raise ValueError("air_speed must be non-negative")


def _calculate_pmv_value(
    air_temp: float,
    mean_radiant_temp: float,
    relative_humidity: float,
    air_speed: float,
    met: float = 1.2,
    clo: float = 0.5,
) -> float:
    """Calculate PMV with the ISO 7730 clothing-temperature iteration."""
    if met <= 0 or clo < 0:
        raise ValueError("met must be positive and clo must be non-negative")

    icl = 0.155 * clo
    metabolic_rate = met * 58.15
    clothing_area = 1.05 + 0.645 * icl if icl > 0.078 else 1.0 + 1.29 * icl
    vapor_pressure = relative_humidity * 10.0 * math.exp(
        16.6536 - 4030.183 / (air_temp + 235.0)
    )
    air_temp_kelvin = air_temp + 273.0
    radiant_temp_kelvin = mean_radiant_temp + 273.0
    clothing_temp = air_temp_kelvin + (
        35.7 - 0.028 * metabolic_rate - air_temp_kelvin
    ) / (3.5 * (6.45 * icl + 0.1))

    p1 = icl * clothing_area
    p2 = p1 * 3.96
    p3 = p1 * 100.0
    p4 = p1 * air_temp_kelvin
    p5 = 308.7 - 0.028 * metabolic_rate + p2 * (radiant_temp_kelvin / 100.0) ** 4
    xn = clothing_temp / 100.0
    xf = clothing_temp / 50.0

    for _ in range(150):
        xf = (xf + xn) / 2.0
        convective_natural = 12.1 * math.sqrt(air_speed)
        convective_forced = 2.38 * abs(100.0 * xf - air_temp_kelvin) ** 0.25
        convective = max(convective_natural, convective_forced)
        xn_next = (p5 + p4 * convective - p2 * xn**4) / (100.0 + p3 * convective)
        if abs(xn_next - xn) <= 0.00015:
            xn = xn_next
            break
        xn = xn_next
    else:
        raise RuntimeError("clothing temperature iteration did not converge")

    clothing_surface_temp = 100.0 * xn - 273.0
    convective = max(
        12.1 * math.sqrt(air_speed),
        2.38 * abs(100.0 * xn - air_temp_kelvin) ** 0.25,
    )
    heat_loss_diffusion = 3.05e-3 * (5733.0 - 6.99 * metabolic_rate - vapor_pressure)
    heat_loss_sweating = (
        0.42 * (metabolic_rate - 58.15) if metabolic_rate > 58.15 else 0.0
    )
    heat_loss_respiration_latent = 1.7e-5 * metabolic_rate * (5867.0 - vapor_pressure)
    heat_loss_respiration_dry = 0.0014 * metabolic_rate * (34.0 - air_temp)
    heat_loss_radiation = 3.96 * clothing_area * (
        xn**4 - (radiant_temp_kelvin / 100.0) ** 4
    )
    heat_loss_convection = clothing_area * convective * (
        clothing_surface_temp - air_temp
    )
    thermal_load = (
        metabolic_rate
        - heat_loss_diffusion
        - heat_loss_sweating
        - heat_loss_respiration_latent
        - heat_loss_respiration_dry
        - heat_loss_radiation
        - heat_loss_convection
    )
    pmv = (0.303 * math.exp(-0.036 * metabolic_rate) + 0.028) * thermal_load
    return max(-3.0, min(3.0, pmv))


def _comfort_level(pmv: float) -> str:
    """Convert PMV to a simple Chinese comfort level."""
    levels = [
        (-2.5, "-3 冷"),
        (-1.5, "-2 凉"),
        (-0.5, "-1 微凉"),
        (0.5, "0 中性"),
        (1.5, "+1 微暖"),
        (2.5, "+2 暖"),
    ]
    for boundary, label in levels:
        if pmv < boundary:
            return label
    return "+3 热"


def calculate_pmv(
    indoor_temp: float,
    indoor_humidity: float,
    outdoor_temp: float,
    air_speed: float = 0.1,
) -> dict[str, Any]:
    """Calculate PMV, PPD, and comfort level for a demonstration condition.

    The current unified data has no usable indoor temperature or humidity, so
    this function is standalone and does not read project data. The outdoor
    temperature is retained for the future page interface but is not used by
    the indoor PMV equation. Until mean radiant temperature, metabolic rate,
    and clothing insulation are provided, the calculation uses tr equal to
    indoor temperature, met equal to 1.2, and clo equal to 0.5.

    Returns a dictionary containing PMV, PPD in percent, and the comfort level.
    """
    _validate_inputs(indoor_temp, indoor_humidity, air_speed)
    if not isinstance(outdoor_temp, (int, float)) or not math.isfinite(outdoor_temp):
        raise ValueError("outdoor_temp must be a finite number")

    pmv = _calculate_pmv_value(
        air_temp=indoor_temp,
        mean_radiant_temp=indoor_temp,
        relative_humidity=indoor_humidity,
        air_speed=air_speed,
    )
    ppd = 100.0 - 95.0 * math.exp(-0.03353 * pmv**4 - 0.2179 * pmv**2)
    return {
        "PMV": pmv,
        "PPD": ppd,
        "舒适度等级": _comfort_level(pmv),
    }


if __name__ == "__main__":
    demo_result = calculate_pmv(
        indoor_temp=24.0,
        indoor_humidity=50.0,
        outdoor_temp=10.0,
    )
    print("PMV 热舒适度演示")
    print(f"PMV: {demo_result['PMV']:.3f}")
    print(f"PPD: {demo_result['PPD']:.2f}%")
    print(f"舒适度等级: {demo_result['舒适度等级']}")
