"""Digital-twin baseline calculations for thermal load comparison."""

import numpy as np
import pandas as pd


def calc_theoretical_load(supply_temp, return_temp, flow_rate=None):
    """Calculate theoretical thermal load with Q = 4.18 * flow * delta_T / 3600.

    The current dataset has no usable flow-rate measurements, so passing None
    returns an all-NaN Series as a reserved baseline until flow data is added.
    """
    if flow_rate is None:
        if isinstance(supply_temp, (pd.Series, pd.Index)):
            return pd.Series(np.nan, index=supply_temp.index, name="theoretical_load")
        return pd.Series([np.nan], name="theoretical_load")

    delta_t = supply_temp - return_temp
    return pd.Series(
        4.18 * flow_rate * delta_t / 3600,
        index=getattr(supply_temp, "index", None),
        name="theoretical_load",
    )


def calc_residual(df):
    """Return measured heat load minus theoretical load for each timestamp."""
    theoretical = calc_theoretical_load(
        df["supply_temp"],
        df["return_temp"],
        df.get("flow_rate"),
    )
    return (df["heat_load"] - theoretical).rename("residual")
