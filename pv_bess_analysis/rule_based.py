import numpy as np
import pandas as pd

from .config import (
    DT_HOURS,
    ETA_C,
    ETA_D,
    MONTH_NAMES,
    P_CH_MAX,
    P_DIS_MAX,
    SOC0,
    SOC_MAX,
    SOC_MIN,
    TOTAL_EB,
)
from .eda import build_daily_wide, plot_battery_profile


def rule_based_MSC(
    df,
    E_b=TOTAL_EB,
    SOC0_pct=SOC0,
    SOC_min_pct=SOC_MIN,
    SOC_max_pct=SOC_MAX,
    P_b_ch=P_CH_MAX,
    P_b_dis=P_DIS_MAX,
    eta_c=ETA_C,
    eta_d=ETA_D,
    dt_hours=DT_HOURS,
    pv_col="Total Solar Power(W)",
    load_col="Total Consumption Power(W)",
):
    P_pv = df[pv_col].to_numpy() / 1000.0
    P_de = df[load_col].to_numpy() / 1000.0
    SOC_min = SOC_min_pct / 100.0
    SOC_max = SOC_max_pct / 100.0
    SOC = SOC0_pct / 100.0
    n = len(df)

    P_p_d = np.zeros(n)
    P_p_b = np.zeros(n)
    P_p_g = np.zeros(n)
    P_b_d = np.zeros(n)
    P_g_d = np.zeros(n)
    SOC_t = np.zeros(n)

    for t in range(n):
        P_p_d[t] = min(P_pv[t], P_de[t])
        if P_pv[t] - P_de[t] > 0:
            surplus = P_pv[t] - P_de[t]
            if SOC < SOC_max:
                P_p_b[t] = min(surplus, P_b_ch)
                P_p_g[t] = max(surplus - P_b_ch, 0.0)
            else:
                P_p_g[t] = max(surplus, 0.0)
        else:
            deficit = P_de[t] - P_pv[t]
            if SOC > SOC_min:
                P_b_d[t] = min(deficit, P_b_dis)
                P_g_d[t] = max(deficit - P_b_dis, 0.0)
            else:
                P_g_d[t] = max(deficit, 0.0)

        E_charge = P_p_b[t] * dt_hours * eta_c
        E_discharge = P_b_d[t] * dt_hours / eta_d
        SOC = SOC + (E_charge - E_discharge) / E_b
        SOC = min(max(SOC, SOC_min), SOC_max)
        SOC_t[t] = SOC

    out = df.copy()
    out["P_pv_kW"] = P_pv
    out["P_de_kW"] = P_de
    out["P_p_d_kW"] = P_p_d
    out["P_p_b_kW"] = P_p_b
    out["P_p_g_kW"] = P_p_g
    out["P_b_d_kW"] = P_b_d
    out["P_g_d_kW"] = P_g_d
    out["SOC"] = SOC_t
    out["SOC_pct"] = SOC_t * 100.0
    out["P_grid_import_kW"] = out["P_g_d_kW"]
    out["P_grid_export_kW"] = out["P_p_g_kW"]
    out["E_import_kWh"] = out["P_grid_import_kW"] * dt_hours
    out["E_export_kWh"] = out["P_grid_export_kW"] * dt_hours
    out["E_charge_kWh"] = out["P_p_b_kW"] * dt_hours
    out["E_discharge_kWh"] = out["P_b_d_kW"] * dt_hours
    return out


def rule_based_TOU(
    df,
    E_b=TOTAL_EB,
    SOC0_pct=SOC0,
    SOC_min_pct=SOC_MIN,
    SOC_max_pct=SOC_MAX,
    P_b_ch=P_CH_MAX,
    P_b_dis=P_DIS_MAX,
    eta_c=ETA_C,
    eta_d=ETA_D,
    dt_hours=DT_HOURS,
    pv_col="Total Solar Power(W)",
    load_col="Total Consumption Power(W)",
    tou_col="tou",
):
    P_pv = df[pv_col].to_numpy() / 1000.0
    P_de = df[load_col].to_numpy() / 1000.0
    tou = df[tou_col].to_numpy()

    months_arr = df["Time"].dt.month.to_numpy()
    is_valley = np.zeros(len(df), dtype=bool)
    for m in np.unique(months_arr):
        mask = months_arr == m
        is_valley[mask] = tou[mask] == tou[mask].min()

    SOC_min = SOC_min_pct / 100.0
    SOC_max = SOC_max_pct / 100.0
    SOC = SOC0_pct / 100.0
    n = len(df)
    P_p_d = np.zeros(n)
    P_p_b = np.zeros(n)
    P_p_g = np.zeros(n)
    P_g_b = np.zeros(n)
    P_b_d = np.zeros(n)
    P_g_d = np.zeros(n)
    SOC_t = np.zeros(n)
    period = np.empty(n, dtype=object)

    for t in range(n):
        P_p_d[t] = min(P_pv[t], P_de[t])
        surplus = P_pv[t] - P_de[t]
        if is_valley[t]:
            period[t] = "valley"
            if surplus > 0:
                if SOC < SOC_max:
                    P_p_b[t] = min(surplus, P_b_ch)
                    P_p_g[t] = max(surplus - P_b_ch, 0.0)
                    P_g_b[t] = max(P_b_ch - surplus, 0.0)
                else:
                    P_p_g[t] = max(surplus, 0.0)
            else:
                P_g_d[t] = max(-surplus, 0.0)
                P_g_b[t] = max(P_b_ch, 0.0)
        else:
            period[t] = "high"
            if surplus > 0:
                if SOC < SOC_max:
                    P_p_b[t] = min(surplus, P_b_ch)
                    P_p_g[t] = max(surplus - P_b_ch, 0.0)
                else:
                    P_p_g[t] = max(surplus, 0.0)
            else:
                if SOC > SOC_min:
                    P_b_d[t] = min(-surplus, P_b_dis)
                    P_g_d[t] = max(-surplus - P_b_dis, 0.0)
                else:
                    P_g_d[t] = max(-surplus, 0.0)

        E_charge_req = (P_p_b[t] + P_g_b[t]) * dt_hours * eta_c
        E_discharge = P_b_d[t] * dt_hours / eta_d
        SOC_raw = SOC + (E_charge_req - E_discharge) / E_b
        SOC_new = min(max(SOC_raw, SOC_min), SOC_max)

        if SOC_raw > SOC_new:
            E_charge_actual = (SOC_new - SOC) * E_b + E_discharge
            actual_charge_power = E_charge_actual / (dt_hours * eta_c)
            P_p_b[t] = min(P_p_b[t], actual_charge_power)
            P_g_b[t] = max(actual_charge_power - P_p_b[t], 0.0)

        SOC = SOC_new
        SOC_t[t] = SOC

    out = df.copy()
    out["P_pv_kW"] = P_pv
    out["P_de_kW"] = P_de
    out["P_p_d_kW"] = P_p_d
    out["P_p_b_kW"] = P_p_b
    out["P_p_g_kW"] = P_p_g
    out["P_g_b_kW"] = P_g_b
    out["P_b_d_kW"] = P_b_d
    out["P_g_d_kW"] = P_g_d
    out["SOC"] = SOC_t
    out["SOC_pct"] = SOC_t * 100.0
    out["period"] = period
    out["P_grid_import_kW"] = out["P_g_d_kW"] + out["P_g_b_kW"]
    out["P_grid_export_kW"] = out["P_p_g_kW"]
    out["E_import_kWh"] = out["P_grid_import_kW"] * dt_hours
    out["E_export_kWh"] = out["P_grid_export_kW"] * dt_hours
    out["E_charge_kWh"] = (out["P_p_b_kW"] + out["P_g_b_kW"]) * dt_hours
    out["E_discharge_kWh"] = out["P_b_d_kW"] * dt_hours
    return out


def plot_rule_profiles(sim, data_rule, method):
    df_rule = sim.copy()
    df_rule["Time"] = data_rule["Time"].values
    df_rule["Pgrid_rule_W"] = (df_rule["P_grid_import_kW"] - df_rule["P_grid_export_kW"]) * 1000
    if method == "tou":
        df_rule["Pb_rule_W"] = (
            df_rule["P_p_b_kW"] + df_rule["P_g_b_kW"] - df_rule["P_b_d_kW"]
        ) * 1000
    else:
        df_rule["Pb_rule_W"] = (df_rule["P_p_b_kW"] - df_rule["P_b_d_kW"]) * 1000
    df_rule["PL_W"] = df_rule["P_de_kW"] * 1000
    df_rule["Ppv_W"] = df_rule["P_pv_kW"] * 1000

    vars_to_pivot = {
        "consumption": "PL_W",
        "solar": "Ppv_W",
        "soc": "SOC_pct",
        "grid": "Pgrid_rule_W",
        "battery": "Pb_rule_W",
    }
    for month, name in MONTH_NAMES.items():
        month_df = df_rule[df_rule["Time"].dt.month == month].copy()
        plot_battery_profile(
            data=build_daily_wide(month_df, vars_to_pivot),
            method=method,
            month_name=name.lower(),
        )
    return df_rule
