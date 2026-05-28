import numpy as np
import pandas as pd
import pulp as pl

from .config import (
    DT_HOURS,
    ETA_C,
    ETA_D,
    P_CH_MAX,
    P_DIS_MAX,
    P_GEX_MAX,
    P_GIM_MAX,
    SOC0,
    SOC_MAX,
    SOC_MIN,
    TOTAL_EB,
)
from .eda import build_daily_wide, plot_battery_profile
from .config import MONTH_NAMES


def solve_daily_milp(day_df, soc0):
    day_df = day_df.sort_values("Time").copy().reset_index(drop=True)
    dt_hours = DT_HOURS
    T = list(day_df.index)
    Ppv = day_df["Ppv"].to_dict()
    PLoad = day_df["PL"].to_dict()
    buy_price = day_df["TOU_price"].to_dict()
    sell_price = day_df["sell_price"].to_dict()

    model = pl.LpProblem("Battery_Optimization_Daily", pl.LpMinimize)
    Ppv2L = pl.LpVariable.dicts("Ppv2L", T, lowBound=0)
    Ppv2b = pl.LpVariable.dicts("Ppv2b", T, lowBound=0)
    Ppv2g = pl.LpVariable.dicts("Ppv2g", T, lowBound=0)
    Ppv2c = pl.LpVariable.dicts("Ppv2c", T, lowBound=0)
    Pg2L = pl.LpVariable.dicts("Pg2L", T, lowBound=0)
    Pg2b = pl.LpVariable.dicts("Pg2b", T, lowBound=0)
    Pb2L = pl.LpVariable.dicts("Pb2L", T, lowBound=0)
    Pb2g = pl.LpVariable.dicts("Pb2g", T, lowBound=0)
    Pgim = pl.LpVariable.dicts("Pgim", T, lowBound=0, upBound=P_GIM_MAX)
    Pgex = pl.LpVariable.dicts("Pgex", T, lowBound=0, upBound=P_GEX_MAX)
    soc = pl.LpVariable.dicts("soc", T, lowBound=SOC_MIN, upBound=SOC_MAX)
    zg = pl.LpVariable.dicts("zg", T, cat="Binary")
    ub = pl.LpVariable.dicts("ub", T, cat="Binary")

    model += pl.lpSum(Pgim[t] * dt_hours * buy_price[t] - Pgex[t] * dt_hours * sell_price[t] for t in T)

    for t in T:
        model += PLoad[t] == Ppv2L[t] + Pb2L[t] + Pg2L[t], f"load_balance_{t}"
        model += Ppv[t] == Ppv2L[t] + Ppv2b[t] + Ppv2g[t] + Ppv2c[t], f"pv_balance_{t}"
        model += Pgim[t] == Pg2L[t] + Pg2b[t], f"grid_import_balance_{t}"
        model += Pgex[t] == Ppv2g[t] + Pb2g[t], f"grid_export_balance_{t}"
        model += Pgim[t] <= zg[t] * P_GIM_MAX, f"grid_import_dir_{t}"
        model += Pgex[t] <= (1 - zg[t]) * P_GEX_MAX, f"grid_export_dir_{t}"
        model += Ppv2b[t] + Pg2b[t] <= ub[t] * P_CH_MAX, f"charge_limit_{t}"
        model += Pb2L[t] + Pb2g[t] <= (1 - ub[t]) * P_DIS_MAX, f"discharge_limit_{t}"

        charge_term = 100 * ETA_C * (Ppv2b[t] + Pg2b[t]) * dt_hours / TOTAL_EB
        discharge_term = 100 * ((Pb2L[t] + Pb2g[t]) / ETA_D) * dt_hours / TOTAL_EB
        if t == 0:
            model += soc[t] == soc0 + charge_term - discharge_term, f"soc_init_{t}"
        else:
            model += soc[t] == soc[t - 1] + charge_term - discharge_term, f"soc_dyn_{t}"

    solver = pl.PULP_CBC_CMD(msg=0)
    model.solve(solver)
    status = pl.LpStatus[model.status]

    result = day_df[["Time", "Ppv", "PL", "SoC(%)", "Battery Power(W)", "Total Grid Power(W)"]].copy()

    result["Pgim_opt"] = [pl.value(Pgim[t]) for t in T]
    result["Pgex_opt"] = [pl.value(Pgex[t]) for t in T]
    result["P_charge_opt"] = [pl.value(Ppv2b[t]) + pl.value(Pg2b[t]) for t in T]
    result["P_discharge_opt"] = [pl.value(Pb2L[t]) + pl.value(Pb2g[t]) for t in T]
    result["Pb_opt"] = result["P_discharge_opt"] - result["P_charge_opt"]
    result["Pgrid_opt"] = result["Pgim_opt"] - result["Pgex_opt"]
    result["soc_opt_pct"] = [pl.value(soc[t]) for t in T]
    result["Pb_opt_W"] = result["Pb_opt"] * 1000
    result["Pgrid_opt_W"] = result["Pgrid_opt"] * 1000
    result["objective_day"] = pl.value(model.objective)
    result["status"] = status
    return result, result["soc_opt_pct"].iloc[-1], pl.value(model.objective), status


def run_rolling_milp(milp_df, soc0=SOC0):
    milp_df = milp_df.sort_values("Time").copy()
    milp_df["date"] = milp_df["Time"].dt.date
    all_results = []
    daily_objectives = []
    soc0_current = soc0

    for day, day_df in milp_df.groupby("date"):
        res_day, soc_end, obj_day, status = solve_daily_milp(day_df=day_df, soc0=soc0_current)
        print(f"{day} | status={status} | objective={obj_day:.4f} | soc_end={soc_end:.2f}")
        all_results.append(res_day)
        daily_objectives.append(
            {
                "date": day,
                "objective_day": obj_day,
                "soc_start": soc0_current,
                "soc_end": soc_end,
                "status": status,
            }
        )
        soc0_current = soc_end

    res_rolling = pd.concat(all_results, ignore_index=True)
    res_rolling["Total Consumption Power(W)"] = milp_df["Total Consumption Power(W)"].values
    daily_summary = pd.DataFrame(daily_objectives)
    return res_rolling, daily_summary


def plot_milp_profiles(res_rolling):
    df = res_rolling.copy()
    df["PL"] = df["PL"] * 1000
    df["Ppv"] = df["Ppv"] * 1000
    vars_to_pivot = {
        "consumption": "PL",
        "solar": "Ppv",
        "soc": "soc_opt_pct",
        "grid": "Pgrid_opt_W",
        "battery": "Pb_opt_W",
    }
    for month, name in MONTH_NAMES.items():
        month_df = df[df["Time"].dt.month == month].copy()
        plot_battery_profile(
            data=build_daily_wide(month_df, vars_to_pivot),
            method="milp",
            month_name=name.lower(),
        )
