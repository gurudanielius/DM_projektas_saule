import pandas as pd

from .config import MONTH_NAMES, MONTHS


def print_milp_cost_summary(milp_df, daily_summary, dt_hours, total_objective_rolling):
    real_grid_import_kW = milp_df["Total Grid Power(W)"].clip(lower=0) / 1000
    real_grid_export_kW = (-milp_df["Total Grid Power(W)"]).clip(lower=0) / 1000
    no_batt_grid_import_kW = (milp_df["PL"] - milp_df["Ppv"]).clip(lower=0)
    no_batt_grid_export_kW = (milp_df["Ppv"] - milp_df["PL"]).clip(lower=0)

    cost_no_battery = (
        no_batt_grid_import_kW * dt_hours * milp_df["fixed_price"]
        - no_batt_grid_export_kW * dt_hours * milp_df["sell_price"]
    )
    cost_no_battery_TOU = (
        no_batt_grid_import_kW * dt_hours * milp_df["TOU_price"]
        - no_batt_grid_export_kW * dt_hours * milp_df["sell_price"]
    )
    real_cost = (real_grid_import_kW * dt_hours * milp_df["fixed_price"]).sum() - (
        real_grid_export_kW * dt_hours * milp_df["sell_price"]
    ).sum()

    print("Kaina be baterijos (€):", round(cost_no_battery.sum(), 2))
    print("Kaina be baterijos (TOU €):", round(cost_no_battery_TOU.sum(), 2))
    print("Reali mėnesio kaina (€):", round(real_cost, 2))
    print("Optimali mėnesio kaina (€):", round(total_objective_rolling, 2))
    print("Potencialus sutaupymas (€):", round(cost_no_battery_TOU.sum() - total_objective_rolling, 2))
    print("==============================\n")
    dfm = milp_df.copy()
    dfm["month"] = dfm["Time"].dt.month
    real_grid_import_kW_m = dfm["Total Grid Power(W)"].clip(lower=0) / 1000.0
    real_grid_export_kW_m = (-dfm["Total Grid Power(W)"]).clip(lower=0) / 1000.0
    dfm["real_cost_fixed"] = (
        real_grid_import_kW_m * dt_hours * dfm["fixed_price"]
        - real_grid_export_kW_m * dt_hours * dfm["sell_price"]
    )
    no_batt_import_kW = (dfm["PL"] - dfm["Ppv"]).clip(lower=0)
    no_batt_export_kW = (dfm["Ppv"] - dfm["PL"]).clip(lower=0)
    dfm["cost_no_batt_fixed"] = (
        no_batt_import_kW * dt_hours * dfm["fixed_price"]
        - no_batt_export_kW * dt_hours * dfm["sell_price"]
    )
    dfm["cost_no_batt_TOU"] = (
        no_batt_import_kW * dt_hours * dfm["TOU_price"]
        - no_batt_export_kW * dt_hours * dfm["sell_price"]
    )

    fixed_cost_m = dfm.groupby("month")["cost_no_batt_fixed"].sum()
    tou_cost_m = dfm.groupby("month")["cost_no_batt_TOU"].sum()
    real_cost_m = dfm.groupby("month")["real_cost_fixed"].sum()
    milp_month = pd.to_datetime(daily_summary["date"]).dt.month
    milp_cost_m = daily_summary.assign(month=milp_month).groupby("month")["objective_day"].sum()

    for m in MONTHS:
        if m not in fixed_cost_m.index and m not in milp_cost_m.index and m not in real_cost_m.index:
            continue
        c_fixed = fixed_cost_m.get(m, float("nan"))
        c_tou = tou_cost_m.get(m, float("nan"))
        c_real = real_cost_m.get(m, float("nan"))
        c_milp = milp_cost_m.get(m, float("nan"))
        saving = c_tou - c_milp if pd.notna(c_real) and pd.notna(c_milp) else float("nan")
        print(MONTH_NAMES[m])
        print(f" Kaina be baterijos (€): {c_fixed:.2f}")
        print(f" Kaina be baterijos (TOU €): {c_tou:.2f}")
        print(f" Reali mėnesio kaina (€): {c_real:.2f}")
        print(f" Optimali mėnesio kaina (€): {c_milp:.2f}")
        print(f" Potencialus sutaupymas (€): {saving:.2f}")
        print("")


def print_rule_cost_summary(sim, data_rule, interval_hours, label):
    sim = sim.copy()
    sim["fixed_price"] = data_rule["fixed_price"].values
    sim["sell_price"] = data_rule["sell_price"].values
    sim["TOU_price"] = data_rule["tou"].values

    rule_grid_import_kW = sim["P_grid_import_kW"]
    rule_grid_export_kW = sim["P_grid_export_kW"]
    real_grid_import_kW = data_rule["Total Grid Power(W)"].clip(lower=0) / 1000
    real_grid_export_kW = (-data_rule["Total Grid Power(W)"]).clip(lower=0) / 1000
    no_batt_grid_import_kW = (sim["P_de_kW"] - sim["P_pv_kW"]).clip(lower=0)
    no_batt_grid_export_kW = (sim["P_pv_kW"] - sim["P_de_kW"]).clip(lower=0)

    cost_no_battery = (
        no_batt_grid_import_kW * interval_hours * sim["fixed_price"]
        - no_batt_grid_export_kW * interval_hours * sim["sell_price"]
    )
    cost_no_battery_TOU = (
        no_batt_grid_import_kW * interval_hours * sim["TOU_price"]
        - no_batt_grid_export_kW * interval_hours * sim["sell_price"]
    )
    real_cost = (real_grid_import_kW * interval_hours * sim["TOU_price"]).sum() - (
        real_grid_export_kW * interval_hours * sim["sell_price"]
    ).sum()
    rule_cost = (rule_grid_import_kW * interval_hours * sim["TOU_price"]).sum() - (
        rule_grid_export_kW * interval_hours * sim["sell_price"]
    ).sum()

    print(f"\n=== Kaštų suvestinė ({label}) ===\n")
    print("Kaina be baterijos (€):", round(cost_no_battery.sum(), 2))
    print("Kaina be baterijos (TOU €):", round(cost_no_battery_TOU.sum(), 2))
    print("Reali mėnesio kaina (€):", round(real_cost, 2))
    print(f"Taisyklėmis pagrįsta kaina ({label} €):", round(rule_cost, 2))
    print("Potencialus sutaupymas (€):", round(cost_no_battery_TOU.sum() - rule_cost, 2))
    print("==============================\n")
    dfm = data_rule.copy()
    dfm["month"] = dfm["Time"].dt.month
    real_grid_import_kW_m = dfm["Total Grid Power(W)"].clip(lower=0) / 1000.0
    real_grid_export_kW_m = (-dfm["Total Grid Power(W)"]).clip(lower=0) / 1000.0
    real_col = "real_cost_TOU" if label == "MSC" else "real_cost_fixed"
    real_price_col = "tou" if label == "MSC" else "fixed_price"
    dfm[real_col] = (
        real_grid_import_kW_m * interval_hours * dfm[real_price_col]
        - real_grid_export_kW_m * interval_hours * dfm["sell_price"]
    )

    no_batt_import_kW = (dfm["load_kw"] - dfm["pv_kw"]).clip(lower=0)
    no_batt_export_kW = (dfm["pv_kw"] - dfm["load_kw"]).clip(lower=0)
    dfm["cost_no_batt_fixed"] = (
        no_batt_import_kW * interval_hours * dfm["fixed_price"]
        - no_batt_export_kW * interval_hours * dfm["sell_price"]
    )
    dfm["cost_no_batt_TOU"] = (
        no_batt_import_kW * interval_hours * dfm["tou"]
        - no_batt_export_kW * interval_hours * dfm["sell_price"]
    )
    dfm["cost_rule_TOU"] = (
        sim["P_grid_import_kW"].values * interval_hours * dfm["tou"]
        - sim["P_grid_export_kW"].values * interval_hours * dfm["sell_price"]
    )

    fixed_cost_m = dfm.groupby("month")["cost_no_batt_fixed"].sum()
    tou_cost_m = dfm.groupby("month")["cost_no_batt_TOU"].sum()
    real_cost_m = dfm.groupby("month")[real_col].sum()
    rule_cost_m = dfm.groupby("month")["cost_rule_TOU"].sum()

    print(f"\n=== Mėnesinė kaštų suvestinė ({label}) ===\n")
    for m in MONTHS:
        if m not in fixed_cost_m.index:
            continue
        c_fixed = fixed_cost_m.get(m, float("nan"))
        c_tou = tou_cost_m.get(m, float("nan"))
        c_real = real_cost_m.get(m, float("nan"))
        c_rule = rule_cost_m.get(m, float("nan"))
        saving = c_tou - c_rule if pd.notna(c_tou) and pd.notna(c_rule) else float("nan")
        print(MONTH_NAMES[m])
        print(f" Kaina be baterijos (€): {c_fixed:.2f}")
        print(f" Kaina be baterijos (TOU €): {c_tou:.2f}")
        print(f" Reali mėnesio kaina (€): {c_real:.2f}")
        print(f" Taisyklėmis pagrįsta kaina (€): {c_rule:.2f}")
        print(f" Potencialus sutaupymas (€): {saving:.2f}")
        print("")

    return sim, dfm
