import json
import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np
import pandas as pd

from .config import (
    DATA_FOLDER,
    FIXED_PRICE,
    LOGGER_FILE,
    PRICE_FILE,
    SELECTED_COLUMNS,
    SECRETS_FILE,
    SUPPLIER_ADMIN_FEE_EUR_KWH,
    VALUE_COLUMNS,
)


def load_logger_data(path=LOGGER_FILE, selected_columns=SELECTED_COLUMNS):
    data_raw = pd.read_csv(path)
    data_raw["Time"] = pd.to_datetime(data_raw["Time"])
    data = data_raw[selected_columns].copy()
    return data_raw, data


def classify_gap(x, expected, tol=0.5):
    if pd.isna(x):
        return "first_row"
    if abs(x - expected) <= tol:
        return "ok"
    if x < expected - tol:
        return "too_short"
    if x > expected + tol:
        return "too_long"
    return "other"


def build_time_step_audit(data, tol=1):
    dt = data.sort_values("Time").copy()
    dt["Time"] = pd.to_datetime(dt["Time"])
    dt["delta_min"] = dt["Time"].diff().dt.total_seconds() / 60
    expected_step = dt["delta_min"].mode().iloc[0]
    dt["gap_type"] = dt["delta_min"].apply(lambda x: classify_gap(x, expected_step, tol=tol))
    return dt, expected_step


def build_regular_grid(data, value_cols=VALUE_COLUMNS):
    """
    DG: Interpoliacijos logika
    """
    data_for_interpolation = data[["Time", *value_cols]].copy()
    data_for_interpolation["source"] = data_for_interpolation["Time"]
    data_for_interpolation["Time"] = pd.to_datetime(data_for_interpolation["Time"]).dt.round("5min")

    agg_dict = {col: "mean" for col in value_cols}
    agg_dict["source"] = "first"
    data_agg = data_for_interpolation.groupby("Time").agg(agg_dict).reset_index() #suvidurkinam tuos kurie pateko i ta pati bucketa

    all_days = data_agg["Time"].dt.normalize().unique() 
    full_range = pd.date_range(
        start=all_days.min(),
        end=all_days.max() + pd.Timedelta("23h 55min"),
        freq="5min",
    )

    data_agg = data_agg.set_index("Time").reindex(full_range).reset_index()
    data_agg.rename(columns={"index": "Time"}, inplace=True)
    data_agg["source"] = data_agg["source"].astype("object")
    data_agg = data_agg[data_agg["Time"].dt.date != pd.Timestamp("2026-05-01").date()]

    for col in value_cols:
        data_agg.loc[data_agg[col].isna(), "source"] = "interpolated"
        data_agg[col] = data_agg[col].interpolate().bfill()

    return add_derived_power_columns(data_agg)


def add_derived_power_columns(df):
    """
    DG: Isvestiniai daikciukai
    """
    out = df.copy()
    out["pgim"] = out["Total Grid Power(W)"].clip(lower=0)
    out["pgex"] = (-out["Total Grid Power(W)"]).clip(lower=0)
    out["Pb_dis"] = out["Battery Power(W)"].clip(lower=0)
    out["Pb_ch"] = (-out["Battery Power(W)"]).clip(lower=0)
    return out


def classify_energy_period(row):
    """
    TOU tarifų klasifikacija pagal datą ir laiką. Pritaikyta 2026 metų vasario-balandžio mėnesiams, atsižvelgiant į vasaros laiko pradžią.
    """
    dt = row["Time"]
    month = dt.month
    day_of_week = dt.weekday()
    hour = dt.hour

    if month == 2:
        day_price, night_price = 0.3197, 0.26162
    elif month == 3:
        day_price, night_price = 0.24642, 0.18834
    elif month == 4:
        day_price, night_price = 0.2225, 0.16442
    else:
        raise ValueError(f"TOU tariff is not configured for month {month}.")

    summer_time = (month == 3 and dt.day >= 29) or (month == 4)
    if day_of_week >= 5:
        return night_price
    if summer_time:
        return day_price if 8 <= hour < 24 else night_price
    return day_price if 7 <= hour < 23 else night_price


def add_tariffs(data_final):
    """
    Prideda TOU tarifų kainas prie duomenų
    """
    out = data_final.copy()
    out["TOU_price"] = out.apply(classify_energy_period, axis=1)
    out["price_fixed"] = FIXED_PRICE
    return out


def load_price_data(path=PRICE_FILE):
    """
    pridedame nordpool
    """
    df_price = pd.read_csv(path)
    df_price["start"] = pd.to_datetime(df_price["datetime_utc"], utc=True).dt.tz_convert("Europe/Vilnius")
    df_price = df_price.sort_values("start").reset_index(drop=True)
    df_price["end"] = df_price["start"].shift(-1)

    last_res = df_price["resolution"].iloc[-1]
    step = pd.Timedelta(minutes=60 if last_res == "PT60M" else 15 if last_res == "PT15M" else 30)
    df_price.loc[df_price.index[-1], "end"] = df_price["start"].iloc[-1] + step
    df_price["nordpool_price"] = df_price["price_eur_mwh"] / 1000.0

    return (
        df_price[["start", "end", "nordpool_price"]]
        .drop_duplicates(subset=["start"])
        .sort_values("start")
        .reset_index(drop=True)
    )


def prepare_milp_dataframe(data_final, df_price):
    milp_cols = [
        "Time",
        "Total Solar Power(W)",
        "Total Consumption Power(W)",
        "Battery Power(W)",
        "Total Grid Power(W)",
        "SoC(%)",
        "TOU_price",
    ]
    milp_df = data_final[milp_cols].copy()
    milp_df["date"] = milp_df["Time"].dt.date
    milp_df["Time"] = pd.to_datetime(milp_df["Time"]).dt.tz_localize(
        "Europe/Vilnius",
        nonexistent="shift_forward",
        ambiguous="NaT",
    )

    milp_df = milp_df.sort_values("Time")
    df_price = df_price.sort_values("start")
    milp_df = pd.merge_asof(
        milp_df,
        df_price[["start", "end", "nordpool_price"]],
        left_on="Time",
        right_on="start",
        direction="backward",
    )
    milp_df["nordpool_price"] = milp_df["nordpool_price"].where(milp_df["Time"] < milp_df["end"])
    milp_df = milp_df.drop(columns=["start", "end"])

    milp_df["sell_price"] = milp_df["nordpool_price"] - SUPPLIER_ADMIN_FEE_EUR_KWH
    milp_df["Ppv"] = milp_df["Total Solar Power(W)"] / 1000.0
    milp_df["PL"] = milp_df["Total Consumption Power(W)"] / 1000.0
    milp_df["fixed_price"] = FIXED_PRICE
    milp_df["date"] = milp_df["Time"].dt.date
    return milp_df


def build_rule_dataframe(data_final, milp_df):
    data_rule_cols = ["Time", "Total Solar Power(W)", "Total Consumption Power(W)"]
    data_rule = data_final[data_rule_cols].copy()
    data_rule = data_rule.sort_values("Time").reset_index(drop=True)
    data_rule["pv_kw"] = data_rule["Total Solar Power(W)"] / 1000.0
    data_rule["load_kw"] = data_rule["Total Consumption Power(W)"] / 1000.0
    data_rule["Time"] = pd.to_datetime(data_rule["Time"])
    data_rule["nordpool_price"] = milp_df["nordpool_price"].values
    data_rule["tou"] = milp_df["TOU_price"].values
    data_rule["sell_price"] = milp_df["sell_price"].values
    data_rule["date"] = data_rule["Time"].dt.date
    data_rule["hour"] = data_rule["Time"].dt.hour
    data_rule["fixed_price"] = FIXED_PRICE
    data_rule["Total Grid Power(W)"] = data_final["Total Grid Power(W)"].values
    return data_rule


def read_secrets(path=SECRETS_FILE):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def fetch_entsoe_prices(
    output_path=PRICE_FILE,
    secrets_path=SECRETS_FILE,
    period_start="202601010000",
    period_end="202608010000",
    zone="10YLT-1001A0008Q",
):
    import requests

    api_key = read_secrets(secrets_path)["ENTSOE_API_KEY"]
    params = {
        "securityToken": api_key,
        "documentType": "A44",
        "in_Domain": zone,
        "out_Domain": zone,
        "periodStart": period_start,
        "periodEnd": period_end,
    }
    r = requests.get("https://web-api.tp.entsoe.eu/api", params=params, timeout=60)
    r.raise_for_status()

    ns = {"ns": "urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3"}
    root = ET.fromstring(r.text)
    rows = []
    for ts in root.findall("ns:TimeSeries", ns):
        period = ts.find("ns:Period", ns)
        p_start = datetime.fromisoformat(
            period.find("ns:timeInterval/ns:start", ns).text.replace("Z", "+00:00")
        )
        resolution = period.find("ns:resolution", ns).text
        step_min = 60 if resolution == "PT60M" else 15 if resolution == "PT15M" else 30
        for pt in period.findall("ns:Point", ns):
            pos = int(pt.find("ns:position", ns).text)
            price = float(pt.find("ns:price.amount", ns).text)
            t_utc = p_start + pd.Timedelta(minutes=step_min * (pos - 1))
            t_local = t_utc.astimezone(pd.Timestamp("now", tz="Europe/Vilnius").tz)
            rows.append(
                {
                    "datetime_local_LT": t_local.strftime("%Y-%m-%d %H:%M"),
                    "datetime_utc": t_utc.strftime("%Y-%m-%d %H:%M"),
                    "price_eur_mwh": price,
                    "price_ct_kwh": round(price / 10, 4),
                    "resolution": resolution,
                }
            )

    df = pd.DataFrame(rows).sort_values("datetime_utc").reset_index(drop=True)
    df.to_csv(output_path, index=False)
    return df
