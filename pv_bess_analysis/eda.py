import calendar

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MultipleLocator

from .config import IMAGE_DIR, MONTH_NAMES, MONTHS


def lt_format(x, pos=None):
    return f"{x:.2f}".replace(".", ",")


def status_checks(data):
    return {
        "grid_export_negative": bool(
            (data.loc[data["Grid Status"] == "Grid connected", "Total Grid Power(W)"] < 0).all()
        ),
        "grid_import_positive": bool(
            (data.loc[data["Grid Status"] == "Purchasing energy", "Total Grid Power(W)"] > 0).all()
        ),
        "grid_static_describe": data.loc[
            data["Grid Status"] == "Static", "Total Grid Power(W)"
        ].describe(),
        "battery_charging_negative": bool(
            (data.loc[data["Battery Status"] == "Charging", "Battery Power(W)"] < 0).all()
        ),
        "battery_discharging_positive": bool(
            (data.loc[data["Battery Status"] == "Discharging", "Battery Power(W)"] > 0).all()
        ),
        "battery_static_describe": data.loc[
            data["Battery Status"] == "Static", "Battery Power(W)"
        ].describe(),
    }


def pivot_day_time(dataframe, value_col, prefix=None):
    tmp = dataframe[["Date", "TimeAxis", value_col]].copy()
    tmp["minute_of_day"] = tmp["TimeAxis"].dt.hour * 60 + tmp["TimeAxis"].dt.minute
    tmp = tmp.sort_values(["Date", "minute_of_day"])
    wide = tmp.pivot_table(
        index="Date",
        columns="minute_of_day",
        values=value_col,
        aggfunc="mean",
    )
    wide.columns = [f"{minute // 60:02d}:{minute % 60:02d}" for minute in wide.columns]
    return wide.reset_index()


def build_daily_profile(daily_df):
    value_cols = [c for c in daily_df.columns if c != "Date"]
    means = daily_df[value_cols].mean()
    stds = daily_df[value_cols].std()
    times = pd.to_datetime(means.index, format="%H:%M")
    return pd.DataFrame({"time": times, "mean": means.values, "std": stds.values}).sort_values("time")


def plot_interpolation_by_month(data_agg, value_col, label, filename, sharey=False):
    IMAGE_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=False, sharey=sharey)
    for ax, (month, month_name) in zip(axes, MONTH_NAMES.items()):
        month_data = data_agg[data_agg["Time"].dt.month == month].copy()
        plot_col = f"{value_col}_kW"
        month_data[plot_col] = month_data[value_col] / 1000

        year = month_data["Time"].dt.year.iloc[0]
        last_day = calendar.monthrange(year, month)[1]
        ax.plot(month_data["Time"], month_data[plot_col], linewidth=0.8, color="steelblue", label=label)

        interp = month_data[month_data["source"] == "interpolated"]
        ax.scatter(interp["Time"], interp[plot_col], s=5, c="red", zorder=3, label="Interpoliuota")

        ax.set_xlim(pd.Timestamp(year=year, month=month, day=1), pd.Timestamp(year=year, month=month, day=last_day))
        ax.set_title(month_name.capitalize(), fontsize=22)
        ax.set_ylabel("Galia (kW)", fontsize=18)
        ax.grid(True)
        ax.tick_params(axis="x", labelsize=18)
        ax.tick_params(axis="y", labelsize=18)
        ax.yaxis.set_major_formatter(FuncFormatter(lt_format))
        ax.legend(loc="upper right", fontsize=18)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center")
    for ax in axes:
        ax.yaxis.set_major_formatter(FuncFormatter(lt_format))

    fig.tight_layout()
    plt.savefig(IMAGE_DIR / filename, dpi=300)
    plt.show()
    return fig, axes


def plot_monthly_power_series(
    data_final,
    value_col,
    label,
    filename_template,
    y_limits,
    y_major_locator=None,
):
    IMAGE_DIR.mkdir(exist_ok=True)
    figures = {}
    for month, month_name in MONTH_NAMES.items():
        month_data = data_final[data_final["Time"].dt.month == month].copy()
        if month_data.empty:
            continue

        plot_col = f"{value_col}_kW"
        month_data[plot_col] = month_data[value_col] / 1000

        year = month_data["Time"].dt.year.iloc[0]
        last_day = calendar.monthrange(year, month)[1]
        x_min = pd.Timestamp(year=year, month=month, day=1)
        x_max = pd.Timestamp(year=year, month=month, day=last_day)

        fig, ax = plt.subplots(figsize=(14, 8))
        ax.plot(month_data["Time"], month_data[plot_col], linewidth=1, color="steelblue", label=label)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_limits[month])
        if y_major_locator is not None:
            ax.yaxis.set_major_locator(MultipleLocator(y_major_locator))
        ax.set_title(month_name.capitalize(), fontsize=26)
        ax.set_ylabel("Galia (kW)", fontsize=24)
        ax.grid(True)
        ax.tick_params(axis="x", labelsize=24)
        ax.tick_params(axis="y", labelsize=24)
        ax.yaxis.set_major_formatter(FuncFormatter(lt_format))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center")
        plt.tight_layout()
        plt.savefig(IMAGE_DIR / filename_template.format(month=month), dpi=300)
        plt.show()
        figures[month] = (fig, ax)
        plt.close(fig)
    return figures


def plot_daily_spaghetti_power(
    data_final,
    value_col,
    filename_template,
    y_limits,
    y_major_locator=None,
):
    IMAGE_DIR.mkdir(exist_ok=True)
    all_days = sorted(data_final["Time"].dt.date.unique())
    colors = plt.cm.tab20.colors
    day_color = {day: colors[i % len(colors)] for i, day in enumerate(all_days)}
    figures = {}

    for month in MONTHS:
        month_df = data_final[data_final["Time"].dt.month == month]
        days = sorted(month_df["Time"].dt.date.unique())
        if not days:
            continue

        fig, ax = plt.subplots(figsize=(14, 8))
        ax.set_ylim(y_limits[month])

        for day in days:
            day_df = month_df[month_df["Time"].dt.date == day].sort_values("Time")
            time_axis = pd.to_datetime(day_df["Time"].dt.strftime("2000-01-01 %H:%M:%S"))
            ax.plot(
                time_axis,
                day_df[value_col] / 1000,
                color=day_color[day],
                marker="o",
                markersize=3,
                linewidth=1.2,
            )

        if y_major_locator is not None:
            ax.yaxis.set_major_locator(MultipleLocator(y_major_locator))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.set_xlim(pd.Timestamp("2000-01-01 00:00:00"), pd.Timestamp("2000-01-02 00:00:00"))
        ax.set_title(MONTH_NAMES[month].capitalize(), fontsize=26)
        ax.set_ylabel("Galia (kW)", fontsize=24)
        ax.tick_params(axis="x", labelsize=24)
        ax.tick_params(axis="y", labelsize=24)
        ax.yaxis.set_major_formatter(FuncFormatter(lt_format))
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center")
        fig.tight_layout()
        fig.savefig(IMAGE_DIR / filename_template.format(month=month), dpi=300)
        plt.show()
        figures[month] = (fig, ax)
        plt.close(fig)
    return figures

def plot_battery_profile(
    data,
    profile_builder=build_daily_profile,
    series_meta=None,
    soc_key="soc",
    soc_label="SoC (%)",
    figsize=(10, 8),
    height_ratios=(3, 1),
    soc_ylim=(0, 115),
    fill_alpha=0.2,
    soc_fill_alpha=0.1,
    method=None,
    month_name=None,
):
    IMAGE_DIR.mkdir(exist_ok=True)
    if series_meta is None:
        series_meta = [
            ("solar", "PV galia (kW)"),
            ("consumption", "Vartojimas (kW)"),
            ("grid", "Tinklas (kW)"),
            ("battery", "Baterija (kW)"),
        ]

    if soc_key is not None:
        fig, (ax, ax_soc) = plt.subplots(
            2,
            1,
            figsize=figsize,
            sharex=True,
            gridspec_kw={"height_ratios": list(height_ratios)},
        )
    else:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax_soc = None

    day_start = day_end = None
    for key, label in series_meta:
        if key not in data:
            continue
        profile = profile_builder(data[key])
        profile["time"] = pd.to_datetime(profile["time"])
        if key == "battery" and method in {"milp", "final", "real"}:
            profile["mean"] = profile["mean"] * -1
            profile["std"] = profile["std"] * -1

        mean_kw = profile["mean"] / 1000
        std_kw = profile["std"] / 1000
        ax.plot(profile["time"], mean_kw, label=label)
        ax.fill_between(profile["time"], mean_kw - std_kw, mean_kw + std_kw, alpha=fill_alpha)
        if day_start is None and len(profile["time"]):
            day_start = profile["time"].iloc[0].normalize()
            day_end = day_start + pd.Timedelta(days=1)

    ax.set_ylabel("Galia (kW)", fontsize=19)
    ax.tick_params(axis="y", labelsize=19)
    ax.yaxis.set_major_formatter(FuncFormatter(lt_format))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=16)

    if soc_key is not None and soc_key in data:
        soc_profile = profile_builder(data[soc_key])
        soc_profile["time"] = pd.to_datetime(soc_profile["time"])
        ax_soc.plot(soc_profile["time"], soc_profile["mean"], color="black", linestyle="--", label=soc_label)
        ax_soc.fill_between(
            soc_profile["time"],
            soc_profile["mean"] - soc_profile["std"],
            soc_profile["mean"] + soc_profile["std"],
            color="black",
            alpha=soc_fill_alpha,
        )
        ax_soc.set_ylabel("SoC (%)", fontsize=19)
        ax_soc.tick_params(axis="y", labelsize=19)
        ax_soc.set_ylim(soc_ylim[0] - 5, soc_ylim[1])
        ax_soc.grid(True, alpha=0.3)
        ax_soc.legend(loc="upper right", fontsize=16)
        if day_start is None and len(soc_profile["time"]):
            day_start = soc_profile["time"].iloc[0].normalize()
            day_end = day_start + pd.Timedelta(days=1)

    bottom = ax_soc if ax_soc is not None else ax
    bottom.tick_params(axis="x", labelsize=19)
    bottom.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    bottom.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.setp(bottom.xaxis.get_majorticklabels(), rotation=0, ha="center")
    if day_start is not None:
        ax.set_xlim(day_start, day_end)

    plt.tight_layout()
    method_str = method if method else "unknown"
    month_str = month_name if month_name else "unknown"
    plt.savefig(IMAGE_DIR / f"battery_profile_{method_str}_{month_str}.png", bbox_inches="tight", dpi=150)
    plt.show()


def plot_price_series(df_price, filename="nordpool_prices_Feb_Mar_Apr_2026.png"):
    IMAGE_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    for ax, month in zip(axes, MONTHS):
        month_data = df_price[df_price["start"].dt.month == month].copy()
        ax.step(month_data["start"], month_data["nordpool_price"], where="post", linewidth=1.2)
        ax.grid(True)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.tick_params(axis="x", labelrotation=45, labelsize=17)
        ax.tick_params(axis="y", labelsize=17)
        ax.set_title(MONTH_NAMES[month].capitalize(), fontsize=22)
        ax.yaxis.set_major_formatter(FuncFormatter(lt_format))
    axes[0].set_ylabel("Kaina (EUR/kWh)", fontsize=17)
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / filename, dpi=300)
    plt.show()


def plot_tariff_weeks(data_final, filename="tarifai.png"):
    IMAGE_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True)
    periods = [
        ("2026-02-09", "2026-02-16", "Vasaris"),
        ("2026-03-02", "2026-03-09", "Kovas"),
        ("2026-04-06", "2026-04-13", "Balandis"),
    ]
    days_lt = ["P", "A", "T", "K", "Pn", "Š", "S"]
    all_prices = data_final["TOU_price"]
    y_min = all_prices.min() - 0.01
    y_max = all_prices.max() + 0.01

    for ax, (start_date, end_date, title) in zip(axes, periods):
        week_data = data_final[(data_final["Time"] >= start_date) & (data_final["Time"] < end_date)].copy()
        ax.step(week_data["Time"], week_data["TOU_price"], where="post", linewidth=1.8)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(pd.date_range(start=start_date, periods=7, freq="D"))
        ax.set_xticklabels(days_lt)
        ax.set_xlim(pd.Timestamp(start_date), pd.Timestamp(end_date))
        ax.set_ylim(y_min, y_max)
        ax.yaxis.set_major_locator(MultipleLocator(0.02))
        ax.yaxis.set_major_formatter(FuncFormatter(lt_format))

    axes[0].set_ylabel("Kaina (EUR/kWh)")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / filename, dpi=300)
    plt.show()


def build_daily_wide(dataframe, vars_to_pivot):
    df = dataframe.copy()
    df["Date"] = df["Time"].dt.date
    df["TimeAxis"] = pd.to_datetime(df["Time"].dt.strftime("2000-01-01 %H:%M:%S"))
    return {prefix: pivot_day_time(df, col, prefix) for prefix, col in vars_to_pivot.items()}


def plot_real_control_profiles(data_final):
    vars_to_pivot = {
        "solar": "Total Solar Power(W)",
        "consumption": "Total Consumption Power(W)",
        "battery": "Battery Power(W)",
        "grid": "Total Grid Power(W)",
        "soc": "SoC(%)",
    }
    figures = {}
    for month, month_name in MONTH_NAMES.items():
        month_df = data_final[data_final["Time"].dt.month == month].copy()
        if month_df.empty:
            continue
        figures[month] = plot_battery_profile(
            data=build_daily_wide(month_df, vars_to_pivot),
            method="final",
            month_name=month_name.lower(),
        )
    return figures

