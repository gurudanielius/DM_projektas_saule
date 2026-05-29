# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter, MultipleLocator

from pv_bess_analysis.config import DT_HOURS, IMAGE_DIR, MONTH_NAMES, MONTHS
from pv_bess_analysis.costs import print_milp_cost_summary, print_rule_cost_summary
from pv_bess_analysis.data_prep import (
    add_tariffs,
    build_regular_grid,
    build_rule_dataframe,
    build_time_step_audit,
    load_logger_data,
    load_price_data,
    prepare_milp_dataframe,
)
from pv_bess_analysis.eda import (
    plot_daily_spaghetti_power,
    plot_interpolation_by_month,
    plot_monthly_power_series,
    plot_price_series,
    plot_real_control_profiles,
    plot_tariff_weeks,
    lt_format,
    status_checks,
)
from pv_bess_analysis.milp import plot_milp_profiles, run_rolling_milp
from pv_bess_analysis.rule_based import plot_rule_profiles, rule_based_MSC, rule_based_TOU

mpl.rcParams["font.size"] = 16

# %% [markdown]
#   # Duomenų nuskaitymas

# %%
data_raw, data = load_logger_data()
data_raw.columns

# %% [markdown]
#   # Pradinė duomenų analizė

# %%
data_with_lag, expected_step = build_time_step_audit(data, tol=1)
data_with_lag["gap_type"].value_counts()

# %%
data_with_lag["delta_min"].describe()

# %%
data_with_lag.loc[data_with_lag["delta_min"] > 30]

# %% [markdown]
#  Turime vieną labai ilgą, bet naktį tai galime įrašyti "0".

# %%
print(data.isnull().sum())
print(data.duplicated().sum())

# %% [markdown]
#  ## Stulpelių logika

# %%
checks = status_checks(data)
checks

# %% [markdown]
#  Total grid power:
#
#  Puchasing - siunčiamės elektrą iš tinklo arba baterijai krauti arba padengti apkrovą. Ženklas "+".
#
#  Static - neveiksni būsena, nedidelės reikšmės.
#
#  Total battery power:
#
#  Baterijos statuso stulpelis įgyja tris reikšmes: Charging, Discharging ir Static
#
#  Kai kraunasi battery power -, t.y. imama energija turi ženklą -
#
#  Kai discharging battery power +, t.y. išleidžiama energija yra +

# %% [markdown]
#   ## Duomenų tvarkymas
#

# %%
data_agg = build_regular_grid(data)
data_final = data_agg.copy()
# %%
data_final

# %%
print(f'interpoliuotų reikšmių skaičius: {len(data_final[data_final["source"] == "interpolated"])}')
print(
    "interpoliuotų reikšmių procentas: "
    f'{len(data_final[data_final["source"] == "interpolated"]) / len(data_final) * 100:.2f}%'
)

# %%
plot_interpolation_by_month(
    data_final,
    value_col="Total Consumption Power(W)",
    label="Apkrova (kW)",
    filename="interpoliacija_apkrova.png",
    sharey=True,
)

# %%
plot_interpolation_by_month(
    data_final,
    value_col="Total Solar Power(W)",
    label="Saulės galia (kW)",
    filename="interpoliacija_saule.png",
    sharey=False,
)

# %% [markdown]
#  ## Skaitiniai ir grafiniai dalykėliai

# %% [markdown]
#  ## TOTAL CONSUMPTION

# %%
data_final.groupby(data_final["Time"].dt.month)["Total Consumption Power(W)"].sum() /1000/1000

# %%
data_final.assign(
    kW=data_final["Total Consumption Power(W)"] / 1000
).groupby(data_final["Time"].dt.month)["kW"].describe().rename(index=MONTH_NAMES).round(2)

# %%
for month, month_name in MONTH_NAMES.items():
    month_data = data_final[data_final["Time"].dt.month == month]["Total Consumption Power(W)"]
    q1 = month_data.quantile(0.25)
    q3 = month_data.quantile(0.75)
    iqr = q3 - q1
    outliers = month_data[(month_data < q1 - 1.5 * iqr) | (month_data > q3 + 1.5 * iqr)]
    print(f"{month_name}: {len(outliers)} išskirtys iš {len(month_data)} ({len(outliers) / len(month_data) * 100:.1f}%)")

# %%
consumption_kwh_month = (
    data_final["Total Consumption Power(W)"] / 1000.0 * DT_HOURS
).groupby(data_final["Time"].dt.month).sum()
consumption_kwh_month.index = consumption_kwh_month.index.map(MONTH_NAMES)
consumption_kwh_month = consumption_kwh_month.rename("consumption_kWh")
consumption_kwh_month

# %%
IMAGE_DIR.mkdir(exist_ok=True)
consumption_plot = data_final.copy()
consumption_plot["month"] = consumption_plot["Time"].dt.month

colors = {
    2: "#1f77b4",
    3: "#ff7f0e",
    4: "#2ca02c",
}

y_limits = {
    2: (None, 10),
    3: (None, 9),
    4: (None, 8),
}

fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)

for ax, month in zip(axes, MONTHS):
    values = consumption_plot.loc[
        consumption_plot["month"] == month,
        "Total Consumption Power(W)",
    ] / 1000

    ax.set_ylim(values.min() - 0.5, y_limits[month][1])
    ax.yaxis.set_major_locator(MultipleLocator(1))

    vp = ax.violinplot(
        [values],
        positions=[1],
        showmeans=False,
        showmedians=False,
        showextrema=False,
        widths=0.8,
    )

    for body in vp["bodies"]:
        body.set_facecolor(colors[month])
        body.set_edgecolor("black")
        body.set_alpha(0.35)

    bp = ax.boxplot(
        [values],
        positions=[1],
        widths=0.22,
        showfliers=True,
        flierprops=dict(
            marker="o",
            markerfacecolor=colors[month],
            markeredgecolor=colors[month],
            alpha=0.3,
            markersize=4,
        ),
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.8},
        whiskerprops={"color": "black", "linewidth": 1.1},
        capprops={"color": "black", "linewidth": 1.1},
    )

    for patch in bp["boxes"]:
        patch.set_facecolor(colors[month])
        patch.set_edgecolor("black")
        patch.set_alpha(0.75)

    ax.set_xticks([1])
    ax.set_xticklabels("")
    ax.set_title(MONTH_NAMES[month].capitalize(), fontsize=17)
    ax.set_xlabel("")
    ax.set_ylabel("Galia, kW", fontsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.yaxis.set_major_formatter(FuncFormatter(lt_format))
    ax.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(
    IMAGE_DIR / "consumption_galios_pasiskirstymas_violin_boxplot.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# %%
plot_monthly_power_series(
            data_final,
            value_col="Total Consumption Power(W)",
            label="Apkrova (kW)",
            filename_template="per_menesi_{month}.png",
            y_limits={
        2: (-0.1, 10.5),
        3: (-0.1, 9),
        4: (-0.1, 8.5),
    },
            y_major_locator=1,
        )

# %%
plot_daily_spaghetti_power(
            data_final,
            value_col="Total Consumption Power(W)",
            filename_template="load_{month:02d}.png",
            y_limits={
        2: (-0.1, 10.5),
        3: (-0.1, 9),
        4: (-0.1, 8.5),
    },
            y_major_locator=1,
        )

# %% [markdown]
#  # TOTAL SOLAR POWER

# %%
pv_active = data_final[data_final["Total Solar Power(W)"] > 0].copy()
pv_active["month"] = pv_active["Time"].dt.month
pv_active_by_month = (
    pv_active.groupby("month")["Total Solar Power(W)"]
    .agg(
        stebejimu_skaicius="count",
        vidurkis_kW=lambda s: s.mean() / 1000,
        mediana_kW=lambda s: s.median() / 1000,
        q25_kW=lambda s: s.quantile(0.25) / 1000,
        q75_kW=lambda s: s.quantile(0.75) / 1000,
        max_kW=lambda s: s.max() / 1000,
    )
    .rename(index=MONTH_NAMES)
    .round(3)
)
pv_active_by_month

# %%
pv_active.groupby("month")["Total Solar Power(W)"].describe().rename(index=MONTH_NAMES)

# %%
for month, month_name in MONTH_NAMES.items():
    month_data = data_final[
        (data_final["Time"].dt.month == month)
        & (data_final["Total Solar Power(W)"] > 0)
    ]["Total Solar Power(W)"]
    q1 = month_data.quantile(0.25)
    q3 = month_data.quantile(0.75)
    iqr = q3 - q1
    outliers = month_data[(month_data < q1 - 1.5 * iqr) | (month_data > q3 + 1.5 * iqr)]
    print(f"{month_name}: {len(outliers)} išskirtys iš {len(month_data)} ({len(outliers) / len(month_data) * 100:.1f}%)")

# %%
pv_energy_kwh_month = (
    data_final["Total Solar Power(W)"] / 1000.0 * DT_HOURS
).groupby(data_final["Time"].dt.month).sum()
pv_energy_kwh_month.index = pv_energy_kwh_month.index.map(MONTH_NAMES)
pv_energy_kwh_month = pv_energy_kwh_month.rename("pv_generation_kWh")
pv_energy_kwh_month

# %%
IMAGE_DIR.mkdir(exist_ok=True)

colors = {
    2: "#1f77b4",
    3: "#ff7f0e",
    4: "#2ca02c",
}

y_limits = {
    2: (-0.01, 0.6),
    3: (-0.5, 22.5),
    4: (-0.5, 25),
}

fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)

for ax, month in zip(axes, MONTHS):
    values = pv_active.loc[pv_active["month"] == month, "Total Solar Power(W)"] / 1000

    vp = ax.violinplot(
        [values],
        positions=[1],
        showmeans=False,
        showmedians=False,
        showextrema=False,
        widths=0.8,
    )

    for body in vp["bodies"]:
        body.set_facecolor(colors[month])
        body.set_edgecolor("black")
        body.set_alpha(0.35)

    bp = ax.boxplot(
        [values],
        positions=[1],
        widths=0.22,
        showfliers=True,
        flierprops=dict(
            marker="o",
            markerfacecolor="black",
            markeredgecolor="black",
            alpha=0.3,
            markersize=4,
        ),
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.8},
        whiskerprops={"color": "black", "linewidth": 1.1},
        capprops={"color": "black", "linewidth": 1.1},
    )

    for patch in bp["boxes"]:
        patch.set_facecolor(colors[month])
        patch.set_edgecolor("black")
        patch.set_alpha(0.75)

    ax.set_xticks([1])
    ax.set_ylim(y_limits[month])
    ax.set_xticklabels("")
    ax.set_title(MONTH_NAMES[month].capitalize(), fontsize=17)
    ax.set_xlabel("")
    ax.set_ylabel("Galia, kW", fontsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.yaxis.set_major_formatter(FuncFormatter(lt_format))
    ax.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(
    IMAGE_DIR / "pv_galios_pasiskirstymas_violin_boxplot_separate.png",
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# %%
plot_monthly_power_series(
            data_final,
            value_col="Total Solar Power(W)",
            label="Saulės elektrinės sugeneruota galia (kW)",
            filename_template="saules_galia_{month:02d}.png",
            y_limits={
        2: (-0.01, 0.5),
        3: (-0.5, 20.5),
        4: (-0.5, 25),
    },
        )

# %%
plot_daily_spaghetti_power(
            data_final,
            value_col="Total Solar Power(W)",
            filename_template="saules_galia_paros_metu_{month:02d}.png",
            y_limits={
        2: (-0.01, 0.6),
        3: (-0.5, 20.5),
        4: (-0.5, 25),
    },
        )

# %% [markdown]
#  # Realus valdymas suvidurkinant

# %%
plot_real_control_profiles(data_final)

# %% [markdown]
#   # Tarifai ir Nord Pool kainos

# %%
data_final = add_tariffs(data_final)
### Jei neturi
# df_price=fetch_entsoe_prices()
df_price = load_price_data()

# %%
plot_price_series(df_price)

# %%
plot_tariff_weeks(data_final)

# %% [markdown]
#   # MILP scenarijus

# %%
milp_df = prepare_milp_dataframe(data_final, df_price)
dt_hours = DT_HOURS
print("Naudojamas dt (val.):", dt_hours)

# %%
res_rolling, daily_summary = run_rolling_milp(milp_df)

# %%
daily_summary["month"] = pd.to_datetime(daily_summary["date"]).dt.month
monthly_summary = daily_summary.groupby("month")["objective_day"].sum().reset_index()
monthly_summary["month_name"] = monthly_summary["month"].map(MONTH_NAMES)
print(monthly_summary[["month_name", "objective_day"]])

# %%
total_objective_rolling = daily_summary["objective_day"].sum()
print("Rolling total objective:", total_objective_rolling)

# %%
plot_milp_profiles(res_rolling)

# %%
print_milp_cost_summary(
    milp_df=milp_df,
    daily_summary=daily_summary,
    dt_hours=dt_hours,
    total_objective_rolling=total_objective_rolling,
)

# %% [markdown]
#   # Taisyklėmis pagrįstos strategijos

# %%
data_rule = build_rule_dataframe(data_final, milp_df)
INTERVAL_HOURS = DT_HOURS
print("Interval hours:", INTERVAL_HOURS)

# %% [markdown]
#   ## Maximizing self-consumption strategija

# %%
sim = rule_based_MSC(data_rule, dt_hours=INTERVAL_HOURS)

# %%
sim, dfm_MSC = print_rule_cost_summary(
    sim=sim,
    data_rule=data_rule,
    interval_hours=INTERVAL_HOURS,
    label="MSC",
)

# %%
df_rule_MSC = plot_rule_profiles(sim, data_rule, method="msc")

# %% [markdown]
#   ## TOU taisyklėmis pagrįsta strategija

# %%
sim_tou = rule_based_TOU(data_rule, dt_hours=INTERVAL_HOURS, tou_col="tou")

# %%
sim_tou, dfm_TOU = print_rule_cost_summary(
    sim=sim_tou,
    data_rule=data_rule,
    interval_hours=INTERVAL_HOURS,
    label="TOU",
)

# %%
df_rule_TOU = plot_rule_profiles(sim_tou, data_rule, method="tou")



