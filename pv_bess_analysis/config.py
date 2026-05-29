from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FOLDER = PROJECT_ROOT / "data_stitched"
LOGGER_FILE = DATA_FOLDER / "Feb_Mar_Apr_2026_Combined_Filtered.csv"
PRICE_FILE = PROJECT_ROOT / "LT_prices_Feb_Mar_Apr_2026.csv"
IMAGE_DIR = PROJECT_ROOT / "images"
SECRETS_FILE = PROJECT_ROOT / "secrets.json"

SELECTED_COLUMNS = [
    "Time",
    "Total Solar Power(W)",
    "DC Power PV1(W)",
    "DC Power PV2(W)",
    "Battery Power(W)",
    "Battery Status",
    "Total Grid Power(W)",
    "SoC(%)",
    "Grid Status",
    "Total Consumption Power(W)",
]

VALUE_COLUMNS = [
    "Total Solar Power(W)",
    "DC Power PV1(W)",
    "DC Power PV2(W)",
    "Battery Power(W)",
    "Total Grid Power(W)",
    "SoC(%)",
    "Total Consumption Power(W)",
]

MONTHS = [2, 3, 4]
MONTH_NAMES = {2: "Vasaris", 3: "Kovas", 4: "Balandis"}
DT_HOURS = 5 / 60

TOTAL_EB = 24.56
SOC0 = 95
SOC_MIN = 10
SOC_MAX = 100
ETA_C = 0.95
ETA_D = 0.95
P_CH_MAX = 12.5
P_DIS_MAX = 12.5
P_GIM_MAX = 28.0
P_GEX_MAX = 20.0

FIXED_PRICE = 0.24
SUPPLIER_ADMIN_FEE_EUR_KWH = 0.02470
