import joblib
import holidays
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Demiryolu Kapasite Planlayıcısı",
    page_icon="🚆",
    layout="wide",
)

st.title("🚆 Demiryolu 10 Dakikalık Kesit Tahmini ve Tren Seti (Consist) Planlayıcı")
st.markdown(
    "Bu karar destek sistemi, seçilen gelecek tarih ve 10 dakikalık zaman dilimindeki "
    "kritik kesit yolcu yükünü öngörür ve planlanan sefer aralığına (headway) göre "
    "en uygun vagon dizisini (**3–10 vagon**) belirler."
)

LINE_MAPPING = {"Blue": 0, "Green": 1, "Orange": 2, "Red": 3, "Yellow": 4}
DEFAULT_FEATURES = [
    "Line_Code", "Direction", "Slot_Of_Day",
    "Day_Of_Week", "Is_Weekend", "Month", "Day_Of_Year", "Is_Holiday",
    "Slot_Sin", "Slot_Cos", "Dow_Sin", "Dow_Cos",
    "Is_AM_Peak", "Is_PM_Peak",
    "Lag_1d", "Lag_7d", "Rolling_Mean_7d",
]

ALIAS_MAP = {
    "Line_Code": ["LineCode", "line_code"],
    "Direction": ["direction"],
    "Slot_Of_Day": ["SlotOfDay", "slot_of_day"],
    "Day_Of_Week": ["DayOfWeek", "day_of_week"],
    "Is_Weekend": ["IsWeekend", "is_weekend"],
    "Month": ["month"],
    "Day_Of_Year": ["DayOfYear", "day_of_year"],
    "Is_Holiday": ["IsHoliday", "is_holiday"],
    "Slot_Sin": ["slot_sin"],
    "Slot_Cos": ["slot_cos"],
    "Dow_Sin": ["dow_sin"],
    "Dow_Cos": ["dow_cos"],
    "Is_AM_Peak": ["IsAMPeak", "is_am_peak"],
    "Is_PM_Peak": ["IsPMPeak", "is_pm_peak"],
    "Lag_1d": ["lag_1d"],
    "Lag_7d": ["lag_7d"],
    "Rolling_Mean_7d": ["rolling_mean_7d"],
}


@st.cache_resource
def load_model_artifact(path: str = "bart_capacity_xgb.joblib"):
    loaded = joblib.load(path)
    if isinstance(loaded, dict):
        model = loaded.get("model")
        features = loaded.get("feature_cols", DEFAULT_FEATURES)
        line_map = loaded.get("line_mapping", LINE_MAPPING)
    else:
        model = loaded
        features = DEFAULT_FEATURES
        line_map = LINE_MAPPING
    return model, list(features), line_map


def recommend_car_consist(
    predicted_10min_load: float,
    headway_minutes: int = 15,
    target_cap_per_car: int = 110,
) -> int:
    trains_in_10min = 10.0 / max(headway_minutes, 1)
    passengers_per_train = predicted_10min_load / max(trains_in_10min, 0.25)
    required_cars = int(np.ceil(passengers_per_train / max(target_cap_per_car, 1)))
    return int(np.clip(required_cars, 3, 10))


def build_feature_row(
    target_date: pd.Timestamp,
    line_name: str,
    direction: int,
    slot_of_day: int,
    line_map: dict,
    lag_1d: float,
    lag_7d: float,
    rolling_mean_7d: float,
) -> dict:
    day_of_week = int(target_date.dayofweek)
    is_weekend = int(day_of_week >= 5)
    month = int(target_date.month)
    day_of_year = int(target_date.dayofyear)

    us_ca_holidays = holidays.US(subdiv="CA", years=[target_date.year])
    is_holiday = int(target_date.normalize() in us_ca_holidays)

    is_workday = (is_weekend == 0) and (is_holiday == 0)
    is_am_peak = int(is_workday and (42 <= slot_of_day <= 57))
    is_pm_peak = int(is_workday and (99 <= slot_of_day <= 114))

    return {
        "Line_Code": int(line_map.get(line_name, 0)),
        "Direction": int(direction),
        "Slot_Of_Day": int(slot_of_day),
        "Day_Of_Week": day_of_week,
        "Is_Weekend": is_weekend,
        "Month": month,
        "Day_Of_Year": day_of_year,
        "Is_Holiday": is_holiday,
        "Slot_Sin": float(np.sin(2 * np.pi * slot_of_day / 144.0)),
        "Slot_Cos": float(np.cos(2 * np.pi * slot_of_day / 144.0)),
        "Dow_Sin": float(np.sin(2 * np.pi * day_of_week / 7.0)),
        "Dow_Cos": float(np.cos(2 * np.pi * day_of_week / 7.0)),
        "Is_AM_Peak": is_am_peak,
        "Is_PM_Peak": is_pm_peak,
        "Lag_1d": float(lag_1d),
        "Lag_7d": float(lag_7d),
        "Rolling_Mean_7d": float(rolling_mean_7d),
    }


def prepare_single_input(row_dict: dict, features: list) -> pd.DataFrame:
    safe_row = dict(row_dict)
    for col in features:
        if col not in safe_row:
            val = None
            for alt in ALIAS_MAP.get(col, []):
                if alt in safe_row:
                    val = safe_row[alt]
                    break
            safe_row[col] = 0.0 if val is None else val
    return pd.DataFrame([safe_row])[features]


try:
    model, features, line_map = load_model_artifact("bart_capacity_xgb.joblib")
except FileNotFoundError:
    st.error("`bart_capacity_xgb.joblib` dosyası bulunamadı. Lütfen model dosyasını `app.py` ile aynı dizine yerleştirin.")
    st.stop()

with st.sidebar:
    st.header("Planlama Parametreleri")
    selected_date = st.date_input("Tarih Seçin", value=pd.Timestamp("2025-12-15").date())
    selected_line = st.selectbox("Hat", options=list(line_map.keys()))
    selected_dir = st.selectbox("Yön", options=[0, 1], format_func=lambda d: "Gidiş (0)" if d == 0 else "Dönüş (1)")

    slot_of_day = st.slider("10 Dakikalık Slot (0–143)", min_value=0, max_value=143, value=51)
    hh = (slot_of_day * 10) // 60
    mm = (slot_of_day * 10) % 60
    st.caption(f"Karşılık gelen saat dilimi: **{hh:02d}:{mm:02d}**")

    headway = st.slider("Sefer Aralığı / Headway (dakika)", min_value=5, max_value=30, value=15, step=5)
    cap_per_car = st.slider("Vagon Başına Hedef Kapasite (yolcu)", min_value=60, max_value=150, value=110, step=5)

    st.subheader("Geçmiş Talep Girdileri (Opsiyonel)")
    lag_1d = st.number_input("1 Gün Önceki Yük (Lag_1d)", min_value=0.0, value=220.0, step=10.0)
    lag_7d = st.number_input("1 Hafta Önceki Yük (Lag_7d)", min_value=0.0, value=240.0, step=10.0)
    rolling_7d = st.number_input("7 Günlük Kaydırılmış Ortalama", min_value=0.0, value=230.0, step=10.0)

row_dict = build_feature_row(
    target_date=pd.Timestamp(selected_date),
    line_name=selected_line,
    direction=selected_dir,
    slot_of_day=slot_of_day,
    line_map=line_map,
    lag_1d=lag_1d,
    lag_7d=lag_7d,
    rolling_mean_7d=rolling_7d,
)

X_single = prepare_single_input(row_dict, features)
pred_log = model.predict(X_single)[0]
pred_load = float(np.clip(np.expm1(pred_log), 0.0, None))
rec_cars = recommend_car_consist(pred_load, headway_minutes=headway, target_cap_per_car=cap_per_car)

col1, col2, col3 = st.columns(3)
col1.metric("Tahmini 10 Dakikalık Kesit Yükü", f"{pred_load:,.1f} yolcu")
col2.metric("Sefer Başına Tahmini Yük", f"{pred_load / max(10.0 / headway, 0.25):,.1f} yolcu")
col3.metric("Önerilen Tren Dizilimi", f"{rec_cars} Vagon")

with st.expander("Modele Gönderilen Özellik Satırı (X_single)"):
    st.dataframe(X_single, use_container_width=True)
