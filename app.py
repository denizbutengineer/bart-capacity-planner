import json
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb


# -----------------------------------------------------------------------------
# Sayfa yapılandırması
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Demiryolu Kapasite Planlayıcısı",
    page_icon="🚆",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(128, 128, 128, 0.25);
        padding: 16px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

MODEL_DIR = Path(__file__).resolve().parent
MODEL_PATH = MODEL_DIR / "bart_model.json"
METADATA_PATH = MODEL_DIR / "model_metadata.json"


# -----------------------------------------------------------------------------
# Model ve metadata yükleme
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Eğitilmiş XGBoost modeli yükleniyor...")
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model dosyası bulunamadı: {MODEL_PATH.name}"
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Model metadata dosyası bulunamadı: {METADATA_PATH.name}"
        )

    model = xgb.XGBRegressor()
    model.load_model(str(MODEL_PATH))

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    feature_cols = metadata.get("feature_cols")
    line_mapping = metadata.get("line_mapping")

    if not feature_cols or not line_mapping:
        raise ValueError(
            "model_metadata.json içinde feature_cols veya line_mapping bulunamadı."
        )

    if len(feature_cols) != 17:
        raise ValueError(
            f"Beklenen özellik sayısı 17; metadata içinde {len(feature_cols)} özellik var."
        )

    return model, feature_cols, line_mapping, metadata


try:
    model, features, line_map, metadata = load_model()
except Exception as exc:
    st.error(f"Model yüklenemedi: {exc}")
    st.stop()


# -----------------------------------------------------------------------------
# Eğitim notebook'u ile aynı özellik üretimi
# -----------------------------------------------------------------------------
def build_feature_row(
    target_date: pd.Timestamp,
    line_name: str,
    direction: int,
    slot_of_day: int,
    line_mapping: dict,
    lag_1d: float,
    lag_7d: float,
    rolling_mean_7d: float,
) -> dict:
    target_date = pd.Timestamp(target_date)

    day_of_week = int(target_date.dayofweek)
    is_weekend = int(day_of_week >= 5)
    month = int(target_date.month)
    day_of_year = int(target_date.dayofyear)

    holiday_calendar = holidays.US(
        subdiv="CA",
        years=[target_date.year]
    )
    is_holiday = int(target_date.date() in holiday_calendar)

    is_workday = (is_weekend == 0) and (is_holiday == 0)
    is_am_peak = int(is_workday and 42 <= slot_of_day <= 57)
    is_pm_peak = int(is_workday and 99 <= slot_of_day <= 114)

    if line_name not in line_mapping:
        raise ValueError(f"Bilinmeyen hat: {line_name}")

    return {
        "Line_Code": int(line_mapping[line_name]),
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


# -----------------------------------------------------------------------------
# Vagon dizilimi hesabı
# -----------------------------------------------------------------------------
def recommend_car_consist(
    predicted_10min_load: float,
    headway_minutes: int = 15,
    target_cap_per_car: int = 110,
):
    headway_minutes = max(float(headway_minutes), 1.0)
    target_cap_per_car = max(float(target_cap_per_car), 1.0)

    trains_in_10min = 10.0 / headway_minutes
    passengers_per_train = predicted_10min_load / max(trains_in_10min, 0.25)

    required_cars = int(
        np.ceil(passengers_per_train / target_cap_per_car)
    )
    recommended_cars = int(np.clip(required_cars, 3, 10))

    return passengers_per_train, required_cars, recommended_cars


# -----------------------------------------------------------------------------
# Arayüz
# -----------------------------------------------------------------------------
st.title("🚆 Demiryolu Kapasite Planlayıcısı")
st.subheader("10 Dakikalık Kesit Yükü Tahmini ve Tren Seti Planlaması")
st.write(
    "Seçilen hat, yön, tarih ve 10 dakikalık zaman dilimi için tahmini kesit yükünü "
    "hesaplar. Ardından sefer aralığı ve vagon başına hedef kapasiteye göre tren "
    "seti önerisi üretir."
)
st.divider()

training_period = metadata.get("training_period", "2025")
st.caption(
    f"Model eğitim dönemi: {training_period}. "
    "Tahminler modelin öğrendiği dönem ve veri varsayımlarıyla sınırlıdır."
)

with st.sidebar:
    st.header("Tahmin Parametreleri")

    selected_date = st.date_input(
        "Tahmin Tarihi",
        value=pd.Timestamp("2025-12-15").date(),
        min_value=pd.Timestamp("2025-01-01").date(),
        max_value=pd.Timestamp("2030-12-31").date(),
    )

    selected_line = st.selectbox(
        "Hat",
        options=list(line_map.keys())
    )

    selected_direction = st.selectbox(
        "Yön",
        options=[0, 1],
        format_func=lambda value: "Gidiş (0)" if value == 0 else "Dönüş (1)"
    )

    slot_of_day = st.slider(
        "10 Dakikalık Slot",
        min_value=0,
        max_value=143,
        value=51,
        help="0 = 00:00, 51 = 08:30, 143 = 23:50"
    )

    hour = (slot_of_day * 10) // 60
    minute = (slot_of_day * 10) % 60
    st.caption(f"Seçilen zaman: **{hour:02d}:{minute:02d}**")

    headway = st.slider(
        "Sefer Aralığı (dakika)",
        min_value=5,
        max_value=30,
        value=15,
        step=5
    )

    cap_per_car = st.slider(
        "Vagon Başına Hedef Kapasite",
        min_value=60,
        max_value=150,
        value=110,
        step=5
    )

    st.subheader("Geçmiş Yük Girdileri")
    st.caption(
        "Aynı hat, yön ve 10 dakikalık slot için geçmiş verileri girin. "
        "Varsayılan değerler yalnızca örnek senaryodur."
    )

    lag_1d = st.number_input(
        "1 Gün Önce — Lag_1d",
        min_value=0.0,
        value=220.0,
        step=10.0
    )

    lag_7d = st.number_input(
        "7 Gün Önce — Lag_7d",
        min_value=0.0,
        value=240.0,
        step=10.0
    )

    rolling_7d = st.number_input(
        "Önceki 7 Günün Aynı Slot Ortalama Yükü",
        min_value=0.0,
        value=230.0,
        step=10.0
    )

# -----------------------------------------------------------------------------
# Tahmin girdisini oluştur ve sırayı metadata'daki feature_cols ile eşleştir.
# -----------------------------------------------------------------------------
target_timestamp = pd.Timestamp(selected_date)

row_dict = build_feature_row(
    target_date=target_timestamp,
    line_name=selected_line,
    direction=selected_direction,
    slot_of_day=slot_of_day,
    line_mapping=line_map,
    lag_1d=lag_1d,
    lag_7d=lag_7d,
    rolling_mean_7d=rolling_7d,
)

X_single = pd.DataFrame([row_dict], columns=features)

if X_single.isna().any().any():
    st.error("Model girdisinde eksik özellik bulundu. Girdileri kontrol edin.")
    st.stop()

try:
    predicted_log = float(model.predict(X_single)[0])
    predicted_load = float(np.clip(np.expm1(predicted_log), 0.0, None))
except Exception as exc:
    st.error(f"Tahmin sırasında hata oluştu: {exc}")
    st.stop()

passengers_per_train, required_cars, recommended_cars = recommend_car_consist(
    predicted_10min_load=predicted_load,
    headway_minutes=headway,
    target_cap_per_car=cap_per_car,
)

col1, col2, col3 = st.columns(3)
col1.metric(
    "10 Dakikalık Tahmini Kesit Yükü",
    f"{predicted_load:,.1f} yolcu"
)
col2.metric(
    "Sefer Başına Tahmini Yük",
    f"{passengers_per_train:,.1f} yolcu"
)
col3.metric(
    "Önerilen Tren Dizilimi",
    f"{recommended_cars} vagon"
)

if required_cars > 10:
    st.error(
        f"Hesaplanan ihtiyaç {required_cars} vagon. Sistem en fazla 10 vagon "
        "önerdiği için bu dizilim hedef kapasiteyi karşılamayabilir."
    )
elif required_cars < 3:
    st.info(
        f"Hesaplanan teorik ihtiyaç {required_cars} vagon. "
        "Sistem alt sınır gereği 3 vagon öneriyor."
    )
else:
    st.success(
        f"Hesaplanan vagon ihtiyacı {required_cars}; önerilen dizilim "
        f"{recommended_cars} vagondur."
    )

if target_timestamp.year != 2025:
    st.warning(
        "Seçilen tarih modelin eğitim yılı olan 2025 dışındadır. "
        "Bu tarihler için tahminler dönem dışı kullanım olarak değerlendirilmelidir."
    )

st.divider()
st.subheader("Modele Gönderilen Özellikler")
st.dataframe(X_single, use_container_width=True)

with st.expander("Tahmin ve kapasite hesaplama ayrıntıları"):
    st.write(f"Tahmin edilen log yük: {predicted_log:.6f}")
    st.write(f"10 dakikalık tahmini yük: {predicted_load:.2f}")
    st.write(f"Sefer aralığı: {headway} dakika")
    st.write(f"Vagon başına hedef kapasite: {cap_per_car} yolcu")
    st.write(f"Sınırlanmamış teorik vagon ihtiyacı: {required_cars}")
    st.write(f"3–10 aralığında önerilen vagon sayısı: {recommended_cars}")

st.caption(
    "Uyarı: Bu uygulama karar destek amaçlı bir prototiptir. "
    "Tahminler saatlik O-D verisinden türetilen 10 dakikalık göstergeye dayanır; "
    "gerçek zamanlı sensör veya turnike ölçümü değildir. Operasyonel kararlar "
    "öncesinde gerçek işletme verileriyle doğrulanmalıdır."
)
