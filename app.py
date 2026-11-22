import streamlit as st
import pandas as pd
import numpy as np
import json
import datetime
from xgboost import XGBRegressor

st.set_page_config(page_title="BART 10-Dk Peron & Vagon Optimizasyonu", page_icon="🚆", layout="wide")

@st.cache_resource
def load_assets():
    xgb_model = XGBRegressor()
    xgb_model.load_model("bart_xgb.json")
    with open("bart_meta.json", "r") as f:
        meta = json.load(f)
    return xgb_model, meta

model, meta = load_assets()

STATIONS = {
    "EMBR": "Embarcadero (San Francisco Downtown)",
    "MONT": "Montgomery St. (Financial District)",
    "POWL": "Powell St. (Union Square)",
    "CIVC": "Civic Center / UN Plaza",
    "16TH": "16th St. Mission",
    "24TH": "24th St. Mission",
    "DALY": "Daly City",
    "SFIA": "San Francisco International Airport (SFO)",
    "MLBR": "Millbrae (Caltrain Aktarma)",
    "WOAK": "West Oakland (Transbay Girişi)",
    "12TH": "12th St. Oakland City Center",
    "19TH": "19th St. Oakland",
    "DBRK": "Downtown Berkeley",
    "OAKL": "Oakland International Airport (OAK)",
    "FRMT": "Fremont",
    "BERY": "Berryessa / North San José"
}

SLOT_KEYS = ["00 - 10", "10 - 20", "20 - 30", "30 - 40", "40 - 50", "50 - 00"]
GUNLER_TR = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]

def get_intrahour_weights(hour: int, is_weekend: bool, is_airport: int) -> dict:
    """Saatin gün içindeki konumuna göre gerçekçi 10 dakikalık yolcu akış ağırlıkları."""
    if is_airport == 1:
        raw = np.array([0.15, 0.18, 0.19, 0.17, 0.16, 0.15])
    elif not is_weekend and hour in (6, 7, 15, 16):
        raw = np.array([0.11, 0.13, 0.16, 0.18, 0.20, 0.22])
    elif not is_weekend and hour in (8, 17):
        raw = np.array([0.14, 0.19, 0.21, 0.18, 0.15, 0.13])
    elif not is_weekend and hour in (9, 18, 19):
        raw = np.array([0.22, 0.20, 0.18, 0.15, 0.13, 0.12])
    elif hour >= 20:
        raw = np.array([0.21, 0.19, 0.17, 0.16, 0.14, 0.13])
    else:
        raw = np.array([0.15, 0.17, 0.18, 0.18, 0.16, 0.16])

    raw = raw / raw.sum()
    return dict(zip(SLOT_KEYS, raw))

st.title("🚆 BART 10 Dakikalık Yolcu Yoğunluğu ve Vagon Optimizasyonu")
st.markdown("Gerçek BART açık verileriyle eğitilmiş ML talep tahmini ve **10 dakikalık sefer penceresi** vagon dizilimi karar destek sistemi.")

with st.sidebar:
    st.header("⚙️ Operasyonel Parametreler")
    origin_code = st.selectbox(
        "Başlangıç İstasyonu",
        options=list(STATIONS.keys()),
        format_func=lambda x: f"{x} - {STATIONS[x]}"
    )
    
    today = datetime.date.today()
    selected_date = st.date_input(
        "Gelecek Planlama Tarihi",
        value=today + datetime.timedelta(days=1),
        min_value=today
    )
    
    hour = st.slider("Saat (05:00 - 23:00)", min_value=5, max_value=23, value=8)
    slot_label = st.selectbox("10 Dakikalık Zaman Dilimi", options=SLOT_KEYS, index=1)
    trains_in_10min = st.selectbox("Bu 10 Dakikadaki Tren Sefer Sayısı", options=[1, 2, 3], index=0)

# Zamansal ve mekansal özellik hesaplamaları
day_idx = selected_date.weekday()
day_name = GUNLER_TR[day_idx]
is_weekend = 1 if day_idx >= 5 else 0
month_val = selected_date.month

is_commute_peak = 1 if ((is_weekend == 0) and ((7 <= hour <= 9) or (16 <= hour <= 19))) else 0
hour_sin = np.sin(2 * np.pi * hour / 24.0)
hour_cos = np.cos(2 * np.pi * hour / 24.0)
is_transbay_core = 1 if origin_code in {'EMBR', 'MONT', 'POWL', 'CIVC', 'WOAK', '12TH', '19TH'} else 0
is_airport = 1 if origin_code in {'SFIA', 'OAKL', 'MLBR'} else 0

center = np.array(meta["center"], dtype=float)
scale = np.array(meta["scale"], dtype=float)

# bart_meta.json içindeki özellik sayısına (9 veya 6) otomatik uyum
feature_map = {
    "Hour": float(hour),
    "DayOfWeek": float(day_idx),
    "Is_Weekend": float(is_weekend),
    "Month": float(month_val),
    "Is_Commute_Peak": float(is_commute_peak),
    "Hour_Sin": float(hour_sin),
    "Hour_Cos": float(hour_cos),
    "Is_Transbay_Core": float(is_transbay_core),
    "Is_Airport": float(is_airport),
}

if "num_features" in meta and len(meta["num_features"]) == len(center):
    num_raw = np.array([feature_map[col] for col in meta["num_features"]], dtype=float)
elif len(center) == 9:
    num_raw = np.array([
        hour, day_idx, is_weekend, month_val,
        is_commute_peak, hour_sin, hour_cos, is_transbay_core, is_airport
    ], dtype=float)
elif len(center) == 6:
    num_raw = np.array([
        hour, is_commute_peak, hour_sin, hour_cos, is_transbay_core, is_airport
    ], dtype=float)
else:
    st.error(f"Beklenmeyen özellik boyutu: bart_meta.json içinde {len(center)} özellik bulundu.")
    st.stop()

# Sıfıra bölünme (scale == 0) koruması
safe_scale = np.where(scale == 0, 1.0, scale)
num_scaled = (num_raw - center) / safe_scale

cat_ohe = np.array([1.0 if st_code == origin_code else 0.0 for st_code in meta["stations"]], dtype=float)
X_input = np.concatenate([num_scaled, cat_ohe]).reshape(1, -1)

# 1. Doğrudan Modelin Gerçek Saatlik Tahmini
raw_hourly_pred = max(0.0, float(model.predict(X_input)[0]))

# Eğer model yalnızca hafta içi verisiyle eğitilmişse (scale[2] == 0 ise Is_Weekend sabittir),
# hafta sonu seçimlerinde BART 2025 resmi katsayıları ile kalibre et
if len(center) == 6 or (len(scale) >= 3 and scale[2] == 0):
    day_factor = 0.60 if day_idx == 5 else (0.43 if day_idx == 6 else 1.00)
else:
    day_factor = 1.00

hourly_pred = max(0, int(round(raw_hourly_pred * day_factor)))

# 2. 10 Dakikalık Dilime Ölçekleme
slot_weights_dict = get_intrahour_weights(hour, bool(is_weekend), is_airport)
slot_weight = slot_weights_dict[slot_label]

slot_passengers = int(round(hourly_pred * slot_weight))
passengers_per_train = slot_passengers / trains_in_10min

# 3. Vagon Optimizasyonu
required_cars = int(np.ceil(passengers_per_train / 150.0))
recommended_consist = min(10, max(3, required_cars))
train_capacity = recommended_consist * 150
occupancy_rate = min(100.0, (passengers_per_train / train_capacity) * 100)

st.subheader(f"📍 {STATIONS[origin_code]} | {selected_date.strftime('%d.%m.%Y')} ({day_name}) | {hour:02d}:{slot_label.split(' - ')[0]} – {hour:02d}:{slot_label.split(' - ')[1]} Penceresi")
col1, col2, col3, col4 = st.columns(4)
col1.metric("10 Dk Tahmini Yolcu", f"{slot_passengers:,} kişi", f"Saatlik: {hourly_pred:,}")
col2.metric("Tren Başına Yolcu", f"{int(passengers_per_train):,} kişi", f"{trains_in_10min} tren/10 dk")
col3.metric("Önerilen Vagon Dizilimi", f"{recommended_consist} Vagonlu Set")
col4.metric("Peron / Vagon Doluluğu", f"%{occupancy_rate:.1f}")

st.divider()

left_col, right_col = st.columns([1.15, 1])

with left_col:
    st.markdown("### 🚦 Operasyonel Karar Özeti")
    if occupancy_rate > 90:
        st.error(f"⚠️ **Kritik Yığılma Riski:** `{hour:02d}:{slot_label}` aralığında yüksek peron yoğunluğu bekleniyor. **10 vagonlu maksimum dizi** ve ek ara sefer planlanmalıdır.")
    elif occupancy_rate > 65:
        st.warning(f"🟡 **Dengeli Pik Operasyon:** `{recommended_consist} vagonlu` tren dizilimi 10 dakikalık yolcu akışını konfor sınırları içinde karşılamaktadır.")
    else:
        st.success(f"🟢 **Enerji Optimizasyonu Fırsatı:** Bu 10 dakikalık dilimde talep düşüktür; **{recommended_consist} vagonlu kısa dizi** kullanılarak cer enerjisi tasarrufu sağlanabilir.")
    
    st.info(f"ℹ️ **Planlama Notu ({day_name}):** Tahmin motoru gerçek BART açık verilerinden eğitilmiş `XGBoost` modelini kullanır ve seçilen tarih/saat profiline göre 10 dakikalık vagon kapasitesini optimize eder.")

with right_col:
    st.markdown(f"### 📊 {hour:02d}:00 Saati İçi 10 Dk Dağılımı")
    chart_df = pd.DataFrame({
        "10 Dk Dilimi": [f"{hour:02d}:{k.split(' - ')[0]}" for k in slot_weights_dict.keys()],
        "Tahmini Yolcu": [int(round(hourly_pred * w)) for w in slot_weights_dict.values()]
    }).set_index("10 Dk Dilimi")
    st.bar_chart(chart_df, color="#38bdf8")

with st.expander("📂 Veri Seti ve Model Özeti (BTK Proje Standartları)"):
    st.markdown("""
    - **Veri Kaynağı & Lisans:** Bay Area Rapid Transit (BART) Resmi Açık Veri Portalı (*Hourly Ridership by Origin-Destination Pairs*, Creative Commons Attribution **CC-BY**).
    - **Hedef Değişken:** İstasyon ve saat bazlı toplam yolcu talebi (`Total_Riders`).
    - **Seçilen Algoritma:** `XGBoost Regressor` (`TimeSeriesSplit` çapraz doğrulama ve `RandomizedSearchCV` ile optimize edilmiştir).
    - **Operasyonel Kural:** Vagon başına maksimum konfor kapasitesi **150 yolcu** kabul edilerek **3 ile 10 vagon** arasında dinamik dizi önerisi üretilir.
    """)
