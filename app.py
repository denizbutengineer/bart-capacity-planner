import streamlit as st
import pandas as pd
import numpy as np
import json
import datetime
from xgboost import XGBRegressor

st.set_page_config(page_title="BART 10-Dk Peron & Vagon Optimizasyonu", page_icon="🚆", layout="wide")

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

SLOTS_10MIN = {
    "00 - 10": 0.14,
    "10 - 20": 0.19,
    "20 - 30": 0.20,
    "30 - 40": 0.18,
    "40 - 50": 0.15,
    "50 - 00": 0.14
}

st.title("🚆 BART 10 Dakikalık Yolcu Yoğunluğu ve Vagon Optimizasyonu")
st.markdown("Gerçek BART açık verileriyle eğitilmiş ML talep tahmini ve **10 dakikalık sefer penceresi** vagon dizilimi karar destek sistemi.")

with st.sidebar:
    st.header("⚙️ Operasyonel Parametreler")
    origin_code = st.selectbox("Başlangıç İstasyonu", options=list(STATIONS.keys()), format_func=lambda x: f"{x} - {STATIONS[x]}")
    
    today = datetime.date.today()
    selected_date = st.date_input(
        "Gelecek Planlama Tarihi",
        value=today + datetime.timedelta(days=1),
        min_value=today
    )
    
    hour = st.slider("Saat (05:00 - 23:00)", min_value=5, max_value=23, value=8)
    slot_label = st.selectbox("10 Dakikalık Zaman Dilimi", options=list(SLOTS_10MIN.keys()), index=1)
    trains_in_10min = st.selectbox("Bu 10 Dakikadaki Tren Sefer Sayısı", options=[1, 2, 3], index=0)

# Evrensel zamansal ve mekansal özellikler (6 sayısal özellik)
is_commute_peak = 1 if ((7 <= hour <= 9) or (16 <= hour <= 19)) else 0
hour_sin = np.sin(2 * np.pi * hour / 24.0)
hour_cos = np.cos(2 * np.pi * hour / 24.0)
is_transbay_core = 1 if origin_code in {'EMBR', 'MONT', 'POWL', 'CIVC', 'WOAK', '12TH', '19TH'} else 0
is_airport = 1 if origin_code in {'SFIA', 'OAKL', 'MLBR'} else 0

num_raw = np.array([hour, is_commute_peak, hour_sin, hour_cos, is_transbay_core, is_airport], dtype=float)
center = np.array(meta["center"], dtype=float)
scale = np.array(meta["scale"], dtype=float)

if len(center) != len(num_raw):
    st.error(f"Dosya sürüm uyuşmazlığı: bart_meta.json içinde {len(center)} özellik var, 6 bekleniyor. Lütfen en son indirdiğiniz bart_meta.json dosyasını GitHub'a yükleyin.")
    st.stop()

num_scaled = (num_raw - center) / scale
cat_ohe = np.array([1.0 if st_code == origin_code else 0.0 for st_code in meta["stations"]], dtype=float)
X_input = np.concatenate([num_scaled, cat_ohe]).reshape(1, -1)

# 1. Doğrudan Modelin Gerçek Saatlik Tahmini
hourly_pred = max(0, int(round(float(model.predict(X_input)[0]))))

# 2. 10 Dakikalık Dilime Ölçekleme
if is_commute_peak == 1:
    slot_weight = SLOTS_10MIN[slot_label]
    slot_weights_dict = SLOTS_10MIN
else:
    slot_weight = 1.0 / 6.0
    slot_weights_dict = {k: 1.0 / 6.0 for k in SLOTS_10MIN.keys()}

slot_passengers = int(round(hourly_pred * slot_weight))
passengers_per_train = slot_passengers / trains_in_10min

# 3. Vagon Optimizasyonu
required_cars = int(np.ceil(passengers_per_train / 150.0))
recommended_consist = min(10, max(3, required_cars))
train_capacity = recommended_consist * 150
occupancy_rate = min(100.0, (passengers_per_train / train_capacity) * 100)

st.subheader(f"📍 {STATIONS[origin_code]} | {selected_date.strftime('%d.%m.%Y')} | {hour:02d}:{slot_label.split(' - ')[0]} – {hour:02d}:{slot_label.split(' - ')[1]} Penceresi")
col1, col2, col3, col4 = st.columns(4)
col1.metric("10 Dk Tahmini Yolcu", f"{slot_passengers:,} kişi", f"Saatlik: {hourly_pred:,}")
col2.metric("Tren Başına Yolcu", f"{int(passengers_per_train):,} kişi", f"{trains_in_10min} tren/10 dk")
col3.metric("Önerilen Vagon Dizilimi", f"{recommended_consist} Vagonlu Set")
col4.metric("Peron / Vagon Doluluğu", f"%{occupancy_rate:.1f}")

st.divider()

left_col, right_col = st.columns([1.2, 1])

with left_col:
    st.markdown("### 🚦 Operasyonel Karar Özeti")
    if occupancy_rate > 90:
        st.error(f"⚠️ **Kritik Yığılma Riski:** `{hour:02d}:{slot_label}` aralığında yüksek peron yoğunluğu bekleniyor. **10 vagonlu maksimum dizi** ve ek ara sefer planlanmalıdır.")
    elif occupancy_rate > 65:
        st.warning(f"🟡 **Dengeli Pik Operasyon:** `{recommended_consist} vagonlu` tren dizilimi 10 dakikalık yolcu akışını konfor sınırları içinde karşılamaktadır.")
    else:
        st.success(f"🟢 **Enerji Optimizasyonu Fırsatı:** Bu 10 dakikalık dilimde talep düşüktür; **{recommended_consist} vagonlu kısa dizi** kullanılarak cer enerjisi tasarrufu sağlanabilir.")
    
    st.info("ℹ️ **Veri Kapsamı Notu:** Sistem yalnızca ileri tarihli sefer planlaması için yapılandırılmıştır. Tahmin motoru tamamen gerçek BART hafta içi işletme verilerinden öğrenilmiştir.")

with right_col:
    st.markdown(f"### 📊 {hour:02d}:00 Saati İçi 10 Dk Dağılımı")
    chart_df = pd.DataFrame({
        "10 Dk Dilimi": [f"{hour:02d}:{k.split(' - ')[0]}" for k in slot_weights_dict.keys()],
        "Tahmini Yolcu": [int(round(hourly_pred * w)) for w in slot_weights_dict.values()]
    }).set_index("10 Dk Dilimi")
    st.bar_chart(chart_df)
