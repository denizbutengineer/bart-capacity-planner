import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime

st.set_page_config(page_title="BART 10-Dk Peron & Vagon Optimizasyonu", page_icon="🚆", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("bart_ridership_model.pkl")

model = load_model()

# BART Ana İstasyon Sözlüğü
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

# 10 Dakikalık Zaman Dilimleri ve Saat İçi Yoğunluk Ağırlıkları (Toplam = 1.00)
# Pik saatlerde ortadaki 10 dakikalık dilimler (:10-:40) daha yüksek peron yığılması taşır
SLOTS_10MIN = {
    "00 - 10": 0.14,
    "10 - 20": 0.19,
    "20 - 30": 0.20,
    "30 - 40": 0.18,
    "40 - 50": 0.15,
    "50 - 00": 0.14
}

st.title("🚆 BART 10 Dakikalık Yolcu Yoğunluğu ve Vagon Optimizasyonu")
st.markdown("Saatlik ML talep tahmini üzerine kurulu **10 dakikalık mikro-zaman dilimi** ve **Tren Dizilimi (Train Consist)** karar destek sistemi.")

with st.sidebar:
    st.header("⚙️ Operasyonel Parametreler")
    origin_code = st.selectbox("Başlangıç İstasyonu", options=list(STATIONS.keys()), format_func=lambda x: f"{x} - {STATIONS[x]}")
    selected_date = st.date_input("Tarih Seçimi", datetime.date(2025, 10, 15))
    hour = st.slider("Saat (05:00 - 23:00)", min_value=5, max_value=23, value=8)
    slot_label = st.selectbox("10 Dakikalık Zaman Dilimi", options=list(SLOTS_10MIN.keys()), index=1)
    trains_in_10min = st.selectbox("Bu 10 Dakikadaki Tren Sefer Sayısı", options=[1, 2, 3], index=0, help="Pik saatlerde koridordan 10 dakikada 2-3 tren geçebilir.")

# Feature Engineering (Colab eğitim pipeline'ı ile birebir aynı)
day_of_week = selected_date.weekday()
month = selected_date.month
is_weekend = 1 if day_of_week >= 5 else 0
is_commute_peak = 1 if (((7 <= hour <= 9) or (16 <= hour <= 19)) and is_weekend == 0) else 0
hour_sin = np.sin(2 * np.pi * hour / 24.0)
hour_cos = np.cos(2 * np.pi * hour / 24.0)
is_transbay_core = 1 if origin_code in {'EMBR', 'MONT', 'POWL', 'CIVC', 'WOAK', '12TH', '19TH'} else 0
is_airport = 1 if origin_code in {'SFIA', 'OAKL', 'MLBR'} else 0

input_df = pd.DataFrame([{
    'Origin': origin_code,
    'Hour': hour,
    'DayOfWeek': day_of_week,
    'Month': month,
    'Is_Weekend': is_weekend,
    'Is_Commute_Peak': is_commute_peak,
    'Hour_Sin': hour_sin,
    'Hour_Cos': hour_cos,
    'Is_Transbay_Core': is_transbay_core,
    'Is_Airport': is_airport
}])

# 1. Saatlik ML Tahmini
hourly_pred = max(0, int(round(model.predict(input_df)[0])))

# 2. 10 Dakikalık Dilime Ölçekleme (Pik saatlerde çan eğrisi ağırlığı, normal saatlerde eşit 1/6 dağılım)
if is_commute_peak == 1:
    slot_weight = SLOTS_10MIN[slot_label]
    slot_weights_dict = SLOTS_10MIN
else:
    slot_weight = 1.0 / 6.0
    slot_weights_dict = {k: 1.0 / 6.0 for k in SLOTS_10MIN.keys()}

slot_passengers = int(round(hourly_pred * slot_weight))
passengers_per_train = slot_passengers / trains_in_10min

# 3. Vagon Optimizasyonu (1 BART Vagonu ≈ 150 yolcu kapasitesi, 3-10 vagon aralığı)
required_cars = int(np.ceil(passengers_per_train / 150.0))
recommended_consist = min(10, max(3, required_cars))
train_capacity = recommended_consist * 150
occupancy_rate = min(100.0, (passengers_per_train / train_capacity) * 100)

# Üst Metrik Kartları
st.subheader(f"📍 {STATIONS[origin_code]} | {hour:02d}:{slot_label.split(' - ')[0]} – {hour:02d}:{slot_label.split(' - ')[1]} Penceresi")
col1, col2, col3, col4 = st.columns(4)
col1.metric("10 Dk Tahmini Yolcu", f"{slot_passengers:,} kişi", f"Saatlik: {hourly_pred:,}")
col2.metric("Tren Başına Yolcu", f"{int(passengers_per_train):,} kişi", f"{trains_in_10min} tren/10 dk")
col3.metric("Önerilen Vagon Dizilimi", f"{recommended_consist} Vagonlu Set")
col4.metric("Peron / Vagon Doluluğu", f"%{occupancy_rate:.1f}")

st.divider()

# Durum Uyarısı ve Saat İçi 10 Dakikalık Dağılım Grafiği
left_col, right_col = st.columns([1.2, 1])

with left_col:
    st.markdown("### 🚦 Operasyonel Karar Özeti")
    if occupancy_rate > 90:
        st.error(f"⚠️ **Kritik Yığılma Riski:** `{hour:02d}:{slot_label}` aralığında yüksek peron yoğunluğu bekleniyor. **10 vagonlu maksimum dizi** ve ek ara sefer planlanmalıdır.")
    elif occupancy_rate > 65:
        st.warning(f"🟡 **Dengeli Pik Operasyon:** `{recommended_consist} vagonlu` tren dizilimi 10 dakikalık yolcu akışını konfor sınırları içinde karşılamaktadır.")
    else:
        st.success(f"🟢 **Enerji Optimizasyonu Fırsatı:** Bu 10 dakikalık dilimde talep düşüktür; **{recommended_consist} vagonlu kısa dizi** kullanılarak cer enerjisi tasarrufu sağlanabilir.")
    
    st.info("ℹ️ **Metodoloji Notu:** BART açık veri seti saatlik çözünürlükte sunulduğu için tahmin motoru saatlik veriyle eğitilmiş, operasyonel katmanda pik saat içi yoğunluk profili ile 10 dakikalık sefer pencerelerine ayrıştırılmıştır.")

with right_col:
    st.markdown(f"### 📊 {hour:02d}:00 Saati İçi 10 Dk Dağılımı")
    chart_df = pd.DataFrame({
        "10 Dk Dilimi": [f"{hour:02d}:{k.split(' - ')[0]}" for k in slot_weights_dict.keys()],
        "Tahmini Yolcu": [int(round(hourly_pred * w)) for w in slot_weights_dict.values()]
    }).set_index("10 Dk Dilimi")
    st.bar_chart(chart_df)