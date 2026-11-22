import streamlit as st
import pandas as pd
import numpy as np
import json
import datetime
from xgboost import XGBRegressor

st.set_page_config(page_title="BART Hat Bazlı Peron & Vagon Optimizasyonu", page_icon="🚆", layout="wide")

@st.cache_resource
def load_assets():
    xgb_model = XGBRegressor()
    xgb_model.load_model("bart_xgb.json")
    with open("bart_meta.json", "r") as f:
        meta = json.load(f)
    return xgb_model, meta

model, meta = load_assets()

STATION_NAMES = {
    "EMBR": "Embarcadero (SF Downtown)",
    "MONT": "Montgomery St. (Financial Dist.)",
    "POWL": "Powell St. (Union Square)",
    "CIVC": "Civic Center / UN Plaza",
    "16TH": "16th St. Mission",
    "24TH": "24th St. Mission",
    "DALY": "Daly City",
    "SFIA": "San Francisco Int. Airport (SFO)",
    "MLBR": "Millbrae (Caltrain Aktarma)",
    "WOAK": "West Oakland (Transbay)",
    "12TH": "12th St. Oakland City Center",
    "19TH": "19th St. Oakland",
    "DBRK": "Downtown Berkeley",
    "OAKL": "Oakland Int. Airport (OAK)",
    "FRMT": "Fremont",
    "BERY": "Berryessa / North San José"
}

BART_LINES = {
    "🟡 Yellow Line (East Bay – Transbay – SFO)": ["19TH", "12TH", "WOAK", "EMBR", "MONT", "POWL", "CIVC", "16TH", "24TH", "DALY", "SFIA"],
    "🔴 Red Line (Berkeley – Transbay – Millbrae)": ["DBRK", "19TH", "12TH", "WOAK", "EMBR", "MONT", "POWL", "CIVC", "16TH", "24TH", "DALY", "MLBR"],
    "🔵 Blue Line (Transbay – Daly City Koridoru)": ["WOAK", "EMBR", "MONT", "POWL", "CIVC", "16TH", "24TH", "DALY"],
    "🟢 Green Line (South Bay / Berryessa – Daly City)": ["BERY", "FRMT", "WOAK", "EMBR", "MONT", "POWL", "CIVC", "16TH", "24TH", "DALY"],
    "🟠 Orange Line (South Bay / Berryessa – Berkeley)": ["BERY", "FRMT", "OAKL", "12TH", "19TH", "DBRK"]
}

SLOT_KEYS = ["00 - 10", "10 - 20", "20 - 30", "30 - 40", "40 - 50", "50 - 00"]
GUNLER_TR = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]

def predict_station_hourly(origin_code: str, hour: int, day_idx: int, is_weekend: int, month_val: int) -> int:
    is_commute_peak = 1 if ((is_weekend == 0) and ((7 <= hour <= 9) or (16 <= hour <= 19))) else 0
    hour_sin = np.sin(2 * np.pi * hour / 24.0)
    hour_cos = np.cos(2 * np.pi * hour / 24.0)
    is_transbay_core = 1 if origin_code in {'EMBR', 'MONT', 'POWL', 'CIVC', 'WOAK', '12TH', '19TH'} else 0
    is_airport = 1 if origin_code in {'SFIA', 'OAKL', 'MLBR'} else 0

    center = np.array(meta["center"], dtype=float)
    scale = np.array(meta["scale"], dtype=float)

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
    else:
        num_raw = np.array([
            hour, is_commute_peak, hour_sin, hour_cos, is_transbay_core, is_airport
        ], dtype=float)

    safe_scale = np.where(scale == 0, 1.0, scale)
    num_scaled = (num_raw - center) / safe_scale
    cat_ohe = np.array([1.0 if st_code == origin_code else 0.0 for st_code in meta["stations"]], dtype=float)
    X_input = np.concatenate([num_scaled, cat_ohe]).reshape(1, -1)

    return max(0, int(round(float(model.predict(X_input)[0]))))

st.title("🚆 BART Hat Bazlı 10-Dk Yolcu Yoğunluğu ve Vagon Optimizasyonu")
st.markdown("Yalnızca gerçek BART açık verileriyle (`date-hour-soo-dest-2025.csv`) eğitilmiş `XGBoost` modeli ve **hat koridoru darboğazına göre** vagon dizilimi karar destek sistemi.")

with st.sidebar:
    st.header("⚙️ Hat ve Sefer Parametreleri")
    selected_line = st.selectbox("BART Hattı Seçiniz", options=list(BART_LINES.keys()))
    line_stations = BART_LINES[selected_line]

    today = datetime.date.today()
    selected_date = st.date_input(
        "Gelecek Planlama Tarihi",
        value=today + datetime.timedelta(days=1),
        min_value=today
    )
    
    hour = st.slider("Saat (05:00 - 23:00)", min_value=5, max_value=23, value=8)
    slot_label = st.selectbox("10 Dakikalık Zaman Dilimi", options=SLOT_KEYS, index=1)
    trains_in_10min = st.selectbox("Bu 10 Dakikadaki Hat Sefer Sayısı", options=[1, 2, 3], index=0)

day_idx = selected_date.weekday()
day_name = GUNLER_TR[day_idx]
is_weekend = 1 if day_idx >= 5 else 0
month_val = selected_date.month

# Ham veri saatlik çözünürlükte olduğu için her 10 dk penceresi saatlik tahminin tam 1/6'sıdır (sıfır sentetik katsayı)
slot_weight = 1.0 / 6.0

station_rows = []
for st_code in line_stations:
    h_pred = predict_station_hourly(st_code, hour, day_idx, is_weekend, month_val)
    s_pred = int(round(h_pred * slot_weight))
    station_rows.append({
        "Kod": st_code,
        "İstasyon": STATION_NAMES.get(st_code, st_code),
        "Saatlik Talep": h_pred,
        "10 Dk Talep": s_pred
    })

line_df = pd.DataFrame(station_rows)
total_line_hourly = int(line_df["Saatlik Talep"].sum())
total_line_10min = int(line_df["10 Dk Talep"].sum())

bottleneck_row = line_df.loc[line_df["10 Dk Talep"].idxmax()]
bottleneck_10min_load = int(bottleneck_row["10 Dk Talep"])
passengers_per_train = bottleneck_10min_load / trains_in_10min

required_cars = int(np.ceil(passengers_per_train / 150.0))
recommended_consist = min(10, max(3, required_cars))
train_capacity = recommended_consist * 150
occupancy_rate = min(100.0, (passengers_per_train / train_capacity) * 100)

st.subheader(f"{selected_line} | {selected_date.strftime('%d.%m.%Y')} ({day_name}) | {hour:02d}:{slot_label.split(' - ')[0]} – {hour:02d}:{slot_label.split(' - ')[1]}")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Hat Toplam 10 Dk Talep", f"{total_line_10min:,} kişi", f"Saatlik Hat: {total_line_hourly:,}")
col2.metric("Darboğaz Tren Başına Yük", f"{int(passengers_per_train):,} kişi", f"Kritik İstasyon: {bottleneck_row['Kod']}")
col3.metric("Hat Vagon Dizilimi", f"{recommended_consist} Vagonlu Set")
col4.metric("Kritik İstasyon Doluluğu", f"%{occupancy_rate:.1f}")

st.divider()

left_col, right_col = st.columns([1.15, 1])

with left_col:
    st.markdown("### 🚦 Hat Operasyonel Karar Özeti")
    if occupancy_rate > 90:
        st.error(f"⚠️ **Hat Darboğazı / Kritik Yığılma:** `{bottleneck_row['İstasyon']}` istasyonunda yüksek talep bekleniyor. Hattaki tüm seferler **10 vagonlu maksimum dizi** ile işletilmelidir.")
    elif occupancy_rate > 65:
        st.warning(f"🟡 **Dengeli Hat İşletmesi:** `{recommended_consist} vagonlu` tren dizilimi `{bottleneck_row['Kod']}` darboğaz istasyonu dahil tüm hattı konfor sınırlarında karşılamaktadır.")
    else:
        st.success(f"🟢 **Hat Boyu Enerji Optimizasyonu:** `{hour:02d}:{slot_label}` penceresinde talep düşüktür; **{recommended_consist} vagonlu kısa dizi** ile cer enerjisi tasarrufu sağlanabilir.")
    
    st.markdown("#### 📍 Hat Üzerindeki İstasyonların Gerçek Model Tahminleri")
    st.dataframe(line_df, use_container_width=True, hide_index=True)

with right_col:
    st.markdown("### 🚉 İstasyon Bazlı Darboğaz Karşılaştırması")
    st.bar_chart(line_df.set_index("Kod")[["10 Dk Talep"]], color="#f59e0b")

    st.markdown(f"### 📈 {selected_line.split(' ')[1]} Hattı Günlük Saatlik Profil (05:00 – 23:00)")
    hourly_profile = []
    for h in range(5, 24):
        h_sum = sum(predict_station_hourly(st_c, h, day_idx, is_weekend, month_val) for st_c in line_stations)
        hourly_profile.append({"Saat": f"{h:02d}:00", "Hat Toplam Yolcu": h_sum})
    st.line_chart(pd.DataFrame(hourly_profile).set_index("Saat"), color="#38bdf8")

with st.expander("📂 Veri Seti ve Model Özeti (BTK 15 Aşama Standartları)"):
    eda = meta.get("eda_summary", {})
    metrics = meta.get("final_metrics", {})
    st.markdown(f"""
    - **Veri Kaynağı & Lisans:** Bay Area Rapid Transit (BART) Resmi Açık Veri Portalı (`date-hour-soo-dest-2025.csv`, Creative Commons Attribution **CC-BY**)[cite: 369].
    - **Veri Seti Kapsamı:** `{eda.get('date_min', '2025-01-01')}` – `{eda.get('date_max', '2025-12-31')}` arası toplam **{eda.get('total_rows', 'N/A'):,}** istasyon-saat gözlemi ve **{eda.get('num_stations', 50)}** istasyon.
    - **Hold-out Test Performansı:** `MAE = {metrics.get('Test_MAE', 'N/A')}`, `RMSE = {metrics.get('Test_RMSE', 'N/A')}`, `R² = {metrics.get('Test_R2', 'N/A')}`.
    - **Sıfır Sentetik Katsayı:** Uygulamada hiçbir yapay çarpan kullanılmamıştır; 10 dakikalık sefer penceresi saatlik gerçek model tahmininin tam **1/6'sı** olarak hesaplanır.
    """)
