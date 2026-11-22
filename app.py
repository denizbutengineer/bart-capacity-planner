import streamlit as st
import pandas as pd
import numpy as np
import joblib
import holidays
import matplotlib.pyplot as plt
import seaborn as sns

# Sayfa Yapılandırması
st.set_page_config(
    page_title="BART Hat Kapasite ve Vagon Planlayıcı",
    page_icon="🚆",
    layout="wide"
)

# 1. Model ve Artefakt Yükleme (Hata Kontrollü)
@st.cache_resource
def load_artifact():
    try:
        return joblib.load("bart_final_artifact.joblib")
    except FileNotFoundError:
        return None

artifact = load_artifact()

# Üst Başlık ve Proje Açıklaması (Yönerge 6.1)
st.title("🚆 BART Hat Bazlı Kapasite Tahmini ve Tren Seti (Consist) Planlayıcı")
st.markdown("""
Bu uygulama, **Bay Area Rapid Transit (BART)** 2025 saatlik Başlangıç-Varış (Origin-Destination) verileri üzerinden 
ağ yönlendirme (Network Routing) yöntemiyle hesaplanan **Hat Maksimum Kesit Yükünü (`Max_Segment_Load`)** tahmin eder 
ve sefer sıklığına göre en uygun **vagon dizisi (3–10 vagon)** önerisini sunar.
""")

if artifact is None:
    st.error("Model dosyası (`bart_final_artifact.joblib`) bulunamadı! Lütfen dosyanın `app.py` ile aynı dizinde olduğundan emin olun.")
    st.stop()

model = artifact["model"]
features = artifact["features"]
line_mapping = artifact["line_mapping"]
sample_df = artifact["sample_data"].copy()

# Tarihsel grup ortalaması (Hist_Group_Mean) lookup tablosu hazırlığı
sample_df["Date"] = pd.to_datetime(sample_df["Date"])
sample_df["Line_Code"] = sample_df["Line"].map(line_mapping)
sample_df["Is_Weekend"] = (sample_df["Date"].dt.dayofweek >= 5).astype(int)
hist_lookup = (
    sample_df.groupby(["Line_Code", "Direction", "Hour", "Is_Weekend"])["Max_Segment_Load"]
    .mean()
    .to_dict()
)
global_mean = float(sample_df["Max_Segment_Load"].mean())

# Kenar Çubuğu (Sidebar) — Kullanıcı Girdileri (Yönerge 6.1)
st.sidebar.header("🛠️ Sefer Parametreleri")

selected_line = st.sidebar.selectbox(
    "Hat Seçimi (BART Line)",
    options=list(line_mapping.keys()),
    index=0
)

direction_label = st.sidebar.radio(
    "Sefer Yönü",
    options=["San Francisco / Güney Yönü (Dir 0)", "Doğu Körfez / Kuzey Yönü (Dir 1)"]
)
selected_dir = 0 if "Dir 0" in direction_label else 1

selected_date = st.sidebar.date_input(
    "Tarih",
    value=pd.to_datetime("2025-11-20")
)

selected_hour = st.sidebar.slider(
    "Saat Dilimi (00:00 - 23:00)",
    min_value=0,
    max_value=23,
    value=8
)

trains_per_hour = st.sidebar.slider(
    "Saatlik Sefer Sıklığı (Tren / Saat)",
    min_value=2,
    max_value=6,
    value=4,
    help="BART pik saatlerde saatte 4-5 tren, normal saatlerde 3 tren işletir."
)

# Tahmin Hesaplama Fonksiyonu
def predict_consist(line_name, direction, dt_val, hour, headway_freq):
    dt = pd.to_datetime(dt_val)
    ca_hol = holidays.US(state="CA", years=[dt.year])
    is_hol = int(dt.date() in ca_hol)
    dow = dt.dayofweek
    is_wknd = int(dow >= 5)
    is_peak = int(((7 <= hour <= 9) or (16 <= hour <= 18)) and (is_wknd == 0) and (is_hol == 0))
    line_code = line_mapping[line_name]
    
    hist_val = hist_lookup.get((line_code, direction, hour, is_wknd), global_mean)
    
    input_dict = {
        "Line_Code": line_code,
        "Direction": direction,
        "Hour": hour,
        "Hour_Sin": np.sin(2 * np.pi * hour / 24.0),
        "Hour_Cos": np.cos(2 * np.pi * hour / 24.0),
        "DayOfWeek": dow,
        "Is_Weekend": is_wknd,
        "Month": dt.month,
        "DayOfMonth": dt.day,
        "Is_Holiday": is_hol,
        "Is_Commute_Peak": is_peak,
        "Hist_Group_Mean": hist_val
    }
    
    X_input = pd.DataFrame([input_dict])[features]
    pred_hourly_load = max(0.0, float(np.expm1(model.predict(X_input)[0])))
    
    # Tek trene düşen maksimum yük (%10 operasyonel emniyet payı dahil)
    load_per_train = (pred_hourly_load / headway_freq) * 1.10
    
    if pred_hourly_load < 15:
        consist = "Sefer Yok / Bakım Aralığı"
        status = "Geceleri (01:00 - 04:00) BART ağır raylı seferleri durdurulur."
    elif load_per_train < 250:
        consist = "3 - 4 Vagon"
        status = "Düşük Talep — Enerji tasarrufu için kısa dizi önerilir."
    elif load_per_train < 550:
        consist = "5 - 6 Vagon"
        status = "Normal Talep — Standart gün ortası / hafta sonu dizisi."
    elif load_per_train < 850:
        consist = "7 - 8 Vagon"
        status = "Yüksek Talep — Yoğun koridor geçişi."
    else:
        consist = "9 - 10 Vagon (Maksimum)"
        status = "Pik Kapasite — Transbay darboğazı için tam uzunlukta set."
        
    return pred_hourly_load, load_per_train, consist, status, bool(is_hol), bool(is_peak)

hourly_load, train_load, consist_rec, status_msg, is_hol_flag, is_peak_flag = predict_consist(
    selected_line, selected_dir, selected_date, selected_hour, trains_per_hour
)

# Ana Panel — Tahmin Sonuçları ve Metrikler
st.subheader("📊 Operasyonel Tahmin ve Vagon Önerisi")

col1, col2, col3 = st.columns(3)
col1.metric("Saatlik Max Kesit Yükü", f"{hourly_load:,.0f} yolcu/saat")
col2.metric("Tren Başına Kritik Yük (+%10 Pay)", f"{train_load:,.0f} yolcu/tren")
col3.metric("Önerilen Tren Seti", consist_rec)

if is_hol_flag:
    st.info("📅 Seçilen tarih Kaliforniya resmi tatil gününe denk gelmektedir; model tatil düşüşünü hesaba katmıştır.")
elif is_peak_flag:
    st.warning("⚡ Seçilen zaman dilimi hafta içi işe gidiş-geliş pik saatidir (Commute Peak).")

st.success(f"**Operasyonel Karar Notu:** {status_msg}")

st.divider()

# İki Kolonlu Alt Bölüm: 24 Saatlik Gün İçi Profil & Feature Importance (Yönerge 6.1 & 6.2)
left_col, right_col = st.columns(2)

with left_col:
    st.subheader(f"📈 {selected_line} Hattı — 24 Saatlik Talep Eğrisi")
    hours_range = list(range(24))
    hourly_curve = [
        predict_consist(selected_line, selected_dir, selected_date, h, trains_per_hour)[0]
        for h in hours_range
    ]
    curve_df = pd.DataFrame({"Saat": hours_range, "Tahmini Kesit Yükü": hourly_curve})
    
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    sns.lineplot(data=curve_df, x="Saat", y="Tahmini Kesit Yükü", marker="o", ax=ax1, color="#005596")
    ax1.axvline(selected_hour, color="red", linestyle="--", label=f"Seçili Saat ({selected_hour}:00)")
    ax1.set_xticks(range(0, 24, 2))
    ax1.set_ylabel("Yolcu / Saat")
    ax1.legend()
    st.pyplot(fig1)

with right_col:
    st.subheader("🧠 Model Özellik Önem Düzeyi (Feature Importance)")
    imp_df = pd.DataFrame({
        "Özellik": features,
        "Önem": model.feature_importances_
    }).sort_values("Önem", ascending=False)
    
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    sns.barplot(data=imp_df, x="Önem", y="Özellik", ax=ax2, palette="viridis")
    ax2.set_xlabel("Bölünme Kazancı (Split Importance)")
    st.pyplot(fig2)

st.divider()

# Örnek Veri Seti İnceleme Bölümü (Yönerge 6.1 Zorunlu Madde)
with st.expander("📂 Örnek Test Veri Seti ve İstatistiksel Özet (BART 2025)"):
    st.write("Modelin değerlendirildiği Hold-out test kesitinden örnek kayıtlar:")
    st.dataframe(sample_df[["Date", "Hour", "Line", "Direction", "Max_Segment_Load"]].head(50), use_container_width=True)
    st.write("**Özet İstatistikler:**")
    st.dataframe(sample_df[["Max_Segment_Load"]].describe().T, use_container_width=True)

# Sınırlılıklar ve Etik Bilgilendirme (Yönerge Bölüm 11)
st.caption("⚠️ Model Sınırlılığı: Tahminler 2025 yılı tarihsel BART O-D akışlarına dayanmaktadır; planlı hat bakımları, hava koşulları veya özel stadyum/konser etkinlikleri anlık sapmalara yol açabilir.")
