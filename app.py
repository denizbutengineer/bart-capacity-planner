import os
import datetime
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import holidays
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="BART İleriye Dönük Vagon & Kapasite Planlayıcı",
    page_icon="🚆",
    layout="wide"
)

@st.cache_resource
def load_artifact():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "bart_final_artifact.joblib")
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

artifact = load_artifact()

st.title("🚆 BART İleriye Dönük Hat Kapasitesi ve Tren Seti (Consist) Planlayıcı")
st.markdown("""
Bu karar destek sistemi, **Bay Area Rapid Transit (BART)** hatları için seçilen **gelecek sefer tarih ve saatindeki** 
maksimum kritik kesit yolcu yükünü (`Max_Segment_Load`) öngörür ve planlanan sefer sıklığına göre en uygun **vagon dizisini (3–10 vagon)** belirler.
""")

if artifact is None:
    st.error("Model dosyası (`bart_final_artifact.joblib`) bulunamadı! Lütfen dosyanın `app.py` ile aynı dizinde olduğundan emin olun.")
    st.stop()

model = artifact["model"]
features = artifact["features"]
line_mapping = artifact["line_mapping"]

# Geriye dönük uyumluluk: Eski artifact yüklüyse Hist_Group_Mean lookup hazırla
hist_lookup = {}
global_mean = 489.27
if "Hist_Group_Mean" in features and "sample_data" in artifact:
    s_df = artifact["sample_data"].copy()
    s_df["Date"] = pd.to_datetime(s_df["Date"])
    s_df["Line_Code"] = s_df["Line"].map(line_mapping)
    s_df["Is_Weekend"] = (s_df["Date"].dt.dayofweek >= 5).astype(int)
    hist_lookup = (
        s_df.groupby(["Line_Code", "Direction", "Hour", "Is_Weekend"])["Max_Segment_Load"]
        .mean()
        .to_dict()
    )
    global_mean = float(s_df["Max_Segment_Load"].mean())

# Kenar Çubuğu — İleriye Dönük Sefer Planlama Girdileri
st.sidebar.header("🗓️ Gelecek Sefer Planlama Girdileri")

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

today = datetime.date.today()
selected_date = st.sidebar.date_input(
    "Planlanan Sefer Tarihi",
    value=today,
    min_value=today
)

selected_hour = st.sidebar.slider(
    "Planlanan Saat Dilimi (00:00 - 23:00)",
    min_value=0,
    max_value=23,
    value=8
)

trains_per_hour = st.sidebar.slider(
    "Planlanan Sefer Sıklığı (Tren / Saat)",
    min_value=2,
    max_value=6,
    value=4,
    help="Pik saatlerde saatte 4-5 tren, normal saatlerde 3 tren işletilir."
)

def build_feature_row(line_name, direction, dt_val, hour):
    dt = pd.to_datetime(dt_val)
    ca_hol = holidays.US(state="CA", years=[dt.year])
    is_hol = int(dt.date() in ca_hol)
    dow = dt.dayofweek
    is_wknd = int(dow >= 5)
    is_peak = int(((7 <= hour <= 9) or (16 <= hour <= 18)) and (is_wknd == 0) and (is_hol == 0))
    line_code = line_mapping[line_name]
    
    row = {
        "Line_Code": line_code,
        "Direction": direction,
        "Hour": hour,
        "Hour_Sin": np.sin(2 * np.pi * hour / 24.0),
        "Hour_Cos": np.cos(2 * np.pi * hour / 24.0),
        "DayOfWeek": dow,
        "Is_Weekend": is_wknd,
        "Month": dt.month,
        "DayOfMonth": dt.day,
        "WeekOfYear": int(dt.isocalendar().week),
        "DayOfYear": int(dt.dayofyear),
        "Is_Holiday": is_hol,
        "Is_Commute_Peak": is_peak,
        "Hist_Group_Mean": hist_lookup.get((line_code, direction, hour, is_wknd), global_mean)
    }
    return row, bool(is_hol), bool(is_peak)

def assign_consist(hourly_load, headway_freq):
    load_per_train = (hourly_load / headway_freq) * 1.10
    if hourly_load < 15:
        return load_per_train, "Sefer Yok / Bakım Aralığı", "Gece bakım penceresi (01:00 - 04:00)."
    elif load_per_train < 250:
        return load_per_train, "3 - 4 Vagon", "Düşük talep — Enerji ve bakım tasarrufu için kısa set."
    elif load_per_train < 550:
        return load_per_train, "5 - 6 Vagon", "Normal talep — Standart operasyonel dizi."
    elif load_per_train < 850:
        return load_per_train, "7 - 8 Vagon", "Yüksek talep — Yoğun koridor geçişi."
    else:
        return load_per_train, "9 - 10 Vagon (Maksimum)", "Pik kapasite — Tam uzunlukta tren seti gerekli."

# Seçili tarih ve saat için tahmin
row_dict, is_hol_flag, is_peak_flag = build_feature_row(selected_line, selected_dir, selected_date, selected_hour)
X_single = pd.DataFrame([row_dict])[features]
hourly_load = max(0.0, float(np.expm1(model.predict(X_single)[0])))
train_load, consist_rec, status_msg = assign_consist(hourly_load, trains_per_hour)

st.subheader(f"📊 {selected_date.strftime('%d.%m.%Y')} — Saat {selected_hour:02d}:00 Operasyonel Planı")

col1, col2, col3 = st.columns(3)
col1.metric("Öngörülen Saatlik Kesit Yükü", f"{hourly_load:,.0f} yolcu/saat")
col2.metric("Tren Başına Kritik Yük (+%10 Pay)", f"{train_load:,.0f} yolcu/tren")
col3.metric("Önerilen Tren Seti", consist_rec)

if is_hol_flag:
    st.info("📅 Seçilen gelecek tarih ABD/Kaliforniya resmi tatiline denk gelmektedir; tatil talep düşüşü modele yansıtılmıştır.")
elif is_peak_flag:
    st.warning("⚡ Hafta içi işe gidiş-geliş pik saati (Commute Peak).")

st.success(f"**Sevk ve Dizilim Kararı:** {status_msg}")

st.divider()

# İki Kolonlu Görselleştirme: Gün İçi 24 Saatlik Profil & 7 Günlük İleriye Dönük Projeksiyon
left_col, right_col = st.columns(2)

with left_col:
    st.subheader(f"📈 {selected_date.strftime('%d.%m.%Y')} — 24 Saatlik Yük Profili")
    rows_24h = [build_feature_row(selected_line, selected_dir, selected_date, h)[0] for h in range(24)]
    preds_24h = np.maximum(0.0, np.expm1(model.predict(pd.DataFrame(rows_24h)[features])))
    df_24h = pd.DataFrame({"Saat": range(24), "Öngörülen Kesit Yükü": preds_24h})
    
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    sns.lineplot(data=df_24h, x="Saat", y="Öngörülen Kesit Yükü", marker="o", ax=ax1, color="#005596")
    ax1.axvline(selected_hour, color="red", linestyle="--", label=f"Seçili Saat ({selected_hour:02d}:00)")
    ax1.set_xticks(range(0, 24, 2))
    ax1.set_ylabel("Yolcu / Saat")
    ax1.legend()
    st.pyplot(fig1)

with right_col:
    st.subheader(f"📅 Önümüzdeki 7 Gün (Saat {selected_hour:02d}:00 Karşılaştırması)")
    next_7_dates = [selected_date + datetime.timedelta(days=i) for i in range(7)]
    rows_7d = [build_feature_row(selected_line, selected_dir, d, selected_hour)[0] for d in next_7_dates]
    preds_7d = np.maximum(0.0, np.expm1(model.predict(pd.DataFrame(rows_7d)[features])))
    
    day_names = ["Pzt", "Sal", "Çar", "Per", "Cum", "Cmt", "Paz"]
    labels_7d = [f"{d.strftime('%d.%m')} ({day_names[d.weekday()]})" for d in next_7_dates]
    df_7d = pd.DataFrame({"Tarih": labels_7d, "Öngörülen Kesit Yükü": preds_7d})
    
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    sns.barplot(data=df_7d, x="Tarih", y="Öngörülen Kesit Yükü", ax=ax2, palette="Blues_d")
    ax2.set_ylabel("Yolcu / Saat")
    ax2.tick_params(axis="x", rotation=25)
    st.pyplot(fig2)

st.divider()

# 7 Günlük İleriye Dönük Vagon Atama Çizelgesi
st.subheader("📋 7 Günlük İleriye Dönük Vagon Atama Çizelgesi")
schedule_rows = []
for d, load_val in zip(next_7_dates, preds_7d):
    t_load, c_rec, _ = assign_consist(load_val, trains_per_hour)
    schedule_rows.append({
        "Tarih": d.strftime("%Y-%m-%d"),
        "Gün": day_names[d.weekday()],
        "Saat": f"{selected_hour:02d}:00",
        "Öngörülen Kesit Yükü": round(float(load_val), 1),
        "Tren Başına Yük": round(float(t_load), 1),
        "Önerilen Vagon Dizisi": c_rec
    })

st.dataframe(pd.DataFrame(schedule_rows), use_container_width=True)

# Alt Bölüm: Özellik Önem Düzeyi & Veri Seti Özeti (BTK Yönerge Bölüm 6.1 Uyumu)
col_exp1, col_exp2 = st.columns(2)

with col_exp1:
    with st.expander("🧠 Model Özellik Önem Düzeyi (Feature Importance)"):
        imp_df = pd.DataFrame({"Özellik": features, "Önem": model.feature_importances_}).sort_values("Önem", ascending=False)
        fig3, ax3 = plt.subplots(figsize=(7, 4))
        sns.barplot(data=imp_df, x="Önem", y="Özellik", ax=ax3, palette="viridis")
        st.pyplot(fig3)

with col_exp2:
    with st.expander("ℹ️ Veri Seti ve Eğitim Özeti (BTK Proje Kapsamı)"):
        st.markdown("""
        * **Veri Kaynağı:** BART 2025 Saatlik Origin-Destination (O-D) Açık Veri Seti (`9.143.257` ham yolculuk kaydı)
        * **İşlenmiş Gözlem Sayısı:** `87.600` hat-yön-saat kesit yükü kaydı (5 Ana Hat × 2 Yön × 365 Gün × 24 Saat)
        * **Hedef Değişken:** `Max_Segment_Load` (Ortalama: `489.3`, Maksimum: `4.739` yolcu/saat)
        * **Seçilen Model:** Tuned LightGBM (`log1p` hedef dönüşümü + kronolojik doğrulama)
        * **Operasyonel Kısıt:** BART yeni nesil Alstom D/E serisi filo sınırlarına uygun 3–10 vagon dizilimi
        """)

st.caption("⚠️ Model Sınırlılığı: Tahminler 2025 yılı tarihsel BART O-D akışlarına dayanmaktadır; planlı hat bakımları, hava koşulları veya özel etkinlikler anlık sapmalara yol açabilir.")
