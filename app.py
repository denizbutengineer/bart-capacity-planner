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
    page_title="BART 10-Dk İleriye Dönük Vagon Planlayıcı",
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

st.title("🚆 BART 10 Dakikalık Kesit Tahmini ve Tren Seti (Consist) Planlayıcı")
st.markdown("""
Bu karar destek sistemi, **Bay Area Rapid Transit (BART)** hatları için seçilen **gelecek tarih ve 10 dakikalık zaman dilimindeki** 
kritik kesit yolcu yükünü öngörür ve planlanan sefer aralığına (headway) göre en uygun **vagon dizisini (3–10 vagon)** belirler.
""")

if artifact is None:
    st.error("Model dosyası (`bart_final_artifact.joblib`) bulunamadı! Lütfen dosyanın `app.py` ile aynı dizinde olduğundan emin olun.")
    st.stop()

model = artifact["model"]
features = artifact["features"]
line_mapping = artifact["line_mapping"]

# Kenar Çubuğu — 10 Dakikalık İleriye Dönük Sefer Planlama Girdileri
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

# 10 dakikalık zaman dilimi seçenekleri (00:00 - 23:50)
time_slots = [f"{h:02d}:{m:02d}" for h in range(24) for m in (0, 10, 20, 30, 40, 50)]
selected_time_str = st.sidebar.select_slider(
    "Planlanan 10 Dakikalık Zaman Dilimi",
    options=time_slots,
    value="08:10"
)
selected_hour, selected_minute = map(int, selected_time_str.split(":"))

headway_min = st.sidebar.selectbox(
    "Planlanan Sefer Aralığı (Headway)",
    options=[10, 15, 20, 30],
    index=1,
    format_func=lambda x: f"Her {x} dakikada 1 tren"
)

def build_feature_row(line_name, direction, dt_val, hour, minute):
    dt = pd.to_datetime(dt_val)
    ca_hol = holidays.US(state="CA", years=[dt.year])
    is_hol = int(dt.date() in ca_hol)
    dow = dt.dayofweek
    is_wknd = int(dow >= 5)
    slot_of_day = hour * 6 + (minute // 10)
    is_peak = int(((42 <= slot_of_day <= 57) or (96 <= slot_of_day <= 113)) and (is_wknd == 0) and (is_hol == 0))
    
    row = {
        "Line_Code": line_mapping[line_name],
        "Direction": direction,
        "Hour": hour,
        "Minute": minute,
        "Slot_Of_Day": slot_of_day,
        "Slot_Sin": np.sin(2 * np.pi * slot_of_day / 144.0),
        "Slot_Cos": np.cos(2 * np.pi * slot_of_day / 144.0),
        "DayOfWeek": dow,
        "Is_Weekend": is_wknd,
        "Month": dt.month,
        "DayOfMonth": dt.day,
        "WeekOfYear": int(dt.isocalendar().week),
        "DayOfYear": int(dt.dayofyear),
        "Is_Holiday": is_hol,
        "Is_Commute_Peak": is_peak
    }
    return row, bool(is_hol), bool(is_peak), slot_of_day

def assign_consist(load_10m, headway_minutes):
    # 10 dakikalık kesit yükünü sefer aralığıyla (headway / 10 dk) ölçekleyerek tek trene düşen yükü bul
    trains_factor = headway_minutes / 10.0
    load_per_train = (load_10m * trains_factor) * 1.10  # +%10 operasyonel emniyet payı
    
    if load_10m < 3:
        return load_per_train, "Sefer Yok / Bakım Aralığı", "Gece bakım penceresi (01:00 - 04:00)."
    elif load_per_train < 250:
        return load_per_train, "3 - 4 Vagon", "Düşük talep — Enerji tasarrufu için kısa set."
    elif load_per_train < 550:
        return load_per_train, "5 - 6 Vagon", "Normal talep — Standart operasyonel dizi."
    elif load_per_train < 850:
        return load_per_train, "7 - 8 Vagon", "Yüksek talep — Yoğun koridor geçişi."
    else:
        return load_per_train, "9 - 10 Vagon (Maksimum)", "Pik kapasite — Tam uzunlukta tren seti gerekli."

# Seçili 10 dakikalık dilim için tahmin
row_dict, is_hol_flag, is_peak_flag, selected_slot = build_feature_row(
    selected_line, selected_dir, selected_date, selected_hour, selected_minute
)
X_single = pd.DataFrame([row_dict])[features]
load_10m_pred = max(0.0, float(np.expm1(model.predict(X_single)[0])))
train_load, consist_rec, status_msg = assign_consist(load_10m_pred, headway_min)

st.subheader(f"📊 {selected_date.strftime('%d.%m.%Y')} — {selected_time_str} (10-Dk Pencere) Operasyonel Planı")

col1, col2, col3 = st.columns(3)
col1.metric("10 Dakikalık Kritik Kesit Yükü", f"{load_10m_pred:,.0f} yolcu / 10 dk")
col2.metric("Tren Başına Kritik Yük (+%10 Pay)", f"{train_load:,.0f} yolcu / tren")
col3.metric("Önerilen Tren Seti", consist_rec)

if is_hol_flag:
    st.info("📅 Seçilen tarih ABD/Kaliforniya resmi tatiline denk gelmektedir; tatil talep düşüşü modele yansıtılmıştır.")
elif is_peak_flag:
    st.warning("⚡ Hafta içi işe gidiş-geliş pik aralığı (Commute Peak).")

st.success(f"**Sevk ve Dizilim Kararı:** {status_msg}")

st.divider()

# İki Kolonlu Görselleştirme: 144 Noktalı Gün İçi Profil & 7 Günlük Aynı 10-Dk Dilim Projeksiyonu
left_col, right_col = st.columns(2)

with left_col:
    st.subheader(f"📈 {selected_date.strftime('%d.%m.%Y')} — 10 Dakikalık Gün İçi Eğri (144 Dilim)")
    rows_144 = [
        build_feature_row(selected_line, selected_dir, selected_date, h, m)[0]
        for h in range(24) for m in (0, 10, 20, 30, 40, 50)
    ]
    preds_144 = np.maximum(0.0, np.expm1(model.predict(pd.DataFrame(rows_144)[features])))
    df_144 = pd.DataFrame({"Slot": range(144), "10-Dk Kesit Yükü": preds_144})
    
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    sns.lineplot(data=df_144, x="Slot", y="10-Dk Kesit Yükü", ax=ax1, color="#005596", linewidth=2)
    ax1.axvline(selected_slot, color="red", linestyle="--", label=f"Seçili Dilim ({selected_time_str})")
    ax1.set_xticks(range(0, 144, 18))
    ax1.set_xticklabels([f"{s//6:02d}:00" for s in range(0, 144, 18)])
    ax1.set_xlabel("Saat")
    ax1.set_ylabel("Yolcu / 10 Dk")
    ax1.legend()
    st.pyplot(fig1)

with right_col:
    st.subheader(f"📅 Önümüzdeki 7 Gün ({selected_time_str} Dilimi Karşılaştırması)")
    next_7_dates = [selected_date + datetime.timedelta(days=i) for i in range(7)]
    rows_7d = [
        build_feature_row(selected_line, selected_dir, d, selected_hour, selected_minute)[0]
        for d in next_7_dates
    ]
    preds_7d = np.maximum(0.0, np.expm1(model.predict(pd.DataFrame(rows_7d)[features])))
    
    day_names = ["Pzt", "Sal", "Çar", "Per", "Cum", "Cmt", "Paz"]
    labels_7d = [f"{d.strftime('%d.%m')} ({day_names[d.weekday()]})" for d in next_7_dates]
    df_7d = pd.DataFrame({"Tarih": labels_7d, "10-Dk Kesit Yükü": preds_7d})
    
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    sns.barplot(data=df_7d, x="Tarih", y="10-Dk Kesit Yükü", ax=ax2, palette="Blues_d")
    ax2.set_ylabel("Yolcu / 10 Dk")
    ax2.tick_params(axis="x", rotation=25)
    st.pyplot(fig2)

st.divider()

# 7 Günlük İleriye Dönük Vagon Atama Çizelgesi
st.subheader(f"📋 7 Günlük İleriye Dönük Vagon Atama Çizelgesi ({selected_time_str})")
schedule_rows = []
for d, load_val in zip(next_7_dates, preds_7d):
    t_load, c_rec, _ = assign_consist(load_val, headway_min)
    schedule_rows.append({
        "Tarih": d.strftime("%Y-%m-%d"),
        "Gün": day_names[d.weekday()],
        "Zaman Dilimi": selected_time_str,
        "10-Dk Kesit Yükü": round(float(load_val), 1),
        "Tren Başına Yük": round(float(t_load), 1),
        "Önerilen Vagon Dizisi": c_rec
    })

st.dataframe(pd.DataFrame(schedule_rows), use_container_width=True)

# Alt Bölüm: Özellik Önem Düzeyi & Veri Seti Özeti
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
        * **Veri Kaynağı:** BART 2025 Saatlik O-D Açık Veri Seti (`9.143.257` ham yolculuk kaydı)
        * **10-Dk Çözünürlük Veri Seti:** `525.600` hat-yön-10dk gözlemi (5 Ana Hat × 2 Yön × 365 Gün × 144 Slot)
        * **Hedef Değişken:** 10 Dakikalık Kritik Kesit Yükü (`Max_Segment_Load_10m`)
        * **Seçilen Model:** Tuned LightGBM (`log1p` hedef dönüşümü + kronolojik doğrulama)
        * **Operasyonel Kısıt:** BART yeni nesil Alstom D/E serisi filo sınırlarına uygun 3–10 vagon dizilimi
        """)

st.caption("⚠️ Model Sınırlılığı: Tahminler 2025 yılı tarihsel BART O-D akışlarının 10 dakikalık alt-saatlik ayrıştırmasına dayanmaktadır; planlı hat bakımları veya özel etkinlikler anlık sapmalara yol açabilir.")
