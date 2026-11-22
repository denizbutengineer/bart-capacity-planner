# 🚆 BART Hat Bazlı 10 Dakikalık Yolcu Yoğunluğu ve Vagon Optimizasyonu

Bay Area Rapid Transit (BART) açık ulaşım verileri (`date-hour-soo-dest-2025.csv`) kullanılarak geliştirilmiş, makine öğrenmesi tabanlı talep tahmini ve **hat koridoru darboğazına göre dinamik vagon dizilimi (3–10 vagon)** karar destek sistemi.

🔗 **Canlı Uygulama (Streamlit Cloud):** [https://denizbutengineer-bart-capacity-planner-app-tjy8q7.streamlit.app](https://denizbutengineer-bart-capacity-planner-app-tjy8q7.streamlit.app)

## 📊 Veri Seti ve Lisans
- **Kaynak:** [BART Official Ridership Reports](https://www.bart.gov/about/reports/ridership) (`date-hour-soo-dest-2025.csv`)
- **Lisans:** Creative Commons Attribution License (**CC-BY**)
- **Özellik:** Tamamen gerçek gözlem verileri kullanılmış olup eğitim ve çıkarım aşamalarında hiçbir sentetik veri veya yapay katsayı eklenmemiştir.

## 🛠️ Metodoloji (15 Aşamalı Veri Bilimi Pipeline'ı)
1. **Veri Kalitesi & Dönüşüm:** Gece işletme dışı saatler (`01:00–04:00`) ve negatif değerler temizlenmiş, başlangıç-varış kayıtları istasyon ve saat bazında (`Total_Riders`) toplulaştırılmıştır.
2. **EDA & Özellik Seçimi:** Sağa çarpık (Right-Skewed) yolcu dağılımı ve çift tepeli (Bimodal) pik saat örüntüleri analiz edilmiş; Filter (`VarianceThreshold`, `SelectKBest`, `Mutual Info`), Wrapper (`RFE`) ve Embedded (`Lasso L1`, `Random Forest Importance`) yöntemleriyle öznitelikler doğrulanmıştır.
3. **Doğrulama Protokolü:** Veri sızıntısını (Data Leakage) önlemek için `shuffle=False` ile kronolojik ayrım (%70 Train / %15 Validation / %15 Test) ve 3 katlı `TimeSeriesSplit` kullanılmıştır.
4. **Model Karşılaştırma & Optimizasyon:** `OLS Linear Regression` baseline dahil 9 regresyon modeli kıyaslanmış; `RandomizedSearchCV` ile optimize edilen **XGBoost Regressor** canlı sisteme entegre edilmiştir.

## 🚀 Yerel Kurulum ve Çalıştırma
```bash
git clone [https://github.com/denizbutengineer/bart-capacity-planner.git](https://github.com/denizbutengineer/bart-capacity-planner.git)
cd bart-capacity-planner
pip install -r requirements.txt
streamlit run app.py
