# 🚆 BART Hat Bazlı 10 Dakikalık Yolcu Yoğunluğu ve Vagon Optimizasyonu

Bay Area Rapid Transit (BART) açık ulaşım verileri (`date-hour-soo-dest-2025.csv`) kullanılarak geliştirilmiş, makine öğrenmesi tabanlı talep tahmini ve **hat koridoru darboğazına göre dinamik vagon dizilimi (3–10 vagon)** karar destek sistemi.

🔗 **Canlı Uygulama (Streamlit Cloud):** [https://denizbutengineer-bart-capacity-planner-app-tjy8q7.streamlit.app](https://denizbutengineer-bart-capacity-planner-app-tjy8q7.streamlit.app)

## 📊 Veri Seti ve Lisans
- **Kaynak:** [BART Official Ridership Reports](https://www.bart.gov/about/reports/ridership) (`date-hour-soo-dest-2025.csv`)
- **Lisans:** Creative Commons Attribution License (**CC-BY**)
- **Özellik:** Tamamen gerçek gözlem verileri kullanılmış olup eğitim ve çıkarım aşamalarında hiçbir sentetik veri veya yapay katsayı eklenmemiştir.

## 🛠️ Metodoloji (15 Aşamalı Veri Bilimi Pipeline'ı)
1. **Veri Kalitesi & Dönüşüm:** Gece işletme dışı saatler (`01:00–04:00`) filtrelenmiş, başlangıç-varış kayıtları istasyon ve saat bazında (`Total_Riders`) toplulaştırılmıştır.
2. **Öznitelik Mühendisliği:** Döngüsel saat temsilleri (`Hour_Sin`, `Hour_Cos`), hafta içi pik saat göstergesi (`Is_Commute_Peak`), hafta sonu (`Is_Weekend`), ay (`Month`) ve koridor göstergeleri (`Is_Transbay_Core`, `Is_Airport`) türetilmiştir.
3. **Doğrulama Protokolü:** Zaman serisi sızıntısını önlemek için `shuffle=False` ile kronolojik %80/%20 ayrım ve 3 katlı `TimeSeriesSplit` kullanılmıştır.
4. **Model Karşılaştırma & Optimizasyon:** `DummyRegressor` baseline dahil 8 farklı regresyon modeli kıyaslanmış; `RandomizedSearchCV` ile optimize edilen **XGBoost Regressor** canlı sisteme entegre edilmiştir.

## 🚀 Yerel Kurulum ve Çalıştırma
```bash
git clone [https://github.com/denizbutengineer/bart-capacity-planner.git](https://github.com/denizbutengineer/bart-capacity-planner.git)
cd bart-capacity-planner
pip install -r requirements.txt
streamlit run app.py
