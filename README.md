# Demiryolu Kapasite Planlayıcısı

10 dakikalık zaman dilimlerinde hat kesit yükünü (segment load) tahmin eden ve öngörülen yolcu talebine göre sefer başına **vagon dizilimi (consist)** önerisi üreten makine öğrenmesi ve raylı sistem planlama uygulaması. Çalışmada örnek raylı ağ veri seti olarak 2025 yılı saatlik Başlangıç-Varış (O-D) yolcu hareketliliği (`9.143.257` satır) kullanılmıştır.

## Öne Çıkan Özellikler

- **Graf Tabanlı Kesit Yükü Hesabı:** Hatlar yönlü graf (`NetworkX DiGraph`) olarak modellenir; ortak koridorlardaki yolcu talebi koridoru paylaşan hat sayısına orantılı dağıtılır ve hat/yön bazında maksimum kesit yükü (`Max_Segment_Load`) çıkarılır.
- **Bellek Dostu İşleme Hattı:** Yüksek hacimli O-D verisi aylık dilimler ve tamsayı `Edge_ID` eşlemeleriyle işlenerek bellek kullanımı sınırlandırılır.
- **Sızıntısız 10 Dakikalık Zaman Dilimleri:** Saatlik yükler 6 adet 10 dakikalık slota ayrılır ve yalnızca geçmişe bakan (`center=False`) hareketli ortalama ile pürüzsüzleştirilir.
- **Zengin Özellik Mühendisliği:** Hat/yön kodları, takvim ve resmi tatil göstergeleri, sabah/akşam pik saat bayrakları, `sin`/`cos` döngüsel zaman özellikleri ve geçmiş gün/hafta gecikme (`Lag_1d`, `Lag_7d`, `Rolling_Mean_7d`) değişkenleri üretilir.
- **Vagon Dizilimi Optimizasyonu:** Tahmin edilen 10 dakikalık kesit yükü ve sefer sıklığı (`headway_minutes`) kullanılarak vagon başına hedef kapasiteye göre `3–10` vagon aralığında dizilim önerisi hesaplanır.

## Veri ve Kronolojik Bölünme

- **Ham Veri:** `date-hour-soo-dest-2025.csv.zip` (`9.143.257` satır)
- **Eğitim (Train):** Ocak – Eylül (`383.040` satır)
- **Doğrulama (Validation):** Ekim – Kasım (`87.840` satır)
- **Test:** Aralık (`44.640` satır)

## Model Karşılaştırması (Validation Seti)

| Model | RMSE | MAE | R² |
| :--- | :---: | :---: | :---: |
| **CatBoost** | **21.34** | **9.83** | **0.9504** |
| **XGBoost** | 21.66 | 10.44 | 0.9489 |
| **LightGBM** | 21.99 | 10.31 | 0.9473 |
| **Baseline (Group Mean)** | 34.94 | 16.80 | 0.8671 |
| **Ridge Regression** | 160.69 | 40.10 | -1.8105 |

Final model (Aralık test seti): **RMSE: 54.01**, **MAE: 23.57**, **R²: 0.57**.

## Proje Yapısı

- `btk_final_proje.ipynb`: Veri yükleme, ağ topolojisi, özellik mühendisliği, model eğitimi ve dışa aktarma iş akışı.
- `bart_model.json` ve `model_metadata.json`: Eğitilmiş final model ve meta veriler.
- `app.py`: Streamlit tabanlı kapasite ve vagon dizilimi planlayıcısı arayüzü.
- `requirements.txt`: Uygulama bağımlılıkları.
