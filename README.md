# Hotel Booking Cancellation Prediction

Model klasifikasi biner untuk mendeteksi pemesanan hotel yang berisiko dibatalkan agar tim Revenue Management dan Customer Service dapat memprioritaskan intervensi.

Capstone Project Module 3, Purwadhika Digital Technology School  
Author: Yonkie Yudha Ardika

## Ringkasan hasil

- Data awal: 83.573 pemesanan dan 11 kolom.
- Setelah imputasi 351 nilai `country` dan penghapusan 73.476 baris identik: 10.097 pemesanan.
- Target: `is_canceled`, dengan 2.405 pembatalan (23,8%) dan 7.692 pemesanan tidak batal (76,2%).
- Split: 8.077 data latih dan 2.020 data uji, memakai `stratify=y` dan `random_state=42`.
- Model akhir: Logistic Regression dengan `class_weight='balanced'`.
- Pemilihan dari tujuh algoritma memakai Stratified 5-Fold Cross-Validation: recall tertinggi di antara model dengan precision minimal 0,40.

## Performa model

### Rata-rata cross-validation

| Model | Recall | Precision | F1 | AUC-ROC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0,822 | 0,453 | 0,585 | 0,826 |
| XGBoost | 0,803 | 0,440 | 0,568 | 0,817 |
| Decision Tree | 0,726 | 0,411 | 0,525 | 0,733 |
| Random Forest | 0,697 | 0,428 | 0,530 | 0,785 |
| Gaussian Naive Bayes | 0,581 | 0,479 | 0,471 | 0,790 |
| K-Nearest Neighbors | 0,475 | 0,500 | 0,487 | 0,780 |
| Gradient Boosting | 0,308 | 0,639 | 0,415 | 0,839 |

Gradient Boosting memiliki AUC tertinggi, tetapi recall-nya paling rendah. Karena tujuan bisnis memprioritaskan deteksi pembatalan, Logistic Regression menjadi model akhir.

### Test set pada threshold 0,5

| Metric | Nilai |
|---|---:|
| Recall kelas pembatalan | 0,830 |
| Precision kelas pembatalan | 0,459 |
| F1 kelas pembatalan | 0,591 |
| Accuracy | 0,726 |

Confusion matrix:

| | Prediksi tidak batal | Prediksi batal |
|---|---:|---:|
| Aktual tidak batal | 1.068 | 471 |
| Aktual batal | 82 | 399 |

Model menangkap 399 dari 481 pembatalan aktual. Sebanyak 870 booking ditandai berisiko, dengan 471 false positive dan 82 false negative.

## Threshold tuning

| Threshold | Recall | Precision | Booking ditandai |
|---:|---:|---:|---:|
| 0,3 | 0,967 | 0,384 | 1.210 |
| 0,4 | 0,913 | 0,419 | 1.048 |
| 0,5 | 0,830 | 0,459 | 870 |
| 0,6 | 0,678 | 0,492 | 662 |
| 0,7 | 0,501 | 0,554 | 435 |

Threshold operasional perlu disesuaikan dengan kapasitas tim. Threshold lebih rendah menangkap lebih banyak pembatalan, tetapi meningkatkan jumlah tamu yang harus dihubungi.

## Preprocessing dan feature engineering

Fitur turunan: `has_previous_cancellations`, `is_domestic` (`country == 'PRT'`), dan `has_high_booking_changes` (`booking_changes > 2`).

Kolom asli `previous_cancellations`, `country`, `days_in_waiting_list`, dan `reserved_room_type` tidak masuk langsung ke model. Pipeline memproses enam fitur numerik dengan median imputation dan StandardScaler, serta fitur kategorikal dengan most-frequent imputation dan OneHotEncoder. Output preprocessing pada notebook berjumlah 22 kolom.

## Interpretasi fitur

Karena model akhir adalah Logistic Regression, interpretasi memakai koefisien, bukan gain-based feature importance. Koefisien dengan magnitudo absolut terbesar pada output notebook:

- `required_car_parking_spaces`: -2,848
- `market_segment_Online TA`: +1,568
- `deposit_type_No Deposit`: -1,471
- `market_segment_Complementary`: -1,196
- `deposit_type_Non Refund`: +1,120

Koefisien menunjukkan asosiasi setelah mengontrol fitur lain, bukan hubungan sebab-akibat. Analisis bivariat juga menemukan cancellation rate `Non Refund` sebesar 89,2%, sehingga anomali ini perlu dikonfirmasi kepada tim operasional sebelum menjadi dasar kebijakan.

## Menjalankan notebook

```bash
pip install -r requirements.txt
jupyter notebook HotelBookingDemand_Capstone_Yonkie.ipynb
```

Notebook menyimpan pipeline lengkap sebagai `model_hotel_cancellation.pkl` setelah seluruh sel dijalankan.

## Menjalankan dashboard Streamlit

```bash
pip install -r requirements.txt
streamlit run app.py
```

Dashboard membaca `data_hotel_booking_demand.csv`, menerapkan proses cleaning dan feature engineering yang sama dengan notebook, lalu melatih Logistic Regression sekali dan menyimpannya di cache Streamlit. Fitur dashboard:

- filter market segment, deposit type, customer type, dan negara;
- KPI serta visualisasi cancellation rate;
- evaluasi model dan simulasi probability threshold;
- form prediksi risiko satu booking;
- tabel serta download data yang sudah difilter.

## Isi repository

- `HotelBookingDemand_Capstone_Yonkie.ipynb`: analisis, preprocessing, benchmarking, evaluasi, dan rekomendasi.
- `app.py`: dashboard Streamlit untuk EDA, evaluasi, dan prediksi booking.
- `.streamlit/config.toml`: tema dan konfigurasi dashboard.
- `data_hotel_booking_demand.csv`: dataset sumber.
- `Prediksi_Pembatalan_Hotel_Capstone3_Yonkie.pptx`: presentasi sumber.
- `Prediksi_Pembatalan_Hotel_Capstone3_FINAL_ipynb_aligned.pptx`: presentasi yang diselaraskan dengan output notebook.
- `Hotel Booking Demand.docx`: dokumen pendukung.
- `QUICKSTART.md`: panduan singkat.

## Limitasi

- Penghapusan 73.476 baris identik perlu diuji ulang karena dataset tidak memiliki ID booking atau tanggal untuk membedakan duplikat teknis dari booking berbeda yang kebetulan sama.
- Dataset tidak memuat `lead_time`, tanggal menginap, ADR, atau identitas tamu.
- Precision 0,459 berarti model cocok sebagai alat penyaring dan prioritisasi, bukan pengambil keputusan otomatis.
- Performa perlu dipantau pada data baru untuk mendeteksi perubahan pola pelanggan.
