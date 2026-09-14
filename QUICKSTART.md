# Quick Start

## Menjalankan dashboard

Gunakan environment Python yang memiliki dependensi dari `requirements.txt`.

```powershell
pip install -r requirements.txt
streamlit run app.py
```

Streamlit akan menampilkan alamat lokal, biasanya `http://localhost:8501`.

Jika perintah `streamlit` tidak dikenali pada environment Anaconda:

```powershell
C:\Users\Yudha\anaconda3\python.exe -m streamlit run app.py
```

## Menjalankan notebook

```powershell
jupyter notebook HotelBookingDemand_Capstone_Yonkie.ipynb
```

Jalankan seluruh sel secara berurutan. Notebook membersihkan data, melakukan EDA, membandingkan tujuh algoritma, memilih Logistic Regression, mengevaluasi threshold, dan menyimpan pipeline sebagai `model_hotel_cancellation.pkl`.

## File yang diperlukan untuk dashboard

- `app.py`
- `data_hotel_booking_demand.csv`
- `requirements.txt`
- `.streamlit/config.toml`

Dashboard tidak menjalankan notebook. Aplikasi membaca CSV secara langsung dan melatih model sekali melalui cache Streamlit.

## Deploy ke Streamlit Community Cloud

1. Push project ke GitHub.
2. Buka Streamlit Community Cloud dan pilih **Create app**.
3. Pilih repository dan branch yang sesuai.
4. Isi main file path dengan `app.py`.
5. Klik **Deploy**.

Pastikan `data_hotel_booking_demand.csv` ikut berada di repository karena dashboard membutuhkannya saat startup.
