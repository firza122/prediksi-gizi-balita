import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.express as plx
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load model dan scaler dari file
model_path = "knn_model_balita.sav"
scaler = MinMaxScaler(feature_range=(0, 1))  # MinMaxScaler manual untuk normalisasi

with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

# Judul aplikasi
st.title("Prediksi Status Gizi Balita")

# Penjelasan singkat
st.write("""
Aplikasi ini memprediksi status gizi balita berdasarkan input umur dan tinggi badan.
Masukkan nilai-nilai berikut untuk mendapatkan prediksi.
""")

# Input dari pengguna
umur = st.number_input("Masukkan umur balita (dalam bulan)", min_value=0, max_value=60, value=12, step=1)
tinggi_badan = st.number_input("Masukkan tinggi badan balita (dalam cm)", min_value=30.0, max_value=150.0, value=75.0, step=0.1)

# Tombol untuk prediksi
if st.button("Prediksi Status Gizi"):
    try:
        # Normalisasi input
        input_data = np.array([[umur, tinggi_badan]])
        scaler.fit([[0, 30], [60, 150]])  # Asumsi rentang data asli: umur (0-60 bulan) dan tinggi badan (30-150 cm)
        normalized_data = scaler.transform(input_data)
        
        # Prediksi menggunakan model
        prediction = model.predict([[umur,tinggi_badan]])
        
        # Menampilkan hasil prediksi
        status_gizi = prediction[0]
        print(status_gizi)
        if status_gizi == 1:
            st.write("Status gizi balita: Stunting")
        elif status_gizi == 0:
            st.write("Status gizi balita: Normal")
        elif status_gizi == 2:
            st.write("Status gizi balita: Tinggi")
        else:
            st.write("Status gizi balita: Severely Stunting")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")

# Informasi tambahan
st.write("""
### Catatan:
- Umur balita dalam rentang 0-60 bulan.
- Tinggi badan balita dalam rentang 30-150 cm.
Pastikan nilai input berada dalam rentang yang valid.
""")
