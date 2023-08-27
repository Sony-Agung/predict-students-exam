import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


# Load model terbaik dari file
model_filename = 'model/best_logistic_regression_model.pkl'
with open(model_filename, 'rb') as model_file:
    best_logreg_model = pickle.load(model_file)




# Fungsi untuk melakukan prediksi
def predict_status(data):
    # Mapping kategori fitur
    category_mappings = {
        "gender": {"male": 0, "female": 1},
        "lunch": {"standard": 0, "free/reduced": 1},
        "test_prep": {"none": 0, "completed": 1}
    }

    data.replace(category_mappings, inplace=True)

    # Daftar urutan pendidikan
    education_order2 = [[
    "some high school",
    "High School",
    "Some High School",  # Tambahkan kategori ini
    "Some College",
    "Associate's Degree",
    "Bachelor's Degree",
    "Master's Degree",
    ]]

    ordinal_encoder = OrdinalEncoder(categories=education_order2)

    data["parent_education"] = ordinal_encoder.fit_transform(data[["parent_education"]])

    # Gunakan model untuk membuat prediksi pada data baru yang sudah diencode
    predicted_status = best_logreg_model.predict(data)

    return predicted_status[0]


# Tampilan halaman utama aplikasi
def main():
    
    st.title("Aplikasi Prediksi Kelulusan")

    st.write("Isi formulir di bawah ini untuk melakukan prediksi kelulusan:")

    # Formulir input
    gender = st.radio("Jenis Kelamin", ("Male", "Female"))
    parent_education = st.selectbox("Pendidikan Orang Tua", ["Some High School", "High School", "Some College", "Associate's Degree", "Bachelor's Degree", "Master's Degree"])
    lunch = st.radio("Jenis Makan Siang", ("Standard", "Free/Reduced"))
    test_prep = st.radio("Persiapan Ujian", ("None", "Completed"))

    # Konversi input menjadi DataFrame
    input_data = pd.DataFrame({
        'gender': [gender.lower()],
        'parent_education': [parent_education],
        'lunch': [lunch.lower()],
        'test_prep': [test_prep.lower()],
        'race/ethnicity_group A': [1],
        'race/ethnicity_group B': [0],
        'race/ethnicity_group C': [0],
        'race/ethnicity_group D': [0],
        'race/ethnicity_group E': [0]
    })

    # Prediksi status
    if st.button("Prediksi"):
        predicted_status = predict_status(input_data)

        if predicted_status == 0:
            st.success("Hasil prediksi: Passed")
        else:
            st.error("Hasil prediksi: Failed")

# Tampilan halaman "About"
# Tampilan halaman "About" dengan HTML kustom
def about():
    
    st.title("Tentang Aplikasi / About the App")
    

    # Menggunakan HTML kustom
    about_html = """
    <div>
        <h2>Ini adalah aplikasi sederhana untuk melakukan prediksi kelulusan berdasarkan beberapa faktor:</h2>
        <ul>
            <li>Jenis kelamin</li>
            <li>Pendidikan orang tua</li>
            <li>Jenis makan siang</li>
            <li>Persiapan ujian</li>
        </ul>
        <p>This is a simple app to predict graduation based on several factors, such as:</p>
        <ul>
            <li>Gender</li>
            <li>Parent's education</li>
            <li>Lunch type</li>
            <li>Test preparation</li>
        </ul>
    </div>
    """
    st.markdown('<link rel="stylesheet" type="style/css" href="static/about.css">', unsafe_allow_html=True)
    st.markdown(about_html, unsafe_allow_html=True)
# Menampilkan pilihan halaman di sidebar
menu = st.sidebar.selectbox("Menu", ["About", "Prediction"])
if menu == "About":
    about()
else:
    main()



