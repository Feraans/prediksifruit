import streamlit as st
import pickle
import numpy as np

# Fungsi untuk memuat model, encoder, dan scaler
def load_model_and_preprocessors(model_name):
    try:
        if model_name == "Perceptron":
            model = pickle.load(open("model_ppn.pkl", "rb"))
            encoder = pickle.load(open("enc_ppn.pkl", "rb"))
            scaler = pickle.load(open("sc_ppn.pkl", "rb"))
        else:
            raise ValueError("Model tidak ditemukan!")
        return model, encoder, scaler
    except FileNotFoundError as e:
        st.error(f"File yang dibutuhkan tidak ditemukan: {e}")
        return None, None, None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {e}")
        return None, None, None

# Fungsi untuk validasi input
def validate_input(features):
    if any(f < 0 for f in features[:2]):  # Diameter dan Weight harus >= 0
        return False, "Diameter dan Weight tidak boleh bernilai negatif."
    if any(f < 0 or f > 1 for f in features[2:]):  # Red, Green, dan Blue harus 0-1
        return False, "Nilai Red, Green, dan Blue harus berada di antara 0 dan 1."
    return True, None

# Judul aplikasi
st.title("Prediksi Jenis Buahmu Disini :)")
st.write("Pilih algoritma dan masukkan data untuk memprediksi.")

# Dropdown untuk memilih algoritma
algorithm = st.selectbox("Pilih Algoritma:", ["Perceptron"])

# Input data
st.write("### Masukkan Data")
feature1 = st.number_input("Diameter", min_value=0.0, step=0.01, format="%.2f")
feature2 = st.number_input("Weight", min_value=0.0, step=0.01, format="%.2f")
feature3 = st.number_input("Red", min_value=0.0, max_value=1.0, step=0.01, format="%.2f")
feature4 = st.number_input("Green", min_value=0.0, max_value=1.0, step=0.01, format="%.2f")
feature5 = st.number_input("Blue", min_value=0.0, max_value=1.0, step=0.01, format="%.2f")

# Tombol untuk prediksi
if st.button("Prediksi"):
    try:
        # Validasi input
        input_features = [feature1, feature2, feature3, feature4, feature5]
        is_valid, error_message = validate_input(input_features)
        if not is_valid:
            st.error(error_message)
        else:
            # Load model, encoder, dan scaler
            model, encoder, scaler = load_model_and_preprocessors(algorithm)
            if model is None or scaler is None:
                st.error("Gagal memuat model atau scaler.")
            else:
                # Scaling data input
                input_data = np.array([input_features])
                st.write("Data sebelum scaling:", input_data)

                input_data_scaled = scaler.transform(input_data)
                st.write("Data setelah scaling:", input_data_scaled)

                # Prediksi
                prediction = model.predict(input_data_scaled)
                st.write("Hasil prediksi numerik:", prediction)

                # Transformasi hasil prediksi ke label jika encoder tersedia
                if encoder:
                    prediction_label = encoder.inverse_transform(prediction)
                    st.success(f"Hasil Prediksi: {prediction_label[0]}")
                else:
                    st.success(f"Hasil Prediksi: {prediction[0]}")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
