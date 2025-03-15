from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Muat model
try:
    model = load_model('batik_model.h5')
    print("Model berhasil dimuat!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Daftar label kelas (sesuaikan dengan dataset Anda)
class_labels = ['Aceh_Pintu_Aceh', 'Bali_Barong', 'Bali_Merak',
                'DKI_Ondel_Ondel', 'Jawa_Barat_Megamendung', 'Jawa_Timur_Pring',
                'Kalimantan_Dayak','Lampung Gajah', 'Madura_Mataketaran',
                'Maluku_Pala', 'NTB_Lumbung', 'Papua_Asmat', 'Papua_Cendrawasih',
                'Papua_tifa', 'Solo_Parang', 'Sulawesi_Selatan_Lontara',
                'Sumatera_Barat_Rumah_Minang', 'Sumatera_Utara_Boraspati',
                'Yogyakarta_Kawung', 'Yogyakarta_Parang']  # Ganti dengan label yang sesuai

# Fungsi untuk memproses gambar dan melakukan prediksi
def predict_image(img_path):
    if model is None:
        return "Model tidak tersedia"

    img = image.load_img(img_path, target_size=(150, 150))  # Sesuaikan ukuran dengan model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalisasi

    prediksi = model.predict(img_array)
    predicted_class = np.argmax(prediksi, axis=1)
    return class_labels[predicted_class[0]]

# Route untuk halaman utama
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Ambil file gambar yang diunggah
        file = request.files['file']
        if file:
            # Simpan file sementara
            upload_folder = os.path.join('static', 'uploads')
            os.makedirs(upload_folder, exist_ok=True)  # Buat folder jika belum ada
            file_path = os.path.join(upload_folder, file.filename)
            file.save(file_path)

            # Lakukan prediksi
            hasil_prediksi = predict_image(file_path)

            # Tampilkan hasil prediksi
            return render_template('index.html', prediction=hasil_prediksi, image_path=file_path)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)


