# 🚗 K-Means Clustering: Go!Track GPS Trajectories

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

## 📌 Deskripsi Proyek (Showcase)
Proyek ini adalah implementasi **Machine Learning (Unsupervised Learning)** menggunakan algoritma **K-Means Clustering** untuk menganalisis dan mengelompokkan data perjalanan pengguna (GPS Trajectories). 

Dengan menganalisis pola jarak tempuh (`distance`), kecepatan (`speed`), dan data perangkat, kita dapat mengidentifikasi karakteristik mobilitas yang berbeda dari setiap perjalanan. Pengelompokan ini sangat berguna untuk:
- Klasifikasi moda transportasi (misalnya: berjalan kaki vs. berkendara).
- Analisis pola mobilitas dan perilaku pengguna aplikasi.
- Sistem rekomendasi rute berdasarkan kebiasaan.

## 📊 Dataset
Dataset yang digunakan adalah **Go!Track Trajectories** dari UCI Machine Learning Repository, yang berisi rekaman lintasan pergerakan dari pengguna perangkat Android.
* **File Utama:** `go_track_tracks.csv`
* **Fitur Analisis:** `id_android`, `speed`, `distance`
* **Pra-pemrosesan:** Melakukan pembersihan data dengan menghapus kolom `linha` karena tingginya *missing values* (NaN) serta format teks yang tidak relevan untuk perhitungan jarak matematis pada algoritma K-Means.

## ⚙️ Metodologi & Tahapan
1. **Eksplorasi Data (EDA):** Memvisualisasikan sebaran awal antara titik-titik perjalanan untuk memahami distribusi data secara mentah.
2. **Normalisasi (Min-Max Scaling):** Menyamakan skala data numerik menjadi rentang 0 hingga 1. *Penting: K-Means mengandalkan Euclidean Distance, sehingga normalisasi adalah tahap yang sangat krusial agar fitur dengan angka besar (seperti kecepatan) tidak mendominasi fitur berangka kecil (seperti jarak pendek).*
3. **Pemodelan K-Means:** * Mengelompokkan data ke dalam **3 klaster** (`n_clusters=3`).
    * Menggunakan `random_state=42` untuk memastikan *seeds* model tetap, sehingga hasil klastering konsisten, dapat direproduksi, dan bukan sekadar estimasi hasil yang berubah-ubah.
4. **Evaluasi Model Unsupervised:** Memperbaiki miskonsepsi umum. Karena K-Means tidak memiliki label target asli, evaluasi tidak menggunakan metrik Supervised (seperti Accuracy atau F1-Score), melainkan dianalisis secara visual melalui pembentukan klaster dan posisi *Centroid*.

## 📈 Hasil dan Visualisasi
Algoritma K-Means berhasil memisahkan data ke dalam 3 profil perjalanan yang berbeda. Sentroid (titik pusat) dari masing-masing klaster ditandai dengan jelas (tanda "X" merah), yang merepresentasikan titik tengah karakteristik (rata-rata kecepatan dan parameter lainnya) untuk setiap kelompok.

### 1. Sebaran Data Awal
<img width="571" height="434" alt="image" src="https://github.com/user-attachments/assets/107f9425-ec45-44a0-adfd-c44ed755afd1" />

### 2. Hasil K-Means Clustering & Centroid
<img width="695" height="547" alt="image" src="https://github.com/user-attachments/assets/eca04903-08fe-44fe-9fc3-c4b4ce11bfe7" />

## 🚀 Cara Menjalankan di Google Colab
1. Buka [Google Colab](https://colab.research.google.com/).
2. Unggah file Notebook (`.ipynb`) dari repositori ini.
3. Pada panel *Files* di sebelah kiri, unggah dataset `go_track_tracks.csv`.
4. Jalankan seluruh sel kode secara berurutan (*Run All*).

## 💡 Kesimpulan
Melalui teknik K-Means Clustering, kumpulan data perjalanan yang mentah dapat diekstraksi menjadi kelompok-kelompok profil pengguna yang bermakna. Proyek ini juga mendemonstrasikan pemahaman dasar yang kuat mengenai pemrosesan data (seperti pentingnya *Scaling*) serta perbedaan mendasar dalam mengevaluasi model *Unsupervised* dibandingkan *Supervised Learning*.
