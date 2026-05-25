# K-Means Clustering: GPS Trajectories (Go!Track)

## Deskripsi Proyek
Proyek ini merupakan implementasi algoritma **K-Means Clustering** menggunakan bahasa pemrograman Python. Model ini dibangun untuk mengelompokkan data perjalanan (GPS trajectories) berdasarkan fitur mobilitas pengguna. Dataset yang digunakan berasal dari aplikasi Android Go!Track (UCI Machine Learning Repository).

## Dataset
Dataset utama yang digunakan adalah `go_track_tracks.csv`. 
- **Pembersihan:** Kolom `linha` dihapus karena mengandung banyak *missing values* (NaN) dan berisi data kategorikal teks yang tidak relevan secara langsung untuk perhitungan jarak Euclidean pada algoritma K-Means.
- **Fitur Numerik:** Variabel difokuskan pada metrik numerik seperti jarak tempuh (`distance`), kecepatan (`speed`), atau id pengguna (`id_android`).

## Teknologi dan Library
- **Environment:** Google Colab / Jupyter Notebook
- **Bahasa:** Python 3.x
- **Manipulasi Data:** Pandas & Numpy
- **Modeling & Scaling:** Scikit-Learn (`KMeans`, `MinMaxScaler`)
- **Visualisasi:** Matplotlib & Seaborn

## Tahapan Implementasi
1. **Persiapan Data:** Memuat dataset ke dalam Pandas DataFrame dan melakukan inspeksi awal (`info()`, `head()`).
2. **Pra-pemrosesan:** Menghapus kolom yang tidak relevan (`linha`).
3. **Seleksi Fitur:** Memilih kolom-kolom numerik yang akan digunakan untuk pembentukan klaster.
4. **Normalisasi Data:** Menggunakan `MinMaxScaler` untuk mengubah rentang seluruh data menjadi skala 0 hingga 1. Hal ini wajib dilakukan karena K-Means sangat sensitif terhadap perbedaan skala data antar variabel.
5. **Pembuatan Model (dengan *Seeds*):** Menginisialisasi algoritma K-Means dengan parameter `n_clusters=3` dan menetapkan *seeds* (`random_state=42`) agar hasil *clustering* bersifat tetap dan dapat direproduksi, bukan sekadar estimasi acak.
6. **Visualisasi:** Menampilkan hasil sebaran klaster menggunakan *scatter plot* yang dilengkapi dengan titik pusat massa (*Centroid*) untuk masing-masing klaster.

## Catatan Penting Mengenai Evaluasi
Karena K-Means merupakan algoritma **Unsupervised Learning** (tidak memiliki label target yang pasti), evaluasi model **tidak dapat** dilakukan menggunakan metrik Supervised Learning seperti `Accuracy`, `Precision`, `Recall`, atau `F1-Score`. 
Pengelompokan (clustering) dievaluasi berdasarkan jarak kedekatan data dalam satu klaster dan jarak antar klaster (misalnya dengan visualisasi plot atau metode *Silhouette Score*).

## Panduan Menjalankan di Google Colab
1. Unduh atau *clone* repositori ini.
2. Buka [Google Colab](https://colab.research.google.com/) di browser.
3. Unggah file notebook (berformat `.ipynb`) ke dalam Colab.
4. Pada panel sebelah kiri (ikon folder / *Files*), unggah file `go_track_tracks.csv` ke dalam *session storage*.
5. Jalankan sel (cell) satu per satu dari atas ke bawah.
