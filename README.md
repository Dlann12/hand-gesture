# 🖐️ Hand Gesture Recognition System — Machine Learning

Sistem pengenalan gestur tangan berbasis Machine Learning yang mampu mengenali gestur tangan manusia dan mengklasifikasikannya menjadi representasi **angka 0–5** secara real-time melalui webcam desktop.

Sistem menggunakan pendekatan **Landmark-Based Classification** dengan mengekstrak 21 koordinat 3D sendi tangan dari **MediaPipe Hands**, menormalisasi koordinat agar invarian terhadap posisi dan skala, kemudian mengklasifikasikan gestur menggunakan model **Random Forest Classifier**.

---

## 📋 Daftar Isi
1. [Fitur Utama](#-fitur-utama)
2. [Arsitektur Machine Learning](#-arsitektur-machine-learning)
3. [Kelas Gestur](#-kelas-gestur)
4. [Hasil Evaluasi Model](#-hasil-evaluasi-model)
5. [Struktur Folder](#-struktur-folder)
6. [Instalasi dan Penggunaan](#-instalasi-dan-penggunaan)
7. [Pengumpulan Data Nyata](#-pengumpulan-data-nyata)
8. [Pengujian Unit (Tests)](#-pengujian-unit-tests)

---

## ✨ Fitur Utama

- **Deteksi Real-Time:** Mengakses webcam dan melacak 21 landmark tangan MediaPipe secara real-time.
- **Klasifikasi Angka 0–5:** Mengidentifikasi gestur tangan mengepal (0) hingga lima jari terbuka (5).
- **Invarian Translasi & Skala:** Normalisasi koordinat berbasis pusat pergelangan tangan (wrist-centering) dan normalisasi jarak euclidean maksimum.
- **Filter Keyakinan (Unknown Rejection):** Menolak prediksi dengan skor keyakinan di bawah ambang batas (default: 60%) untuk mencegah false positives.
- **Dashboard Streamlit Interaktif:**
  - Preview webcam dengan skeleton landmark dan badge hasil prediksi besar.
  - Mode unggah gambar untuk pengujian file foto.
  - Halaman evaluasi interaktif (Confusion matrix, grafik akurasi, perbandingan model).
  - Slider pengaturan ambang keyakinan (confidence threshold).
  - Ekspor dan unduh laporan klasifikasi.

---

## 🧠 Arsitektur Machine Learning

```text
[ Webcam / Foto ]
       │
       ▼
[ MediaPipe Hands ] ────────► 21 Koordinat Sendi 3D (x, y, z) = 63 Nilai
       │
       ▼
[ Normalisasi Fitur ] ──────► 1. Wrist-Centering (Pergelangan = 0, 0, 0)
       │                      2. Scale-Invariance (Dibagi Jarak Maksimum)
       ▼
[ Random Forest Model ] ────► Estimasi Probabilitas 6 Kelas (predict_proba)
       │
       ├── Jika Max Prob < Threshold ──► "Unknown"
       └── Jika Max Prob ≥ Threshold ──► Angka 0–5 + Confidence Score
```

---

## 🖐️ Kelas Gestur

| Label | Nama Gestur | Deskripsi Posisi Tangan |
|---|---|---|
| **0** | `number_0` | Tangan mengepal rapat |
| **1** | `number_1` | Satu jari terangkat (telunjuk) |
| **2** | `number_2` | Dua jari terangkat (telunjuk & tengah) |
| **3** | `number_3` | Tiga jari terangkat (telunjuk, tengah, manis) |
| **4** | `number_4` | Empat jari terangkat (tanpa jempol) |
| **5** | `number_5` | Lima jari terbuka lebar |

---

## 📊 Hasil Evaluasi Model

Model dievaluasi pada **270 sampel data uji held-out (15%)** yang tidak pernah dilihat model saat training.

### 1. Metrik Keseluruhan (Test Set)
- **Akurasi Pengujian (Accuracy):** `81.48%`
- **Macro Precision:** `81.45%`
- **Macro Recall:** `81.48%`
- **Macro F1-Score:** `81.40%`

### 2. Performa Per Kelas
| Kelas | Label Gestur | Precision | Recall | F1-Score | Support |
|:---:|---|:---:|:---:|:---:|:---:|
| **0** | Tangan Mengepal | 100.0% | 100.0% | **100.0%** | 45 |
| **1** | Satu Jari Terangkat | 100.0% | 100.0% | **100.0%** | 45 |
| **2** | Dua Jari Terangkat | 45.1% | 51.1% | **47.9%** | 45 |
| **3** | Tiga Jari Terangkat | 43.6% | 37.8% | **40.5%** | 45 |
| **4** | Empat Jari Terangkat | 100.0% | 100.0% | **100.0%** | 45 |
| **5** | Lima Jari Terbuka | 100.0% | 100.0% | **100.0%** | 45 |

> **Catatan Dataset:** Kelas 0, 1, 4, dan 5 mencapai F1-Score 100%. Terdapat tumpang tindih antara kelas 2 dan 3 pada dataset geometri sintetis. Pada pengambilan landmark kamera nyata, akurasi kelas 2 dan 3 dapat ditingkatkan lebih lanjut dengan merekam data tangan asli via `src/collect_data.py`.

### 3. Perbandingan Model (Eksperimen Tambahan)
| Model | Akurasi Validasi | Waktu Training |
|---|:---:|:---:|
| **Random Forest (Primary)** | **84.81%** | 0.26s |
| Random Forest Deep | 83.70% | 0.40s |
| Support Vector Machine (SVM RBF) | 83.70% | 0.06s |
| Multi-Layer Perceptron (MLP) | 81.48% | 0.09s |

---

## 📁 Struktur Folder

```text
hand-number/
├── app.py                      # Aplikasi web dashboard Streamlit
├── requirements.txt            # Daftar dependensi teruji
├── README.md                   # Dokumentasi proyek lengkap
├── configs/
│   └── config.yaml             # Konfigurasi data, model, dan inferensi
├── data/
│   ├── raw/                    # Data rekaman mentah per kelas
│   └── processed/              # Dataset siap latih (CSV)
├── models/
│   ├── gesture_classifier.joblib  # Model Random Forest tersimpan
│   └── metadata.json           # Metadata versi, konfigurasi, dan metrik
├── notebooks/
│   └── exploratory_analysis.ipynb # Notebook analisis data eksploratif
├── reports/
│   ├── classification_report.txt  # Laporan metrik teks Scikit-Learn
│   ├── confusion_matrix.png       # Grafik heatmap confusion matrix
│   ├── metrics.json               # Data metrik format JSON
│   └── model_comparison.csv       # Tabel perbandingan model
├── src/
│   ├── __init__.py
│   ├── collect_data.py         # Tool perekam landmark dari webcam
│   ├── extract_landmarks.py    # Ekstraksi koordinat MediaPipe
│   ├── preprocess.py           # Normalisasi invarian & split dataset
│   ├── train.py                # Pipeline pelatihan model ML
│   ├── evaluate.py             # Perhitungan metrik evaluasi
│   └── predict.py              # Inferensi dan logika penolakan unknown
└── tests/
    ├── test_preprocess.py      # Pengujian unit normalisasi & splitting
    └── test_predict.py         # Pengujian unit prediksi model
```

---

## 🚀 Instalasi dan Penggunaan

### 1. Prasyarat
- Python 3.11 atau 3.12
- Webcam (untuk inferensi real-time)

### 2. Setup Environment
```bash
# Clone atau buka direktori proyek
cd /home/kerja/Documents/hand-number

# Buat virtual environment
python3 -m venv .venv

# Aktifkan virtual environment
source .venv/bin/activate

# Pasang dependensi
pip install -r requirements.txt
```

### 3. Menjalankan Pelatihan Model
Untuk melatih ulang model dan memperbarui laporan metrik:
```bash
python src/train.py
```

### 4. Menjalankan Dashboard Streamlit
```bash
streamlit run app.py
```
Aplikasi akan terbuka otomatis di peramban pada alamat `http://localhost:8501`.

---

## 📹 Pengumpulan Data Nyata (Opsional)

Untuk memperkaya dataset menggunakan gestur tangan sendiri melalui webcam:

```bash
# Rekam 300 sampel untuk label 1 (Satu Jari)
python src/collect_data.py --label 1 --samples 300

# Kontrol:
# [SPACE] : Ambil 1 sampel landmark
# [A]     : Aktifkan mode auto-capture
# [Q/ESC] : Selesai dan simpan
```
File CSV hasil rekaman akan tersimpan di `data/raw/gestures_1.csv`.

---

## 🧪 Pengujian Unit (Tests)

Jalankan seluruh test suite untuk memastikan fungsi normalisasi dan inferensi bekerja sesuai spesifikasi:
```bash
PYTHONPATH=. pytest tests/ -v
```
Seluruh 10 pengujian unit tervalidasi sukses (10 passed).
