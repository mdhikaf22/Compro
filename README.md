# Face Recognition System - MTCNN + ViT

Sistem pengenalan wajah real-time menggunakan **MTCNN** untuk deteksi wajah dan **Vision Transformer (ViT)** untuk klasifikasi.

## ⚡ Quick Start

```bash
# 1. Clone repository
git clone https://github.com/mdhikaf22/Compro.git
cd Compro

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download model (WAJIB!)
# Download dari: https://drive.google.com/file/d/YOUR_FILE_ID/view
# Letakkan file 'best_vit_mtcnn.pth' di folder root project

# 4. Jalankan server
python app.py

# 5. Buka browser
# http://localhost:5000/api/webcam
```

## 📥 Download Model

**PENTING:** File model tidak termasuk di repository karena terlalu besar.

📁 **Download model di sini:** [Google Drive Link - best_vit_mtcnn.pth](https://drive.google.com/YOUR_LINK)

Setelah download, letakkan file `best_vit_mtcnn.pth` di folder root project.

## 📋 Deskripsi

Proyek ini adalah sistem pengenalan wajah yang dirancang untuk keperluan akses kontrol (misalnya di depan pintu lab). Sistem dapat mendeteksi wajah dari webcam secara real-time dan mengklasifikasikan apakah orang tersebut **Authorized** atau **Not Authorized**.

### Fitur Utama
- **Face Detection**: Menggunakan MTCNN (Multi-task Cascaded Convolutional Networks)
- **Face Classification**: Menggunakan ViT (Vision Transformer) dari Google
- **Real-time Inference**: Mendukung webcam untuk deteksi langsung
- **REST API**: Backend Flask dengan endpoint untuk integrasi
- **Database Logging**: SQLite untuk menyimpan log akses
- **Web Interface**: UI webcam berbasis browser
- **Authorization System**: Menampilkan status otorisasi berdasarkan role

## 👥 Kelas yang Dikenali

| Nama | Role | Status |
|------|------|--------|
| Iksan | Aslab | ✅ Authorized |
| Akbar | Aslab | ✅ Authorized |
| Aprilianza | Aslab | ✅ Authorized |
| Bian | Dosen | ✅ Authorized |
| Fadhilah | Aslab | ✅ Authorized |
| Falah | Aslab | ✅ Authorized |
| Imelda | Aslab | ✅ Authorized |
| Rifqy | Aslab | ✅ Authorized |
| Yolanda | Aslab | ✅ Authorized |

## 🛠️ Requirements

```
torch
torchvision
transformers
facenet-pytorch
opencv-python
Pillow
matplotlib
numpy
psutil
roboflow
scikit-learn
seaborn
```

### Instalasi Dependencies

```bash
pip install torch torchvision transformers facenet-pytorch opencv-python Pillow matplotlib numpy psutil roboflow scikit-learn seaborn
```

## 📁 Struktur Proyek

```
compro/
├── app.py                      # Main entry point (Flask)
├── requirements.txt            # Dependencies
├── best_vit_mtcnn.pth          # Model weights terbaik
├── access_logs.db              # SQLite database (auto-generated)
│
├── api/                        # API Package (Modular)
│   ├── __init__.py
│   ├── config.py               # Konfigurasi aplikasi
│   ├── database.py             # Database operations
│   ├── model.py                # Face detection & classification
│   └── routes/                 # API Routes
│       ├── __init__.py
│       ├── main.py             # Home & health check
│       ├── detection.py        # Face detection endpoints
│       ├── logs.py             # Access logs & statistics
│       └── webcam.py           # Webcam interface & stream
│
├── Compro_MTCNN.ipynb          # Notebook (training & evaluation)
├── webcam_inference.py         # Script webcam (standalone)
├── webcam_cell_code.py         # Code untuk cell notebook
│
├── vit_mtcnn_model/            # Folder model
│   ├── model.pth
│   └── config.json
├── vit_dataset/                # Dataset untuk training
│   └── train/
│       ├── akbar/
│       ├── aprilianza/
│       └── ...
├── Computing_project-2/        # Dataset asli dari Roboflow
├── screenshots/                # Folder screenshot webcam
└── README.md
```

## 🚀 Cara Penggunaan

### 1. Training Model (Notebook)

1. Buka `Compro_MTCNN.ipynb` di Jupyter Notebook atau Google Colab
2. Jalankan semua cell secara berurutan
3. Model akan disimpan di `best_vit_mtcnn.pth`

### 2. Menjalankan API Server (Recommended)

```bash
# Install dependencies
pip install -r requirements.txt

# Jalankan server
python app.py
```

Server akan berjalan di `http://localhost:5000`

### 3. Webcam Inference (Standalone)

#### Option A: Via Web Browser (Recommended)
1. Jalankan `python app.py`
2. Buka browser ke `http://localhost:5000/api/webcam`
3. Klik "Start Camera" lalu "Capture & Detect"

#### Option B: Script Standalone
```bash
python webcam_inference.py
```

#### Option C: Di Notebook
1. Jalankan semua cell training terlebih dahulu
2. Tambahkan cell baru di akhir notebook
3. Copy code dari `webcam_cell_code.py`
4. Jalankan cell tersebut

### 4. Kontrol Webcam (Standalone Script)

| Key | Fungsi |
|-----|--------|
| `q` | Keluar dari webcam |
| `s` | Simpan screenshot |

## 📊 Hasil Training

- **Model**: ViT (google/vit-base-patch16-224-in21k)
- **Epochs**: 30
- **Best Validation Accuracy**: ~100%
- **Total Classes**: 9

## 🔧 Konfigurasi

### Threshold
```python
CONFIDENCE_THRESHOLD = 0.5      # Threshold klasifikasi
FACE_DETECTION_THRESHOLD = 0.7  # Threshold deteksi wajah MTCNN
```

### MTCNN Parameters
```python
min_face_size = 20              # Ukuran minimum wajah
thresholds = [0.5, 0.6, 0.6]    # Threshold per stage
```

## 📝 Catatan

- Pastikan webcam terhubung dan berfungsi dengan baik
- Model membutuhkan GPU untuk performa optimal (CUDA)
- Jika menggunakan CPU, inference akan lebih lambat

## 🏗️ Arsitektur

```
Input Image
    ↓
MTCNN (Face Detection)
    ↓
Face Cropping + Padding (15%)
    ↓
Preprocessing (Resize 224x224, Normalize)
    ↓
ViT (Classification)
    ↓
Output: Name, Role, Authorization Status
```

## 📜 License

Project ini dibuat untuk keperluan Computing Project.

## 👨‍💻 Author

MAHARDHIKA
