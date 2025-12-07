"""
============================================================
CONFIG - Konfigurasi Aplikasi
============================================================
"""

import os

# Role mapping
ROLE_MAPPING = {
    "iksan": "Aslab",
    "akbar": "Aslab",
    "aprilianza": "Aslab",
    "bian": "Dosen",
    "fadhilah": "Aslab",
    "falah": "Aslab",
    "imelda": "Aslab",
    "rifqy": "Aslab",
    "yolanda": "Aslab",
}

# Class names (sesuai urutan di dataset)
CLASS_NAMES = ['akbar', 'aprilianza', 'bian', 'fadhilah', 'falah', 'iksan', 'imelda', 'rifqy', 'yolanda']

# Thresholds
CONFIDENCE_THRESHOLD = 0.5
FACE_DETECTION_THRESHOLD = 0.7

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_PATH = os.path.join(BASE_DIR, "access_logs.db")
MODEL_PATH = os.path.join(BASE_DIR, "best_vit_mtcnn.pth")
SCREENSHOTS_DIR = os.path.join(BASE_DIR, "screenshots")

# Server config
HOST = '0.0.0.0'
PORT = 5000
DEBUG = False
