import cv2
import os
import numpy as np

# Folder dataset
DATASET_DIR = "dataset_wajah"
MODEL_PATH = "model_lbph.yml"
LABELS_PATH = "labels.npy"

# Inisialisasi recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
label_dict = {}
label_id = 0

# Detektor wajah untuk cropping otomatis
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print("[INFO] Memproses dataset...")
for person_name in os.listdir(DATASET_DIR):
    person_folder = os.path.join(DATASET_DIR, person_name)
    if not os.path.isdir(person_folder):
        continue

    label_dict[label_id] = person_name

    for img_name in os.listdir(person_folder):
        path = os.path.join(person_folder, img_name)
        img = cv2.imread(path)
        if img is None:
            print(f"[WARNING] Gagal membaca {path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces_detected = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Kalau wajah terdeteksi, crop dan resize
        if len(faces_detected) > 0:
            for (x, y, w, h) in faces_detected:
                face_crop = gray[y:y+h, x:x+w]
                # Samakan ukuran semua wajah
                face_resized = cv2.resize(face_crop, (200, 200))
                faces.append(face_resized)
                labels.append(label_id)
        else:
            # Jika tidak terdeteksi, tetap resize seluruh gambar
            face_resized = cv2.resize(gray, (200, 200))
            faces.append(face_resized)
            labels.append(label_id)

    label_id += 1

print(f"[INFO] Total data wajah: {len(faces)}")
print(f"[INFO] Jumlah orang: {len(label_dict)}")

# Ubah list ke array NumPy (sekarang ukurannya seragam)
faces = np.array(faces, dtype='uint8')
labels = np.array(labels, dtype='int32')

# Latih model
print("[INFO] Melatih model LBPH...")
recognizer.train(faces, labels)

# Simpan model dan label
recognizer.write(MODEL_PATH)
np.save(LABELS_PATH, label_dict)

print("[INFO] Model tersimpan sebagai model_lbph.yml dan labels.npy ✅")
