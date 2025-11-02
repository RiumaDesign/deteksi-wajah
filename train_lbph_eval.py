import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Konfigurasi folder dan file
DATASET_DIR = "dataset_wajah"
MODEL_PATH = "model_lbph.yml"
LABELS_PATH = "labels.npy"

# Inisialisasi LBPH
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

faces = []
labels = []
label_dict = {}
label_id = 0

print("[INFO] Memproses dataset...")

# Loop setiap folder (nama orang)
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

        if len(faces_detected) > 0:
            for (x, y, w, h) in faces_detected:
                face_crop = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face_crop, (200, 200))
                faces.append(face_resized)
                labels.append(label_id)
        else:
            # Jika wajah tidak terdeteksi
            face_resized = cv2.resize(gray, (200, 200))
            faces.append(face_resized)
            labels.append(label_id)

    label_id += 1

print(f"[INFO] Total data wajah: {len(faces)}")
print(f"[INFO] Jumlah kelas (orang): {len(label_dict)}")

faces = np.array(faces, dtype='uint8')
labels = np.array(labels, dtype='int32')

# Bagi dataset menjadi train & test
X_train, X_test, y_train, y_test = train_test_split(faces, labels, test_size=0.2, random_state=42)

print(f"[INFO] Data train: {len(X_train)}, Data test: {len(X_test)}")

# Latih model LBPH
print("[INFO] Melatih model LBPH...")
recognizer.train(X_train, y_train)

# Simpan model dan label
recognizer.write(MODEL_PATH)
np.save(LABELS_PATH, label_dict)
print(f"[INFO] Model tersimpan sebagai '{MODEL_PATH}' dan label '{LABELS_PATH}'")

# Uji model
print("[INFO] Menguji model...")
y_pred = []
for img in X_test:
    label_pred, confidence = recognizer.predict(img)
    y_pred.append(label_pred)

# Hitung akurasi
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Akurasi Model LBPH: {acc*100:.2f}%")

# Laporan klasifikasi
print("\n[INFO] Laporan Klasifikasi:")
print(classification_report(y_test, y_pred, target_names=list(label_dict.values())))

# Matriks kebingungan
print("[INFO] Matriks Kebingungan:")
print(confusion_matrix(y_test, y_pred))
