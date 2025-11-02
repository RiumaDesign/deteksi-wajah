import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# Konfigurasi Aplikasi

MODEL_PATH = "model_lbph.yml"
LABELS_PATH = "labels.npy"

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_PATH)
label_dict = np.load(LABELS_PATH, allow_pickle=True).item()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

st.set_page_config(page_title="Face Recognition LBPH", layout="wide")
st.title("Face Recognition")
st.markdown("Gunakan aplikasi ini untuk **mengenali wajah** sekaligus **melihat tahapan pengolahan citra digital**.")


# Fungsi Proses Wajah

def recognize_faces(image):
    img_rgb = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    img_result = img_rgb.copy()
    recognized = []

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        label_id, conf = recognizer.predict(roi)
        name = label_dict.get(label_id, "Unknown") if conf < 80 else "Unknown"

        recognized.append(name)
        cv2.rectangle(img_result, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img_result, f"{name} ({conf:.0f})", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return img_rgb, gray, img_result, recognized


# Sidebar Menu

menu = st.sidebar.radio("Pilih Mode", ["Upload Gambar", "Webcam"])
st.sidebar.markdown("---")
st.sidebar.markdown("**Dibuat oleh Rifaudin**<br>Pengolahan Citra Digital", unsafe_allow_html=True)


# MODE 1: Upload Gambar

if menu == "Upload Gambar":
    uploaded_file = st.file_uploader("Upload gambar wajah", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Citra Asli", use_container_width=True)

        if st.button("Deteksi & Kenali"):
            rgb, gray, result, names = recognize_faces(image)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(rgb, caption="Citra RGB", use_container_width=True)
            with col2:
                st.image(gray, caption="Citra Grayscale", use_container_width=True)
            with col3:
                st.image(result, caption="Hasil Deteksi Wajah", use_container_width=True)

            # Histogram Grayscale
            st.markdown("Histogram Intensitas")
            fig, ax = plt.subplots()
            ax.hist(gray.ravel(), bins=256, range=[0, 256], color='gray')
            ax.set_title("Histogram Intensitas (Grayscale)")
            ax.set_xlabel("Intensitas Piksel")
            ax.set_ylabel("Frekuensi")
            st.pyplot(fig)

            st.success(f"Jumlah wajah terdeteksi: {len(names)}")
            st.write("Nama yang dikenali:", names)

        if st.button("Reset"):
            st.rerun()


# MODE 2: Webcam Realtime

elif menu == "Webcam":
    run = st.checkbox("Mulai Webcam")
    FRAME_WINDOW = st.image([])

    camera = cv2.VideoCapture(0)
    while run:
        ret, frame = camera.read()
        if not ret:
            st.warning("Tidak dapat mengakses kamera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            label_id, conf = recognizer.predict(roi)
            name = label_dict.get(label_id, "Unknown") if conf < 80 else "Unknown"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} ({conf:.0f})", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    camera.release()


# Footer

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:gray;'>"
    "Dibuat dengan oleh <b>Rifaudin</b> | "
    "<i>Pengolahan Citra Digital - Deteksi & Pengenalan Wajah</i></p>",
    unsafe_allow_html=True
)
