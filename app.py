import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ==============================
# KONFIGURASI HALAMAN
# ==============================
st.set_page_config(page_title="Face Recognition - PCD", layout="wide")
st.title("Deployment Sistem Deteksi & Pengenalan Wajah")
st.markdown("""
Aplikasi ini merupakan hasil **deployment pengolahan citra digital** menggunakan **OpenCV + LBPH (Local Binary Pattern Histogram)**.
Dapat digunakan untuk **mendeteksi dan mengenali wajah** baik melalui **unggahan gambar** maupun **kamera realtime**.
""")

# ==============================
# LOAD MODEL DAN LABEL
# ==============================
MODEL_PATH = "model_lbph.yml"
LABELS_PATH = "labels.npy"

try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)
    label_dict = np.load(LABELS_PATH, allow_pickle=True).item()
except Exception as e:
    st.error("[!] Gagal memuat model atau label. Pastikan file `model_lbph.yml` dan `labels.npy` tersedia di direktori aplikasi.")
    st.stop()

# ==============================
# LOAD CLASSIFIER WAJAH
# ==============================
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ==============================
# FUNGSI PENGENALAN WAJAH
# ==============================
def recognize_faces(image):
    img = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        st.warning("Tidak ada wajah terdeteksi pada gambar.")
        return img, gray, []

    recognized = []
    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        label_id, conf = recognizer.predict(roi)

        name = label_dict.get(label_id, "Unknown") if conf < 80 else "Unknown"
        recognized.append(name)

        cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(img, f"{name} ({conf:.1f})", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    return img, gray, recognized

# ==============================
# SIDEBAR MENU
# ==============================
menu = st.sidebar.radio("Pilih Mode:", ["Upload Gambar", "Webcam"])

# ==============================
# MODE UPLOAD GAMBAR
# ==============================
if menu == "Upload Gambar":
    uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang Diupload", use_container_width=True)

        if st.button("Deteksi dan Kenali"):
            result_img, gray_img, results = recognize_faces(image)

            if results:
                # Visualisasi Tahapan Pengolahan Citra
                st.subheader("Tahapan Pengolahan Citra Digital")
                edges = cv2.Canny(gray_img, 100, 200)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(image, caption="Citra Asli", use_container_width=True)
                with col2:
                    st.image(gray_img, caption="Grayscale", use_container_width=True)
                with col3:
                    st.image(edges, caption="Deteksi Tepi (Canny)", use_container_width=True)

                # Histogram
                fig, ax = plt.subplots()
                ax.hist(gray_img.ravel(), bins=256, color='gray')
                ax.set_title("Histogram Grayscale")
                st.pyplot(fig)

                # Hasil deteksi wajah
                st.subheader("Hasil Pengenalan Wajah")
                st.image(result_img, caption="Hasil Deteksi & Pengenalan", use_container_width=True)
                st.success(f"Jumlah wajah terdeteksi: {len(results)}")
                st.write("Nama terdeteksi:", results)
            else:
                st.warning("Tidak ditemukan wajah yang dapat dikenali.")

        if st.button("Reset"):
            st.rerun()

# ==============================
# MODE WEBCAM
# ==============================
elif menu == "Webcam":
    run = st.checkbox("Aktifkan Kamera")
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
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, f"{name} ({conf:.0f})", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    camera.release()
    cv2.destroyAllWindows()

# ==============================
# FOOTER
# ==============================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:gray;'>Dibuat oleh <b>Rifaudin</b> | Mata Kuliah Pengolahan Citra Digital | Universitas XYZ</p>",
    unsafe_allow_html=True
)
