import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import math
import os
import streamlit as st
from PIL import Image
import io

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="İnek Ağırlık Tahmini",
    page_icon="🐄",
    layout="wide"
)

# Custom CSS ile sayfa tasarımını güzelleştirme
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1 {
        color: #2e5e4e;
        text-align: center;
        margin-bottom: 30px;
        font-family: 'Arial', sans-serif;
    }
    .stButton button {
        background-color: #2e5e4e;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
        margin: 10px 0;
    }
    .info-box {
        background-color: #ebf5f0;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border: 1px solid #2e5e4e;
    }
    .weight-display {
        background-color: #2e5e4e;
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 24px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Load YOLOv8 models
@st.cache_resource
def load_models():
    model1 = YOLO("models/eye.pt")
    model2 = YOLO("models/cow.pt")
    return model1, model2

model1, model2 = load_models()

# Calculate the Euclidean distance between two points
def euclidean(pt1, pt2):
    return math.sqrt((pt1[0] - pt2[0])**2 + abs(pt1[1] - pt2[1])**2)

# Function to visualize instance segmentation and keypoint detection
def visualize_combined_results(img, model1, model2):
    results1 = model1(img, save=False)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Perform instance segmentation
    # Perform keypoint detection
    results2 = model2(img, save=False)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    dist = None
    dist1 = None
    dist2 = None
    
    # Visualize instance segmentation results
    for result in results1:
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for mask, box, score, cls in zip(masks, boxes, scores, classes):
                mask_resized = cv2.resize(mask, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
                mask_resized = mask_resized.astype(bool)

                img_mask = np.zeros_like(img_rgb)
                img_mask[mask_resized] = [229, 22, 122]  # Orange color for mask

                img_rgb = cv2.addWeighted(img_rgb, 1.0, img_mask, 1.0, 1)

                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (229, 22, 122), 2)

                pt1 = (x1, y1)
                pt2 = (x2, y2)
                dist = euclidean(pt1, pt2)
                
                label = f'{model1.names[int(cls)]} {score:.2f}'
                cv2.putText(img_rgb, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        else:
            st.warning("Bu görüntüde göz tespit edilemedi.")

    # Visualize keypoint detection results
    for result in results2:
        if result.keypoints is not None and result.boxes is not None:
            keypoints = result.keypoints.data.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            point1 = keypoints[0][1][0], keypoints[0][1][1]
            point2 = keypoints[0][2][0], keypoints[0][2][1]
            point3 = keypoints[0][3][0], keypoints[0][3][1]
            point4 = keypoints[0][4][0], keypoints[0][4][1]
            
            dist1 = euclidean(point1, point2)
            dist2 = euclidean(point3, point4)

            for keypoint, box, score, cls in zip(keypoints, boxes, scores, classes):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                for kp in keypoint:
                    kp_x, kp_y = int(kp[0]), int(kp[1])
                    cv2.circle(img_rgb, (kp_x, kp_y), 3, (0, 0, 255), -1)

                label = f'{model2.names[int(cls)]} {score:.2f}'
                cv2.putText(img_rgb, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    if dist and dist1 and dist2:
        x = 4
        lb = 0.45359237
        dist1cm = (x * dist1) / dist
        dist2cm = (x * dist2) / dist
        
        _weight = (dist1cm * dist2cm * dist2cm * lb) / 300
        return img_rgb, _weight
    return img_rgb, None

# Process image function
def process_image(img):
    # New image size
    new_width = 1040
    new_height = 640

    # Resize the image
    resized_image = cv2.resize(img, (new_width, new_height))

    # Call the combined visualization function
    with st.spinner('Görüntü işleniyor...'):
        processed_image, weight = visualize_combined_results(resized_image, model1, model2)

    # Display the processed image
    st.image(processed_image, caption='İşlenmiş Görüntü', use_container_width=True)

    if weight is not None:
        st.markdown(f"""
        <div class="weight-display">
            Tahmini Ağırlık: {weight:.2f} kg
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("Ağırlık hesaplanamadı. Lütfen ineğin tüm vücudu görünecek şekilde başka bir fotoğraf deneyin.")
    
    return processed_image, weight

# Streamlit app
st.title("🐄 İnek Ağırlık Tahmini Uygulaması")

st.markdown("""
<div class="info-box">
    <h3>Bu uygulama nasıl çalışır?</h3>
    <p>Yapay zeka modelleri kullanarak bir ineğin fotoğrafından ağırlığını tahmin eder. 
    İnek gözü ve vücut noktaları tespit edilerek matematiksel bir model ile ağırlık hesaplanır.</p>
    <p>En iyi sonuç için ineğin yanından çekilmiş, tüm vücudu görünen ve net bir fotoğraf yükleyin.</p>
</div>
""", unsafe_allow_html=True)

# Sekme oluşturma
tab1, tab2 = st.tabs(["📁 Dosyadan Yükle", "📷 Kameradan Çek"])

with tab1:
    st.subheader("Fotoğraf Yükle")
    uploaded_file = st.file_uploader("İnek fotoğrafı seçin...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the uploaded image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        
        # Process the image
        process_image(img)

with tab2:
    st.subheader("Kamera ile Fotoğraf Çek")
    
    # Kamera seçenekleri
    camera_options = ["Ana Kamera", "Ön Kamera", "Harici Kamera"]
    selected_camera = st.selectbox("Kamera seçin:", camera_options)
    
    # Kamera indeksi belirleme
    camera_index = 0
    if selected_camera == "Ön Kamera":
        camera_index = 1
    elif selected_camera == "Harici Kamera":
        camera_index = 2
    
    # Kamera çekimi
    if st.button("Fotoğraf Çek", key="camera_button"):
        try:
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                st.error(f"Kamera açılamadı (indeks: {camera_index}). Lütfen başka bir kamera seçin.")
            else:
                # Görüntüyü oku
                ret, frame = cap.read()
                if ret:
                    # Process the image
                    process_image(frame)
                else:
                    st.error("Görüntü okunamadı. Lütfen tekrar deneyin.")
                # Kamerayı serbest bırak
                cap.release()
        except Exception as e:
            st.error(f"Kamera erişim hatası: {str(e)}")

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 40px; padding: 20px; color: #666;">
    <hr>
    <p>© 2025 İnek Ağırlık Tahmini Uygulaması </p>
</div>
""", unsafe_allow_html=True)

