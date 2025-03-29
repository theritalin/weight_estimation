# 🐄 Cattle Weight Estimation Application

A simple and easy-to-use application that estimates cattle weight from photos using artificial intelligence.

## 📋 Features

- Automatic weight estimation from cattle images
- Visual segmentation and body keypoint detection
- Photo upload from files
- Take photos directly from camera
- User-friendly interface

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/username/weight-estimation.git
cd weight-estimation

# Install required packages
pip install -r requirements.txt
```

## 📝 Requirements

- Python 3.8+
- OpenCV
- Streamlit
- PyTorch
- Ultralytics YOLO
- PIL

## 🚀 Usage

To start the application:

```bash
streamlit run webapp.py
```

The application will automatically open in your browser at http://localhost:8501.

## 📊 How It Works

1. AI models detect the cow's eye and body keypoints
2. The eye size is used to calculate a scale factor
3. Distances between body keypoints are measured
4. A mathematical model is used to estimate weight

## 🧪 Model Files

Two model files must be in the `models` folder for the application to work:

- `eye.pt`: Cattle eye detection model
- `cow.pt`: Cattle body keypoint detection model

## 📸 Recommended Photo Characteristics

- Cattle should be photographed from the side
- The entire body should be visible
- Clear and well-lit images

## 📜 License

MIT
