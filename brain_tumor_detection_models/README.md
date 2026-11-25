<p align="center">
  <img src="screenshots/frontend_view.png" alt="Neuromind AI — Upload Interface" width="700" />
</p>

<h1 align="center">Neuromind AI — Brain Tumor Detection</h1>

<p align="center">
  <em>A CNN-powered web application that detects brain tumors from MRI scans in real time.</em>
</p>

<p align="center">
  <a href="https://braintumordetection-production-1192.up.railway.app/">
    <img src="https://img.shields.io/badge/Live_Demo-Railway-blueviolet?style=for-the-badge&logo=railway&logoColor=white" alt="Live Demo" />
  </a>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License" />
</p>

---

## 🔬 Project Overview

**Neuromind AI** is a full-stack medical imaging application that classifies brain MRI scans as **tumorous** or **non-tumorous** using a custom Convolutional Neural Network built with TensorFlow/Keras. The model is served through a high-performance **FastAPI** backend and presented via a cyberpunk-themed glassmorphism frontend with real-time scanning animations.

> **🌐 Try it live:** [braintumordetection-production-1192.up.railway.app](https://braintumordetection-production-1192.up.railway.app/)

---

## ✨ Core Features

| Feature | Description |
|---|---|
| **CNN Binary Classifier** | Custom-built Sequential CNN with brain-contour preprocessing for tumor detection |
| **Real-Time Inference** | Upload an MRI scan and receive predictions with confidence scores in seconds |
| **Drag & Drop Upload** | Intuitive drag-and-drop interface supporting JPEG, PNG, and JPG formats |
| **Scanning Animation** | Immersive neon scanner line effect during model inference |
| **Confidence Visualization** | Animated progress bar displaying AI confidence percentage |
| **Responsive Design** | Fully responsive layout optimized for desktop and mobile viewports |
| **Medical Disclaimer** | Built-in advisory prompting users to consult qualified professionals |

---

## 🖼️ Screenshots

<table>
  <tr>
    <td align="center"><strong>⚠️ Tumor Detected</strong></td>
    <td align="center"><strong>✅ No Tumor Detected</strong></td>
  </tr>
  <tr>
    <td><img src="screenshots/frontend_tumor_detected.png" alt="Tumor Detected — 99.93% Confidence" width="450" /></td>
    <td><img src="screenshots/frontend_no_tumor_detected.png" alt="No Tumor — 82.76% Confidence" width="450" /></td>
  </tr>
</table>

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python, FastAPI, Uvicorn |
| **ML Framework** | TensorFlow / Keras (Sequential API) |
| **Image Processing** | OpenCV, imutils, NumPy |
| **Frontend** | HTML5, Vanilla CSS3, Vanilla JavaScript |
| **Fonts** | Google Fonts — Inter, Outfit |
| **Icons** | Font Awesome 6 |
| **Deployment** | Railway |

---

## 🏗️ Architecture & Workflow

```
┌──────────────┐       POST /predict       ┌──────────────────────┐
│              │  ───────────────────────►  │   FastAPI Backend    │
│   Browser    │       (multipart/form)     │                      │
│   Frontend   │                            │  1. Decode image     │
│  (HTML/CSS/  │  ◄───────────────────────  │  2. Crop brain ROI   │
│    JS)       │       JSON response        │  3. Resize 240×240   │
└──────────────┘   {has_tumor, confidence}  │  4. Normalize [0,1]  │
                                            │  5. CNN inference    │
                                            └──────────┬───────────┘
                                                       │
                                              ┌────────▼────────┐
                                              │   model.h5      │
                                              │  (CNN weights)   │
                                              └─────────────────┘
```

### Inference Pipeline

1. **Upload** — User drops an MRI scan image via the browser interface
2. **Contour Cropping** — OpenCV isolates the brain region by detecting the largest contour after thresholding, erosion, and dilation
3. **Resize & Normalize** — Image is resized to `240×240×3` and pixel values are scaled to `[0, 1]`
4. **Prediction** — The CNN outputs a sigmoid probability; values `> 0.5` indicate a tumor
5. **Response** — JSON payload with `has_tumor` (bool), `confidence` (float), and `tumor_probability` (float)

---

## 🚀 Installation & Local Setup

### Prerequisites

- Python 3.10+
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/priyalshrivastava10/brain_tumor_detection.git
cd brain_tumor_detection

# 2. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the development server
python -m uvicorn app:app --port 8000 --reload
```

Open **[http://localhost:8000](http://localhost:8000)** in your browser.

---

## 📡 API Reference

### `POST /predict`

Upload an MRI image for tumor classification.

**Request**

```
Content-Type: multipart/form-data
Field: file (image/jpeg | image/png)
```

**Response** `200 OK`

```json
{
  "has_tumor": true,
  "confidence": 0.9993,
  "tumor_probability": 0.9993
}
```

| Field | Type | Description |
|---|---|---|
| `has_tumor` | `bool` | `true` if tumor probability > 0.5 |
| `confidence` | `float` | Model confidence in the predicted class (0–1) |
| `tumor_probability` | `float` | Raw sigmoid output for the tumor class |

**Error Responses**

| Code | Condition |
|---|---|
| `400` | Non-image file uploaded or image decode failure |
| `500` | Internal prediction error |

---

## 🧠 Model Details

### Architecture

A lightweight custom CNN built with the Keras Sequential API:

```
Input (240 × 240 × 3)
  │
  ├─► ZeroPadding2D (2, 2)
  ├─► Conv2D — 32 filters, 7×7 kernel, stride 1
  ├─► BatchNormalization (axis=3)
  ├─► ReLU Activation
  ├─► MaxPooling2D — pool 4×4, stride 4
  ├─► MaxPooling2D — pool 4×4, stride 4
  ├─► Flatten
  └─► Dense (1 neuron, sigmoid) → Binary output
```

### Dataset

[Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection) — 253 brain MRI images:

| Class | Samples |
|---|---|
| Tumorous (`yes`) | 155 |
| Non-Tumorous (`no`) | 98 |

### Preprocessing Pipeline

| Step | Operation | Purpose |
|---|---|---|
| 1 | Grayscale + Gaussian Blur | Reduce noise for contour detection |
| 2 | Binary Threshold → Erode → Dilate | Isolate brain region from background |
| 3 | Contour Detection → Crop | Extract brain ROI, remove black borders |
| 4 | Resize to `240×240` | Uniform input shape for CNN |
| 5 | Normalize to `[0, 1]` | Scale pixel values for stable training |

### Performance

| Metric | Validation Set | Test Set |
|---|---|---|
| **Accuracy** | 91% | 89% |
| **F1 Score** | 0.91 | 0.88 |

---

## 📂 Folder Structure

```
brain_tumor_detection/
├── app.py                  # FastAPI application — model loading, preprocessing, prediction endpoint
├── requirements.txt        # Python dependencies
├── models/
│   └── model.h5            # Pre-trained CNN weights (~170 KB)
├── templates/
│   └── index.html          # Main HTML page served at /
├── static/
│   ├── style.css           # Cyberpunk glassmorphism stylesheet (919 lines)
│   └── script.js           # Client-side upload, drag-drop, and result rendering logic
└── screenshots/            # UI screenshots for documentation
```

---

## ☁️ Deployment

The application is deployed on **Railway** with zero additional configuration required.

**Live URL:** [braintumordetection-production-1192.up.railway.app](https://braintumordetection-production-1192.up.railway.app/)

The server binds to `0.0.0.0` on the port provided by the `PORT` environment variable (defaults to `8000` locally), making it compatible with Railway, Render, and similar PaaS platforms out of the box.

---

## 👥 Team

| Member | Roll Number |
|---|---|
| **Priyal** | S24cseu2582 |
| **Astha Karanwal** | S24cseu2570 |

> Semester 4 Project

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">
  <strong>Contributions are welcome!</strong><br/>
  If you find this project useful, consider giving it a ⭐
</p>
