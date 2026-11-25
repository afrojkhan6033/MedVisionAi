# 🩺 MedvisionAi — Unified Medical Imaging Platform

A full-stack AI-powered medical imaging analysis platform that unifies **four trained deep-learning models** into a single clinical workspace. Upload a scan, get instant predictions with explainability overlays.

> **⚠️ Research Tool Only** — Not a medical device. Do not use for clinical diagnosis.

---

## 🎯 Supported Analyses

| Task | Modality | Models | Classes | Status |
|------|----------|--------|---------|--------|
| **Alzheimer's Stage Detection** | Brain MRI | ResNet-50 + VGG-16 | Non/Very Mild/Mild/Moderate Demented | ⏳ Needs weights |
| **Brain Tumor Detection** | Brain MRI | Custom CNN | No Tumor / Tumor | ✅ Ready |
| **Pneumonia Detection** | Chest Scan | DenseNet-121 + VGG-16 | Normal / Pneumonia | ⏳ Needs weights |
| **COVID-19 Detection** | Chest Scan | ResNet-50 + VGG-16 | COVID / Normal / Pneumonia | ⏳ Needs weights |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    React + Tailwind UI                   │
│  ┌──────────┐  ┌─────────────────┐  ┌────────────────┐  │
│  │ Task     │  │  Scan Upload &  │  │  Results:      │  │
│  │ Sidebar  │  │  Viewer + XAI   │  │  Verdict +     │  │
│  │          │  │  Overlay Toggle │  │  Prob Bars +   │  │
│  │          │  │                 │  │  Consensus     │  │
│  └──────────┘  └─────────────────┘  └────────────────┘  │
└──────────────────────┬──────────────────────────────────┘
                       │  /api/v1/*
┌──────────────────────▼──────────────────────────────────┐
│                   FastAPI Backend                        │
│  ┌────────────┐  ┌──────────┐  ┌──────────────────────┐ │
│  │ Input      │  │ LRU      │  │ Pipeline Adapters    │ │
│  │ Guard      │→ │ Registry │→ │ (PyTorch / Keras)    │ │
│  │ (validate) │  │ (4 slots)│  │ + Grad-CAM / ROI     │ │
│  └────────────┘  └──────────┘  └──────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Backend

```bash
cd a:\Projects\MedvisionAi

# Install Python dependencies
pip install -r backend/requirements.txt

# Start the API server
python -m backend.main
```

The API runs at **http://localhost:8000** — interactive docs at **/docs**.

### 2. Frontend

```bash
cd frontend

# Install dependencies
npm install

# Start dev server
npm run dev
```

Opens at **http://localhost:5173** with API proxy to the backend.

## 📂 Project Structure

```
MedvisionAi/
├── backend/
│   ├── main.py              # FastAPI app — routes, lifespan, CORS
│   ├── config.py            # Paths, limits, CORS origins
│   ├── schemas.py           # Pydantic v2 response models
│   ├── input_guard.py       # Upload validation (magic bytes, size, PIL)
│   ├── registry.py          # Thread-safe LRU model registry
│   ├── requirements.txt     # Python dependencies
│   └── pipelines/
│       ├── base.py          # Abstract BasePipeline (run → canonical response)
│       ├── alzheimer.py     # PyTorch ResNet50+VGG16, Grad-CAM
│       ├── brain_tumor.py   # Keras CNN, contour crop, ROI overlay
│       ├── covid.py         # Keras ResNet50+VGG16, majority vote
│       └── pneumonia.py     # PyTorch DenseNet121+VGG16, Grad-CAM
├── frontend/
│   ├── src/
│   │   ├── App.tsx          # 3-zone clinical workspace layout
│   │   ├── api/             # Typed fetch client + TS interfaces
│   │   ├── hooks/           # useDiagnosis state machine
│   │   └── components/      # TopBar, Sidebar, Dropzone, Viewer, etc.
│   ├── package.json
│   ├── vite.config.ts       # Proxy /api → localhost:8000
│   └── tailwind.config.js
├── artifacts/               # Model manifests + weight files
│   ├── alzheimer/manifest.json
│   ├── brain_tumor/manifest.json + model.h5
│   ├── covid/manifest.json
│   └── pneumonia/manifest.json
└── README.md
```

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/health` | Server health, device info, loaded models |
| `GET` | `/api/v1/catalog` | All registered tasks, models, availability |
| `POST` | `/api/v1/predict/{task_id}` | Single-task inference on uploaded image |
| `POST` | `/api/v1/predict/panel/{modality}` | Run all available tasks for a modality |

## 🧠 Adding Model Weights

To enable a task, train the model and place the checkpoint files in the corresponding `artifacts/` directory:

```
artifacts/alzheimer/resnet50_best.pt    # PyTorch checkpoint
artifacts/alzheimer/vgg16_best.pt       # PyTorch checkpoint
artifacts/pneumonia/densenet121_best.pt # PyTorch checkpoint
artifacts/pneumonia/vgg16_best.pt       # PyTorch checkpoint
artifacts/covid/resnet50_best.h5        # Keras weights
artifacts/covid/vgg16_best.h5           # Keras weights
```

The registry auto-detects available weights at startup.

## 👤 Author

**Afroj Khan** — All four original detection models + unified platform.

## 📜 License

For research and educational purposes only.
