# VisionLabel AI — Zero-Shot Vision-Language Auto-Annotator

> An AI-powered image annotation suite that autonomously detects and annotates specified objects across $N$ images using fast Vision-Language (VL) retrieval and open-vocabulary prompts.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20Framework-lightgrey.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🌟 Key Features

- **Autonomous Zero-Shot Detection**: Type any natural language prompt (e.g., *"snake"*, *"taxi"*, *"traffic light"*, *"laptop"*) and let the model detect and label objects autonomously without task-specific training.
- **Batch Processing across $N$ Images**: Upload or drop up to 1,000 images and annotate all of them in a single click with real-time progress tracking.
- **Interactive Annotation Studio**:
  - **Direct Middle-Point Resizing**: Drag the middle point of any side (top, bottom, left, right) or any corner to adjust bounding boxes just like industry-standard tools (CVAT, LabelMe, Roboflow).
  - **Direct Edge Dragging**: Left-click and drag along any edge to resize that side directly.
  - **Move & Translate**: Left-click and drag inside the box to reposition.
  - **Smooth Zoom & Pan**: Infinite canvas with mouse wheel zoom and <kbd>Space</kbd>+drag panning.
  - **Manual Annotation Tool**: Draw new custom bounding boxes with the pencil tool.
- **Real-Time Sliders**: Adjust Cosine Similarity Threshold and Non-Maximum Suppression (NMS IoU) dynamically.
- **Multi-Format Dataset Export**:
  - **COCO 1.0 JSON**
  - **YOLO Format** (`labels/*.txt` + `data.yaml`)
  - **Custom JSON**
  - **ZIP Archive** of annotated images

---

## 🏗️ Architecture Pipeline

The system uses a two-stage Fast Retrieval Vision-Language pipeline:

```
  Text Prompts ("snake", "car")
               ↓
    [CLIP Text Encoder]
               ↓
     Cached Text Embedding
─────────────────────────────────────────────
          Input Image
               ↓
    [YOLO / MobileNetV3 FPN] ──> Class-Agnostic Proposals
               ↓
       [Embedding Head]      ──> Proposal Crop Embeddings (CLIP Vision)
               ↓
  [Fast Cosine Similarity Matching & NMS]
               ↓
       Final Bounding Boxes & Confidence Scores
```

1. **Prompt Cache**: Natural language queries are pre-encoded into joint VL embedding space.
2. **Proposal Generator**: Fast MobileNet-V3 FPN backbone generates candidate bounding box proposals with objectness scores.
3. **Embedding Head**: Crops are mapped into the joint 512-D Vision-Language space via CLIP.
4. **Fast Retrieval**: Rapid dot-product matrix multiplication matches visual proposals to query prompts with low latency (~1.4 ms retrieval).

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.9+ installed:
```bash
python --version
```

### 2. Clone the Repository
```bash
git clone https://github.com/robin1278/object-detection-app.git
cd object-detection-app
```

### 3. Install Dependencies
```bash
pip install torch torchvision transformers pillow numpy flask
```

### 4. Run the Application
Navigate to the app directory and start the server:
```bash
cd "FAST RETRIEVAL VL DETECTOR"
python start_app.py
```

The application will automatically start the server at `http://127.0.0.1:5000` and open it in your browser.

---

## ⌨️ Canvas Shortcuts & Controls

| Action | Control |
| :--- | :--- |
| **Resize Side** | Left-Click & Drag the **middle point** of any side |
| **Resize Corners** | Left-Click & Drag any of the 4 corner handles |
| **Move Box** | Left-Click & Drag inside the bounding box |
| **Rename Label** | Double-Click the bounding box or its label in the sidebar |
| **Delete Box** | Select box and press <kbd>Del</kbd> or <kbd>Backspace</kbd> |
| **Pan Canvas** | Hold <kbd>Space</kbd> and drag, or drag empty background |
| **Zoom** | Mouse Scroll Wheel |
| **Next / Prev Image** | <kbd>→</kbd> / <kbd>←</kbd> Arrow Keys |

---

## 📁 Project Structure

```
object-detection-app/
│
├── .gitignore                      # Git ignore rules (weights, caches, uploads)
├── README.md                       # Documentation
│
└── FAST RETRIEVAL VL DETECTOR/
    ├── app.py                      # Flask backend API & endpoints
    ├── model.py                    # Proposal Generator, CLIP Embedding & Retrieval Engine
    ├── start_app.py                # Launch script with pre-warming & browser opener
    ├── architecture.txt            # System architecture overview
    ├── problem_statement.txt       # Project problem statement
    │
    └── static/                     # Web Application Frontend
        ├── index.html              # Annotation Studio UI structure
        ├── css/
        │   └── style.css           # Premium dark-mode UI & vector canvas styling
        └── js/
            └── app.js              # Vector SVG canvas, direct handle resizing & API sync
```

---

## 📦 Export Formats

- **COCO 1.0**: Standard annotations JSON with `images`, `annotations` (`bbox: [x, y, w, h]`), and `categories`.
- **YOLO**: Normalized coordinates `<class_id> <x_center> <y_center> <width> <height>` per image, plus `data.yaml`.
- **JSON**: Comprehensive per-image detection metadata including similarity scores and objectness metrics.

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
