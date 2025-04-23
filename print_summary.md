# 🔍 ComV - Crime Detection Models Documentation

This documentation outlines the structure, usage, and performance of the two core models used in the ComV system: **Fight Detection (3D CNN + Keras)** and **Object Detection (YOLOv8)**.

---

## 🥊 1. Fight Detection Model

### 📌 Overview
- **Architecture**: Custom 3D CNN using **Keras** with **TensorFlow backend**
- **Classification**: Binary (`fight` / `no fighting`)
- **Input Shape**: Resized to `(30, 96, 96, 3)`
- **Frame Sampling**: 50 frames per video extracted with OpenCV
- **Execution**: Fully local, runs in offline environments

---

### 📊 1.1 Performance Metrics

**Model:** `final_model_2.h5`

| Metric                    | Value    | Epoch |
|---------------------------|----------|-------|
| Best Training Accuracy    | 0.8583   | 7     |
| Best Validation Accuracy  | 0.9167   | 10    |
| Lowest Training Loss      | 0.3636   | 7     |
| Lowest Validation Loss    | 0.2805   | 8     |

---

### 📊 1.2 Dataset Composition

| Category       | Count | Percentage | Formats       | Avg Duration |
|----------------|-------|------------|---------------|--------------|
| Fight Videos   | 2,340 | 61.9%      | .mp4, .mpeg   | 5.2 sec      |
| Normal Videos  | 1,441 | 38.1%      | .mp4, .mpeg   | 6.1 sec      |
| **Total**      | **3,781** | **100%**  |               |              |

---

### ⚙️ 1.3 Technical Specifications

- **Resolution**: 64×64 pixels (resized)
- **Color Space**: RGB
- **Frame Rate**: ~30 FPS
- **Input Shape to Model**: `(30, 96, 96, 3)` (resized before feeding into the model)

---

### 📁 Key Files
- `project.py`: Main logic, includes `fight_detection()` function
- `Fight_detection_run.py`: Extracts and processes frames
- `func_Fight_detection.py`: Helper utilities
- `action/`: Trained model directory (`final_model_2.h5`)

---

## 🔫 2. Object Detection (YOLOv8)

### 📌 Overview

- **Model Type**: YOLOv8 (Ultralytics)
- **Use Case**: Real-time object/crime/weapon detection
- **Input**: RGB videos
- **Output**: Annotated video + list of detected crime-related objects
- **Interface**: Called via `Crime()` in `project.py`

---

### 📊 2.1 Detection Accuracy

| Metric       | Value | Interpretation                       |
|--------------|-------|----------------------------------------|
| mAP@0.5      | 0.80  | Excellent for standard object detection |
| mAP@0.5:0.95 | 0.50  | Needs tuning for small/fine objects     |
| Precision    | 0.80  | Low false positives                    |
| Recall       | 0.70  | Misses ~30% of relevant objects        |

---

### 📁 Key Files

- `project.py`: Calls `Crime()` for YOLO inference
- `yolo/best.pt`: Trained weights file (required)
- `Object_detection.py`: (Optional, modular structure)

---

## 🛠️ Environment & Deployment

- **Python**: 3.10
- **Execution**: Local environment via Conda or virtualenv
- **Interface**: Web interface via Django for video upload and analysis
- **Phase 2 Plan**: CCTV integration with real-time alerts and secure personnel access

---

## 👥 Users

- **Public Users**: Upload videos for detection
- **Security Personnel**: Receive alerts, monitor logs (Phase 2)

---

## 🧠 Limitations

### Fight Detection:
- Binary output only (no multi-action support yet)
- Doesn’t recognize weapons or group behaviors

### YOLO Object Detection:
- Requires `best.pt` to be added manually
- Struggles slightly with small or fast-moving objects

---

## 🧩 Future Work

- [ ] Real-time video processing from CCTV
- [ ] Behavior + object hybrid classification
- [ ] Night-time enhancement integration (Retinexformer)
- [ ] Web dashboard for personnel insights and alerts

