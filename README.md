# AI Visual Quality Inspection System

An end-to-end AI-powered visual inspection system for automated industrial defect detection. The system combines **ResNet18 image classification**, **YOLO11 defect localization**, **OpenCV image processing**, **defect analysis**, and **Grad-CAM explainability** to identify defective products and provide interpretable quality assessments.

---

## 🚀 Project Overview

Manual visual inspection in manufacturing can be time-consuming, inconsistent, and difficult to scale.

This project develops an AI-based quality inspection pipeline that analyzes product images and determines whether a product is **GOOD** or **DEFECTIVE**, while also attempting to localize the defect using an object detection model.

The pipeline is designed around the following workflow:

```text
Product Image
      ↓
OpenCV Preprocessing
      ↓
ResNet18 Classification
      ↓
GOOD / DEFECTIVE
      ↓
YOLO11 Defect Detection
      ↓
Defect Localization
      ↓
Grad-CAM Explainability
      ↓
Quality Assessment
      ↓
API Deployment
```

---

## ✨ Key Features

* Binary product classification using **ResNet18**
* Defect localization using **YOLO11**
* OpenCV-based image preprocessing
* Ground-truth defect mask analysis
* Automatic conversion of MVTec pixel masks into YOLO bounding-box annotations
* Defect area calculation
* CNN + detector-based quality assessment
* **Grad-CAM** visual explanations
* Structured inspection results
* Designed for FastAPI deployment
* Suitable for future Gradio and Docker deployment

---

## 🧠 Models Used

### ResNet18

A pretrained ResNet18 model is fine-tuned for binary classification:

```text
0 → GOOD
1 → DEFECTIVE
```

The classification model provides:

* Predicted class
* Classification confidence

### YOLO11

YOLO11 is used to localize defects within the product image.

The current detection task uses a single class:

```text
0 → defect
```

YOLO outputs:

* Bounding box
* Detection confidence
* Defect location

### Grad-CAM

Grad-CAM is used to visualize the regions of the image that contributed to the CNN's classification decision.

This provides an explainability layer for the classification model.

---

## 📊 Dataset

This project uses the **MVTec AD dataset**, specifically the:

```text
metal_nut
```

category.

MVTec AD provides industrial product images together with pixel-level anomaly masks.

For the YOLO detection component, the available pixel-level ground-truth masks were converted into bounding-box annotations.

### Dataset Structure

```text
metal_nut/
│
├── train/
│   └── good/
│
├── test/
│   ├── good/
│   ├── bent/
│   ├── color/
│   ├── flip/
│   └── ...
│
└── ground_truth/
    ├── bent/
    ├── color/
    ├── flip/
    └── ...
```

### Important Note

MVTec AD is primarily an **industrial anomaly detection and localization dataset**, not originally a YOLO object-detection dataset.

Therefore, the YOLO annotations in this project are derived from the original pixel-level ground-truth masks.

---

## 🔄 Data Processing Pipeline

The original MVTec masks follow a naming pattern such as:

```text
000_mask.png
```

while the corresponding test image is:

```text
000.png
```

The project maps the mask to its corresponding image and extracts the defect bounding box.

```text
Ground Truth Mask
       ↓
Binary Defect Region
       ↓
Find Non-Zero Pixels
       ↓
Calculate Bounding Box
       ↓
Normalize Coordinates
       ↓
YOLO Annotation
```

YOLO annotation format:

```text
class_id x_center y_center width height
```

All coordinates are normalized between `0` and `1`.

---

## 🏗️ Project Architecture

```text
                   ┌──────────────────────┐
                   │     Product Image    │
                   └──────────┬───────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │  OpenCV Processing   │
                   └──────────┬───────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
                    ▼                   ▼
          ┌─────────────────┐   ┌─────────────────┐
          │    ResNet18     │   │     YOLO11      │
          │ Classification  │   │ Defect Detection│
          └────────┬────────┘   └────────┬────────┘
                   │                     │
                   ▼                     ▼
            GOOD / DEFECTIVE       Defect Bounding Box
                   │                     │
                   └──────────┬──────────┘
                              ▼
                   ┌──────────────────────┐
                   │ Quality Assessment   │
                   └──────────┬───────────┘
                              │
                              ▼
                   ┌──────────────────────┐
                   │    Grad-CAM          │
                   │  Explainability      │
                   └──────────────────────┘
```

---

## 🛠️ Technologies

| Technology   | Purpose                                      |
| ------------ | -------------------------------------------- |
| Python       | Core programming language                    |
| PyTorch      | CNN training and inference                   |
| Torchvision  | ResNet18 architecture and pretrained weights |
| Ultralytics  | YOLO11 training and inference                |
| OpenCV       | Image processing                             |
| NumPy        | Numerical operations                         |
| Pandas       | Dataset analysis                             |
| Scikit-learn | Classification evaluation                    |
| Matplotlib   | Visualization                                |
| Seaborn      | Confusion matrix visualization               |
| Grad-CAM     | Model explainability                         |

## 📈 Model Evaluation

The project evaluates the classification model using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

The YOLO detector is evaluated using:

* Precision
* Recall
* mAP@50
* mAP@50-95


## 🔍 Explainability

Grad-CAM is applied to the final convolutional layer of ResNet18.

Example workflow:

```text
Input Image
     ↓
ResNet18
     ↓
Predicted Class
     ↓
Grad-CAM
     ↓
Activation Heatmap
     ↓
Important Image Regions
```

This helps visualize which regions influenced the classification decision.

---

## 📊 Quality Assessment

The system combines classification and detection results to produce an application-defined quality assessment.

Possible outcomes include:

```text
PASS
REVIEW
DEFECTIVE
```

Examples:

```text
GOOD + No YOLO Detection
        ↓
PASS
```

```text
DEFECTIVE + YOLO Detection
        ↓
DEFECTIVE
```

```text
DEFECTIVE + No YOLO Detection
        ↓
REVIEW
```

The quality score and thresholds are **application-specific** and are not intended to represent an industry-standard manufacturing quality specification.

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/ai-visual-quality-inspection.git
cd ai-visual-quality-inspection
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

Run the main notebook:

```text
notebooks/visual_quality_inspection.ipynb
```

The notebook covers:

1. Dataset loading
2. Dataset exploration
3. Image preprocessing
4. Data augmentation
5. ResNet18 training
6. Classification evaluation
7. YOLO dataset generation
8. YOLO11 training
9. Defect localization
10. Defect analysis
11. Quality assessment
12. Grad-CAM explainability

---

## 🌐 API Deployment

The project is designed to expose the trained models through a FastAPI service.

Planned API workflow:

```text
Client
  ↓
POST /predict
  ↓
FastAPI
  ↓
ResNet18 + YOLO11
  ↓
Inspection Result
  ↓
JSON Response
```

Example response:

```json
{
    "product_id": "sample.png",
    "classification": "DEFECTIVE",
    "classification_confidence": 0.98,
    "detections": [
        {
            "class_id": 0,
            "confidence": 0.91,
            "bbox": [120, 80, 340, 290]
        }
    ],
    "quality": {
        "quality_score": 42.5,
        "status": "DEFECTIVE"
    }
}
```

---

## 🔮 Future Improvements

* [ ] Complete FastAPI deployment
* [ ] Build Gradio web interface
* [ ] Add Docker deployment
* [ ] Add real-time camera inspection
* [ ] Add multiple defect classes
* [ ] Improve YOLO annotation strategy
* [ ] Add automated model versioning
* [ ] Add experiment tracking
* [ ] Add CI/CD pipeline
* [ ] Add production monitoring
* [ ] Add defect severity classification
* [ ] Add database-backed inspection history

---

## ⚠️ Limitations

This project is a portfolio-oriented prototype and has several limitations.

1. The current dataset focuses on the MVTec AD `metal_nut` category.
2. YOLO annotations are derived from pixel-level anomaly masks.
3. The current YOLO model uses a single `defect` class.
4. The quality score is an application-defined heuristic rather than an industrial standard.
5. The classification train/validation split is a supervised modeling baseline and should not be interpreted as the canonical MVTec AD evaluation protocol.
6. Real manufacturing deployment would require validation on production-specific data and domain-specific quality thresholds.

---

This project is a FastAPI-based API for inspecting product images and classifying them as either GOOD or DEFECTIVE.

## Project structure

- `api.py` — FastAPI application
- `models/` — folder for the trained model file
- `Vision_Inspection_System.ipynb` — training notebook with the inspection workflow

## Setup

1. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Train the model using the notebook or place a trained model in:

   ```text
   models/resnet18_best.pth
   ```

## Run the API

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Then open:

- http://localhost:8000/
- http://localhost:8000/health

## Prediction endpoint

### Upload an image

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_image.jpg"
```

### Example response

```json
{
  "label": "DEFECTIVE",
  "confidence": 0.9342,
  "probabilities": {
    "GOOD": 0.0658,
    "DEFECTIVE": 0.9342
  },
  "filename": "sample_image.jpg"
}
```

## Notes

- The API expects a trained model file named `resnet18_best.pth`.
- The notebook contains the training pipeline and dataset preparation logic.
- For production use, it is recommended to add validation, logging, and model versioning.
