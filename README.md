# Vision Inspection System

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
