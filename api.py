from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import models, transforms

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODEL_DIR / "resnet18_best.pth"

app = FastAPI(
    title="Vision Inspection API",
    version="1.0.0",
    description="API for classifying metal nut images as GOOD or DEFECTIVE.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_model(model_path: Path = MODEL_PATH):
    """Load the trained inspection model if it exists."""
    if not model_path.exists():
        return None

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


app.state.model = load_model()


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    image = Image.open(__import__('io').BytesIO(image_bytes)).convert("RGB")
    return IMAGE_TRANSFORM(image).unsqueeze(0).to(device)


@app.get("/health")
def health_check() -> dict[str, Any]:
    return {
        "status": "ok",
        "device": str(device),
        "model_loaded": app.state.model is not None,
        "model_path": str(MODEL_PATH),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    if app.state.model is None:
        raise HTTPException(
            status_code=500,
            detail="No trained model found. Train the model in the notebook and save it at models/resnet18_best.pth.",
        )

    if not file.content_type or "image" not in file.content_type:
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    image_bytes = await file.read()

    try:
        tensor = preprocess_image(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(exc)}") from exc

    with torch.no_grad():
        logits = app.state.model(tensor)
        probabilities = torch.softmax(logits, dim=1)
        predicted_index = int(torch.argmax(probabilities, dim=1).item())
        confidence = float(probabilities[0, predicted_index].item())

    label = "DEFECTIVE" if predicted_index == 1 else "GOOD"

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "probabilities": {
            "GOOD": round(float(probabilities[0, 0].item()), 4),
            "DEFECTIVE": round(float(probabilities[0, 1].item()), 4),
        },
        "filename": file.filename,
    }


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "Vision Inspection API is running."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
