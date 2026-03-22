from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import resnet18
from pathlib import Path
import io

app = FastAPI()

# Paths
BASE_DIR = Path(__file__).parent
model_path = BASE_DIR / "model" / "plantmodel.pth" / "PlantVillage"

# Class names (from PlantVillage dataset)
class_names = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___healthy",
    "Potato___Late_blight",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_mosaic_virus",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
]

# Transform
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# Load model
checkpoint = torch.load(model_path, map_location="cpu")
num_classes = checkpoint["fc.weight"].shape[0]
model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(checkpoint)
model.eval()

# Prediction function
def predict(image: Image.Image):
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        _, pred = torch.max(outputs, 1)

    return class_names[pred.item()]

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# API routes
@app.get("/")
def root():
    return FileResponse("static/index.html")

@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    prediction = predict(image)

    return {"prediction": prediction}