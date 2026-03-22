import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms

#paths
BASE_DIR = Path(__file__).parent
model_path = BASE_DIR / "model" / "plantmodel.pth"
data_dir = BASE_DIR / "data" / "PlantVillage" / "PlantVillage"

#load class names
from torchvision import datasets
dataset = datasets.ImageFolder(data_dir)
class_names = dataset.classes

#transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),                                 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#load model
model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features,len(class_names))

model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

#prediction function
def predict(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(image)
        _,predicted = torch.max(outputs,1)
    return class_names[predicted.item()]

if __name__ == '__main__':
    test_image = input("Enter the path of the image to predict: ").strip().strip('"').strip("'")
    if Path(test_image).is_file():
        result = predict(test_image)
        print(f"The predicted class is: {result}")
    else:
        print("Invalid image path.")