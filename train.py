import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader,random_split
import os
import random
from pathlib import Path
import numpy as np
from tqdm import tqdm

#reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


#paths
BASE_DIR = Path(__file__).parent
data_dir = BASE_DIR / "data" / "PlantVillage" / "PlantVillage"
model_path = BASE_DIR / "model" / "plantmodel.pth"

#transform
from torchvision.models import resnet18, ResNet18_Weights

weights = ResNet18_Weights.DEFAULT



from torchvision import transforms

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

if __name__ == '__main__':
    model = resnet18(weights=weights)

    #load dataset
    full_dataset = datasets.ImageFolder(data_dir)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_indices, test_indices = random_split(range(len(full_dataset)), [train_size, test_size])

    train_dataset = datasets.ImageFolder(data_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(data_dir, transform=test_transform)

    train_data = torch.utils.data.Subset(train_dataset, train_indices)
    test_data = torch.utils.data.Subset(test_dataset, test_indices)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    #model
    for par in model.parameters():
        par.requires_grad = False
    for par in model.layer3.parameters():
        par.requires_grad = True
    for par in model.layer4.parameters():
        par.requires_grad = True

    num_classes = len(full_dataset.classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    #training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam([
        {"params": model.layer3.parameters(), "lr": 1e-4},
        {"params": model.layer4.parameters(), "lr": 1e-4},
        {"params": model.fc.parameters(), "lr": 1e-3},
    ])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    #train loop
    print(f"Training on: {device}")
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            loop.set_postfix(loss=running_loss/len(train_loader))

        scheduler.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

    #saving model
    os.makedirs("model", exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print("Model trained and saved successfully!")

    #testing
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy}%")


