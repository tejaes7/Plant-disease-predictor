# 🌿 Plant Disease Classifier

A deep learning web app that classifies plant leaf diseases from images. Built with PyTorch and FastAPI, this project lets you upload a photo of a plant leaf and instantly get a prediction of what disease (if any) it has.

---

## 💡 About the Project

I built this project to explore how deep learning can be applied to real-world agricultural problems. The idea is simple — farmers and gardeners often struggle to identify plant diseases early. With this tool, you can just take a photo of a leaf and get an instant diagnosis.

I trained a ResNet-18 model on the PlantVillage dataset, then wrapped it in a FastAPI backend and built a clean frontend so anyone can use it without touching any code.

---

## 🖼️ Screenshots

![Home Page](screenshots/Screenshot%202026-03-22%20205207.png)
![Prediction 1](screenshots/Screenshot%202026-03-22%20203556.png)
![Prediction 2](screenshots/Screenshot%202026-03-22%20203647.png)

---

## 🌱 Classes It Can Predict

The model can identify the following 15 plant conditions with 97% accuracy:

| # | Class |
|---|-------|
| 1 | Pepper Bell - Bacterial Spot |
| 2 | Pepper Bell - Healthy |
| 3 | Potato - Early Blight |
| 4 | Potato - Healthy |
| 5 | Potato - Late Blight |
| 6 | Tomato - Bacterial Spot |
| 7 | Tomato - Early Blight |
| 8 | Tomato - Healthy |
| 9 | Tomato - Late Blight |
| 10 | Tomato - Leaf Mold |
| 11 | Tomato - Septoria Leaf Spot |
| 12 | Tomato - Spider Mites (Two Spotted Spider Mite) |
| 13 | Tomato - Target Spot |
| 14 | Tomato - Tomato Mosaic Virus |
| 15 | Tomato - Tomato Yellow Leaf Curl Virus |

---

## 🛠️ Tech Stack

- **Model:** ResNet-18 (PyTorch)
- **Dataset:** PlantVillage
- **Backend:** FastAPI
- **Frontend:** HTML, CSS, JavaScript
- **Server:** Uvicorn

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone <your-repo-url>
cd Image-classifierproject
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

> For PyTorch with CUDA support, install from [https://pytorch.org](https://pytorch.org) based on your GPU/CUDA version.

**3. Make sure the model file is in place**
```
model/plantmodel.pth
```

**4. Start the server**
```bash
python -m uvicorn app:app --reload
```

**5. Open your browser**
```
http://127.0.0.1:8000
```

---

## 📁 Project Structure

```
Image-classifierproject/
├── app.py                  # FastAPI backend
├── requirements.txt        # Python dependencies
├── static/
│   └── index.html          # Frontend UI
├── model/
│   └── plantmodel.pth      # Trained model weights
└── data/
    └── PlantVillage/
        └── PlantVillage/   # Dataset folders (15 classes)
```

---

## 📌 Notes

- The model was trained on the PlantVillage dataset and works best with clear, close-up photos of individual leaves.
- Currently supports pepper, potato, and tomato plants.
- GPU is used automatically if available, otherwise falls back to CPU.
