## 🌐 Live Demo  

[![Hugging Face Spaces](https://img.shields.io/badge/🚀%20Try%20Demo-HuggingFace-blue?logo=huggingface)](https://huggingface.co/spaces/demodemodemo123/Pneumonia-Detector)


---

# 🩺 Pneumonia Detection from Chest X-Rays

A deep learning project that leverages **Convolutional Neural Networks (CNNs)** to automatically detect **Pneumonia** from chest X-ray images.
Our model is trained on a publicly available dataset and achieves an impressive **97.12% accuracy** on the test set.

---

## 🚀 Features

✅ Preprocessing and augmentation of X-ray images
✅ Convolutional Neural Network (CNN) for classification
✅ Achieved **97.12% accuracy** on test dataset
✅ Training & evaluation pipeline included
✅ Works on **Google Colab** and **local environments**

---

## 📊 Dataset

* Chest X-ray images (Normal vs Pneumonia)
* Preprocessing applied: resizing, normalization, and data augmentation (flip, rotation, zoom, etc.)
* Balanced dataset for robust training

---

## 🏗️ Model Architecture

* **Input Layer** → Preprocessed chest X-ray images
* **Convolutional + Pooling Layers** → Feature extraction
* **Fully Connected Layers** → Classification
* **Output Layer** → Binary prediction: `Normal ✅` or `Pneumonia ⚠️`

---

## 📈 Results

* **Accuracy:** `97.12%`
* **Evaluation Metrics:** Loss curve & Accuracy curve (visualized in the notebook)

---

## 📷 Example Predictions

| Image                 | Prediction   |
| --------------------- | ------------ |
| Normal Chest X-ray    | ✅ Normal     |
| Pneumonia Chest X-ray | ⚠️ Pneumonia |

---

## ⚡ How to Run

### 🔹 On Google Colab

1. Open the notebook in Google Colab
2. Upload dataset or mount Google Drive
3. Run all cells to train & test the model

### 🔹 On Local Machine

```bash
# Clone this repository
git clone https://github.com/Aan9758/pneumonia-detection.git

# Navigate to project folder
cd pneumonia-detection

# Install dependencies
pip install -r requirements.txt

# Run the notebook / script
jupyter notebook pneumonia_detection.ipynb
```

---

## 🛠️ Tech Stack

* **Python** 🐍
* **TensorFlow / Keras** 🤖
* **OpenCV** 🔍
* **Matplotlib / Seaborn** 📊

---

## 📌 Future Improvements

* Deploy model as a **web app / mobile app**
* Use **transfer learning** with pretrained CNNs (ResNet, VGG16, EfficientNet)
* Expand dataset for higher generalization

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork this repo and submit a pull request.

---

## 📜 License

This project is licensed under the **MIT License** – you’re free to use, modify, and distribute.

---
