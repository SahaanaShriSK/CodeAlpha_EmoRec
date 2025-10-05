# 🗣️ Emotion Recognition from Speech using Deep Learning

This project recognizes human emotions such as **Anger**, **Happiness**, **Sadness**, **Fear**, **Disgust**, **Boredom**, and **Neutral** from speech audio.  
It uses **MFCC (Mel-Frequency Cepstral Coefficients)** as acoustic features and a **1D Convolutional Neural Network (CNN)** for classification.

---

## 🚀 Project Overview

| Task | Details |
|------|----------|
| **Objective** | Detect emotions from speech recordings |
| **Dataset** | [Berlin Database of Emotional Speech (EmoDB)](https://www.kaggle.com/datasets/piyushagni5/berlin-database-of-emotional-speech-emodb) |
| **Features** | MFCCs + Data Augmentation (Time Stretching, Pitch Shifting) |
| **Model** | CNN with Batch Normalization & Dropout |
| **Framework** | TensorFlow / Keras |
| **Environment** | Google Colab (Kaggle Dataset API integrated) |

---


## 🧠 Model Architecture

- **Input:** 120 MFCC-based feature values (augmented and concatenated)
- **Layers:**
  - Conv1D (64 filters, ReLU, BatchNorm, MaxPooling)
  - Conv1D (128 filters, ReLU, BatchNorm, MaxPooling)
  - Dense (128 units, Dropout)
  - Output: Softmax (7 emotion classes)

---

## 🧰 Installation

Run this in **Google Colab** (recommended) or locally with Python 3.10+.

```bash
# Clone this repository
git clone https://github.com/<your-username>/emotion-recognition.git
cd emotion-recognition

# Install dependencies
pip install librosa tensorflow numpy pandas scikit-learn matplotlib kagglehub

```
---

## 📈 Results

| Metric                          | Score               |
|---------------------------------|---------------------|
| **Training Accuracy**           | ~73%                |
| **Validation Accuracy**         | ~66%                |
| **Test Accuracy (after tuning)**| ~75–80% possible    |

---

## 👩‍💻 Contact:
Sahaana Shri S K

📧 sahaanashrisk@gmail.com  
🌐 [GitHub Profile](https://github.com/SahaanaShriSK)

