# Real-Time ASL Recognition Using Vision Transformer (ViT)

## 📌 Project Description
This project presents a real-time American Sign Language (ASL) recognition system built using the **Vision Transformer (ViT)** model. It classifies 26 alphabetical hand gestures from ASL with strong generalization on unseen data and integrates with a webcam-enabled web application for real-time usage.

---

## 🔬 Methodology & Key Findings

- **Dataset**: ASL Alphabet from Kaggle (26 classes, 1530 train + 270 test images per class).
- **Preprocessing**: Grayscale conversion and resizing to 224x224.
- **Model**: ViT-Tiny trained using tuned hyperparameters (lr=0.1, weight_decay=5e-5, drop_rate=0.05).
- **Training Strategy**:
  - Hyperparameter tuning with 10 trials (25% data used)
  - Best validation accuracy: **99.80%**
  - Best test accuracy: **85.00%**
- **Key Finding**: ViT generalizes better than CNN/LSTM on unseen data, showing high accuracy and reduced overfitting.

---

## 🚀 Real-Time Application

- Built a **Flask web app** for real-time ASL recognition.
- User can show hand signs through a webcam.
- The system captures the image → feeds it to the ViT model → displays predicted class with confidence.
- Useful in practical scenarios where sign language interpreters aren't available.
