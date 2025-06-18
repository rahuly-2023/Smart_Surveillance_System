# 🛡️ Smart Surveillance System

A real-time Streamlit app for detecting potential threats like violence, distress gestures, and high-risk gender situations using webcam input and AI models.

---

## 🚀 Features

- **👤 Gender Detection** — Classifies faces as Male/Female.
- **✋ Hand Wave Detection** — Detects distress hand signals using MediaPipe.
- **🤜 Violence Detection** — Uses deep learning to detect violent activity from video.
- **📢 Alerts** — Shows real-time warnings and plays sound on threat detection.

---

## 🧠 Models Used

- `Gender_detection.keras`: CNN model to classify gender from faces.
- `Violence_detection.h5`: ConvLSTM model using a sequence of frames to detect violence.

---

## 🛠️ Tech Stack

- **Streamlit** — UI Framework
- **OpenCV** — Video processing
- **MediaPipe** — Hand landmark detection
- **TensorFlow/Keras** — Model inference

---

## 📁 Folder Structure

```
app/
├── main.py                  # Streamlit app
├── models/
│   ├── Gender_detection.keras
│   └── Violence_detection.h5
.gitignore
readme.MD
requirements.txt
```

> ⚠️ Note: Face detection uses OpenCV's built-in Haar Cascade.
> You do not need to manually include `haarcascade_frontalface_default.xml`.

---

## 📦 Installation

```bash
# Clone this repo
$ git clone https://github.com/yourusername/smart-surveillance-system
$ cd smart-surveillance-system

# Install dependencies
$ pip install -r requirements.txt

# Run the app
$ streamlit run app/main.py
```

> ✅ Allow webcam access in your browser when prompted.

---

## ✅ Accuracy

| Task                | Accuracy  |
|---------------------|-----------|
| Gender Detection    | 92.99%    |
| Violence Detection  | 91.59%    |

---

## 🎓 Credits

- **Team Members**: Kumar Chirag, Rahul Yadav, Naman Karara
- **Institution**: Netaji Subhas University of Technology
- **Research Base**: SUSAN: Deep Learning for Violence Detection Against Women (2024)

---

