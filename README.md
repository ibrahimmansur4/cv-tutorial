# 🔥 Complete Computer Vision Starter Repo (Python + OpenCV + VS Code)

This repository is a **ready-to-run starter kit** for beginners learning **Computer Vision** using Python and OpenCV.  
It includes a clean project structure, sample scripts, and installation files to help you start instantly.

---

## 🚀 Features

- ✔️ Fully isolated Python environment (venv-ready)  
- ✔️ `opencv-python`, `numpy`, `matplotlib` pre-configured  
- ✔️ Webcam live-feed script  
- ✔️ Basic image-processing example  
- ✔️ Clean project structure for easy scaling  
- ✔️ Perfect for tutorials, workshops, and Medium articles  

---

## 📁 Project Structure

```
cv_full_repo/
│
├── src/
│   ├── cv_demo.py              # Webcam: original + grayscale
│   └── image_processing.py     # Blur + edge detection
│
├── data/
│   └── test.jpg (add your own)
│
├── requirements.txt
└── README.md
```

---

## 🧰 Installation

### 1️⃣ Create & activate a virtual environment

```
python -m venv venv
venv\Scripts\activate
```

### 2️⃣ Install dependencies

```
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Run webcam demo:

```
python src/cv_demo.py
```

### Run image processing demo:

Add `test.jpg` to the `data/` folder, then run:

```
python src/image_processing.py
```

---

## 🖥️ Recommended Setup

Use **VS Code** with:

- Python extension  
- Pylance  
- Black formatter (optional)

Select interpreter:

```
Ctrl + Shift + P → Python: Select Interpreter → venv/Scripts/python.exe
```

---

## ⭐ Next Steps (Level Up Your CV Skills)

You can expand this repo into:

- Object detection (YOLOv11, YOLO-NAS)  
- Face detection (Haar cascades + DNN models)  
- Image classification with CNNs  
- Real-time gesture recognition  
- Robotics applications (OpenCV + ROS)  

Want me to generate a full **YOLO-ready starter kit**?  
Just ask! 🔥

---

## 👨‍💻 Author  
Built by someone passionate about **Computer Vision, Robotics, and Embedded Systems**.

Happy coding!
