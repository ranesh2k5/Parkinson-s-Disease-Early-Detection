#  Parkinson's Early Detection System

> *Using AI to catch Parkinson’s before it catches you.*

---

##  TL;DR

A dual-model, AI-powered system that predicts early signs of **Parkinson’s Disease** using:

 Voice recordings (via CNN)  
 Hand tremor sensor data (via DNN)  
 Smart fusion for high-accuracy, real-world-ready results.

---

##  Table of Contents

- [ Project Highlights](#-project-highlights)  
- [ Tech Stack](#️-tech-stack)
- [ Hardware ](#️-hardware)
- [ Getting Started](#-getting-started)  
- [ How to Use](#-how-to-use)  
- [ Results](#-results)  
- [ Model Insight](#-model-insight)  
- [ Demo](#-demo)  
- [ License](#-license)  
- [ Contact](#-contact)  

---

##  Project Highlights

This project is built for **early diagnosis** of Parkinson’s — with **multimodal AI**

- CNN-based voice analysis  
- DNN-based IMU tremor classification  
- Fused predictions = smarter decisions  
- Modular design — drop in your own models or datasets  
- CLI ready, research ready, demo ready

---

##  Tech Stack

Built with a no-nonsense combo of ML and signal processing libs:

- Python 3.x  
- TensorFlow / Keras  
- Scikit-learn  
- NumPy, Pandas  
- Librosa 🎵 (voice features)  
- OpenCV + SciPy (signal magic)  
- Matplotlib (for visuals that actually matter)  
- Jupyter Notebook (exploration mode)

---

---

## Hardware

- **ESP32**: The core microcontroller, chosen for its low-latency wireless communication and onboard processing capabilities.

- **MPU9250**: A 9-axis IMU sensor used to capture **hand tremor data** — includes accelerometer, gyroscope, and magnetometer.

- **Mounting**: The ESP32 and MPU9250 are securely mounted on a **custom 3D-printed frame**, designed to sit snugly on a patient’s hand for stable readings.

- **BOYA BY-M1 Lavalier Microphone**: A high-quality, omnidirectional mic used to collect **clean voice samples** for CNN-based analysis.

Together, this setup ensures reliable, high-resolution signal acquisition — vital for both training and real-time inference.

<p align="center">
  <img src="path/to/hardware.png" width="450" alt="Hardware Setup">
</p>

---

##  Getting Started

Clone it, set it up and just run it.

```bash
# 1. Clone the repo
git clone https://github.com/ranesh2k5/Parkinson-s-Disease-Early-Detection
cd ParkinsonsEarlyPrediction

# 2. (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 3. Prepare your data:
# └── data/
#     ├── voice/
#     └── imu/
```

---

##  How to Use

Run the pipeline:

```bash
python pipeline.py
```

Want voice-only? Just comment out the tremor block. Want to tweak weights? Do it in `pipeline.py`.  
Everything’s in plain sight.

---

## 📊 Results

| Input Type     | Model     | Accuracy |
|----------------|-----------|----------|
| Voice          | CNN       | **88%**  |
| Voice + Tremor | CNN + DNN | **91%**  |


---

##  Model Insight

 **Final Prediction = 60% Voice + 40% Tremor**

| Example | Voice Says |    Tremor Says     |   Final Call  |
|-------|-------------|-------------------|-----------------------|
| 1️⃣    | Parkinson's | Parkinson's       | ✅ Parkinson's (85.6%) |
| 2️⃣    | Healthy     | Parkinson's (56%) | ✅ Healthy (26.8%)     |
| 3️⃣    | Healthy     | Parkinson's (70%) | ✅ Healthy (33.7%)     |

Why? Because voice changes show up earlier and are more stable.  
Weighting is **medically inspired**, not just data-driven.

---

##  Sample Output

<p align="center"> 
<img src="media/image.png" width="300"> 
<img src="media/Healthy2.png" width="300"> 
<img src="media/Healthy.png" width="300"> 
</p>

Crystal-clear visual breakdowns of model outputs.


##  License

MIT. Use it, modify it, remix it. Just don’t claim you built it.

---

## 📬 Contact

**Ranesh Prashar**  
 raneshprashar140@gmail.com  
 [GitHub](https://github.com/ranesh2k5)
