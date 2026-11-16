# Focus Mode AI

**Your phone is distracting you. Let's catch it in the act. **

Focus Mode AI is a fun computer vision project that uses your laptop webcam to detect your phone and explore whether you're interacting with it while working.


## What It Does

- 📷 **Webcam Detection** — Uses your laptop's built-in webcam.
- 📱 **Phone Detection** — Uses YOLO11 to detect phones in real time.
- ✋ **Hand Detection** — Uses MediaPipe to detect hand landmarks.
- 🤝 **Hand-Phone Interaction** — Experiments with detecting when your hand gets close to your phone.
- 🔔 **Distraction Alerts** — Working towards visual and audio alerts when phone usage.

🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| OpenCV | Webcam access and image processing |
| YOLO11n | Phone object detection |
| MediaPipe | Hand landmark detection |
| ByteTrack | Object tracking experiments |
| Pygame | Audio alerts |
| PyInstaller | Packaging the application |

## Getting Started

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
cd focus-mode-fun-ai
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the project

```bash
python phone_hand_detection_2.py
```

Make sure your webcam is connected and allow camera access when prompted.


## Why I Built This

I am using this project to learn computer vision by building something practical (and hopefully useful!).

Along the way, I'm exploring object detection, model training, image processing, and real-time tracking — learning one feature and one bug at a time.

**Built with curiosity, Python, and probably too many distractions. ☕**

---

If you find this project interesting, feel free to explore, give feedback, or contribute!
