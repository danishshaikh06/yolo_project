# yolo_project
# YOLO Exam Proctoring Project

This repository contains Python prototypes for webcam-based exam proctoring using OpenCV, MediaPipe, and YOLO models. The scripts detect different kinds of potentially suspicious behavior such as head movement, eye/gaze movement, body movement, phone presence, and posture changes.

## Features

- Head pose detection using YOLO person detection, MediaPipe Face Mesh, and OpenCV `solvePnP`
- Eye movement and blink tracking using MediaPipe face landmarks and iris landmarks
- Body/posture monitoring using MediaPipe Pose
- Phone detection using YOLO object detection with configurable alert zones
- Voice/audio alert prototype for head movement and phone detection
- Memory-optimized YOLO training helper for low-VRAM systems

## Repository Contents

| File | Purpose |
| --- | --- |
| `headmovement_prototype.py` | Detects whether the user is looking left, right, up, down, or at the screen. |
| `headmovementsound_prototype.py` | Head movement detector with text-to-speech voice feedback. |
| `eyemovement.py` | Tracks gaze direction, pupil/iris landmarks, and blink count. |
| `body_movement.py` | Monitors body movement, reaching, turning, and fidgeting after calibration. |
| `suspicious.py` | More advanced posture and suspicious behavior monitor with statistical calibration. |
| `phonedetection.py` | Detects phones with YOLO and triggers alerts inside defined camera zones. |
| `training.py` | Trains a YOLO model with memory-optimized settings for smaller GPUs. |
| `requirements.txt` | Base dependency list. |
| `*.pt` | YOLO model weights used by the detection and training scripts. |
| `shape_predictor.dat` | Face landmark model asset included in the repo. |

## Requirements

- Python 3.9 or newer
- Webcam
- Windows, macOS, or Linux with camera access
- Optional NVIDIA GPU for training or faster inference

The current `requirements.txt` includes the core OpenCV, MediaPipe, NumPy, SciPy, and Matplotlib packages. Some scripts also import extra packages that may need to be installed manually:

```bash
pip install ultralytics torch pygame pyttsx3
```

## Installation

Clone the repository:

```bash
git clone https://github.com/danishshaikh06/yolo_project.git
cd yolo_project
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install ultralytics torch pygame pyttsx3
```

## Usage

Run any prototype directly with Python. Most scripts open the default webcam at camera index `0`.

### Head Movement Detection

```bash
python headmovement_prototype.py
```

Controls:

- `q`: quit

### Head Movement With Voice Alerts

```bash
python headmovementsound_prototype.py
```

Controls:

- `q`: quit
