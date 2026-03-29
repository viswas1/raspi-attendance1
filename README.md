# AI Attendance System - Knowledge Base & AI Training Guide

This README is explicitly designed to teach an AI agent or a developer about the architecture, execution flow, and deep mechanics of the AI Attendance System.

## 1. Project Overview & Purpose

The **AI Attendance System** is a real-time, computer vision-based application designed to automatically log student attendance using facial recognition. 

It accomplishes this through a two-part system:
1. **Real-time Recognition Engine:** Uses OpenCV to capture webcam feeds and the `face_recognition` library to identify faces against a pre-encoded database.
2. **Interactive Dashboard:** A Streamlit web application that provides a real-time view of attendance logs, summary metrics, and CSV export functionality.

## 2. Technology Stack
- **Python 3.x**: Core language.
- **OpenCV (`cv2`)**: Used for webcam video capture and drawing bounding boxes/labels on the video frame.
- **`face_recognition` (built on dlib)**: Used for extracting face bounding boxes (using HOG model) and generating/comparing 128-dimensional face encodings.
- **Streamlit**: Used to build the reactive web dashboard.
- **Pandas**: Used for reading, filtering, and manipulating attendance CSV data in the dashboard.

## 3. Project Structure

```text
attendance-system/
│── main.py               # Main entry point (Webcam + Recognition loop)
│── app.py                # Streamlit Web Dashboard
│── config.py             # Global constants and tunable thresholds
│── requirements.txt      # Python dependencies
│
├── data/
│   ├── students/         # Raw images organized by student name (e.g., students/John_Doe/img.jpg)
│   └── encodings.pkl     # Serialized dictionary of {"encodings": [...], "names": [...]}
│
├── src/
│   ├── face_encoder.py   # Script to parse `data/students/` and generate `encodings.pkl`
│   ├── face_recognizer.py# Core face detection, encoding, and distance comparison logic
│   ├── attendance.py     # Session state management and CSV appending logic
│   └── utils.py          # Helper functions (e.g., ensuring directories exist)
│
└── logs/
    └── attendance.csv    # Generated attendance records (Name, Date, Time)
```

## 4. Deep Dive: Key Components

### `config.py`
The central source of truth for paths and tunable parameters:
- `TOLERANCE` (0.5): The threshold for face matching distance. Lower means stricter matching.
- `FRAME_RESIZE_FACTOR` (0.25): Downscales the video frame before processing to significantly speed up detection.
- `PROCESS_ALTERNATE_FRAMES` (True): Optimization toggle to process face recognition on every *other* frame to save CPU cycles.

### `src/face_encoder.py` (`encode_faces`)
- **Purpose**: Pre-processes raw images into mathematical embeddings.
- **Logic**: Iterates over subdirectories in `data/students/`. Uses OpenCV to load images, converts BGR to RGB, finds face locations using the standard "hog" model, computes the 128-d encodings, and serializes the arrays into `data/encodings.pkl` using `pickle`.

### `src/face_recognizer.py` (`FaceRecognizer` class)
- **Purpose**: Compares real-time frames against known encodings.
- **Logic**:
  - Initializes by loading `encodings.pkl`.
  - `recognize_faces(frame)`:
    - Resizes the frame and converts to RGB.
    - Extracts face locations and encodings for the current frame.
    - Compares each detected face to known encodings using `face_recognition.compare_faces` and `face_distance`.
    - Selects the best match (lowest distance) if it falls within the `TOLERANCE` config.
    - Scales the bounding box coordinates back to the original frame size.
  - Returns a list of tuples: `(name, (top, right, bottom, left))`.

### `src/attendance.py` (`AttendanceManager` class)
- **Purpose**: Safely logs attendance to a CSV file and prevents duplicate entries.
- **Logic**:
  - Ensures the CSV (`attendance.csv`) and headers exist.
  - Maintains an in-memory `session_marked` set to track who has already been logged during the *current script execution*.
  - `mark_attendance(name)` skips "Unknown" faces and individuals already in `session_marked`. If new, it appends the Name, Date (YYYY-MM-DD), and Time (HH:MM:SS) to the CSV.

### `main.py`
- **Purpose**: The orchestration loop.
- **Logic**:
  - Initializes the camera (`cv2.VideoCapture(0)`).
  - Enters a continuous `while True` loop to grab frames.
  - Implements the alternate frame logic to decide whether to process the current frame.
  - Passes the frame to `FaceRecognizer`.
  - Passes recognized names to `AttendanceManager`.
  - Draws visual feedback (bounding boxes and names) on the frame using OpenCV.
  - Displays the frame and listens for the 'q' key to gracefully shutdown.

### `app.py`
- **Purpose**: The Streamlit user interface for viewing data.
- **Logic**:
  - `load_data()` safely reads the CSV (handling empty file cases).
  - Uses `st.sidebar` for controls (auto-refresh toggle, date filters).
  - Calculates summary metrics (Total Records, Today's Attendance, Unique Students).
  - Renders a data table and provides a CSV download button.
  - Uses `st.rerun()` paired with `time.sleep(2)` if auto-refresh is enabled.

## 5. Execution & Data Flow
1. **Setup Phase**: Developer places student images in `data/students/` and runs `src/face_encoder.py`. This transforms visual payload into a lightweight, searchable index (`encodings.pkl`).
2. **Recognition Phase**: Developer runs `main.py`. The webcam initializes. 
3. **Pipeline (Per Frame)**:
   - Frame -> Resized & RGB Converted -> HOG Face Detection -> 128-d Encoding -> Distance Calculation vs `encodings.pkl` -> Name Classification.
4. **Logging Phase**: If a classified name is not in the current session's memory set, it is appended to `logs/attendance.csv`.
5. **Monitoring Phase**: User views `app.py` in the browser. Streamlit continuously polls `logs/attendance.csv` and updates the UI metrics.

## 6. Notes for AI Finetuning & Enhancement
If you are an AI agent analyzing this codebase to suggest improvements, consider the following vectors:
- **Model Upgrades**: The current system uses "hog" for face detection which is CPU-friendly but fails on profile faces or poor lighting. Upgrading to a CNN model (`model="cnn"`) in `face_recognition` would improve accuracy if a GPU is available.
- **Database Scalability**: Transitioning `attendance.csv` to SQLite or PostgreSQL for robust querying if the dataset grows large.
- **Session Persistence**: Currently, restarting `main.py` resets the `session_marked` set, potentially allowing duplicates across multiple script restarts purely separated by time. State should ideally be checked against the database for the current date.
- **Liveness Detection**: The current system might be spoofed by presenting a photograph to the webcam. Adding anti-spoofing logic would make it production-ready.
