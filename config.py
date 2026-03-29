import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
STUDENTS_DIR = os.path.join(DATA_DIR, "students")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
SRC_DIR = os.path.join(BASE_DIR, "src")

# Files
ENCODINGS_FILE = os.path.join(DATA_DIR, "encodings.pkl")
ATTENDANCE_LOG_FILE = os.path.join(LOGS_DIR, "attendance.csv")
MEDIAPIPE_MODEL_PATH = os.path.join(DATA_DIR, "face_detector.tflite")

# Face Recognition Settings
TOLERANCE = 0.5  # For legacy face_recognition
MATCH_THRESHOLD = 0.6  # For Euclidean distance (lower is better)
FRAME_RESIZE_FACTOR = 0.5  # Resize frame for faster face detection
PROCESS_ALTERNATE_FRAMES = True  # Process every Nth frame
FRAME_SKIP = 2  # Skip every N frames
FACE_SIZE = (100, 100)  # Size to resize cropped face for encoding
MAX_FACES_PER_FRAME = 2  # Limit number of faces processed

# Networking Settings
STREAM_HOST = "0.0.0.0"
STREAM_PORT = 8000
PI_IP_ADDRESS = "localhost"  # Change to your Pi's IP when running on laptop
