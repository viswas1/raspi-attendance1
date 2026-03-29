import os
import pickle
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.utils import ensure_directories

def get_hog_descriptor():
    """Returns a HOG descriptor configured for our face size."""
    # winSize, blockSize, blockStride, cellSize, nbins
    return cv2.HOGDescriptor(config.FACE_SIZE, (16, 16), (8, 8), (8, 8), 9)

def extract_features(face_roi):
    """Extracts normalized HOG features from a face ROI."""
    # Preprocessing: Resize and Grayscale
    face_resized = cv2.resize(face_roi, config.FACE_SIZE)
    face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
    
    # Lighting Normalization: Histogram Equalization
    face_equ = cv2.equalizeHist(face_gray)
    
    # Feature Extraction: HOG
    hog = get_hog_descriptor()
    features = hog.compute(face_equ)
    
    # Ensure it's a 1D array and L2 normalized
    features = features.flatten()
    norm = np.linalg.norm(features)
    if norm > 0:
        features = features / norm
        
    return features

def encode_faces():
    """Reads student images and generates robust HOG-based face encodings."""
    ensure_directories()
    
    # Initialize MediaPipe Face Detector
    base_options = python.BaseOptions(model_asset_path=config.MEDIAPIPE_MODEL_PATH)
    options = vision.FaceDetectorOptions(base_options=base_options)
    detector = vision.FaceDetector.create_from_options(options)
    
    known_encodings = []
    known_names = []
    
    students_dir = config.STUDENTS_DIR
    
    if not os.path.exists(students_dir):
        print(f"[ERROR] Directory not found: {students_dir}")
        return
        
    print("[INFO] Indexing student images using HOG features...")
    
    # Iterate over each student folder
    for student_name in os.listdir(students_dir):
        student_path = os.path.join(students_dir, student_name)
        
        # Skip if not a directory
        if not os.path.isdir(student_path):
            continue
            
        print(f"[INFO] Processing student: {student_name}")
        
        # Iterate over images in the student folder
        for filename in os.listdir(student_path):
            image_path = os.path.join(student_path, filename)
            
            # Simple check for image files
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            try:
                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    print(f"[WARNING] Could not read image: {image_path}")
                    continue
                    
                # Convert BGR (OpenCV) to RGB (MediaPipe)
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
                
                # Perform detection
                detection_result = detector.detect(mp_image)
                
                if detection_result.detections:
                    for detection in detection_result.detections:
                        # Get bounding box
                        bbox = detection.bounding_box
                        
                        top = bbox.origin_y
                        left = bbox.origin_x
                        width = bbox.width
                        height = bbox.height
                        
                        # Crop face ROI
                        face_roi = image[max(0, top):min(image.shape[0], top+height), 
                                         max(0, left):min(image.shape[1], left+width)]
                        
                        if face_roi.size == 0:
                            continue
                            
                        # Extract HOG features
                        encoding = extract_features(face_roi)
                        
                        known_encodings.append(encoding)
                        known_names.append(student_name)
                        
                        # Use all faces found in a calibration image (or just the first one)
                        break
                        
            except Exception as e:
                print(f"[ERROR] Failed to process {image_path}: {str(e)}")
    
    # Save the encodings
    print("[INFO] Saving robust encodings...")
    data = {"encodings": known_encodings, "names": known_names}
    
    with open(config.ENCODINGS_FILE, "wb") as f:
        pickle.dump(data, f)
        
    print(f"[SUCCESS] Saved {len(known_encodings)} robust encodings to {config.ENCODINGS_FILE}")

if __name__ == "__main__":
    encode_faces()
