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

class FaceRecognizer:
    def __init__(self):
        """Initialize the FaceRecognizer, MediaPipe Face Detection, and load encodings."""
        self.known_face_encodings = []
        self.known_face_names = []
        
        # Initialize MediaPipe Face Detection (Tasks API)
        base_options = python.BaseOptions(model_asset_path=config.MEDIAPIPE_MODEL_PATH)
        options = vision.FaceDetectorOptions(base_options=base_options)
        self.detector = vision.FaceDetector.create_from_options(options)
        
        self._load_encodings()
        
    def _load_encodings(self):
        """Load known face encodings from the pickle file."""
        if not os.path.exists(config.ENCODINGS_FILE):
            print(f"[WARNING] Encodings file not found: {config.ENCODINGS_FILE}")
            return
            
        try:
            with open(config.ENCODINGS_FILE, "rb") as f:
                data = pickle.load(f)
                self.known_face_encodings = [np.array(e) for e in data["encodings"]]
                self.known_face_names = data["names"]
            print(f"[INFO] Loaded {len(self.known_face_encodings)} encodings.")
        except Exception as e:
            print(f"[ERROR] Could not load encodings: {e}")
            
    def recognize_faces(self, frame):
        """
        Detect and recognize faces in the given frame using MediaPipe Tasks API.
        Returns a list of tuples: (name, (top, right, bottom, left))
        """
        if not self.known_face_encodings:
            return []
            
        # Resize frame for faster detection
        h, w, _ = frame.shape
        small_frame = cv2.resize(frame, (0, 0), fx=config.FRAME_RESIZE_FACTOR, fy=config.FRAME_RESIZE_FACTOR)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_small_frame)
        
        # Perform detection
        detection_result = self.detector.detect(mp_image)
        
        recognized_faces = []
        
        if detection_result.detections:
            # Limit number of faces processed
            detections = detection_result.detections[:config.MAX_FACES_PER_FRAME]
            
            for detection in detections:
                # Get bounding box
                bbox = detection.bounding_box
                
                # Scale relative to small_frame back up to original frame
                # Actually bbox from detector is in pixels of the input image (small_frame)
                inv_factor = 1.0 / config.FRAME_RESIZE_FACTOR
                top = int(bbox.origin_y * inv_factor)
                left = int(bbox.origin_x * inv_factor)
                width = int(bbox.width * inv_factor)
                height = int(bbox.height * inv_factor)
                bottom = top + height
                right = left + width
                
                # Crop and generate encoding
                y1, y2 = max(0, top), min(h, bottom)
                x1, x2 = max(0, left), min(w, right)
                
                face_roi = frame[y1:y2, x1:x2]
                
                if face_roi.size == 0:
                    continue
                    
                # Resize and flatten
                face_resized = cv2.resize(face_roi, config.FACE_SIZE)
                face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
                current_encoding = face_gray.flatten()
                
                # Comparison using Euclidean distance
                name = "Unknown"
                if len(self.known_face_encodings) > 0:
                    distances = [np.linalg.norm(current_encoding - known) for known in self.known_face_encodings]
                    best_match_index = np.argmin(distances)
                    
                    norm_dist = distances[best_match_index] / len(current_encoding)
                    if norm_dist < config.MATCH_THRESHOLD * 255:
                         name = self.known_face_names[best_match_index]
                
                recognized_faces.append((name, (top, right, bottom, left)))
                
        return recognized_faces
