import cv2
import sys
import os

from src.face_recognizer import FaceRecognizer
from src.attendance import AttendanceManager
from src.utils import ensure_directories
import config

def main():
    print("[INFO] Starting AI Attendance System...")
    ensure_directories()
    
    # Initialize components
    recognizer = FaceRecognizer()
    attendance_manager = AttendanceManager()
    
    # Start webcam. Usually index 0 or 1
    video_capture = cv2.VideoCapture(0)
    
    if not video_capture.isOpened():
        print("[ERROR] Could not open webcam.")
        sys.exit(1)
        
    print("[INFO] Press 'q' to quit.")
    
    frame_count = 0
    recognized_faces = []
    
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()
        
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break
            
        # Optimization: process every N frames
        if frame_count % config.FRAME_SKIP == 0:
            # Recognize faces
            recognized_faces = recognizer.recognize_faces(frame)
            
            # Log attendance
            for name, _ in recognized_faces:
                if name != "Unknown":
                    attendance_manager.mark_attendance(name)
        
        frame_count += 1
                    
        # Draw boxes (we do this every frame based on the last processed result if alternating)
        for name, (top, right, bottom, left) in recognized_faces:
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)
            
        # Display the resulting image
        cv2.imshow('AI Attendance System', frame)
        
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()
    print("[INFO] System shutdown.")

if __name__ == "__main__":
    main()
