import cv2
import time
from flask import Flask, Response
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from src.face_recognizer import FaceRecognizer
from src.attendance import AttendanceManager
from src.utils import ensure_directories

app = Flask(__name__)

# Initialize components
ensure_directories()
recognizer = FaceRecognizer()
attendance_manager = AttendanceManager()

def generate_frames():
    video_capture = cv2.VideoCapture(0)
    
    if not video_capture.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    frame_count = 0
    recognized_faces = []

    while True:
        success, frame = video_capture.read()
        if not success:
            break
        
        # Recognition logic (same as main.py)
        if frame_count % config.FRAME_SKIP == 0:
            recognized_faces = recognizer.recognize_faces(frame)
            
            # Log attendance
            for name, _ in recognized_faces:
                if name != "Unknown":
                    attendance_manager.mark_attendance(name)
        
        frame_count += 1

        # Draw boxes on the frame for the stream
        for name, (top, right, bottom, left) in recognized_faces:
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield frame in MJPEG format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Video streaming home page."""
    return """
    <html>
      <head>
        <title>AI Attendance Stream</title>
      </head>
      <body>
        <h1>AI Attendance Stream Server</h1>
        <p>The server is running correctly.</p>
        <ul>
          <li><b>Video Feed:</b> <a href="/video">/video</a></li>
        </ul>
        <hr>
        <p>Use this URL in your Streamlit dashboard: <code>http://&lt;your-pi-ip&gt;:8000/video</code></p>
      </body>
    </html>
    """

@app.route('/video')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print(f"[INFO] Starting Stream Server on {config.STREAM_HOST}:{config.STREAM_PORT}...")
    print(f"[INFO] Access the stream at http://<pi-ip>:{config.STREAM_PORT}/video")
    app.run(host=config.STREAM_HOST, port=config.STREAM_PORT, threaded=True)
