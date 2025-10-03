#!/usr/bin/env python3
"""
Web-based Face Recognition System
================================
Flask web application with camera switching functionality
"""

from flask import Flask, render_template, request, jsonify, Response
import cv2
import face_recognition
import numpy as np
import mysql.connector
from datetime import datetime
import logging
import threading
import time
import base64
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class WebFaceRecognition:
    def __init__(self):
        self.db = None
        self.known_face_encodings = []
        self.known_face_names = []
        self.current_camera = None
        self.camera_active = False
        self.current_frame = None
        self.detection_results = []
        self.last_detection = {}  # Track last detection time for each person
        
    def connect_database(self):
        """Connect to MySQL database"""
        try:
            self.db = mysql.connector.connect(
                host='localhost',
                user='root',
                password='root',
                database='hrm_database'
            )
            logger.info("✅ Database connected")
            return True
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            return False
    
    def load_face_data(self):
        """Load face embeddings from database"""
        if not self.connect_database():
            return False
            
        try:
            cursor = self.db.cursor()
            cursor.execute("""
                SELECT e.employee_id, e.full_name, f.embedding_vector 
                FROM employees e 
                JOIN employee_face_embeddings f ON e.employee_id = f.employee_id 
                WHERE f.embedding_vector IS NOT NULL
            """)
            
            results = cursor.fetchall()
            self.known_face_encodings = []
            self.known_face_names = []
            
            for emp_id, name, encoding_json in results:
                # Parse JSON encoding
                try:
                    encoding_list = json.loads(encoding_json)
                    encoding = np.array(encoding_list, dtype=np.float64)
                    self.known_face_encodings.append(encoding)
                    self.known_face_names.append(f"{name} (ID: {emp_id})")
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON encoding for employee {emp_id}")
            
            logger.info(f"✅ Loaded {len(self.known_face_encodings)} employee faces")
            return len(self.known_face_encodings) > 0
            
        except Exception as e:
            logger.error(f"❌ Error loading face data: {e}")
            return False
    
    def log_detection(self, employee_name, camera_source):
        """Log face detection to database"""
        try:
            if not self.db:
                self.connect_database()
            cursor = self.db.cursor()
            cursor.execute("""
                INSERT INTO employee_movements 
                (employee_id, employee_name, camera_id, camera_location, movement_type) 
                VALUES (%s, %s, %s, %s, %s)
            """, (1, employee_name, 1, camera_source, 'entry'))
            self.db.commit()
        except Exception as e:
            logger.error(f"Error logging detection: {e}")
    
    def get_camera_source(self, camera_type, channel=None):
        """Get camera source URL"""
        if camera_type == "webcam":
            return 0, "Webcam"
        elif camera_type == "cctv" and channel:
            cctv_url = f"rtsp://cctv1:cctv%254321@160.191.137.18:8554/cam/realmonitor?channel={channel}&subtype=0"
            return cctv_url, f"CCTV Channel {channel}"
        else:
            # Default to webcam if invalid configuration
            logger.warning(f"Invalid camera configuration: {camera_type}, {channel}. Using webcam.")
            return 0, "Webcam"
    
    def start_camera(self, camera_type, channel=None):
        """Start camera with given parameters"""
        try:
            # Force stop current camera and cleanup
            if self.current_camera:
                self.current_camera.release()
                self.current_camera = None
            
            # Stop camera flag
            self.camera_active = False
            time.sleep(0.5)  # Wait for cleanup
            
            camera_source, camera_name = self.get_camera_source(camera_type, channel)
            if camera_source is None:
                return False, "Invalid camera configuration"
            
            self.current_camera = cv2.VideoCapture(camera_source)
            
            # RTSP optimizations
            if isinstance(camera_source, str) and camera_source.startswith('rtsp://'):
                self.current_camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                try:
                    self.current_camera.set(cv2.CAP_PROP_TIMEOUT, 10000)
                except AttributeError:
                    pass
            
            if not self.current_camera.isOpened():
                return False, f"Cannot open {camera_name}"
            
            self.camera_active = True
            self.camera_name = camera_name
            logger.info(f"✅ Camera started: {camera_name}")
            return True, f"Camera started: {camera_name}"
            
        except Exception as e:
            logger.error(f"Camera start error: {e}")
            return False, str(e)
    
    def stop_camera(self):
        """Stop current camera"""
        self.camera_active = False
        if self.current_camera:
            try:
                self.current_camera.release()
            except:
                pass  # Ignore any release errors
            self.current_camera = None
        
        # Clear any detection tracking
        self.last_detection = {}
        
        logger.info("🛑 Camera stopped")
    
    def process_frame(self):
        """Process current frame for face recognition"""
        if not self.current_camera or not self.camera_active:
            return None, []
        
        ret, frame = self.current_camera.read()
        if not ret:
            return None, []
        
        # Resize for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Find faces
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        detections = []
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Scale back up
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            # Compare with known faces
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"
            confidence = 0
            
            if True in matches:
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index] and face_distances[best_match_index] < 0.6:
                    name = self.known_face_names[best_match_index]
                    confidence = round((1 - face_distances[best_match_index]) * 100, 1)
                    
                    # Log detection only if not detected recently (avoid spam)
                    current_time = time.time()
                    if name not in self.last_detection or (current_time - self.last_detection[name]) > 5:  # 5 second cooldown
                        self.log_detection(name, self.camera_name)
                        self.last_detection[name] = current_time
            
            # Draw on frame
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, f"{name} ({confidence}%)", (left + 6, bottom - 6), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            
            detections.append({
                'name': name,
                'confidence': confidence,
                'location': [left, top, right, bottom]
            })
        
        # Add camera info
        cv2.putText(frame, f"Camera: {self.camera_name}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Time: {datetime.now().strftime('%H:%M:%S')}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Employees: {len(self.known_face_encodings)}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame, detections

# Global face recognition instance
face_recognizer = WebFaceRecognition()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/load_faces', methods=['POST'])
def load_faces():
    """Load face data"""
    success = face_recognizer.load_face_data()
    return jsonify({
        'success': success,
        'count': len(face_recognizer.known_face_encodings),
        'message': f'Loaded {len(face_recognizer.known_face_encodings)} employee faces' if success else 'Failed to load faces'
    })

@app.route('/api/start_camera', methods=['POST'])
def start_camera():
    """Start camera"""
    data = request.get_json()
    camera_type = data.get('camera_type')
    channel = data.get('channel')
    
    success, message = face_recognizer.start_camera(camera_type, channel)
    return jsonify({
        'success': success,
        'message': message
    })

@app.route('/api/stop_camera', methods=['POST'])
def stop_camera():
    """Stop camera"""
    face_recognizer.stop_camera()
    return jsonify({
        'success': True,
        'message': 'Camera stopped'
    })

@app.route('/api/camera_frame')
def camera_frame():
    """Get current camera frame"""
    def generate():
        try:
            while face_recognizer.camera_active and face_recognizer.current_camera:
                frame, detections = face_recognizer.process_frame()
                if frame is not None:
                    # Convert frame to JPEG
                    ret, buffer = cv2.imencode('.jpg', frame)
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                else:
                    # If no frame, break the loop
                    break
                time.sleep(0.1)  # 10 FPS
        except Exception as e:
            logger.error(f"Camera frame error: {e}")
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/detections')
def get_detections():
    """Get recent detections"""
    try:
        if not face_recognizer.db:
            face_recognizer.connect_database()
        
        cursor = face_recognizer.db.cursor()
        cursor.execute("""
            SELECT employee_name, camera_location, timestamp 
            FROM employee_movements 
            ORDER BY timestamp DESC 
            LIMIT 10
        """)
        
        detections = []
        for name, camera, time_stamp in cursor.fetchall():
            detections.append({
                'name': name,
                'camera': camera,
                'time': time_stamp.strftime('%H:%M:%S')
            })
        
        return jsonify({'detections': detections})
    except:
        return jsonify({'detections': []})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)