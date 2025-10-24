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
import os
from ultralytics import YOLO

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
        self.face_crop_counter = 0  # Counter for saving face crops
        
        # Create face crops directories
        self.face_crops_dir = "face_crops"
        self.detected_faces_dir = os.path.join(self.face_crops_dir, "detected_faces")
        self.unknown_faces_dir = os.path.join(self.face_crops_dir, "unknown_faces")
        
        # Initialize YOLO model for human detection
        try:
            self.yolo_model = YOLO('yolov8n.pt')  # Load YOLOv8 nano model
            logger.info("✅ YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load YOLO model: {e}")
            self.yolo_model = None
        
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
    
    def save_face_crop(self, frame, face_box, name="Unknown", confidence=0):
        """Save cropped face image"""
        try:
            left, top, right, bottom = face_box
            
            # Ensure coordinates are valid
            height, width = frame.shape[:2]
            left = max(0, min(left, width))
            right = max(0, min(right, width))
            top = max(0, min(top, height))
            bottom = max(0, min(bottom, height))
            
            if left >= right or top >= bottom:
                logger.warning("Invalid face box coordinates")
                return None
            
            # Add padding around face
            padding = 30
            y1 = max(0, top - padding)
            y2 = min(height, bottom + padding)
            x1 = max(0, left - padding)
            x2 = min(width, right + padding)
            
            # Crop face with padding
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                logger.warning("Empty face crop")
                return None
            
            # Resize face crop for consistency
            face_crop = cv2.resize(face_crop, (150, 150))
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
            self.face_crop_counter += 1
            
            # Clean name for filename
            clean_name = name.replace(' ', '_').replace('(', '').replace(')', '').replace(':', '')
            
            if name != "Unknown":
                # Save in detected_faces folder
                filename = f"{clean_name}_{confidence:.1f}pct_{timestamp}_{self.face_crop_counter:04d}.jpg"
                save_path = os.path.join(self.detected_faces_dir, filename)
            else:
                # Save in unknown_faces folder
                filename = f"unknown_{timestamp}_{self.face_crop_counter:04d}.jpg"
                save_path = os.path.join(self.unknown_faces_dir, filename)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save the cropped face
            success = cv2.imwrite(save_path, face_crop)
            if success:
                logger.info(f"💾 Face crop saved: {filename}")
                return save_path
            else:
                logger.error(f"Failed to save face crop: {filename}")
                return None
            
        except Exception as e:
            logger.error(f"Error saving face crop: {e}")
            return None
    
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
    
    def detect_humans_yolo(self, frame):
        """Use YOLO to detect human bodies first"""
        if not self.yolo_model:
            return []
        
        try:
            # Run YOLO detection
            results = self.yolo_model(frame, verbose=False)
            human_boxes = []
            
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        # Class 0 is 'person' in COCO dataset
                        if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.5:
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            human_boxes.append((int(x1), int(y1), int(x2), int(y2)))
            
            return human_boxes
        except Exception as e:
            logger.error(f"YOLO detection error: {e}")
            return []
    
    def process_frame(self):
        """Process current frame with YOLO + face recognition"""
        if not self.current_camera or not self.camera_active:
            return None, []
        
        ret, frame = self.current_camera.read()
        if not ret:
            return None, []
        
        detections = []
        
        # SIMPLE APPROACH: Skip YOLO for now, use direct face recognition
        # This ensures face detection works even if YOLO fails
        
        # Resize for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Find faces directly
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        logger.info(f"🔍 Found {len(face_locations)} faces in frame")
        
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
                    
                    # Log detection
                    current_time = time.time()
                    if name not in self.last_detection or (current_time - self.last_detection[name]) > 5:
                        self.log_detection(name, self.camera_name)
                        self.last_detection[name] = current_time
                        logger.info(f"✅ Detected: {name} with {confidence}% confidence")
            
            # Save face crop for every detection
            face_box = [left, top, right, bottom]
            crop_path = self.save_face_crop(frame, face_box, name, confidence)
            if crop_path:
                logger.info(f"💾 Face crop saved: {crop_path}")
            
            # Draw on frame
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, f"{name} ({confidence}%)", (left + 6, bottom - 6), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            
            detections.append({
                'name': name,
                'confidence': confidence,
                'location': [left, top, right, bottom]
            })
        
        # Optional: Try YOLO if available and no faces found with traditional method
        if len(detections) == 0 and self.yolo_model:
            logger.info("🤖 No faces found with traditional method, trying YOLO...")
            try:
                human_boxes = self.detect_humans_yolo(frame)
                logger.info(f"🤖 YOLO found {len(human_boxes)} human bodies")
                
                # Process each detected human area
                for (x1, y1, x2, y2) in human_boxes:
                    # Draw YOLO detection box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)  # Blue box for human detection
                    
                    # Crop human region for face detection
                    human_crop = frame[y1:y2, x1:x2]
                    if human_crop.size == 0:
                        continue
                    
                    # Focus on upper part of body for face
                    height, width = human_crop.shape[:2]
                    face_area = human_crop[0:int(height*0.4), :]  # Top 40% of human body
                    
                    if face_area.size == 0:
                        continue
                    
                    # Process face area
                    small_face_area = cv2.resize(face_area, (0, 0), fx=0.5, fy=0.5)
                    rgb_small_face = cv2.cvtColor(small_face_area, cv2.COLOR_BGR2RGB)
                    
                    face_locations = face_recognition.face_locations(rgb_small_face)
                    face_encodings = face_recognition.face_encodings(rgb_small_face, face_locations)
                    
                    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                        # Scale back to original coordinates
                        top = int(top * 2) + y1
                        right = int(right * 2) + x1
                        bottom = int(bottom * 2) + y1
                        left = int(left * 2) + x1
                        
                        # Face recognition
                        matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                        name = "Unknown"
                        confidence = 0
                        
                        if True in matches:
                            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                            best_match_index = np.argmin(face_distances)
                            
                            if matches[best_match_index] and face_distances[best_match_index] < 0.6:
                                name = self.known_face_names[best_match_index]
                                confidence = round((1 - face_distances[best_match_index]) * 100, 1)
                                
                                # Log detection
                                current_time = time.time()
                                if name not in self.last_detection or (current_time - self.last_detection[name]) > 5:
                                    self.log_detection(name, self.camera_name)
                                    self.last_detection[name] = current_time
                        
                        # Save face crop
                        face_box = [left, top, right, bottom]
                        crop_path = self.save_face_crop(frame, face_box, name, confidence)
                        if crop_path:
                            logger.info(f"💾 YOLO Face crop saved: {crop_path}")
                        
                        # Draw face detection
                        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                        cv2.putText(frame, f"{name} ({confidence}%)", (left + 6, bottom - 6), 
                                   cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                        
                        detections.append({
                            'name': name,
                            'confidence': confidence,
                            'location': [left, top, right, bottom]
                        })
            except Exception as e:
                logger.error(f"YOLO processing error: {e}")
        
        # Add camera info and debug information
        cv2.putText(frame, f"Camera: {self.camera_name}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Time: {datetime.now().strftime('%H:%M:%S')}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Employees: {len(self.known_face_encodings)}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Faces Found: {len(detections)}", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Crops Saved: {self.face_crop_counter}", (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
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

@app.route('/api/face_crops')
def get_face_crops():
    """Get list of saved face crops"""
    try:
        crops_data = {
            'detected_faces': [],
            'unknown_faces': []
        }
        
        # Get detected faces
        if os.path.exists(face_recognizer.detected_faces_dir):
            detected_files = os.listdir(face_recognizer.detected_faces_dir)
            detected_files.sort(reverse=True)  # Latest first
            crops_data['detected_faces'] = detected_files[:50]  # Latest 50
        
        # Get unknown faces
        if os.path.exists(face_recognizer.unknown_faces_dir):
            unknown_files = os.listdir(face_recognizer.unknown_faces_dir)
            unknown_files.sort(reverse=True)  # Latest first
            crops_data['unknown_faces'] = unknown_files[:50]  # Latest 50
        
        return jsonify(crops_data)
    except Exception as e:
        logger.error(f"Error getting face crops: {e}")
        return jsonify({'detected_faces': [], 'unknown_faces': []})

@app.route('/api/face_crop/<path:filename>')
def serve_face_crop(filename):
    """Serve face crop image"""
    try:
        # Check in detected_faces first
        detected_path = os.path.join(face_recognizer.detected_faces_dir, filename)
        if os.path.exists(detected_path):
            return app.send_static_file(f"../face_crops/detected_faces/{filename}")
        
        # Check in unknown_faces
        unknown_path = os.path.join(face_recognizer.unknown_faces_dir, filename)
        if os.path.exists(unknown_path):
            return app.send_static_file(f"../face_crops/unknown_faces/{filename}")
        
        return "File not found", 404
    except Exception as e:
        logger.error(f"Error serving face crop: {e}")
        return "Error", 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)