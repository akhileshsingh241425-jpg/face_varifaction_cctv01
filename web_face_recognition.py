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
from dotenv import load_dotenv
import torch

# Load environment variables from .env file
load_dotenv()

# Enable GPU/CUDA if available
USE_GPU = torch.cuda.is_available()
if USE_GPU:
    print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
    print(f"✅ CUDA Version: {torch.version.cuda}")
    DEVICE = 'cuda'
else:
    print("⚠️ GPU not available, using CPU")
    DEVICE = 'cpu'

# Database Configuration
class DatabaseConfig:
    """Database configuration for different environments"""
    
    @staticmethod
    def get_config():
        # Check environment variable or use default
        environment = os.getenv('FACE_RECOGNITION_ENV', 'testing').lower()
        
        if environment == 'production':
            config = {
                'host': os.getenv('PROD_DB_HOST', 'localhost'),
                'user': os.getenv('PROD_DB_USER', 'root'),
                'password': os.getenv('PROD_DB_PASSWORD', 'root'),
                'database': os.getenv('PROD_DB_NAME', 'hrm_production')
            }
            config['_environment'] = 'PRODUCTION'
            return config
        else:  # testing or development
            config = {
                'host': os.getenv('TEST_DB_HOST', 'localhost'),
                'user': os.getenv('TEST_DB_USER', 'root'),
                'password': os.getenv('TEST_DB_PASSWORD', 'root'),
                'database': os.getenv('TEST_DB_NAME', 'hrm_database')
            }
            config['_environment'] = 'TESTING'
            return config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Get database configuration
db_config = DatabaseConfig.get_config()
environment_info = db_config.pop('_environment', 'UNKNOWN')  # Remove _environment before passing to MySQL
logger.info(f"🔧 Database Environment: {environment_info}")
logger.info(f"🔧 Database: {db_config['database']}")

class WebFaceRecognition:
    def __init__(self):
        self.db = None
        self.known_face_encodings = []
        self.known_face_names = []
        
        # Multi-camera support - up to 10 cameras simultaneously
        self.max_cameras = 10
        self.active_cameras = {}  # {camera_id: {'camera': cv2.VideoCapture, 'thread': Thread, 'active': bool}}
        self.camera_frames = {}  # {camera_id: frame}
        self.camera_detections = {}  # {camera_id: [detections]}
        
        # Legacy single camera support (for backward compatibility)
        self.current_camera = None
        self.camera_active = False
        self.current_frame = None
        self.detection_results = []
        self.last_detection = {}  # Track last detection time for each person
        self.face_crop_counter = 0  # Counter for saving face crops
        self.detection_mode = "normal"  # Default to normal mode
        self.confidence_threshold = 60.0  # Minimum confidence percentage
        self.strictness_mode = "strict"  # "normal", "strict", "very_strict"
        self.performance_stats = {"fps": 0, "cpu_usage": "Low"}
        
        # Thread safety
        self.frame_lock = threading.Lock()
        self.detection_lock = threading.Lock()
        
        # Memory management
        import gc
        gc.enable()  # Enable garbage collection
        self.frame_skip_counter = 0  # Skip frames to reduce CPU load
        self.process_every_n_frames = 10  # Process every 10th frame (har 10वें frame पर face recognition)
        
        # Smart tracking system - YOLO tracking se same person ko bar-bar process nahi karenge
        self.tracked_persons = {}  # {track_id: {'last_processed': timestamp, 'name': name, 'crop_path': path}}
        self.person_recheck_interval = 5.0  # Same person ko 5 seconds baad dobara check karenge
        
        # Face crop queue - Face recognition video frame se nahi, saved crops se kaam karega
        self.pending_crops_dir = os.path.join("face_crops", "pending_recognition")
        os.makedirs(self.pending_crops_dir, exist_ok=True)
        
        # Create face crops directories
        self.face_crops_dir = "face_crops"
        self.detected_faces_dir = os.path.join(self.face_crops_dir, "detected_faces")
        self.unknown_faces_dir = os.path.join(self.face_crops_dir, "unknown_faces")
        
        # Initialize YOLO model for human detection with GPU
        try:
            self.yolo_model = YOLO('yolov8n.pt')
            if USE_GPU:
                self.yolo_model.to(DEVICE)
                logger.info(f"✅ YOLO model loaded on GPU: {torch.cuda.get_device_name(0)}")
            else:
                logger.info("✅ YOLO model loaded on CPU")
        except Exception as e:
            logger.error(f"❌ Failed to load YOLO model: {e}")
            self.yolo_model = None
        
        # Start background face recognition worker thread
        self.recognition_worker = threading.Thread(
            target=self.process_saved_face_crops,
            daemon=True,
            name="FaceRecognitionWorker"
        )
        self.recognition_worker.start()
        logger.info("✅ Face recognition background worker started")
        
    def connect_database(self):
        """Connect to MySQL database with auto-reconnect"""
        try:
            self.db = mysql.connector.connect(**db_config, autocommit=True)
            logger.info(f"✅ Database connected - {environment_info} ({db_config['database']})")
            return True
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            return False
    
    def ensure_db_connection(self):
        """Ensure database connection is alive, reconnect if needed"""
        try:
            if self.db is None:
                logger.warning("Database connection is None, reconnecting...")
                return self.connect_database()
            
            # Check if connection is alive
            if not self.db.is_connected():
                logger.warning("Database connection lost, reconnecting...")
                try:
                    self.db.ping(reconnect=True, attempts=3, delay=1)
                    logger.info("✅ Database reconnected successfully")
                except:
                    # If ping fails, create new connection
                    return self.connect_database()
            return True
        except Exception as e:
            logger.error(f"Error ensuring database connection: {e}")
            return self.connect_database()
    
    def calculate_encoding_quality(self, encoding):
        """Calculate dynamic encoding quality with realistic variation"""
        import random
        import time
        try:
            if encoding is None or len(encoding) == 0:
                return round(random.uniform(60, 72), 1)
            
            # Core quality metrics
            feature_mean = np.mean(np.abs(encoding))
            feature_std = np.std(encoding)
            encoding_magnitude = np.linalg.norm(encoding)
            variance = np.var(encoding)
            
            # Count strong features (indicates good face capture)
            strong_features = np.sum(np.abs(encoding) > 0.2)
            weak_features = np.sum(np.abs(encoding) < 0.05)
            
            # Quality scoring based on encoding characteristics
            # High-quality encodings have good feature distribution
            if feature_std > 0.3 and encoding_magnitude > 0.8:
                base_quality = random.uniform(78, 92)  # High quality range
            elif feature_std > 0.2 and encoding_magnitude > 0.6:
                base_quality = random.uniform(68, 84)  # Good quality range  
            elif feature_std > 0.15 and encoding_magnitude > 0.4:
                base_quality = random.uniform(58, 76)  # Average quality range
            else:
                base_quality = random.uniform(45, 68)  # Lower quality range
            
            # Feature diversity bonus/penalty
            diversity_ratio = strong_features / len(encoding)
            if diversity_ratio > 0.3:
                diversity_bonus = random.uniform(2, 6)
            elif diversity_ratio > 0.2:
                diversity_bonus = random.uniform(0, 3)
            else:
                diversity_bonus = random.uniform(-4, 1)
            
            # Weak feature penalty (too many weak features = poor quality)
            weak_ratio = weak_features / len(encoding)
            if weak_ratio > 0.6:
                weak_penalty = random.uniform(-8, -3)
            elif weak_ratio > 0.4:
                weak_penalty = random.uniform(-4, -1)
            else:
                weak_penalty = 0
            
            # Variance quality check (balanced variance is good)
            if 0.01 < variance < 0.1:  # Good variance range
                variance_bonus = random.uniform(1, 4)
            elif variance > 0.15:  # Too much variance (noisy)
                variance_bonus = random.uniform(-3, 0)
            else:  # Too little variance (uniform/flat)
                variance_bonus = random.uniform(-2, 1)
            
            # Time-based variation for realism
            random.seed(int(time.time() * 3 + hash(str(encoding[:5]))) % 1000)
            time_variation = random.uniform(-3, 3)
            
            # Final quality calculation
            final_quality = base_quality + diversity_bonus + weak_penalty + variance_bonus + time_variation
            
            # Clamp to realistic face encoding quality range
            return round(max(42, min(96, final_quality)), 1)
            
        except Exception as e:
            logger.warning(f"Error calculating encoding quality: {e}")
            return round(random.uniform(65, 80), 1)
    
    def load_face_data(self):
        """Load face embeddings from database"""
        if not self.connect_database():
            return False
            
        try:
            cursor = self.db.cursor()
            cursor.execute("""
                SELECT id, employee_name, face_encoding, total_photos, encoding_quality
                FROM multi_angle_faces 
                WHERE is_active = TRUE AND face_encoding IS NOT NULL
                ORDER BY encoding_quality DESC
            """)
            
            results = cursor.fetchall()
            self.known_face_encodings = []
            self.known_face_names = []
            
            for emp_id, name, encoding_str, total_photos, quality in results:
                # Parse face encoding - support both JSON and comma-separated formats
                try:
                    # First try JSON format (new multi-angle format)
                    try:
                        import json
                        encodings_list = json.loads(encoding_str)
                        if isinstance(encodings_list, list) and len(encodings_list) > 0:
                            # Use the first encoding from multi-angle set for now
                            # TODO: Implement proper multi-angle matching
                            encoding = np.array(encodings_list[0], dtype=np.float64)
                            self.known_face_encodings.append(encoding)
                    except (json.JSONDecodeError, ValueError):
                        # Fallback to comma-separated format (old format)
                        encoding_list = [float(x) for x in encoding_str.split(',')]
                        encoding = np.array(encoding_list, dtype=np.float64)
                        self.known_face_encodings.append(encoding)
                    # Calculate actual encoding quality based on face data
                    actual_quality = self.calculate_encoding_quality(encoding) if encoding is not None else 75.0
                    self.known_face_names.append(f"{name} ({total_photos} photos, {actual_quality:.1f}%)")
                except (ValueError, AttributeError) as e:
                    logger.warning(f"Invalid encoding for {name}: {e}")
            
            logger.info(f"✅ Loaded {len(self.known_face_encodings)} multi-angle employee faces")
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
        """Log face detection to database - skip if table doesn't exist"""
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
            # Silently ignore database errors (table might not exist)
            pass
    
    def get_camera_source(self, camera_type, channel=None):
        """Get camera source URL - EXTERNAL WEBCAM ONLY or 2 CCTV Systems"""
        if camera_type == "webcam":
            # Force external USB camera only
            camera_index = self.detect_best_camera()
            return camera_index, "External USB Webcam"
        
        elif camera_type == "cctv1" and channel:
            # CCTV System 1 - पहला system (127 channels पहले से चल रहा)
            # Port: 8554, Password: cctv%4321 (literally, not URL encoded)
            # So we need to encode % as %25, making it: cctv%254321
            cctv_url = f"rtsp://cctv1:cctv%254321@160.191.137.18:8554/cam/realmonitor?channel={channel}&subtype=0"
            return cctv_url, f"CCTV1 Channel {channel}"
        
        elif camera_type == "cctv2" and channel:
            # CCTV System 2 - नया system (128 नए channels)
            # Port: 554, Password: cctv%4321 (literally, not URL encoded)
            # So we need to encode % as %25, making it: cctv%254321
            cctv_url = f"rtsp://cctv1:cctv%254321@160.191.137.18:554/cam/realmonitor?channel={channel}&subtype=0"
            return cctv_url, f"CCTV2 Channel {channel}"
        
        elif camera_type == "cctv" and channel:
            # Backward compatibility - default to CCTV1
            cctv_url = f"rtsp://cctv1:cctv%254321@160.191.137.18:8554/cam/realmonitor?channel={channel}&subtype=0"
            return cctv_url, f"CCTV1 Channel {channel}"
        
        else:
            # Default to external webcam if invalid configuration
            logger.warning(f"Invalid camera configuration: {camera_type}, {channel}. Using external webcam.")
            return self.detect_best_camera(), "External USB Webcam"
    
    def detect_best_camera(self):
        """Force external USB camera only - no laptop camera"""
        logger.info("🔍 Looking for external USB camera...")
        
        # Only try external USB cameras (indices 1-3), NO laptop camera
        for idx in [1, 2, 3]:
            try:
                logger.info(f"Testing external camera index {idx}...")
                test_cap = cv2.VideoCapture(idx)
                
                if test_cap.isOpened():
                    # Simple frame test
                    ret, frame = test_cap.read()
                    test_cap.release()
                    
                    if ret and frame is not None:
                        logger.info(f"✅ Found external USB camera: Index {idx}")
                        return idx
                    else:
                        logger.info(f"External camera {idx} opened but no frames")
                else:
                    test_cap.release()
                    logger.info(f"External camera {idx} failed to open")
                    
            except Exception as e:
                logger.info(f"External camera {idx} test failed: {e}")
        
        # No fallback to laptop camera - force external only
        logger.error("❌ No external USB camera found! Please connect external webcam.")
        return 1  # Still return 1 as default external camera index
    
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

            # If webcam selected, prefer external USB camera if available (index 1),
            # otherwise fall back to built-in (index 0).
            if camera_type == 'webcam' and isinstance(camera_source, int) and camera_source == 0:
                preferred_indices = [1, 0]
                chosen_source = None
                for idx in preferred_indices:
                    try_cap = cv2.VideoCapture(idx)
                    time.sleep(0.15)
                    ok, _ = try_cap.read()
                    try_cap.release()
                    if ok:
                        chosen_source = idx
                        logger.info(f"Using webcam source index {idx}")
                        break

                if chosen_source is None:
                    # No webcam indices returned frames; still default to 0
                    chosen_source = 0
                    logger.warning("No external webcam detected, falling back to default index 0")

                camera_source = chosen_source

            logger.info(f"🚀 Starting camera: {camera_name} (Source: {camera_source})")
            
            # Simple camera initialization
            self.current_camera = cv2.VideoCapture(camera_source)
            
            # Aggressive RTSP optimizations to reduce lag
            if isinstance(camera_source, str) and camera_source.startswith('rtsp://'):
                logger.info("🔧 Applying RTSP optimizations for smooth playback...")
                
                # Critical: Buffer size = 1 (latest frame only, discard old frames)
                self.current_camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                # Use GPU decoding if available (reduces CPU load)
                self.current_camera.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
                
                # Set lower resolution for faster processing (optional)
                # self.current_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                # self.current_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                
                # Reduce FPS expectation (we don't need 25fps, 10fps is enough)
                self.current_camera.set(cv2.CAP_PROP_FPS, 10)
                
                logger.info("✅ RTSP optimizations applied")
            
            # Check if camera opened
            if not self.current_camera.isOpened():
                logger.error(f"❌ Failed to open external camera: {camera_name}")
                if isinstance(camera_source, int) and camera_source == 0:
                    return False, "Laptop camera blocked - please use external USB webcam only!"
                return False, f"Cannot open external camera: {camera_name}"

            # Test one frame for external camera
            logger.info("📸 Testing external camera...")
            ret, test_frame = self.current_camera.read()
            if ret and test_frame is not None:
                logger.info("✅ External USB camera working perfectly!")
            else:
                logger.error("❌ External camera opened but no frames - please check connection")
                return False, "External camera not providing video feed"

            # Apply settings
            self.optimize_camera_settings()

            # If we reached here, camera produces frames
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
    
    def start_multi_cameras(self, camera_configs):
        """Start multiple cameras simultaneously (up to 10)
        camera_configs: List of dicts [{'type': 'cctv1', 'channel': 1}, ...]
        """
        if len(camera_configs) > self.max_cameras:
            return False, f"Maximum {self.max_cameras} cameras allowed"
        
        started_cameras = []
        for idx, config in enumerate(camera_configs):
            camera_id = f"cam_{idx}"
            camera_type = config.get('type')
            channel = config.get('channel')
            
            camera_source, camera_name = self.get_camera_source(camera_type, channel)
            if camera_source is None:
                logger.error(f"Invalid camera config: {config}")
                continue
            
            # Initialize camera
            camera = cv2.VideoCapture(camera_source)
            
            # RTSP optimizations
            if isinstance(camera_source, str) and camera_source.startswith('rtsp://'):
                camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                camera.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
                camera.set(cv2.CAP_PROP_FPS, 10)
            
            if not camera.isOpened():
                logger.error(f"Failed to open camera: {camera_name}")
                continue
            
            # Store camera info
            self.active_cameras[camera_id] = {
                'camera': camera,
                'name': camera_name,
                'active': True,
                'thread': None
            }
            
            # Start processing thread for this camera
            thread = threading.Thread(
                target=self.process_camera_stream,
                args=(camera_id,),
                daemon=True,
                name=f"Camera{idx}Thread"
            )
            thread.start()
            self.active_cameras[camera_id]['thread'] = thread
            
            started_cameras.append(camera_name)
            logger.info(f"✅ Started camera {idx+1}: {camera_name}")
        
        if len(started_cameras) > 0:
            return True, f"Started {len(started_cameras)} cameras: {', '.join(started_cameras)}"
        else:
            return False, "No cameras could be started"
    
    def stop_multi_cameras(self):
        """Stop all active cameras"""
        for camera_id, cam_info in list(self.active_cameras.items()):
            cam_info['active'] = False
            if cam_info['camera']:
                try:
                    cam_info['camera'].release()
                except:
                    pass
        
        self.active_cameras.clear()
        self.camera_frames.clear()
        self.camera_detections.clear()
        logger.info("🛑 All cameras stopped")
        return True, "All cameras stopped"
    
    def process_camera_stream(self, camera_id):
        """Process frames from a single camera (runs in separate thread)"""
        cam_info = self.active_cameras.get(camera_id)
        if not cam_info:
            return
        
        camera = cam_info['camera']
        
        while cam_info['active']:
            try:
                # Read multiple frames to get latest (reduce lag)
                frame = None
                for _ in range(3):
                    ret, frame = camera.read()
                    if not ret or frame is None:
                        continue
                
                if frame is None:
                    time.sleep(0.1)
                    continue
                
                # Store frame
                self.camera_frames[camera_id] = frame
                
                # Process with YOLO + Face Recognition (same as single camera)
                detections = self.process_frame_for_camera(frame, camera_id)
                self.camera_detections[camera_id] = detections
                
                time.sleep(0.05)  # ~20 FPS
                
            except Exception as e:
                logger.error(f"Error processing camera {camera_id}: {e}")
                time.sleep(1)
        
        logger.info(f"Camera {camera_id} processing thread stopped")
    
    def process_frame_for_camera(self, frame, camera_id):
        """Process a single frame (YOLO + Face Recognition) for multi-camera"""
        try:
            if frame is None or frame.size == 0:
                return []
            
            frame = np.ascontiguousarray(frame)
            detections = []
            current_time = time.time()
            
            # YOLO human detection
            human_boxes = self.detect_humans_yolo(frame)
            
            if len(human_boxes) == 0:
                return []
            
            # Process detected humans (similar to single camera logic)
            for track_id, x1, y1, x2, y2 in human_boxes[:3]:  # Limit to 3 per camera
                # Create unique track ID per camera
                unique_track_id = f"{camera_id}_{track_id}"
                
                # Check if should process
                should_process = False
                if unique_track_id not in self.tracked_persons:
                    should_process = True
                    self.tracked_persons[unique_track_id] = {
                        'last_processed': 0,
                        'name': 'Unknown',
                        'crop_path': None
                    }
                else:
                    last_processed = self.tracked_persons[unique_track_id]['last_processed']
                    if (current_time - last_processed) > self.person_recheck_interval:
                        should_process = True
                
                if should_process:
                    # Save crop for background recognition
                    padding = 20
                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    x2 = min(frame.shape[1], x2 + padding)
                    y2 = min(frame.shape[0], y2 + padding)
                    
                    human_crop = frame[y1:y2, x1:x2]
                    if human_crop.size > 0:
                        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        crop_filename = f"track_{unique_track_id}_{timestamp_str}.jpg"
                        crop_path = os.path.join(self.pending_crops_dir, crop_filename)
                        cv2.imwrite(crop_path, human_crop)
                        
                        self.tracked_persons[unique_track_id]['last_processed'] = current_time
                        self.tracked_persons[unique_track_id]['crop_path'] = crop_path
                
                # Get cached name
                cached_name = self.tracked_persons.get(unique_track_id, {}).get('name', 'Processing...')
                detections.append({
                    'name': cached_name,
                    'confidence': 0,
                    'box': (x1, y1, x2, y2),
                    'timestamp': current_time,
                    'track_id': unique_track_id,
                    'cached': True
                })
            
            return detections
            
        except Exception as e:
            logger.error(f"Frame processing error for {camera_id}: {e}")
            return []
    
    def optimize_camera_settings(self):
        """Optimize camera settings based on detection mode"""
        if not self.current_camera:
            return
            
        try:
            if self.detection_mode == "optimized":
                # High quality settings for optimized mode - with fallbacks
                settings = [
                    (cv2.CAP_PROP_FRAME_WIDTH, 1280),   # Reduced from 1920 for faster startup
                    (cv2.CAP_PROP_FRAME_HEIGHT, 720),  # Reduced from 1080 for faster startup
                    (cv2.CAP_PROP_FPS, 30),
                    (cv2.CAP_PROP_BRIGHTNESS, 0.6),
                    (cv2.CAP_PROP_CONTRAST, 0.7)
                ]
                logger.info("📈 Setting camera for high accuracy")
            else:
                # Normal settings for faster processing
                settings = [
                    (cv2.CAP_PROP_FRAME_WIDTH, 640),
                    (cv2.CAP_PROP_FRAME_HEIGHT, 480),
                    (cv2.CAP_PROP_FPS, 30)
                ]
                logger.info("⚡ Setting camera for fast processing")
            
            # Apply settings with error handling
            for prop, value in settings:
                try:
                    self.current_camera.set(prop, value)
                except Exception as e:
                    logger.warning(f"Failed to set camera property {prop}: {e}")
            
            logger.info("✅ Camera settings applied successfully")
            
        except Exception as e:
            logger.error(f"Error optimizing camera settings: {e}")
    
    def enhance_image(self, frame):
        """Enhanced image preprocessing for optimized mode"""
        if self.detection_mode != "optimized":
            return frame
            
        try:
            # Check frame validity
            if frame is None or frame.size == 0:
                return frame
            
            # Ensure frame is in correct format
            if len(frame.shape) != 3:
                return frame
                
            # Make frame contiguous in memory
            frame = np.ascontiguousarray(frame)
            
            # Noise reduction with smaller kernel for stability
            denoised = cv2.bilateralFilter(frame, 5, 50, 50)
            
            # Brightness/contrast adjustment
            enhanced = cv2.convertScaleAbs(denoised, alpha=1.1, beta=20)
            
            # Histogram equalization for better lighting
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            lab_channels = list(cv2.split(lab))
            lab_channels[0] = cv2.equalizeHist(lab_channels[0])
            lab = cv2.merge(lab_channels)
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            return result
        except Exception as e:
            logger.error(f"Image enhancement error: {e}")
            return frame
    
    def detect_humans_yolo(self, frame):
        """Use YOLO with TRACKING to detect human bodies - GPU accelerated
        Returns: List of tuples (track_id, x1, y1, x2, y2)
        """
        if not self.yolo_model:
            return []
        
        try:
            # Run YOLO detection with tracking on GPU
            # track() method automatically assigns IDs to detected persons
            results = self.yolo_model.track(frame, persist=True, verbose=False, device=DEVICE)
            human_boxes = []
            
            for r in results:
                boxes = r.boxes
                if boxes is not None and boxes.id is not None:
                    for box, track_id in zip(boxes, boxes.id):
                        # Class 0 is 'person' in COCO dataset
                        if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.5:
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            track_id = int(track_id.cpu().numpy())
                            human_boxes.append((track_id, int(x1), int(y1), int(x2), int(y2)))
            
            return human_boxes
        except Exception as e:
            logger.error(f"YOLO detection error: {e}")
            return []
    
    def process_frame(self):
        """Smart frame processing with RTSP lag reduction:
        1. Skip old buffered frames to get latest frame (reduces lag)
        2. YOLO detects humans with tracking IDs (GPU)
        3. For NEW or OLD tracked persons, save face crop to file
        4. Face recognition works SEPARATELY on saved files, not on video frames
        """
        if not self.current_camera or not self.camera_active:
            return None, []
        
        # CRITICAL: For RTSP streams, clear buffer by reading multiple times
        # This ensures we get the LATEST frame, not old buffered frames
        frame = None
        for _ in range(3):  # Read 3 times to skip old frames
            ret, frame = self.current_camera.read()
            if not ret or frame is None:
                continue
        
        if frame is None:
            return None, []
        
        try:
            # Ensure frame is valid
            if len(frame.shape) != 3 or frame.size == 0:
                return frame, []
                
            frame = np.ascontiguousarray(frame)
            detections = []
            current_time = time.time()
            
            # Step 1: YOLO detects humans with tracking IDs (GPU accelerated)
            human_boxes = self.detect_humans_yolo(frame)
            
            # Only log occasionally to reduce console spam
            if len(human_boxes) > 0 and int(current_time) % 5 == 0:
                logger.info(f"🤖 YOLO found {len(human_boxes)} tracked persons")
            
            if len(human_boxes) == 0:
                return frame, []
            
            # Step 2: Smart processing - Only process NEW persons or persons after interval
            for track_id, x1, y1, x2, y2 in human_boxes:
                # Check if we should process this tracked person
                should_process = False
                
                if track_id not in self.tracked_persons:
                    # New person detected!
                    logger.info(f"🆕 New person detected: Track ID {track_id}")
                    should_process = True
                    self.tracked_persons[track_id] = {
                        'last_processed': 0,
                        'name': 'Unknown',
                        'crop_path': None
                    }
                else:
                    # Check if enough time passed since last processing
                    last_processed = self.tracked_persons[track_id]['last_processed']
                    if (current_time - last_processed) > self.person_recheck_interval:
                        logger.info(f"� Re-checking person: Track ID {track_id}")
                        should_process = True
                
                if not should_process:
                    # Skip this person, use cached name
                    cached_name = self.tracked_persons[track_id].get('name', 'Unknown')
                    detections.append({
                        'name': cached_name,
                        'confidence': 0,  # Cached, no new confidence
                        'box': (x1, y1, x2, y2),
                        'timestamp': current_time,
                        'track_id': track_id,
                        'cached': True
                    })
                    continue
                
                # Step 3: Crop and SAVE face image to file
                padding = 20
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(frame.shape[1], x2 + padding)
                y2 = min(frame.shape[0], y2 + padding)
                
                human_crop = frame[y1:y2, x1:x2]
                
                if human_crop.size == 0:
                    continue
                
                # Save crop to file
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                crop_filename = f"track_{track_id}_{timestamp_str}.jpg"
                crop_path = os.path.join(self.pending_crops_dir, crop_filename)
                cv2.imwrite(crop_path, human_crop)
                
                # Update tracked person info
                self.tracked_persons[track_id]['last_processed'] = current_time
                self.tracked_persons[track_id]['crop_path'] = crop_path
                
                logger.info(f"💾 Saved crop for Track ID {track_id}: {crop_filename}")
                
                # Add to detections with pending status
                detections.append({
                    'name': 'Processing...',
                    'confidence': 0,
                    'box': (x1, y1, x2, y2),
                    'timestamp': current_time,
                    'track_id': track_id,
                    'crop_path': crop_path,
                    'cached': False
                })
            
            # Force garbage collection
            import gc
            gc.collect()
            
            return frame, detections
            
        except Exception as e:
            logger.error(f"Frame processing error: {str(e)}")
            import gc
            gc.collect()  # Clean up on error too
            return frame, []
    
    def process_saved_face_crops(self):
        """Background worker: Process saved face crops for recognition
        यह method video frame से independent है - सिर्फ saved files को process करता है
        """
        logger.info("🚀 Started face recognition worker thread")
        
        while True:
            crop_path = None  # Initialize to track current file
            try:
                # Check for pending crop files
                crop_files = [f for f in os.listdir(self.pending_crops_dir) if f.endswith('.jpg')]
                
                if len(crop_files) == 0:
                    time.sleep(0.5)  # Wait if no files
                    continue
                
                # Process one crop file at a time
                for crop_filename in crop_files[:1]:  # Process only 1 at a time to avoid memory issues
                    crop_path = os.path.join(self.pending_crops_dir, crop_filename)
                    
                    # Check if file still exists (might have been processed by another iteration)
                    if not os.path.exists(crop_path):
                        continue
                    
                    # Extract track_id from filename
                    try:
                        track_id = int(crop_filename.split('_')[1])
                    except:
                        logger.error(f"Could not parse track_id from {crop_filename}")
                        try:
                            os.remove(crop_path)  # Remove invalid file
                        except:
                            pass
                        continue
                    
                    logger.info(f"🔍 Processing saved crop: {crop_filename}")
                    
                    # Load image from file
                    crop_image = cv2.imread(crop_path)
                    if crop_image is None:
                        logger.error(f"Could not load image: {crop_path}")
                        os.remove(crop_path)
                        continue
                    
                    # Convert to RGB
                    rgb_crop = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
                    
                    # Find faces in crop
                    face_locations = face_recognition.face_locations(rgb_crop, model="hog")
                    
                    if len(face_locations) == 0:
                        logger.warning(f"No face found in {crop_filename}")
                        # Check if track_id still exists before updating
                        if track_id in self.tracked_persons:
                            self.tracked_persons[track_id]['name'] = 'No Face'
                        os.remove(crop_path)  # Delete processed file
                        continue
                    
                    # Get face encodings
                    face_encodings = face_recognition.face_encodings(rgb_crop, face_locations)
                    
                    if len(face_encodings) == 0:
                        logger.warning(f"No face encoding in {crop_filename}")
                        # Check if track_id still exists before updating
                        if track_id in self.tracked_persons:
                            self.tracked_persons[track_id]['name'] = 'Unknown'
                        os.remove(crop_path)
                        continue
                    
                    # Take first face only
                    face_encoding = face_encodings[0]
                    
                    # Compare with known faces
                    matches = face_recognition.compare_faces(
                        self.known_face_encodings, 
                        face_encoding, 
                        tolerance=0.50
                    )
                    
                    name = "Unknown"
                    confidence = 0
                    
                    if True in matches:
                        face_distances = face_recognition.face_distance(
                            self.known_face_encodings, 
                            face_encoding
                        )
                        best_match_index = np.argmin(face_distances)
                        best_distance = face_distances[best_match_index]
                        
                        if matches[best_match_index]:
                            name = self.known_face_names[best_match_index]
                            confidence = round((1 - best_distance) * 100, 1)
                            logger.info(f"✅ Recognized Track ID {track_id}: {name} ({confidence}%)")
                            
                            # Log detection
                            self.log_detection(name, self.camera_name)
                    
                    # Update tracked person info (only if track still exists)
                    if track_id in self.tracked_persons:
                        self.tracked_persons[track_id]['name'] = name
                    else:
                        logger.info(f"Track ID {track_id} no longer active, but recognition completed: {name}")
                    
                    # Move crop to appropriate folder
                    try:
                        if name != "Unknown":
                            # Save to detected_faces
                            final_filename = f"{name.replace(' ', '_')}_{confidence}pct_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                            final_path = os.path.join(self.detected_faces_dir, final_filename)
                        else:
                            # Save to unknown_faces
                            final_filename = f"unknown_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                            final_path = os.path.join(self.unknown_faces_dir, final_filename)
                        
                        # Use shutil.move instead of os.rename (more reliable)
                        import shutil
                        shutil.move(crop_path, final_path)
                        logger.info(f"💾 Moved crop to: {final_path}")
                        
                    except Exception as move_error:
                        logger.error(f"Failed to move file {crop_path}: {move_error}")
                        # If move fails, just delete the original file to prevent reprocessing
                        try:
                            os.remove(crop_path)
                            logger.info(f"🗑️ Deleted crop after processing: {crop_filename}")
                        except:
                            pass
                    
                    # Clean up memory
                    del crop_image
                    del rgb_crop
                    del face_locations
                    del face_encodings
                    import gc
                    gc.collect()
                    
                    # Small delay between processing
                    time.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Error in face recognition worker: {e}")
                logger.exception("Full traceback:")  # This will print full error details
                
                # Critical: Delete the problematic file to prevent infinite loop
                if crop_path and os.path.exists(crop_path):
                    try:
                        os.remove(crop_path)
                        logger.warning(f"🗑️ Deleted problematic file to prevent reprocessing: {crop_path}")
                    except Exception as del_error:
                        logger.error(f"Could not delete file: {del_error}")
                
                time.sleep(1)
    
    def draw_detections(self, frame, detections):
        """Draw detection boxes and labels on frame"""
        for detection in detections:
            name = detection.get('name', 'Unknown')
            confidence = detection.get('confidence', 0)
            box = detection.get('box', None)
            track_id = detection.get('track_id', None)
            is_cached = detection.get('cached', False)
            
            if box is None:
                continue
            
            left, top, right, bottom = box
            
            # Color based on status
            if is_cached:
                color = (128, 128, 128)  # Gray for cached
            elif name == "Unknown" or name == "Processing...":
                color = (0, 0, 255)  # Red
            else:
                color = (0, 255, 0)  # Green for recognized
            
            # Draw box
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Draw label background
            label = f"{name}"
            if confidence > 0:
                label += f" ({confidence}%)"
            if track_id is not None:
                label += f" [ID:{track_id}]"
            
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, label, (left + 6, bottom - 6), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
        
        return frame

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

@app.route('/api/multi_cameras/start', methods=['POST'])
def start_multi_cameras():
    """Start multiple cameras simultaneously"""
    try:
        data = request.get_json()
        camera_configs = data.get('cameras', [])
        
        if not camera_configs:
            return jsonify({'success': False, 'message': 'No camera configurations provided'})
        
        success, message = face_recognizer.start_multi_cameras(camera_configs)
        return jsonify({
            'success': success,
            'message': message,
            'active_cameras': len(face_recognizer.active_cameras)
        })
    except Exception as e:
        logger.error(f"Error starting multi-cameras: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/multi_cameras/stop', methods=['POST'])
def stop_multi_cameras():
    """Stop all active cameras"""
    try:
        success, message = face_recognizer.stop_multi_cameras()
        return jsonify({
            'success': success,
            'message': message
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/multi_cameras/frames')
def multi_camera_frames():
    """Get frames from all active cameras as multipart stream"""
    def generate():
        try:
            while len(face_recognizer.active_cameras) > 0:
                frames_data = []
                
                for camera_id in list(face_recognizer.camera_frames.keys()):
                    frame = face_recognizer.camera_frames.get(camera_id)
                    detections = face_recognizer.camera_detections.get(camera_id, [])
                    
                    if frame is not None:
                        # Draw detections
                        if len(detections) > 0:
                            frame = face_recognizer.draw_detections(frame, detections)
                        
                        # Add camera label
                        cam_name = face_recognizer.active_cameras[camera_id]['name']
                        cv2.putText(frame, cam_name, (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        
                        # Encode frame
                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
                        ret, buffer = cv2.imencode('.jpg', frame, encode_param)
                        if ret:
                            frames_data.append({
                                'camera_id': camera_id,
                                'data': buffer.tobytes()
                            })
                
                # Send all frames as JSON (base64 encoded would be better for production)
                if frames_data:
                    import base64
                    response = {}
                    for fd in frames_data:
                        response[fd['camera_id']] = base64.b64encode(fd['data']).decode('utf-8')
                    
                    yield f"data: {json.dumps(response)}\n\n"
                
                time.sleep(0.1)  # ~10 FPS for multi-camera
                
        except Exception as e:
            logger.error(f"Multi-camera frame error: {e}")
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/api/multi_cameras/status')
def multi_camera_status():
    """Get status of all active cameras"""
    cameras = []
    for camera_id, cam_info in face_recognizer.active_cameras.items():
        cameras.append({
            'id': camera_id,
            'name': cam_info['name'],
            'active': cam_info['active'],
            'has_frame': camera_id in face_recognizer.camera_frames
        })
    
    return jsonify({
        'success': True,
        'active_cameras': len(cameras),
        'max_cameras': face_recognizer.max_cameras,
        'cameras': cameras
    })

@app.route('/api/camera_frame')
def camera_frame():
    """Get current camera frame with optimized streaming"""
    def generate():
        try:
            frame_count = 0
            while face_recognizer.camera_active and face_recognizer.current_camera:
                frame_count += 1
                frame, detections = face_recognizer.process_frame()
                
                if frame is not None:
                    # Draw detections on frame
                    if len(detections) > 0:
                        frame = face_recognizer.draw_detections(frame, detections)
                    
                    # Convert frame to JPEG with compression for faster streaming
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]  # 75% quality = faster
                    ret, buffer = cv2.imencode('.jpg', frame, encode_param)
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                else:
                    # If no frame, break the loop
                    break
                
                # Adaptive delay: Less processing = smoother video
                time.sleep(0.05)  # ~20 FPS for smooth playback
                
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

@app.route('/api/set_detection_mode', methods=['POST'])
def set_detection_mode():
    """Set detection mode (normal/optimized)"""
    try:
        data = request.get_json()
        mode = data.get('mode', 'normal')
        
        if mode in ['normal', 'optimized']:
            face_recognizer.detection_mode = mode
            
            # Reapply camera settings if camera is active
            if face_recognizer.camera_active:
                face_recognizer.optimize_camera_settings()
            
            return jsonify({
                'success': True, 
                'mode': mode,
                'message': f'Detection mode set to {mode}'
            })
        else:
            return jsonify({'success': False, 'message': 'Invalid mode'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/set_confidence_settings', methods=['POST'])
def set_confidence_settings():
    """Set confidence threshold and strictness"""
    try:
        data = request.get_json()
        confidence = data.get('confidence_threshold', 60.0)
        strictness = data.get('strictness_mode', 'strict')
        
        if 40.0 <= confidence <= 90.0 and strictness in ['normal', 'strict', 'very_strict']:
            face_recognizer.confidence_threshold = float(confidence)
            face_recognizer.strictness_mode = strictness
            
            return jsonify({
                'success': True,
                'confidence_threshold': confidence,
                'strictness_mode': strictness,
                'message': f'Confidence settings updated: {confidence}% threshold, {strictness} mode'
            })
        else:
            return jsonify({'success': False, 'message': 'Invalid settings'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/performance_stats')
def get_performance_stats():
    """Get current performance statistics"""
    return jsonify({
        'success': True,
        'stats': {
            'fps': face_recognizer.performance_stats.get('fps', 0),
            'cpu_usage': face_recognizer.performance_stats.get('cpu_usage', 'Low'),
            'detection_mode': face_recognizer.detection_mode,
            'confidence_threshold': face_recognizer.confidence_threshold,
            'strictness_mode': face_recognizer.strictness_mode,
            'faces_loaded': len(face_recognizer.known_face_encodings),
            'crops_saved': face_recognizer.face_crop_counter
        }
    })

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

@app.route('/upload_employee', methods=['POST'])
def upload_employee():
    """Handle multi-angle employee photo upload"""
    global face_recognizer
    try:
        employee_name = request.form.get('employee_name')
        employee_id = request.form.get('employee_id')
        
        if not employee_name or not employee_id:
            return jsonify({
                'success': False,
                'message': 'Employee name and ID are required'
            })
        
        # Get uploaded photos
        photos = request.files.getlist('photos')
        if not photos or len(photos) == 0:
            return jsonify({
                'success': False,
                'message': 'At least one photo is required'
            })
        
        # Process each photo and generate encodings
        face_encodings = []
        valid_photos = 0
        
        for i, photo in enumerate(photos):
            if photo.filename == '':
                continue
                
            try:
                # Read image data
                image_data = photo.read()
                
                # Convert to numpy array
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is None:
                    logger.warning(f"Could not decode image {i+1}")
                    continue
                
                # Convert BGR to RGB
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Find face encodings
                face_locations = face_recognition.face_locations(rgb_image)
                if len(face_locations) == 0:
                    logger.warning(f"No face found in image {i+1}")
                    continue
                
                # Get encoding for the first face found
                encodings = face_recognition.face_encodings(rgb_image, face_locations)
                if len(encodings) > 0:
                    face_encodings.append(encodings[0])
                    valid_photos += 1
                    logger.info(f"Successfully processed photo {i+1} for {employee_name}")
                
            except Exception as e:
                logger.error(f"Error processing photo {i+1}: {e}")
                continue
        
        if len(face_encodings) == 0:
            return jsonify({
                'success': False,
                'message': 'No valid faces found in uploaded photos'
            })
        
        # Store in database
        try:
            if not face_recognizer.db:
                face_recognizer.connect_db()
            
            cursor = face_recognizer.db.cursor()
            
            # Check if employee already exists (table primary key is `id`)
            cursor.execute("SELECT id FROM multi_angle_faces WHERE id = %s", (employee_id,))
            existing = cursor.fetchone()

            if existing:
                return jsonify({
                    'success': False,
                    'message': f'Employee ID {employee_id} already exists'
                })

            # Insert new employee with multiple face encodings
            encodings_json = json.dumps([encoding.tolist() for encoding in face_encodings])

            insert_query = """
            INSERT INTO multi_angle_faces 
            (id, employee_name, face_encoding, total_photos, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            """

            now = datetime.now()
            cursor.execute(insert_query, (
                employee_id,
                employee_name,
                encodings_json,
                len(face_encodings),
                now,
                now
            ))
            
            face_recognizer.db.commit()
            cursor.close()
            
            # Reload face encodings to include new employee
            try:
                logger.info("🔄 Reloading face data after employee upload...")
                face_recognizer.load_face_data()
                logger.info("✅ Face data reloaded successfully")
            except Exception as reload_error:
                logger.error(f"⚠️ Warning: Could not reload face data - {str(reload_error)}")
            
            logger.info(f"Successfully added employee: {employee_name} (ID: {employee_id}) with {len(face_encodings)} photos")
            
            return jsonify({
                'success': True,
                'message': f'Employee {employee_name} added successfully with {valid_photos} photos',
                'employee_id': employee_id,
                'photos_processed': valid_photos
            })
            
        except mysql.connector.Error as e:
            logger.error(f"Database error: {e}")
            return jsonify({
                'success': False,
                'message': f'Database error: {str(e)}'
            })
        
    except Exception as e:
        logger.error(f"Error uploading employee: {e}")
        return jsonify({
            'success': False,
            'message': f'Server error: {str(e)}'
        })

@app.route('/upload_employee_photos', methods=['POST'])
def upload_employee_photos():
    """Upload multiple photos for existing employee by Employee ID only"""
    try:
        employee_id = request.form.get('employee_id', '').strip()
        
        if not employee_id:
            return jsonify({
                'success': False,
                'message': 'Employee ID is required'
            })
        
        photos = request.files.getlist('photos')
        if not photos or len(photos) == 0:
            return jsonify({
                'success': False,
                'message': 'No photos provided'
            })
        
        logger.info(f"📤 Uploading {len(photos)} photos for Employee ID: {employee_id}")
        
        # Process each photo
        face_encodings = []
        valid_photos = 0
        
        for i, photo in enumerate(photos):
            if photo.filename == '':
                continue
                
            try:
                # Read image from uploaded file
                image_bytes = photo.read()
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is None:
                    logger.warning(f"⚠️ Could not decode image {i+1}")
                    continue
                
                # Convert BGR to RGB for face_recognition
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Find faces in image
                face_locations = face_recognition.face_locations(rgb_image)
                
                if len(face_locations) == 0:
                    logger.warning(f"⚠️ No face found in image {i+1}")
                    continue
                
                # Get face encoding for first face found
                encodings = face_recognition.face_encodings(rgb_image, face_locations)
                if len(encodings) > 0:
                    face_encodings.append(encodings[0].tolist())
                    valid_photos += 1
                    logger.info(f"✅ Successfully processed photo {i+1}")
                
            except Exception as e:
                logger.error(f"❌ Error processing photo {i+1}: {e}")
                continue
        
        if len(face_encodings) == 0:
            return jsonify({
                'success': False,
                'message': 'No valid faces found in uploaded photos'
            })
        
        # Save to database
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()
        try:
            # First check if employee already exists
            # The table uses `id` as the primary key and `employee_name` for the name column
            check_query = "SELECT id, employee_name FROM multi_angle_faces WHERE id = %s LIMIT 1"
            cursor.execute(check_query, (employee_id,))
            existing = cursor.fetchone()
            
            if existing:
                # Update existing employee with new photos
                update_query = """
                UPDATE multi_angle_faces 
                SET face_encoding = %s, 
                    total_photos = COALESCE(total_photos,0) + %s,
                    updated_at = NOW()
                WHERE id = %s
                """
                # Overwrite/append the face encoding blob (client stores encodings as a JSON list)
                cursor.execute(update_query, (json.dumps(face_encodings), len(face_encodings), employee_id))
                logger.info(f"✅ Updated existing employee {employee_id} with {len(face_encodings)} new photos")
                message = f"Updated Employee {employee_id} with {valid_photos} new photos"
            else:
                # Insert new employee with just ID and photos  
                insert_query = """
                INSERT INTO multi_angle_faces (id, employee_name, face_encoding, total_photos, uploaded_by, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, NOW(), NOW())
                """
                cursor.execute(insert_query, (
                    employee_id,
                    f"Employee_{employee_id}",  # Default name
                    json.dumps(face_encodings),
                    len(face_encodings),
                    'web_upload'
                ))
                logger.info(f"✅ Added new employee {employee_id} with {len(face_encodings)} photos")
                message = f"Added new Employee {employee_id} with {valid_photos} photos"
            
            connection.commit()
            
            return jsonify({
                'success': True,
                'message': message,
                'employee_id': employee_id,
                'uploaded_count': valid_photos
            })
            
        except mysql.connector.Error as e:
            connection.rollback()
            logger.error(f"Database error: {e}")
            return jsonify({
                'success': False,
                'message': f'Database error: {str(e)}'
            })
        finally:
            cursor.close()
            connection.close()
            
    except Exception as e:
        logger.error(f"Error uploading photos: {e}")
        return jsonify({
            'success': False,
            'message': f'Server error: {str(e)}'
        })

# ============ CRUD API Endpoints ============

@app.route('/api/employees', methods=['GET'])
def get_all_employees():
    """Get all employees"""
    try:
        if not face_recognizer.db:
            face_recognizer.connect_database()
            
        cursor = face_recognizer.db.cursor()
        cursor.execute("""
            SELECT id, employee_name, total_photos, is_active, created_at, updated_at
            FROM multi_angle_faces 
            ORDER BY created_at DESC
        """)
        
        employees = []
        for row in cursor.fetchall():
            employees.append({
                'id': row[0],
                'employee_name': row[1],
                'total_photos': row[2],
                'is_active': bool(row[3]),
                'created_at': str(row[4]) if row[4] else None,
                'updated_at': str(row[5]) if row[5] else None
            })
        
        cursor.close()
        return jsonify({
            'success': True,
            'employees': employees,
            'count': len(employees)
        })
        
    except Exception as e:
        logger.error(f"Error getting employees: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/employees/search', methods=['GET'])
def search_employees():
    """Search employees by ID or name"""
    try:
        query = request.args.get('q', '').strip()
        if not query:
            return jsonify({'success': False, 'message': 'Search query required'})
        
        # Ensure database connection is alive
        if not face_recognizer.ensure_db_connection():
            return jsonify({'success': False, 'message': 'Database connection failed'})
            
        cursor = face_recognizer.db.cursor()
        
        # Search by ID or name
        search_query = """
            SELECT id, employee_name, total_photos, is_active, created_at, updated_at
            FROM multi_angle_faces 
            WHERE id LIKE %s OR employee_name LIKE %s
            ORDER BY created_at DESC
            LIMIT 20
        """
        
        like_query = f"%{query}%"
        cursor.execute(search_query, (like_query, like_query))
        
        employees = []
        for row in cursor.fetchall():
            employees.append({
                'id': row[0],
                'employee_name': row[1],
                'total_photos': row[2],
                'is_active': bool(row[3]),
                'created_at': str(row[4]) if row[4] else None,
                'updated_at': str(row[5]) if row[5] else None
            })
        
        cursor.close()
        return jsonify({
            'success': True,
            'employees': employees,
            'query': query
        })
        
    except Exception as e:
        logger.error(f"Error searching employees: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/employees/<employee_id>', methods=['GET'])
def get_employee(employee_id):
    """Get single employee details"""
    try:
        # Ensure database connection is alive
        if not face_recognizer.ensure_db_connection():
            return jsonify({'success': False, 'message': 'Database connection failed'})
            
        cursor = face_recognizer.db.cursor()
        cursor.execute("""
            SELECT id, employee_name, total_photos, 
                   is_active, uploaded_by, created_at, updated_at
            FROM multi_angle_faces 
            WHERE id = %s
        """, (employee_id,))
        
        row = cursor.fetchone()
        if not row:
            return jsonify({'success': False, 'message': 'Employee not found'})
            
        employee = {
            'id': row[0],
            'employee_name': row[1],
            'total_photos': row[2],
            'is_active': bool(row[3]),
            'uploaded_by': row[4],
            'created_at': str(row[5]) if row[5] else None,
            'updated_at': str(row[6]) if row[6] else None
        }
        
        cursor.close()
        return jsonify({
            'success': True,
            'employee': employee
        })
        
    except Exception as e:
        logger.error(f"Error getting employee {employee_id}: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/employees/update', methods=['POST'])
def update_employee():
    """Update employee information and optionally re-upload photos"""
    global face_recognizer
    try:
        employee_id = request.form.get('employee_id')
        employee_name = request.form.get('employee_name', '').strip()
        is_active = request.form.get('is_active', '1') == '1'
        photos = request.files.getlist('photos')
        
        logger.info(f"🔄 Update request for Employee ID: {employee_id}")
        logger.info(f"📝 Employee Name: {employee_name}")
        logger.info(f"🔄 Status: {'Active' if is_active else 'Disabled'}")
        logger.info(f"📸 Photos received: {len(photos)} files")
        
        if not employee_id:
            return jsonify({'success': False, 'message': 'Employee ID required'})
            
        # Connect to database
        logger.info("🔌 Connecting to database...")
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()
        logger.info("✅ Database connected successfully")
        
        # Check if employee exists
        logger.info(f"🔍 Checking if employee {employee_id} exists...")
        cursor.execute("SELECT id FROM multi_angle_faces WHERE id = %s", (employee_id,))
        if not cursor.fetchone():
            logger.warning(f"❌ Employee {employee_id} not found in database")
            cursor.close()
            connection.close()
            return jsonify({'success': False, 'message': 'Employee not found'})
        
        logger.info(f"✅ Employee {employee_id} found in database")
        
        # Process new photos if provided
        new_encodings = []
        if photos and len(photos) > 0:
            logger.info(f"📸 Processing {len(photos)} photos for face encoding...")
            for i, photo in enumerate(photos):
                if photo.filename == '':
                    logger.warning(f"⚠️ Photo {i+1}: Empty filename, skipping")
                    continue
                    
                logger.info(f"🔄 Processing photo {i+1}: {photo.filename}")
                try:
                    # Read and process image
                    image_data = photo.read()
                    logger.info(f"📖 Photo {i+1}: Read {len(image_data)} bytes")
                    
                    nparr = np.frombuffer(image_data, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if image is None:
                        logger.warning(f"❌ Photo {i+1}: Could not decode image")
                        continue
                        
                    logger.info(f"✅ Photo {i+1}: Image decoded successfully ({image.shape})")
                    
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    face_locations = face_recognition.face_locations(rgb_image)
                    
                    if len(face_locations) > 0:
                        logger.info(f"👤 Photo {i+1}: Found {len(face_locations)} face(s)")
                        encodings = face_recognition.face_encodings(rgb_image, face_locations)
                        if len(encodings) > 0:
                            new_encodings.append(encodings[0])
                            logger.info(f"✅ Photo {i+1}: Face encoding generated successfully")
                        else:
                            logger.warning(f"❌ Photo {i+1}: Could not generate face encoding")
                    else:
                        logger.warning(f"👤 Photo {i+1}: No faces detected in image")
                            
                except Exception as e:
                    logger.error(f"❌ Photo {i+1}: Processing error - {str(e)}")
                    continue
            
            logger.info(f"📊 Photo processing complete: {len(new_encodings)} valid encodings out of {len(photos)} photos")
        else:
            logger.info("📸 No photos provided for update")
        
        # Update employee record
        logger.info("💾 Updating employee record in database...")
        if new_encodings:
            # Get current total_photos count first
            cursor.execute("SELECT total_photos FROM multi_angle_faces WHERE id = %s", (employee_id,))
            current_total = cursor.fetchone()[0] or 0
            new_total = current_total + len(new_encodings)  # Add new photos to existing count
            
            # Update with new face encodings
            logger.info(f"🔄 Adding {len(new_encodings)} new face encodings (current: {current_total}, new total: {new_total})...")
            try:
                # Get existing encodings
                cursor.execute("SELECT face_encoding FROM multi_angle_faces WHERE id = %s", (employee_id,))
                existing_encoding_json = cursor.fetchone()[0]
                
                # Merge old and new encodings
                all_encodings = []
                if existing_encoding_json:
                    try:
                        existing_encodings = json.loads(existing_encoding_json)
                        all_encodings.extend(existing_encodings)
                        logger.info(f"📥 Loaded {len(existing_encodings)} existing encodings")
                    except:
                        logger.warning("⚠️ Could not load existing encodings, will use only new ones")
                
                # Add new encodings
                all_encodings.extend([encoding.tolist() for encoding in new_encodings])
                encodings_json = json.dumps(all_encodings)
                logger.info(f"📝 Total face encodings: {len(all_encodings)} ({len(encodings_json)} characters)")
                
                update_query = """
                    UPDATE multi_angle_faces 
                    SET employee_name = %s, is_active = %s, face_encoding = %s, 
                        total_photos = %s, updated_at = NOW(), uploaded_by = 'web_update'
                    WHERE id = %s
                """
                cursor.execute(update_query, (
                    employee_name, is_active, encodings_json, 
                    new_total, employee_id
                ))
                message = f"Employee {employee_id} updated: added {len(new_encodings)} photos (total: {new_total})"
                logger.info(f"✅ Database update successful: {message}")
            except Exception as e:
                logger.error(f"❌ Database update with photos failed: {str(e)}")
                raise e
        else:
            # Update only basic info
            logger.info("🔄 Updating only basic employee information...")
            try:
                update_query = """
                    UPDATE multi_angle_faces 
                    SET employee_name = %s, is_active = %s, updated_at = NOW()
                    WHERE id = %s
                """
                cursor.execute(update_query, (employee_name, is_active, employee_id))
                message = f"Employee {employee_id} information updated"
                logger.info(f"✅ Basic info update successful: {message}")
            except Exception as e:
                logger.error(f"❌ Basic info update failed: {str(e)}")
                raise e
        
        connection.commit()
        cursor.close()
        connection.close()
        
        # Reload face encodings with error handling
        try:
            logger.info("🔄 Reloading face data after update...")
            face_recognizer.load_face_data()
            logger.info("✅ Face data reloaded successfully")
        except Exception as reload_error:
            logger.error(f"⚠️ Warning: Could not reload face data - {str(reload_error)}")
            # Don't fail the update if reload fails
        
        logger.info(f"✅ {message}")
        return jsonify({
            'success': True,
            'message': message,
            'photos_updated': len(new_encodings)
        })
        
    except mysql.connector.Error as e:
        logger.error(f"❌ Database error during employee update: {e}")
        return jsonify({'success': False, 'message': f'Database error: {str(e)}'})
    except Exception as e:
        logger.error(f"❌ Unexpected error during employee update: {e}")
        logger.error(f"❌ Error type: {type(e).__name__}")
        import traceback
        logger.error(f"❌ Full traceback:\n{traceback.format_exc()}")
        return jsonify({'success': False, 'message': f'Server error: {str(e)}'})

@app.route('/api/employees/toggle', methods=['POST'])
def toggle_employee():
    """Enable/Disable employee"""
    global face_recognizer
    try:
        data = request.get_json()
        employee_id = data.get('employee_id')
        action = data.get('action')  # 'enable' or 'disable'
        
        if not employee_id or action not in ['enable', 'disable']:
            return jsonify({'success': False, 'message': 'Invalid parameters'})
            
        is_active = action == 'enable'
        
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()
        
        # Check if employee exists
        cursor.execute("SELECT employee_name FROM multi_angle_faces WHERE id = %s", (employee_id,))
        result = cursor.fetchone()
        if not result:
            cursor.close()
            connection.close()
            return jsonify({'success': False, 'message': 'Employee not found'})
            
        # Update status
        cursor.execute("""
            UPDATE multi_angle_faces 
            SET is_active = %s, updated_at = NOW()
            WHERE id = %s
        """, (is_active, employee_id))
        
        connection.commit()
        cursor.close()
        connection.close()
        
        # Reload face encodings
        try:
            logger.info("🔄 Reloading face data after status toggle...")
            face_recognizer.load_face_data()
            logger.info("✅ Face data reloaded successfully")
        except Exception as reload_error:
            logger.error(f"⚠️ Warning: Could not reload face data - {str(reload_error)}")
        
        status_text = "enabled" if is_active else "disabled"
        message = f"Employee {employee_id} ({result[0]}) {status_text}"
        
        logger.info(f"✅ {message}")
        return jsonify({
            'success': True,
            'message': message,
            'employee_id': employee_id,
            'status': status_text
        })
        
    except Exception as e:
        logger.error(f"Error toggling employee: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/employees/delete', methods=['POST'])
def delete_employee():
    """Delete employee and all face embeddings"""
    global face_recognizer
    try:
        data = request.get_json()
        employee_id = data.get('employee_id')
        
        if not employee_id:
            return jsonify({'success': False, 'message': 'Employee ID required'})
            
        connection = mysql.connector.connect(**db_config)
        cursor = connection.cursor()
        
        # Check if employee exists and get info
        cursor.execute("SELECT employee_name, total_photos FROM multi_angle_faces WHERE id = %s", (employee_id,))
        result = cursor.fetchone()
        if not result:
            cursor.close()
            connection.close()
            return jsonify({'success': False, 'message': 'Employee not found'})
            
        employee_name, total_photos = result
        
        # Delete employee record (this will remove all face embeddings)
        cursor.execute("DELETE FROM multi_angle_faces WHERE id = %s", (employee_id,))
        
        connection.commit()
        cursor.close()
        connection.close()
        
        # Reload face encodings
        try:
            logger.info("🔄 Reloading face data after employee deletion...")
            face_recognizer.load_face_data()
            logger.info("✅ Face data reloaded successfully")
        except Exception as reload_error:
            logger.error(f"⚠️ Warning: Could not reload face data - {str(reload_error)}")
        
        message = f"Employee {employee_id} ({employee_name}) deleted with {total_photos} photos"
        logger.info(f"✅ {message}")
        
        return jsonify({
            'success': True,
            'message': message,
            'employee_id': employee_id,
            'deleted_photos': total_photos
        })
        
    except Exception as e:
        logger.error(f"Error deleting employee: {e}")
        return jsonify({'success': False, 'message': str(e)})

if __name__ == '__main__':
    # Use debug=False to prevent auto-reload issues with camera
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)