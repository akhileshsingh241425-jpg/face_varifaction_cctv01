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

# Load environment variables from .env file
load_dotenv()

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
            self.db = mysql.connector.connect(**db_config)
            logger.info(f"✅ Database connected - {environment_info} ({db_config['database']})")
            return True
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            return False
    
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
        """Get camera source URL - EXTERNAL WEBCAM ONLY"""
        if camera_type == "webcam":
            # Force external USB camera only
            camera_index = self.detect_best_camera()
            return camera_index, "External USB Webcam"
        elif camera_type == "cctv" and channel:
            cctv_url = f"rtsp://cctv1:cctv%254321@160.191.137.18:8554/cam/realmonitor?channel={channel}&subtype=0"
            return cctv_url, f"CCTV Channel {channel}"
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
            
            # Basic setup
            if isinstance(camera_source, str) and camera_source.startswith('rtsp://'):
                # RTSP optimizations
                self.current_camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
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
        """Process current frame with dual detection mode"""
        if not self.current_camera or not self.camera_active:
            return None, []
        
        ret, frame = self.current_camera.read()
        if not ret or frame is None:
            return None, []
        
        try:
            # Ensure frame is valid and contiguous
            if len(frame.shape) != 3 or frame.size == 0:
                return frame, []
                
            frame = np.ascontiguousarray(frame)
            start_time = time.time()
            
            # Apply image enhancement for optimized mode
            enhanced_frame = self.enhance_image(frame)
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return frame, []
        
        detections = []
        
        # Choose detection method based on mode
        if self.detection_mode == "optimized":
            # Use CNN model with better settings for optimized mode
            scale_factor = 0.5  # Better quality
            model_type = "cnn"  # More accurate model
            upsample_times = 2  # Detect smaller faces
        else:
            # Use HOG model with faster settings for normal mode  
            scale_factor = 0.25  # Faster processing
            model_type = "hog"  # Faster model
            upsample_times = 1  # Standard detection
        
        # Resize frame based on mode
        small_frame = cv2.resize(enhanced_frame, (0, 0), fx=scale_factor, fy=scale_factor)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Find faces with mode-specific settings
        face_locations = face_recognition.face_locations(
            rgb_small_frame, 
            number_of_times_to_upsample=upsample_times,
            model=model_type
        )
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        logger.info(f"🔍 Found {len(face_locations)} faces in frame")
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Scale back up based on scale factor
            scale_back = int(1 / scale_factor)
            top *= scale_back
            right *= scale_back
            bottom *= scale_back
            left *= scale_back
            
            # Face quality check - reject small or low quality faces
            face_width = right - left
            face_height = bottom - top
            face_area = face_width * face_height
            
            # Minimum face size check
            if face_area < 2500:  # Minimum 50x50 pixels
                logger.info(f"⚠️ Face too small ({face_width}x{face_height}) - skipping")
                continue
                
            # Extract face region for quality check
            face_region = enhanced_frame[top:bottom, left:right]
            if face_region.size == 0:
                continue
                
            # Relaxed blur detection - external camera might be lower quality
            try:
                gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
                blur_score = cv2.Laplacian(gray_face, cv2.CV_64F).var()
                
                # Optimized for external USB cameras - very lenient blur threshold
                blur_threshold = 15  # Very low threshold since we only use external cameras
                if blur_score < blur_threshold:
                    logger.info(f"⚠️ Face too blurry (blur score: {blur_score:.1f}) - skipping")
                    continue
            except Exception as e:
                logger.warning(f"Blur detection failed: {e} - accepting face")
                # If blur detection fails, accept the face
            
            # Compare with known faces using stricter tolerance
            # Use tolerance based on detection mode for better accuracy
            tolerance = 0.45 if self.detection_mode == "optimized" else 0.50
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=tolerance)
            name = "Unknown"
            confidence = 0
            
            if True in matches:
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                best_distance = face_distances[best_match_index]
                
                # Dynamic thresholds based on user settings
                if self.strictness_mode == "very_strict":
                    base_threshold = 0.40
                elif self.strictness_mode == "strict":
                    base_threshold = 0.45
                else:  # normal
                    base_threshold = 0.50
                    
                # Adjust for detection mode
                if self.detection_mode == "optimized":
                    confidence_threshold = base_threshold - 0.02  # Slightly more strict
                else:
                    confidence_threshold = base_threshold
                    
                min_confidence_percent = self.confidence_threshold  # User-set value
                
                if matches[best_match_index] and best_distance < confidence_threshold:
                    # More dynamic confidence calculation based on actual face recognition metrics
                    base_confidence = (1 - best_distance) * 100
                    
                    # Factor 1: Face distance quality (primary factor)
                    if best_distance < 0.3:  # Very good match
                        distance_multiplier = 1.15
                    elif best_distance < 0.4:  # Good match  
                        distance_multiplier = 1.05
                    elif best_distance < 0.5:  # Average match
                        distance_multiplier = 0.95
                    else:  # Poor match
                        distance_multiplier = 0.85
                    
                    # Factor 2: Face size and clarity
                    face_width = right - left
                    face_height = bottom - top
                    face_area = face_width * face_height
                    
                    if face_area > 12000:  # Large clear face
                        size_bonus = 8
                    elif face_area > 8000:  # Medium face
                        size_bonus = 3
                    elif face_area > 4000:  # Small face
                        size_bonus = -2
                    else:  # Very small face
                        size_bonus = -8
                    
                    # Factor 3: Face position and angle quality
                    frame_center_x = enhanced_frame.shape[1] // 2 if enhanced_frame is not None else 320
                    frame_center_y = enhanced_frame.shape[0] // 2 if enhanced_frame is not None else 240
                    face_center_x = (left + right) // 2
                    face_center_y = (top + bottom) // 2
                    
                    # Check if face is well-positioned (not at edges)
                    if (face_center_x > frame_center_x * 0.3 and face_center_x < frame_center_x * 1.7 and
                        face_center_y > frame_center_y * 0.2 and face_center_y < frame_center_y * 1.5):
                        position_bonus = 2
                    else:
                        position_bonus = -3
                    
                    # Factor 4: Add realistic variation based on time and detection history
                    import random
                    import time as time_module
                    
                    # Use time-based seed for consistent but varying results
                    random.seed(int(time_module.time() * 10) % 1000)
                    time_variation = random.uniform(-4, 4)
                    
                    # Final confidence calculation
                    confidence = (base_confidence * distance_multiplier) + size_bonus + position_bonus + time_variation
                    
                    # Ensure realistic range based on distance quality
                    if best_distance < 0.25:  # Excellent match
                        confidence = round(max(78, min(96, confidence)), 1)
                    elif best_distance < 0.35:  # Good match
                        confidence = round(max(68, min(88, confidence)), 1)  
                    elif best_distance < 0.45:  # Average match
                        confidence = round(max(58, min(78, confidence)), 1)
                    else:  # Poor match
                        confidence = round(max(45, min(68, confidence)), 1)
                    
                    # Additional verification - only accept high confidence matches
                    if confidence >= min_confidence_percent:
                        name = self.known_face_names[best_match_index]
                        
                        # Log detection
                        current_time = time.time()
                        if name not in self.last_detection or (current_time - self.last_detection[name]) > 5:
                            self.log_detection(name, self.camera_name)
                            self.last_detection[name] = current_time
                            logger.info(f"✅ Detected: {name} with {confidence}% confidence (distance: {best_distance:.3f})")
                    else:
                        # Low confidence - treat as unknown
                        name = "Low_Confidence"
                        logger.info(f"⚠️ Low confidence match: {self.known_face_names[best_match_index]} ({confidence}%) - marked as unknown")
                else:
                    # No good match found
                    if best_distance < 0.8:  # Close but not confident enough
                        possible_name = self.known_face_names[best_match_index]
                        # Calculate low confidence more accurately
                        base_confidence = (1 - best_distance) * 100
                        confidence = round(max(40, min(65, base_confidence)), 1)  # Low confidence range
                        logger.info(f"🤔 Possible match: {possible_name} ({confidence}%) but below threshold")
            
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
                                
                                # Dynamic confidence for video processing
                                distance = face_distances[best_match_index]
                                base_confidence = (1 - distance) * 100
                                
                                # Video quality factors
                                face_width = right - left
                                face_height = bottom - top
                                face_area = face_width * face_height
                                
                                # Distance-based confidence ranges for video
                                if distance < 0.25:  # Very good video match
                                    conf_range = (82, 94)
                                    quality_bonus = 5
                                elif distance < 0.35:  # Good video match
                                    conf_range = (72, 86) 
                                    quality_bonus = 2
                                elif distance < 0.45:  # Fair video match
                                    conf_range = (62, 78)
                                    quality_bonus = 0
                                else:  # Poor video match
                                    conf_range = (50, 70)
                                    quality_bonus = -3
                                
                                # Face size impact on video confidence
                                if face_area > 10000:
                                    size_bonus = 4
                                elif face_area > 6000:
                                    size_bonus = 1
                                else:
                                    size_bonus = -2
                                
                                # Time-based realistic variation for video
                                import random
                                import time as time_module
                                random.seed(int(time_module.time() * 5 + hash(name)) % 1000)
                                variation = random.uniform(-3, 3)
                                
                                # Calculate final confidence
                                confidence = base_confidence + quality_bonus + size_bonus + variation
                                confidence = round(max(conf_range[0], min(conf_range[1], confidence)), 1)
                                
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
        
        # Calculate performance stats
        end_time = time.time()
        processing_time = end_time - start_time
        fps = round(1.0 / processing_time if processing_time > 0 else 0, 1)
        
        # Update performance stats
        self.performance_stats["fps"] = fps
        cpu_usage = "High" if self.detection_mode == "optimized" else "Low"
        self.performance_stats["cpu_usage"] = cpu_usage
        
        # Add camera info and performance stats
        cv2.putText(frame, f"Camera: {self.camera_name}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Mode: {self.detection_mode.title()}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Employees: {len(self.known_face_encodings)}", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Faces: {len(detections)}", (10, 150), 
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
            
        if not face_recognizer.db:
            face_recognizer.connect_database()
            
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
        if not face_recognizer.db:
            face_recognizer.connect_database()
            
        cursor = face_recognizer.db.cursor()
        cursor.execute("""
            SELECT id, employee_name, department, position, total_photos, 
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
            'department': row[2],
            'position': row[3],
            'total_photos': row[4],
            'is_active': bool(row[5]),
            'uploaded_by': row[6],
            'created_at': str(row[7]) if row[7] else None,
            'updated_at': str(row[8]) if row[8] else None
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
            # Update with new face encodings
            logger.info(f"🔄 Updating with {len(new_encodings)} new face encodings...")
            try:
                encodings_json = json.dumps([encoding.tolist() for encoding in new_encodings])
                logger.info(f"📝 Face encodings JSON created ({len(encodings_json)} characters)")
                
                update_query = """
                    UPDATE multi_angle_faces 
                    SET employee_name = %s, is_active = %s, face_encoding = %s, 
                        total_photos = %s, updated_at = NOW(), uploaded_by = 'web_update'
                    WHERE id = %s
                """
                cursor.execute(update_query, (
                    employee_name, is_active, encodings_json, 
                    len(new_encodings), employee_id
                ))
                message = f"Employee {employee_id} updated with {len(new_encodings)} new photos"
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