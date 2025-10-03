#!/usr/bin/env python3
"""
Video Recording and Analysis System
==================================
Record 1-minute CCTV videos and analyze them offline
"""

import cv2
import face_recognition
import numpy as np
import mysql.connector
from datetime import datetime
import logging
import threading
import time
import json
import os
from flask import Flask, render_template, request, jsonify, send_file

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class VideoAnalysisSystem:
    def __init__(self):
        self.db = None
        self.known_face_encodings = []
        self.known_face_names = []
        self.recording = False
        self.analyzing = False
        self.videos_folder = "recorded_videos"
        
        # Create videos folder if not exists
        if not os.path.exists(self.videos_folder):
            os.makedirs(self.videos_folder)
    
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
    
    def get_cctv_url(self, channel):
        """Get CCTV URL for given channel"""
        return f"rtsp://cctv1:cctv%254321@160.191.137.18:8554/cam/realmonitor?channel={channel}&subtype=0"
    
    def record_video(self, channel, duration_seconds=60):
        """Record video from CCTV for specified duration"""
        try:
            self.recording = True
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.videos_folder}/cctv_ch{channel}_{timestamp}.mp4"
            
            # Get CCTV URL
            cctv_url = self.get_cctv_url(channel)
            logger.info(f"🎥 Starting recording from Channel {channel}")
            
            # Open CCTV stream
            cap = cv2.VideoCapture(cctv_url)
            
            # RTSP optimizations
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            try:
                cap.set(cv2.CAP_PROP_TIMEOUT, 10000)
            except AttributeError:
                pass
            
            if not cap.isOpened():
                logger.error(f"❌ Cannot connect to CCTV Channel {channel}")
                return None, "Cannot connect to CCTV"
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
            
            logger.info(f"📹 Recording: {width}x{height} @ {fps}FPS for {duration_seconds}s")
            
            # Setup video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
            
            start_time = time.time()
            frame_count = 0
            
            while self.recording and (time.time() - start_time) < duration_seconds:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("⚠️ Failed to read frame, continuing...")
                    continue
                
                out.write(frame)
                frame_count += 1
                
                # Log progress every 10 seconds
                elapsed = time.time() - start_time
                if int(elapsed) % 10 == 0 and int(elapsed) > 0:
                    logger.info(f"🎬 Recording... {int(elapsed)}/{duration_seconds}s ({frame_count} frames)")
            
            # Cleanup
            cap.release()
            out.release()
            
            if frame_count > 0:
                logger.info(f"✅ Recording completed: {filename} ({frame_count} frames)")
                return filename, f"Recording completed: {frame_count} frames"
            else:
                logger.error("❌ No frames recorded")
                return None, "No frames recorded"
            
        except Exception as e:
            logger.error(f"❌ Recording error: {e}")
            return None, str(e)
        finally:
            self.recording = False
    
    def analyze_video(self, video_path):
        """Analyze recorded video for face recognition"""
        try:
            self.analyzing = True
            logger.info(f"🔍 Starting video analysis: {video_path}")
            
            if not os.path.exists(video_path):
                return False, "Video file not found"
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False, "Cannot open video file"
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"📊 Video info: {total_frames} frames @ {fps:.1f} FPS")
            
            detections = {}
            processed_frames = 0
            
            # Process every 30th frame (1 frame per second if 30 FPS)
            frame_skip = int(fps) if fps > 0 else 30
            
            while self.analyzing:
                ret, frame = cap.read()
                if not ret:
                    break
                
                processed_frames += 1
                
                # Skip frames for performance
                if processed_frames % frame_skip != 0:
                    continue
                
                # Resize for faster processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                # Find faces
                face_locations = face_recognition.face_locations(rgb_small_frame)
                if len(face_locations) == 0:
                    continue
                
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    
                    if True in matches:
                        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        
                        if matches[best_match_index] and face_distances[best_match_index] < 0.6:
                            name = self.known_face_names[best_match_index]
                            confidence = round((1 - face_distances[best_match_index]) * 100, 1)
                            
                            # Count detections
                            if name not in detections:
                                detections[name] = {'count': 0, 'max_confidence': 0}
                            
                            detections[name]['count'] += 1
                            detections[name]['max_confidence'] = max(detections[name]['max_confidence'], confidence)
                
                # Log progress
                progress = (processed_frames / total_frames) * 100
                if int(progress) % 20 == 0:
                    logger.info(f"🔄 Analysis progress: {progress:.1f}% ({len(detections)} people found)")
            
            cap.release()
            
            # Log detections to database
            self.log_video_analysis(video_path, detections)
            
            logger.info(f"✅ Analysis completed: {len(detections)} unique people detected")
            return True, detections
            
        except Exception as e:
            logger.error(f"❌ Analysis error: {e}")
            return False, str(e)
        finally:
            self.analyzing = False
    
    def log_video_analysis(self, video_path, detections):
        """Log video analysis results to database"""
        try:
            if not self.db:
                self.connect_database()
            
            cursor = self.db.cursor()
            video_name = os.path.basename(video_path)
            
            for name, data in detections.items():
                cursor.execute("""
                    INSERT INTO employee_movements 
                    (employee_id, employee_name, camera_id, camera_location, movement_type) 
                    VALUES (%s, %s, %s, %s, %s)
                """, (1, f"{name} (Video: {data['count']} detections, {data['max_confidence']:.1f}%)", 
                      2, f"Video Analysis: {video_name}", 'entry'))
            
            self.db.commit()
            logger.info(f"📝 Logged {len(detections)} detections to database")
            
        except Exception as e:
            logger.error(f"❌ Error logging analysis: {e}")
    
    def get_recorded_videos(self):
        """Get list of recorded videos"""
        try:
            videos = []
            for filename in os.listdir(self.videos_folder):
                if filename.endswith('.mp4'):
                    filepath = os.path.join(self.videos_folder, filename)
                    size = os.path.getsize(filepath) / (1024 * 1024)  # MB
                    mtime = os.path.getmtime(filepath)
                    videos.append({
                        'filename': filename,
                        'path': filepath,
                        'size_mb': round(size, 2),
                        'created': datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
                    })
            
            # Sort by creation time (newest first)
            videos.sort(key=lambda x: x['created'], reverse=True)
            return videos
        except Exception as e:
            logger.error(f"Error getting videos: {e}")
            return []

# Global video analysis instance
video_analyzer = VideoAnalysisSystem()

@app.route('/')
def index():
    """Main page"""
    return render_template('video_analysis.html')

@app.route('/api/load_faces', methods=['POST'])
def load_faces():
    """Load face data"""
    success = video_analyzer.load_face_data()
    return jsonify({
        'success': success,
        'count': len(video_analyzer.known_face_encodings),
        'message': f'Loaded {len(video_analyzer.known_face_encodings)} employee faces' if success else 'Failed to load faces'
    })

@app.route('/api/record_video', methods=['POST'])
def record_video():
    """Start video recording"""
    data = request.get_json()
    channel = data.get('channel', '127')
    duration = data.get('duration', 60)
    
    if video_analyzer.recording:
        return jsonify({'success': False, 'message': 'Recording already in progress'})
    
    # Start recording in background thread
    def record_thread():
        filename, message = video_analyzer.record_video(channel, duration)
        logger.info(f"Recording result: {message}")
    
    threading.Thread(target=record_thread, daemon=True).start()
    
    return jsonify({
        'success': True,
        'message': f'Started recording Channel {channel} for {duration} seconds'
    })

@app.route('/api/analyze_video', methods=['POST'])
def analyze_video():
    """Analyze video"""
    data = request.get_json()
    video_path = data.get('video_path')
    
    if not video_path:
        return jsonify({'success': False, 'message': 'Video path required'})
    
    if video_analyzer.analyzing:
        return jsonify({'success': False, 'message': 'Analysis already in progress'})
    
    # Start analysis in background thread
    def analyze_thread():
        success, result = video_analyzer.analyze_video(video_path)
        if success:
            logger.info(f"Analysis completed: {len(result)} people detected")
        else:
            logger.error(f"Analysis failed: {result}")
    
    threading.Thread(target=analyze_thread, daemon=True).start()
    
    return jsonify({
        'success': True,
        'message': 'Started video analysis'
    })

@app.route('/api/status')
def get_status():
    """Get system status"""
    return jsonify({
        'recording': video_analyzer.recording,
        'analyzing': video_analyzer.analyzing,
        'faces_loaded': len(video_analyzer.known_face_encodings)
    })

@app.route('/api/videos')
def get_videos():
    """Get recorded videos list"""
    videos = video_analyzer.get_recorded_videos()
    return jsonify({'videos': videos})

@app.route('/api/download_video/<path:filename>')
def download_video(filename):
    """Download recorded video"""
    try:
        filepath = os.path.join(video_analyzer.videos_folder, filename)
        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=True)
        else:
            return jsonify({'error': 'Video not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/detections')
def get_detections():
    """Get recent detections"""
    try:
        if not video_analyzer.db:
            video_analyzer.connect_database()
        
        cursor = video_analyzer.db.cursor()
        cursor.execute("""
            SELECT employee_name, camera_location, timestamp 
            FROM employee_movements 
            ORDER BY timestamp DESC 
            LIMIT 20
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
    app.run(debug=True, host='0.0.0.0', port=5001, threaded=True)