#!/usr/bin/env python3
"""
Face Recognition and Embedding Script for HRM Employee Data
==========================================================
This script processes employee images to extract face embeddings and store them in database.

Author: Your Name
Date: October 3, 2025
"""

import cv2
import face_recognition
import numpy as np
import mysql.connector
from mysql.connector import Error
import requests
import logging
from datetime import datetime
import os
import tempfile
from typing import Dict, List, Optional, Tuple
import json

# Configuration for Face Recognition
FACE_CONFIG = {
    # Database Configuration (same as main script)
    'DB_HOST': 'localhost',
    'DB_USER': 'root', 
    'DB_PASSWORD': 'root',
    'DB_NAME': 'hrm_database',
    'DB_PORT': 3306,
    
    # Face Recognition Settings
    'FACE_MODEL': 'hog',  # 'hog' for CPU, 'cnn' for GPU
    'NUM_JITTERS': 1,     # Number of times to re-sample face for encoding
    'TOLERANCE': 0.6,     # Face matching tolerance (lower = more strict)
    
    # Image Processing
    'MAX_IMAGE_SIZE': (800, 600),  # Max image dimensions for processing
    'SUPPORTED_FORMATS': ['.jpg', '.jpeg', '.png', '.bmp'],
    
    # Logging
    'LOG_LEVEL': 'INFO',
    'LOG_FORMAT': '%(asctime)s - %(levelname)s - %(message)s'
}

class FaceEmbeddingProcessor:
    """Process employee images to extract and store face embeddings"""
    
    def __init__(self):
        """Initialize the face recognition processor"""
        self.setup_logging()
        self.db_connection = None
        self.processing_stats = {
            'total_employees': 0,
            'images_processed': 0,
            'faces_detected': 0,
            'embeddings_stored': 0,
            'errors': 0,
            'error_details': []
        }
        
    def setup_logging(self):
        """Setup logging configuration"""
        import sys
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        file_handler = logging.FileHandler(
            f'face_processing_{datetime.now().strftime("%Y%m%d")}.log', 
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter(FACE_CONFIG['LOG_FORMAT'])
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, FACE_CONFIG['LOG_LEVEL']))
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        
    def connect_database(self) -> bool:
        """Connect to MySQL database"""
        try:
            self.logger.info("[DB] Connecting to MySQL database...")
            
            self.db_connection = mysql.connector.connect(
                host=FACE_CONFIG['DB_HOST'],
                user=FACE_CONFIG['DB_USER'],
                password=FACE_CONFIG['DB_PASSWORD'],
                database=FACE_CONFIG['DB_NAME'],
                port=FACE_CONFIG['DB_PORT'],
                autocommit=True
            )
            
            self.logger.info("[OK] Database connection successful")
            return True
            
        except Error as e:
            self.logger.error(f"[ERROR] Database connection failed: {e}")
            return False
            
    def create_face_tables(self) -> bool:
        """Create face embeddings table"""
        try:
            self.logger.info("[DB] Creating face embeddings table...")
            
            cursor = self.db_connection.cursor()
            
            # Create face embeddings table
            face_embeddings_table = """
                CREATE TABLE IF NOT EXISTS employee_face_embeddings (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    employee_id VARCHAR(50) NOT NULL,
                    embedding_vector JSON NOT NULL,
                    confidence_score FLOAT,
                    face_locations JSON,
                    image_url TEXT,
                    image_processed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processing_status ENUM('success', 'no_face', 'multiple_faces', 'error') DEFAULT 'success',
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    FOREIGN KEY (employee_id) REFERENCES employees(employee_id) ON DELETE CASCADE,
                    INDEX idx_employee_id (employee_id),
                    INDEX idx_processing_status (processing_status)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
            
            # Create face matching results table for future use
            face_matching_table = """
                CREATE TABLE IF NOT EXISTS face_matching_results (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    source_employee_id VARCHAR(50),
                    matched_employee_id VARCHAR(50),
                    similarity_score FLOAT,
                    match_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_verified BOOLEAN DEFAULT FALSE,
                    notes TEXT,
                    INDEX idx_source_employee (source_employee_id),
                    INDEX idx_matched_employee (matched_employee_id),
                    INDEX idx_similarity (similarity_score)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
            
            cursor.execute(face_embeddings_table)
            cursor.execute(face_matching_table)
            cursor.close()
            
            self.logger.info("[OK] Face recognition tables created successfully")
            return True
            
        except Error as e:
            self.logger.error(f"[ERROR] Failed to create face tables: {e}")
            return False
            
    def get_employees_with_images(self) -> List[Dict]:
        """Get all employees who have images"""
        try:
            cursor = self.db_connection.cursor(dictionary=True)
            
            query = """
                SELECT employee_id, full_name, user_img 
                FROM employees 
                WHERE user_img IS NOT NULL 
                AND user_img != '' 
                AND user_img NOT LIKE '%default%'
                ORDER BY employee_id
            """
            
            cursor.execute(query)
            employees = cursor.fetchall()
            cursor.close()
            
            self.logger.info(f"[INFO] Found {len(employees)} employees with images")
            return employees
            
        except Error as e:
            self.logger.error(f"[ERROR] Failed to get employees with images: {e}")
            return []
            
    def download_image(self, image_url: str) -> Optional[np.ndarray]:
        """Download and process employee image"""
        try:
            # Handle relative URLs by adding base URL if needed
            if image_url.startswith('uploads/'):
                # You may need to adjust this base URL according to your HRM system
                base_url = 'https://hrm.umanerp.com/'
                image_url = base_url + image_url
                
            self.logger.debug(f"[DOWNLOAD] Downloading image: {image_url}")
            
            # Download image
            response = requests.get(image_url, timeout=30, stream=True)
            response.raise_for_status()
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)
                temp_path = temp_file.name
                
            # Load image using OpenCV
            image = cv2.imread(temp_path)
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            if image is None:
                raise ValueError("Could not load image")
                
            # Resize if too large
            height, width = image.shape[:2]
            max_width, max_height = FACE_CONFIG['MAX_IMAGE_SIZE']
            
            if width > max_width or height > max_height:
                scale = min(max_width / width, max_height / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height))
                
            # Convert BGR to RGB for face_recognition
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            return image_rgb
            
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to download image {image_url}: {e}")
            return None
            
    def extract_face_embedding(self, image: np.ndarray) -> Tuple[Optional[List], str, Optional[List]]:
        """Extract face embedding from image"""
        try:
            # Find face locations
            face_locations = face_recognition.face_locations(
                image, 
                model=FACE_CONFIG['FACE_MODEL']
            )
            
            if len(face_locations) == 0:
                return None, 'no_face', None
            elif len(face_locations) > 1:
                self.logger.warning("[WARN] Multiple faces detected, using largest face")
                # Use the largest face (by area)
                largest_face = max(face_locations, key=lambda loc: (loc[2] - loc[0]) * (loc[1] - loc[3]))
                face_locations = [largest_face]
                
            # Extract face encoding
            face_encodings = face_recognition.face_encodings(
                image, 
                face_locations,
                num_jitters=FACE_CONFIG['NUM_JITTERS']
            )
            
            if len(face_encodings) == 0:
                return None, 'error', face_locations
                
            # Convert numpy array to list for JSON storage
            embedding_list = face_encodings[0].tolist()
            
            return embedding_list, 'success', face_locations
            
        except Exception as e:
            self.logger.error(f"[ERROR] Face embedding extraction failed: {e}")
            return None, 'error', None
            
    def store_face_embedding(self, employee_id: str, embedding: List, status: str, 
                           face_locations: Optional[List] = None, 
                           image_url: str = "", error_msg: str = "") -> bool:
        """Store face embedding in database"""
        try:
            cursor = self.db_connection.cursor()
            
            # Check if embedding already exists
            cursor.execute(
                "SELECT id FROM employee_face_embeddings WHERE employee_id = %s",
                (employee_id,)
            )
            existing = cursor.fetchone()
            
            if existing:
                # Update existing record
                query = """
                    UPDATE employee_face_embeddings 
                    SET embedding_vector = %s, face_locations = %s, image_url = %s,
                        processing_status = %s, error_message = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE employee_id = %s
                """
                values = (
                    json.dumps(embedding) if embedding else None,
                    json.dumps(face_locations) if face_locations else None,
                    image_url,
                    status,
                    error_msg,
                    employee_id
                )
            else:
                # Insert new record
                query = """
                    INSERT INTO employee_face_embeddings 
                    (employee_id, embedding_vector, face_locations, image_url, processing_status, error_message)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """
                values = (
                    employee_id,
                    json.dumps(embedding) if embedding else None,
                    json.dumps(face_locations) if face_locations else None,
                    image_url,
                    status,
                    error_msg
                )
                
            cursor.execute(query, values)
            cursor.close()
            return True
            
        except Error as e:
            self.logger.error(f"[ERROR] Failed to store embedding for {employee_id}: {e}")
            return False
            
    def process_employee_image(self, employee: Dict) -> bool:
        """Process single employee image"""
        try:
            employee_id = employee['employee_id']
            full_name = employee['full_name']
            image_url = employee['user_img']
            
            self.logger.info(f"[PROCESS] Processing {employee_id} - {full_name}")
            
            # Download image
            image = self.download_image(image_url)
            if image is None:
                self.store_face_embedding(
                    employee_id, None, 'error', None, image_url, 
                    "Failed to download image"
                )
                self.processing_stats['errors'] += 1
                return False
                
            self.processing_stats['images_processed'] += 1
            
            # Extract face embedding
            embedding, status, face_locations = self.extract_face_embedding(image)
            
            if status == 'success':
                self.processing_stats['faces_detected'] += 1
                
            # Store in database
            success = self.store_face_embedding(
                employee_id, embedding, status, face_locations, image_url
            )
            
            if success and embedding:
                self.processing_stats['embeddings_stored'] += 1
                self.logger.info(f"[OK] Face embedding stored for {employee_id}")
            elif status == 'no_face':
                self.logger.warning(f"[WARN] No face detected for {employee_id}")
            else:
                self.processing_stats['errors'] += 1
                self.logger.error(f"[ERROR] Failed to process {employee_id}")
                
            return success
            
        except Exception as e:
            error_msg = f"Employee {employee.get('employee_id', 'Unknown')}: {str(e)}"
            self.processing_stats['error_details'].append(error_msg)
            self.processing_stats['errors'] += 1
            self.logger.error(f"[ERROR] {error_msg}")
            return False
            
    def run_face_processing(self):
        """Main face processing function"""
        start_time = datetime.now()
        
        try:
            self.logger.info("[START] Face Recognition Processing Started")
            self.logger.info("=" * 50)
            
            # Connect to database
            if not self.connect_database():
                raise Exception("Database connection failed")
                
            # Create face tables
            if not self.create_face_tables():
                raise Exception("Face table creation failed")
                
            # Get employees with images
            employees = self.get_employees_with_images()
            if not employees:
                self.logger.warning("[WARN] No employees found with images")
                return True
                
            self.processing_stats['total_employees'] = len(employees)
            
            # Process each employee
            for i, employee in enumerate(employees, 1):
                try:
                    self.process_employee_image(employee)
                    
                    # Progress update every 10 employees
                    if i % 10 == 0:
                        self.logger.info(f"[PROGRESS] {i}/{len(employees)} processed")
                        
                except Exception as e:
                    self.logger.error(f"[ERROR] Failed to process employee {i}: {e}")
                    
            # Calculate duration and show results
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            self.logger.info("[COMPLETE] Face Processing Completed!")
            self.logger.info("=" * 50)
            self.logger.info("[SUMMARY] Results:")
            self.logger.info(f"   Total Employees: {self.processing_stats['total_employees']}")
            self.logger.info(f"   Images Processed: {self.processing_stats['images_processed']}")
            self.logger.info(f"   Faces Detected: {self.processing_stats['faces_detected']}")
            self.logger.info(f"   Embeddings Stored: {self.processing_stats['embeddings_stored']}")
            self.logger.info(f"   Errors: {self.processing_stats['errors']}")
            self.logger.info(f"   Duration: {duration:.2f} seconds")
            
            # Show some error details if any
            if self.processing_stats['error_details']:
                self.logger.warning("[WARN] Sample errors:")
                for error in self.processing_stats['error_details'][:5]:
                    self.logger.warning(f"   - {error}")
                if len(self.processing_stats['error_details']) > 5:
                    remaining = len(self.processing_stats['error_details']) - 5
                    self.logger.warning(f"   ... and {remaining} more errors")
                    
            return self.processing_stats['errors'] == 0
            
        except Exception as e:
            self.logger.error(f"[ERROR] Face processing failed: {e}")
            return False
            
        finally:
            if self.db_connection and self.db_connection.is_connected():
                self.db_connection.close()
                self.logger.info("[OK] Database connection closed")
                
    def display_help(self):
        """Display help information"""
        help_text = """
Face Recognition Processing - Help
=================================

This script processes employee images to extract face embeddings.

Requirements:
- face_recognition library
- opencv-python
- numpy
- All employee images should be accessible via URL

Usage:
    python face_processing.py

Features:
- Automatic face detection and embedding extraction
- Handles multiple faces (uses largest face)
- Stores embeddings as JSON in database
- Error handling for download failures
- Progress tracking
- Comprehensive logging

Database Tables Created:
- employee_face_embeddings: Face embeddings and metadata
- face_matching_results: For future face matching operations

Install Requirements:
    pip install face_recognition opencv-python numpy requests
        """
        print(help_text)

def main():
    """Main function"""
    import sys
    
    # Check for help
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        processor = FaceEmbeddingProcessor()
        processor.display_help()
        return
        
    # Run face processing
    processor = FaceEmbeddingProcessor()
    success = processor.run_face_processing()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()