#!/usr/bin/env python3
"""
HRM Employee Data Sync Script
=============================
Single Python script to fetch             self.logger.info("[DB] Creating database tables...")mployee data from HRM API and store in MySQL database.

Author: Your Name
Date: October 3, 2025
"""

import requests
import mysql.connector
from mysql.connector import Error
import json
import logging
from datetime import datetime
import time
import os
from typing import Dict, List, Optional, Tuple

# Configuration
CONFIG = {
    # Database Configuration
    'DB_HOST': 'localhost',
    'DB_USER': 'root',
    'DB_PASSWORD': 'root',  # Root password is 'root'
    'DB_NAME': 'hrm_database',
    'DB_PORT': 3306,
    
    # API Configuration
    'API_URL': 'https://hrm.umanerp.com/api/users/getEmployee',
    'API_TIMEOUT': 30,
    'BATCH_SIZE': 100,
    
    # Logging
    'LOG_LEVEL': 'INFO',
    'LOG_FORMAT': '%(asctime)s - %(levelname)s - %(message)s'
}

class HRMDataSync:
    """Main class for HRM employee data synchronization"""
    
    def __init__(self):
        """Initialize the sync system"""
        self.setup_logging()
        self.db_connection = None
        self.sync_stats = {
            'total_records': 0,
            'success_count': 0,
            'error_count': 0,
            'errors': []
        }
        
    def setup_logging(self):
        """Setup logging configuration"""
        # Create console handler with UTF-8 encoding
        import sys
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler(
            f'hrm_sync_{datetime.now().strftime("%Y%m%d")}.log', 
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        
        # Create formatter
        formatter = logging.Formatter(CONFIG['LOG_FORMAT'])
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Setup logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, CONFIG['LOG_LEVEL']))
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        
    def connect_database(self) -> bool:
        """Connect to MySQL database"""
        try:
            self.logger.info("[DB] Connecting to MySQL database...")
            
            # First, create database if it doesn't exist
            temp_connection = mysql.connector.connect(
                host=CONFIG['DB_HOST'],
                user=CONFIG['DB_USER'],
                password=CONFIG['DB_PASSWORD'],
                port=CONFIG['DB_PORT']
            )
            
            cursor = temp_connection.cursor()
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{CONFIG['DB_NAME']}`")
            cursor.close()
            temp_connection.close()
            
            # Connect to the database
            self.db_connection = mysql.connector.connect(
                host=CONFIG['DB_HOST'],
                user=CONFIG['DB_USER'],
                password=CONFIG['DB_PASSWORD'],
                database=CONFIG['DB_NAME'],
                port=CONFIG['DB_PORT'],
                autocommit=True
            )
            
            self.logger.info("[OK] Database connection successful")
            return True
            
        except Error as e:
            self.logger.error(f"[ERROR] Database connection failed: {e}")
            # Check if MySQL server is running
            if "Can't connect to MySQL server" in str(e):
                self.logger.error("[ERROR] MySQL server is not running. Please start MySQL service.")
            elif "Access denied" in str(e):
                self.logger.error("[ERROR] MySQL access denied. Check username/password or run: 'mysql -u root -p'")
            return False
            
    def create_tables(self) -> bool:
        """Create necessary database tables"""
        try:
            self.logger.info("[DB] Creating database tables...")
            
            cursor = self.db_connection.cursor()
            
            # Create employees table
            employees_table = """
                CREATE TABLE IF NOT EXISTS employees (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    employee_id VARCHAR(50) UNIQUE NOT NULL,
                    company_name VARCHAR(255),
                    department VARCHAR(255),
                    designation VARCHAR(255),
                    user_img TEXT,
                    line_unit VARCHAR(100),
                    date_of_joining DATE,
                    mobile_number VARCHAR(20),
                    reporting_head TEXT,
                    full_name VARCHAR(255) NOT NULL,
                    father_name VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    INDEX idx_employee_id (employee_id),
                    INDEX idx_department (department),
                    INDEX idx_full_name (full_name)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
            
            # Create sync log table
            sync_log_table = """
                CREATE TABLE IF NOT EXISTS sync_log (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    sync_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    total_records INT,
                    success_count INT,
                    error_count INT,
                    status ENUM('success', 'partial', 'failed') DEFAULT 'success',
                    error_message TEXT,
                    INDEX idx_sync_date (sync_date)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
            
            cursor.execute(employees_table)
            cursor.execute(sync_log_table)
            cursor.close()
            
            self.logger.info("[OK] Database tables created successfully")
            return True
            
        except Error as e:
            self.logger.error(f"[ERROR] Failed to create tables: {e}")
            return False
            
    def create_face_tables(self) -> bool:
        """Create face embeddings table"""
        try:
            self.logger.info("[DB] Creating face recognition tables...")
            
            cursor = self.db_connection.cursor()
            
            # Create face embeddings table
            face_embeddings_table = """
                CREATE TABLE IF NOT EXISTS employee_face_embeddings (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    employee_id VARCHAR(50) NOT NULL,
                    embedding_vector JSON,
                    confidence_score FLOAT,
                    face_locations JSON,
                    image_url TEXT,
                    image_processed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processing_status ENUM('pending', 'success', 'no_face', 'multiple_faces', 'error') DEFAULT 'pending',
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    FOREIGN KEY (employee_id) REFERENCES employees(employee_id) ON DELETE CASCADE,
                    INDEX idx_employee_id (employee_id),
                    INDEX idx_processing_status (processing_status)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
            
            # Create face matching results table
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
            
        except Error as e:
            self.logger.error(f"[ERROR] Failed to create tables: {e}")
            return False
            
    def fetch_api_data(self) -> Optional[Dict]:
        """Fetch employee data from HRM API"""
        try:
            self.logger.info("[API] Fetching data from HRM API...")
            
            headers = {
                'Accept': 'application/json',
                'Content-Type': 'application/json',
                'User-Agent': 'HRM-Data-Sync-Python/1.0'
            }
            
            response = requests.get(
                CONFIG['API_URL'],
                headers=headers,
                timeout=CONFIG['API_TIMEOUT']
            )
            
            response.raise_for_status()
            data = response.json()
            
            total_employees = data.get('total', 0)
            employees = data.get('employees', [])
            
            self.logger.info(f"[OK] API data fetched successfully. Total employees: {total_employees}")
            return {
                'total': total_employees,
                'employees': employees
            }
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"[ERROR] API request failed: {e}")
            return None
        except json.JSONDecodeError as e:
            self.logger.error(f"[ERROR] Failed to parse API response: {e}")
            return None
            
    def validate_employee_data(self, employee: Dict) -> Tuple[bool, List[str]]:
        """Validate employee data"""
        errors = []
        
        if not employee.get('employeeId'):
            errors.append('Missing employee ID')
            
        if not employee.get('fullName'):
            errors.append('Missing full name')
            
        mobile = employee.get('mobileNumber', '')
        if mobile and not mobile.isdigit():
            errors.append('Invalid mobile number format')
            
        return len(errors) == 0, errors
        
    def format_date(self, date_string: str) -> Optional[str]:
        """Format date string to MySQL date format"""
        if not date_string:
            return None
            
        try:
            # Handle different date formats
            date_formats = [
                '%d %B %Y',  # "03 October 2025"
                '%Y-%m-%d',  # "2025-10-03"
                '%d-%m-%Y',  # "03-10-2025"
                '%d/%m/%Y'   # "03/10/2025"
            ]
            
            for date_format in date_formats:
                try:
                    parsed_date = datetime.strptime(date_string, date_format)
                    return parsed_date.strftime('%Y-%m-%d')
                except ValueError:
                    continue
                    
            self.logger.warning(f"[WARN] Could not parse date: {date_string}")
            return None
            
        except Exception as e:
            self.logger.warning(f"[WARN] Date parsing error: {e}")
            return None
            
    def insert_employee(self, employee: Dict) -> bool:
        """Insert or update employee in database"""
        try:
            cursor = self.db_connection.cursor()
            
            # Format the date
            formatted_date = self.format_date(employee.get('dateOfJoining'))
            
            query = """
                INSERT INTO employees (
                    employee_id, company_name, department, designation,
                    user_img, line_unit, date_of_joining, mobile_number,
                    reporting_head, full_name, father_name
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    company_name = VALUES(company_name),
                    department = VALUES(department),
                    designation = VALUES(designation),
                    user_img = VALUES(user_img),
                    line_unit = VALUES(line_unit),
                    date_of_joining = VALUES(date_of_joining),
                    mobile_number = VALUES(mobile_number),
                    reporting_head = VALUES(reporting_head),
                    full_name = VALUES(full_name),
                    father_name = VALUES(father_name),
                    updated_at = CURRENT_TIMESTAMP
            """
            
            values = (
                employee.get('employeeId'),
                employee.get('companyName'),
                employee.get('department'),
                employee.get('designation'),
                employee.get('userImg'),
                employee.get('lineUnit'),
                formatted_date,
                employee.get('mobileNumber'),
                employee.get('reportingHead'),
                employee.get('fullName'),
                employee.get('fatherName')
            )
            
            cursor.execute(query, values)
            cursor.close()
            return True
            
        except Error as e:
            self.logger.error(f"[ERROR] Failed to insert employee {employee.get('employeeId', 'Unknown')}: {e}")
            return False
            
    def log_sync_result(self, error_message: Optional[str] = None):
        """Log synchronization result"""
        try:
            if not self.db_connection or not self.db_connection.is_connected():
                self.logger.warning("[WARN] Cannot log sync result - no database connection")
                return
                
            cursor = self.db_connection.cursor()
            
            if error_message:
                status = 'failed'
            elif self.sync_stats['error_count'] > 0:
                status = 'partial'
            else:
                status = 'success'
                
            query = """
                INSERT INTO sync_log (total_records, success_count, error_count, status, error_message)
                VALUES (%s, %s, %s, %s, %s)
            """
            
            # Limit error message length
            error_msg = error_message or '; '.join(self.sync_stats['errors'][:5])
            if len(error_msg) > 500:
                error_msg = error_msg[:497] + '...'
                
            values = (
                self.sync_stats['total_records'],
                self.sync_stats['success_count'],
                self.sync_stats['error_count'],
                status,
                error_msg if error_msg else None
            )
            
            cursor.execute(query, values)
            cursor.close()
            
        except Error as e:
            self.logger.error(f"[ERROR] Failed to log sync result: {e}")
            
    def process_employees(self, employees: List[Dict]):
        """Process all employees data"""
        self.sync_stats['total_records'] = len(employees)
        
        self.logger.info(f"[PROCESS] Processing {len(employees)} employees...")
        
        for i, employee in enumerate(employees, 1):
            try:
                # Validate employee data
                is_valid, errors = self.validate_employee_data(employee)
                
                if not is_valid:
                    error_msg = f"Employee {employee.get('employeeId', 'Unknown')}: {', '.join(errors)}"
                    self.sync_stats['errors'].append(error_msg)
                    self.sync_stats['error_count'] += 1
                    continue
                    
                # Insert employee
                if self.insert_employee(employee):
                    self.sync_stats['success_count'] += 1
                else:
                    self.sync_stats['error_count'] += 1
                    
                # Progress update every 100 records
                if i % 100 == 0:
                    self.logger.info(f"[PROGRESS] {i}/{len(employees)} processed")
                    
            except Exception as e:
                error_msg = f"Employee {employee.get('employeeId', 'Unknown')}: {str(e)}"
                self.sync_stats['errors'].append(error_msg)
                self.sync_stats['error_count'] += 1
                
    def get_database_stats(self) -> Dict:
        """Get current database statistics"""
        try:
            cursor = self.db_connection.cursor()
            
            # Get employee count
            cursor.execute("SELECT COUNT(*) FROM employees")
            employee_count = cursor.fetchone()[0]
            
            # Get last sync info
            cursor.execute("""
                SELECT sync_date, total_records, success_count, error_count, status
                FROM sync_log 
                ORDER BY sync_date DESC 
                LIMIT 1
            """)
            last_sync = cursor.fetchone()
            
            cursor.close()
            
            return {
                'employee_count': employee_count,
                'last_sync': {
                    'date': last_sync[0] if last_sync else None,
                    'total': last_sync[1] if last_sync else 0,
                    'success': last_sync[2] if last_sync else 0,
                    'errors': last_sync[3] if last_sync else 0,
                    'status': last_sync[4] if last_sync else 'none'
                } if last_sync else None
            }
            
        except Error as e:
            self.logger.error(f"[ERROR] Failed to get database stats: {e}")
            return {'employee_count': 0, 'last_sync': None}
            
    def run_sync(self):
        """Main synchronization process"""
        start_time = datetime.now()
        
        try:
            self.logger.info("[START] Starting HRM Employee Data Synchronization")
            self.logger.info("=" * 50)
            
            # Connect to database
            if not self.connect_database():
                raise Exception("Database connection failed")
                
            # Create tables
            if not self.create_tables():
                raise Exception("Table creation failed")
                
            # Create face recognition tables
            self.create_face_tables()
                
            # Fetch API data
            api_data = self.fetch_api_data()
            if not api_data:
                raise Exception("API data fetch failed")
                
            # Process employees
            self.process_employees(api_data['employees'])
            
            # Log results
            self.log_sync_result()
            
            # Calculate duration
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Print summary
            self.logger.info("[COMPLETE] Synchronization Completed!")
            self.logger.info("=" * 50)
            self.logger.info(f"[SUMMARY] Results:")
            self.logger.info(f"   Total Records: {self.sync_stats['total_records']}")
            self.logger.info(f"   Successful: {self.sync_stats['success_count']}")
            self.logger.info(f"   Errors: {self.sync_stats['error_count']}")
            self.logger.info(f"   Duration: {duration:.2f} seconds")
            
            if self.sync_stats['error_count'] > 0:
                self.logger.warning("[WARN] Errors encountered:")
                for i, error in enumerate(self.sync_stats['errors'][:10], 1):
                    self.logger.warning(f"   {i}. {error}")
                if len(self.sync_stats['errors']) > 10:
                    self.logger.warning(f"   ... and {len(self.sync_stats['errors']) - 10} more errors")
                    
            # Show current database stats
            stats = self.get_database_stats()
            self.logger.info(f"[INFO] Current Database: {stats['employee_count']} total employees")
            
            return self.sync_stats['error_count'] == 0
            
        except Exception as e:
            self.logger.error(f"[ERROR] Synchronization failed: {e}")
            self.log_sync_result(str(e))
            return False
            
        finally:
            if self.db_connection and self.db_connection.is_connected():
                self.db_connection.close()
                self.logger.info("[OK] Database connection closed")
                
    def display_help(self):
        """Display help information"""
        help_text = """
HRM Employee Data Sync - Help
=============================

This script synchronizes employee data from HRM API to MySQL database.

Configuration:
- Update the CONFIG dictionary in this script with your database credentials
- Set DB_PASSWORD to your MySQL root password
- Adjust other settings as needed

Usage:
    python hrm_sync.py

Features:
- Automatic database and table creation
- Data validation and error handling  
- Duplicate prevention (updates existing records)
- Comprehensive logging
- Progress tracking
- Sync history tracking

Requirements:
- Python 3.6+
- MySQL Server
- Required packages: requests, mysql-connector-python

Install requirements:
    pip install requests mysql-connector-python

Database Configuration:
    Update these values in CONFIG dictionary:
    - DB_HOST: MySQL server host
    - DB_USER: MySQL username  
    - DB_PASSWORD: MySQL password
    - DB_NAME: Database name
    - DB_PORT: MySQL port (default 3306)

Log Files:
- Sync logs are saved to: hrm_sync_YYYYMMDD.log
- Database sync history is stored in: sync_log table
        """
        print(help_text)

def main():
    """Main function"""
    import sys
    import getpass
    
    # Check for help argument
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        sync = HRMDataSync()
        sync.display_help()
        return
        
    # Check if we need to ask for MySQL password
    if not CONFIG['DB_PASSWORD']:
        print("[INFO] MySQL Database Connection")
        print(f"[INFO] Host: {CONFIG['DB_HOST']}")
        print(f"[INFO] User: {CONFIG['DB_USER']}")
        print(f"[INFO] Database: {CONFIG['DB_NAME']}")
        
        # Ask for password
        try:
            password = getpass.getpass("[INPUT] Enter MySQL password (press Enter if no password): ")
            CONFIG['DB_PASSWORD'] = password
        except KeyboardInterrupt:
            print("\n[EXIT] Cancelled by user")
            return
            
    # Run synchronization
    sync = HRMDataSync()
    success = sync.run_sync()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()