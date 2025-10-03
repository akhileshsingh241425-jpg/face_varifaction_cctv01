#!/usr/bin/env python3
"""
Complete Database Setup for Factory System
=========================================
Creates all necessary tables for the complete factory monitoring system.
"""

import mysql.connector
from datetime import datetime, date

def create_all_tables():
    """Create all required tables for factory monitoring system"""
    
    try:
        # Database connection
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='root',
            database='hrm_database'
        )
        
        cursor = connection.cursor()
        
        print("🏭 Creating factory monitoring tables...")
        
        # 1. face_embeddings table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS face_embeddings (
                id INT AUTO_INCREMENT PRIMARY KEY,
                employee_id INT NOT NULL,
                embedding TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                FOREIGN KEY (employee_id) REFERENCES employees(id),
                UNIQUE KEY unique_employee (employee_id)
            )
        """)
        print("✅ face_embeddings table created")
        
        # 2. employee_movements table  
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS employee_movements (
                id INT AUTO_INCREMENT PRIMARY KEY,
                employee_id INT NOT NULL,
                employee_name VARCHAR(255) NOT NULL,
                confidence FLOAT DEFAULT 0.0,
                camera_id INT NOT NULL,
                camera_location VARCHAR(100) NULL,
                movement_type ENUM('entry', 'exit') NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (employee_id) REFERENCES employees(id),
                INDEX idx_employee_id (employee_id),
                INDEX idx_timestamp (timestamp),
                INDEX idx_movement_type (movement_type)
            )
        """)
        print("✅ employee_movements table created")
        
        # 3. daily_attendance table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_attendance (
                id INT AUTO_INCREMENT PRIMARY KEY,
                employee_id INT NOT NULL,
                employee_name VARCHAR(255) NOT NULL,
                date DATE NOT NULL,
                first_entry TIME NULL,
                last_exit TIME NULL,
                total_duration_minutes INT NULL,
                entry_count INT DEFAULT 0,
                exit_count INT DEFAULT 0,
                current_status ENUM('inside', 'outside') DEFAULT 'outside',
                last_location VARCHAR(100) NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                FOREIGN KEY (employee_id) REFERENCES employees(id),
                UNIQUE KEY unique_employee_date (employee_id, date),
                INDEX idx_date (date),
                INDEX idx_status (current_status)
            )
        """)
        print("✅ daily_attendance table created")
        
        # 4. camera_status table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS camera_status (
                camera_id INT PRIMARY KEY,
                camera_name VARCHAR(100) NOT NULL,
                location VARCHAR(100) NOT NULL,
                status ENUM('online', 'offline', 'error') DEFAULT 'offline',
                last_online TIMESTAMP NULL,
                last_detection TIMESTAMP NULL,
                detection_count_today INT DEFAULT 0,
                total_detections INT DEFAULT 0,
                error_message TEXT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
            )
        """)
        print("✅ camera_status table created")
        
        # 5. system_logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                log_level ENUM('INFO', 'WARNING', 'ERROR', 'DEBUG') DEFAULT 'INFO',
                component VARCHAR(50) NOT NULL,
                message TEXT NOT NULL,
                camera_id INT NULL,
                employee_id INT NULL,
                additional_data JSON NULL
            )
        """)
        print("✅ system_logs table created")
        
        # Insert default camera data
        cursor.execute("""
            INSERT INTO camera_status (camera_id, camera_name, location, status) VALUES
            (0, 'Main Gate Entry', 'Main Gate Entry Point', 'offline'),
            (1, 'Main Gate Exit', 'Main Gate Exit Point', 'offline'),
            (2, 'Side Gate', 'Side Gate Entry/Exit', 'offline'),
            (3, 'Production Entry', 'Production Area Entry', 'offline'),
            (4, 'Production Exit', 'Production Area Exit', 'offline'),
            (5, 'Warehouse Entry', 'Warehouse Entry Point', 'offline'),
            (6, 'Canteen Area', 'Canteen Access Point', 'offline'),
            (7, 'Emergency Exit', 'Emergency Exit Door', 'offline')
            ON DUPLICATE KEY UPDATE
            camera_name = VALUES(camera_name),
            location = VALUES(location)
        """)
        print("✅ Default camera data inserted")
        
        # Create indexes for performance
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_camera_status ON camera_status(status)",
            "CREATE INDEX IF NOT EXISTS idx_last_detection ON camera_status(last_detection)",
            "CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON system_logs(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_logs_level ON system_logs(log_level)",
            "CREATE INDEX IF NOT EXISTS idx_logs_component ON system_logs(component)"
        ]
        
        for index_query in indexes:
            try:
                cursor.execute(index_query)
            except mysql.connector.Error as e:
                print(f"Index creation note: {e}")
        
        print("✅ Database indexes created")
        
        connection.commit()
        print("✅ All tables created successfully!")
        
        return True
        
    except mysql.connector.Error as error:
        print(f"❌ Error creating tables: {error}")
        return False
        
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

def verify_database():
    """Verify all tables are created correctly"""
    
    try:
        connection = mysql.connector.connect(
            host='localhost',
            user='root',
            password='root',
            database='hrm_database'
        )
        
        cursor = connection.cursor()
        
        print("\n📊 Database Verification:")
        print("=" * 50)
        
        # Check all tables
        tables = [
            'employees', 'face_embeddings', 'employee_movements', 
            'daily_attendance', 'camera_status', 'system_logs'
        ]
        
        all_good = True
        
        for table in tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"✅ {table}: {count} records")
            except mysql.connector.Error as e:
                print(f"❌ {table}: ERROR - {e}")
                all_good = False
                
        # Show camera status
        print("\n📹 Camera Configuration:")
        try:
            cursor.execute("SELECT camera_id, camera_name, status FROM camera_status ORDER BY camera_id")
            cameras = cursor.fetchall()
            
            for camera_id, name, status in cameras:
                status_emoji = "🟢" if status == "online" else "🔴"
                print(f"   {status_emoji} Camera {camera_id}: {name}")
        except:
            print("❌ Could not load camera configuration")
            all_good = False
            
        return all_good
        
    except mysql.connector.Error as error:
        print(f"❌ Verification error: {error}")
        return False
        
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

def main():
    """Main setup function"""
    
    print("🏭 FACTORY MONITORING DATABASE SETUP")
    print("=" * 50)
    
    # Create all tables
    if create_all_tables():
        print("\n✅ Database setup completed successfully!")
        
        # Verify setup
        if verify_database():
            print("\n🎉 Factory monitoring system is ready!")
            print("\nNext steps:")
            print("   1. Run: python factory_tracking_system.py")
            print("   2. Run: python factory_dashboard.py")
            print("   3. Or run: python run_factory_system.py")
        else:
            print("\n⚠️  Database verification failed. Please check manually.")
    else:
        print("\n❌ Database setup failed!")

if __name__ == "__main__":
    main()