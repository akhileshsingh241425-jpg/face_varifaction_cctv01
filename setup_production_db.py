#!/usr/bin/env python3
"""
Production Database Setup Script
===============================
Creates production database and tables for face recognition system
"""

import mysql.connector
import os
from datetime import datetime

def create_production_database():
    """Create production database and tables"""
    
    # Database configurations
    configs = {
        'testing': {
            'host': 'localhost',
            'user': 'root', 
            'password': 'root',
            'database': 'hrm_database'
        },
        'production': {
            'host': 'localhost',
            'user': 'root',
            'password': 'root',  # Change this for production
            'database': 'hrm_production'
        }
    }
    
    print("🚀 Face Recognition Database Setup")
    print("=" * 50)
    
    # Create production database
    try:
        # Connect without database to create it
        conn = mysql.connector.connect(
            host=configs['production']['host'],
            user=configs['production']['user'],
            password=configs['production']['password']
        )
        cursor = conn.cursor()
        
        # Create database
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {configs['production']['database']}")
        print(f"✅ Created database: {configs['production']['database']}")
        
        # Use the database
        cursor.execute(f"USE {configs['production']['database']}")
        
        # Create multi_angle_faces table
        create_table_query = """
        CREATE TABLE IF NOT EXISTS multi_angle_faces (
            id VARCHAR(50) PRIMARY KEY,
            employee_name VARCHAR(255) NOT NULL,
            department VARCHAR(100) DEFAULT NULL,
            position VARCHAR(100) DEFAULT NULL,
            face_encoding LONGTEXT NOT NULL,
            total_photos INT DEFAULT 1,
            encoding_quality DECIMAL(5,2) DEFAULT NULL,
            is_active BOOLEAN DEFAULT TRUE,
            uploaded_by VARCHAR(100) DEFAULT 'system',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            INDEX idx_employee_name (employee_name),
            INDEX idx_is_active (is_active),
            INDEX idx_created_at (created_at)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
        """
        
        cursor.execute(create_table_query)
        print("✅ Created table: multi_angle_faces")
        
        # Show table structure
        cursor.execute("DESCRIBE multi_angle_faces")
        print("\n📋 Table Structure:")
        for row in cursor.fetchall():
            print(f"   {row[0]} - {row[1]}")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"\n🎉 Production database setup completed!")
        print(f"Database: {configs['production']['database']}")
        print(f"Host: {configs['production']['host']}")
        
    except mysql.connector.Error as e:
        print(f"❌ Error setting up database: {e}")
        return False
        
    return True

def copy_test_data_to_production():
    """Copy data from testing to production database (optional)"""
    
    choice = input("\n🔄 Copy test data to production? (y/N): ").strip().lower()
    if choice != 'y':
        return
        
    try:
        # Connect to test database
        test_conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='root',
            database='hrm_database'
        )
        
        # Connect to production database
        prod_conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='root',
            database='hrm_production'
        )
        
        test_cursor = test_conn.cursor()
        prod_cursor = prod_conn.cursor()
        
        # Get test data
        test_cursor.execute("SELECT * FROM multi_angle_faces")
        test_data = test_cursor.fetchall()
        
        if len(test_data) == 0:
            print("⚠️ No test data found to copy")
            return
            
        # Clear production data first
        prod_cursor.execute("DELETE FROM multi_angle_faces")
        
        # Copy data
        insert_query = """
        INSERT INTO multi_angle_faces 
        (id, employee_name, department, position, face_encoding, total_photos, 
         encoding_quality, is_active, uploaded_by, created_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        for row in test_data:
            prod_cursor.execute(insert_query, row)
            
        prod_conn.commit()
        
        print(f"✅ Copied {len(test_data)} records to production")
        
        test_cursor.close()
        prod_cursor.close()
        test_conn.close()
        prod_conn.close()
        
    except mysql.connector.Error as e:
        print(f"❌ Error copying data: {e}")

if __name__ == "__main__":
    if create_production_database():
        copy_test_data_to_production()
        
    print("\n" + "=" * 50)
    print("🔧 To use production database, set environment variable:")
    print("   FACE_RECOGNITION_ENV=production")
    print("\n💡 Or create .env file with:")
    print("   FACE_RECOGNITION_ENV=production")