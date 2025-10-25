#!/usr/bin/env python3
"""
Create missing employee_movements table in production database
"""
import mysql.connector
import os
from dotenv import load_dotenv

load_dotenv()

# Database config
db_config = {
    'host': os.getenv('PROD_DB_HOST', 'localhost'),
    'user': os.getenv('PROD_DB_USER', 'root'),
    'password': os.getenv('PROD_DB_PASSWORD', 'root'),
    'database': os.getenv('PROD_DB_NAME', 'hrm_production')
}

print("🔧 Creating employee_movements table...")

try:
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    
    # Create table
    create_table_query = """
    CREATE TABLE IF NOT EXISTS employee_movements (
        id INT AUTO_INCREMENT PRIMARY KEY,
        employee_name VARCHAR(255) NOT NULL,
        detection_time DATETIME NOT NULL,
        confidence_score FLOAT,
        camera_source VARCHAR(100),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_employee (employee_name),
        INDEX idx_time (detection_time)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """
    
    cursor.execute(create_table_query)
    conn.commit()
    
    print("✅ Table 'employee_movements' created successfully")
    
    cursor.close()
    conn.close()
    
except Exception as e:
    print(f"❌ Error: {e}")
