#!/usr/bin/env python3
"""
Detection Results Checker
========================
Check detection results from database
"""

import mysql.connector
from datetime import datetime, timedelta

def check_detections():
    """Check recent detection results"""
    try:
        # Connect to database
        db = mysql.connector.connect(
            host='localhost',
            user='root',
            password='root',
            database='hrm_database'
        )
        
        cursor = db.cursor()
        
        print("🎯 DETECTION RESULTS")
        print("=" * 50)
        
        # Get today's detections
        cursor.execute("""
            SELECT employee_name, camera_location, timestamp, movement_type
            FROM employee_movements 
            WHERE DATE(timestamp) = CURDATE()
            ORDER BY timestamp DESC 
            LIMIT 20
        """)
        
        results = cursor.fetchall()
        
        if results:
            print(f"\n📊 Today's Detections ({len(results)} found):")
            print("-" * 80)
            
            for name, camera, timestamp, movement in results:
                time_str = timestamp.strftime('%H:%M:%S')
                print(f"👤 {name}")
                print(f"   📹 Camera: {camera}")
                print(f"   ⏰ Time: {time_str}")
                print(f"   🚪 Type: {movement}")
                print("-" * 40)
        else:
            print("\n❌ No detections found for today")
        
        # Get summary stats
        cursor.execute("""
            SELECT COUNT(DISTINCT employee_name) as unique_people,
                   COUNT(*) as total_detections,
                   MIN(timestamp) as first_detection,
                   MAX(timestamp) as last_detection
            FROM employee_movements 
            WHERE DATE(timestamp) = CURDATE()
        """)
        
        stats = cursor.fetchone()
        if stats and stats[1] > 0:
            print(f"\n📈 Today's Summary:")
            print(f"   👥 Unique People: {stats[0]}")
            print(f"   🎯 Total Detections: {stats[1]}")
            print(f"   🌅 First Detection: {stats[2].strftime('%H:%M:%S') if stats[2] else 'N/A'}")
            print(f"   🌆 Last Detection: {stats[3].strftime('%H:%M:%S') if stats[3] else 'N/A'}")
        
        # Get camera-wise stats
        cursor.execute("""
            SELECT camera_location, COUNT(*) as count
            FROM employee_movements 
            WHERE DATE(timestamp) = CURDATE()
            GROUP BY camera_location
            ORDER BY count DESC
        """)
        
        camera_stats = cursor.fetchall()
        if camera_stats:
            print(f"\n📹 Camera-wise Detections:")
            for camera, count in camera_stats:
                print(f"   📺 {camera}: {count} detections")
        
        db.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    check_detections()