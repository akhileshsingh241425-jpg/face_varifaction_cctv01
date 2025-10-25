#!/usr/bin/env python3
"""
Production Environment Switcher
Easily switch between testing and production environments
"""

import os
import shutil
from datetime import datetime

def backup_current_env():
    """Backup current .env file"""
    if os.path.exists('.env'):
        backup_name = f'.env.backup.{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        shutil.copy('.env', backup_name)
        print(f"✅ Current .env backed up as: {backup_name}")
        return backup_name
    return None

def switch_to_production():
    """Switch to production environment"""
    print("🔄 Switching to PRODUCTION environment...")
    
    # Backup current
    backup_current_env()
    
    # Create production .env
    prod_env = """# Face Recognition System - PRODUCTION Environment
FACE_RECOGNITION_ENV=production

# Testing Database (Local Development)  
TEST_DB_HOST=localhost
TEST_DB_USER=root
TEST_DB_PASSWORD=root
TEST_DB_NAME=hrm_database

# Production Database (Hostinger) - UPDATE THESE VALUES
PROD_DB_HOST=your_hostinger_mysql_host
PROD_DB_USER=your_hostinger_db_user
PROD_DB_PASSWORD=your_hostinger_db_password
PROD_DB_NAME=your_hostinger_db_name

# Server Configuration
SERVER_HOST=0.0.0.0
SERVER_PORT=5000
DEBUG_MODE=false

# Security
SECRET_KEY=your_production_secret_key_here
FLASK_ENV=production
"""
    
    with open('.env', 'w') as f:
        f.write(prod_env)
    
    print("✅ Switched to PRODUCTION environment")
    print("⚠️  IMPORTANT: Update database credentials in .env file")

def switch_to_testing():
    """Switch to testing environment"""
    print("🔄 Switching to TESTING environment...")
    
    # Backup current
    backup_current_env()
    
    # Create testing .env
    test_env = """# Face Recognition System - TESTING Environment
FACE_RECOGNITION_ENV=testing

# Testing Database (Local Development)
TEST_DB_HOST=localhost
TEST_DB_USER=root
TEST_DB_PASSWORD=root
TEST_DB_NAME=hrm_database

# Production Database (Not used in testing)
PROD_DB_HOST=localhost
PROD_DB_USER=root
PROD_DB_PASSWORD=your_production_password
PROD_DB_NAME=hrm_production

# Server Configuration
SERVER_HOST=0.0.0.0
SERVER_PORT=5000
DEBUG_MODE=true

# Security
SECRET_KEY=testing_key_not_for_production
FLASK_ENV=development
"""
    
    with open('.env', 'w') as f:
        f.write(test_env)
    
    print("✅ Switched to TESTING environment")

def show_current_env():
    """Show current environment configuration"""
    if not os.path.exists('.env'):
        print("❌ No .env file found")
        return
    
    with open('.env', 'r') as f:
        content = f.read()
    
    # Extract environment
    env_line = [line for line in content.split('\n') if 'FACE_RECOGNITION_ENV=' in line]
    if env_line:
        env = env_line[0].split('=')[1]
        print(f"📊 Current Environment: {env.upper()}")
        
        if env == 'production':
            print("🔴 PRODUCTION MODE - Using production database")
        else:
            print("🟡 TESTING MODE - Using local database")
    else:
        print("⚠️  Environment not configured")

def main():
    """Main environment switcher"""
    print("🔧 Face Recognition Environment Switcher")
    print("=" * 45)
    
    while True:
        print("\n📋 Options:")
        print("1. Switch to PRODUCTION")
        print("2. Switch to TESTING") 
        print("3. Show current environment")
        print("4. Exit")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == '1':
            switch_to_production()
        elif choice == '2':
            switch_to_testing()
        elif choice == '3':
            show_current_env()
        elif choice == '4':
            break
        else:
            print("❌ Invalid option")

if __name__ == "__main__":
    main()