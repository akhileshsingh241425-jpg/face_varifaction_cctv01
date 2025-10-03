#!/usr/bin/env python3
"""
Factory Employee Recogni    print("\n📋 Available Operations:")
    print("1. Setup Database")
    print("2. Sync HRM Data")  
    print("3. Process Face Embeddings")
    print("4. 🌐 Web Interface (Live Camera)")
    print("5. 🎥 Video Analysis System (Record & Analyze)")
    print("6. 🎯 Check Detection Results")
    print("7. Run Complete Setup (All steps)")
    
    choice = input("\nSelect operation (1-7): ").strip()tem
=================================
Complete employee recognition system for factory monitoring.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_script(script_name, description):
    """Run a Python script"""
    print(f"\n🔄 {description}...")
    
    try:
        result = subprocess.run([
            sys.executable, script_name
        ], cwd=os.getcwd(), check=True)
        
        print(f"✅ {description} completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with error code {e.returncode}")
        return False
    except Exception as e:
        print(f"❌ Error running {script_name}: {e}")
        return False

def main():
    """Main function"""
    
    print("🏭 FACTORY EMPLOYEE RECOGNITION SYSTEM")
    print("=" * 50)
    
    # Check if required files exist
    required_files = [
        'complete_db_setup.py',
        'hrm_sync.py', 
        'face_processing.py',
        'web_face_recognition.py',
        'video_analysis_system.py'
    ]
    
    missing_files = [f for f in required_files if not Path(f).exists()]
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return
    
    print("\n📋 Available Operations:")
    print("1. Setup Database")
    print("2. Sync HRM Data")  
    print("3. Process Face Embeddings")
    print("4.  Web Interface (Camera Switching UI)")
    print("5. Run Complete Setup (All steps)")
    
    choice = input("\nSelect operation (1-5): ").strip()
    
    if choice == "1":
        run_script("complete_db_setup.py", "Setting up database")
        
    elif choice == "2":
        run_script("hrm_sync.py", "Syncing HRM data")
        
    elif choice == "3":
        run_script("face_processing.py", "Processing face embeddings")
        
    elif choice == "4":
        print("\n🌐 Starting Web Interface...")
        run_script("web_face_recognition.py", "Starting Web Interface")
        
    elif choice == "5":
        print("\n🎥 Starting Video Analysis System...")
        run_script("video_analysis_system.py", "Starting Video Analysis System")
        
    elif choice == "6":
        print("\n🎯 Checking Detection Results...")
        run_script("check_detections.py", "Checking Detection Results")
        
    elif choice == "7":
        print("\n🚀 Running complete setup...")
        
        steps = [
            ("complete_db_setup.py", "Setting up database"),
            ("hrm_sync.py", "Syncing HRM data"),
            ("face_processing.py", "Processing face embeddings")
        ]
        
        all_success = True
        for script, desc in steps:
            if not run_script(script, desc):
                all_success = False
                break
                
        if all_success:
            print("\n✅ Complete setup finished successfully!")
            
            start_integrated = input("\nStart integrated CCTV face recognition? (y/n): ").lower().strip()
            if start_integrated == 'y':
                run_script("integrated_cctv_recognition.py", "Starting integrated CCTV face recognition")
        else:
            print("\n❌ Setup failed. Please check the errors above.")
            
    else:
        print("❌ Invalid choice. Please select 1-8.")

if __name__ == "__main__":
    main()