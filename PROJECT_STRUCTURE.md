# 📁 CCTV Live Face Recognition - Project Structure

## 🎯 Core Files (Live Recognition Only)

### 🚀 **Launchers**
- `main.py` - Main system launcher with menu options
- `web_face_recognition.py` - Live face recognition web interface (Port 5000)

### 🔧 **Setup & Configuration**
- `complete_db_setup.py` - Database initialization
- `hrm_sync.py` - Sync employees from HRM API
- `face_processing.py` - Process employee face embeddings

### 🛠️ **Utilities**
- `check_detections.py` - Check detection results
- `requirements.txt` - Python dependencies

## 📂 **Folders**

### 🌐 **Web Interface**
```
templates/
└── index.html          # Live camera interface only
```

### ⚙️ **System Files**
```
.venv/                  # Virtual environment (Python packages)
.git/                   # Git version control
.gitignore             # Git ignore rules
```

## 🎯 **How to Use**

### ✅ **Quick Start**
```bash
# 1. Run main system
python main.py

# 2. Or directly start web interface
python web_face_recognition.py
```

### 🌐 **Web Access**
- **Live Face Recognition**: http://127.0.0.1:5000

## 📊 **System Components**

| File | Purpose | Status |
|------|---------|--------|
| `main.py` | System launcher | ✅ Ready |
| `web_face_recognition.py` | Live recognition | ✅ Ready |
| `hrm_sync.py` | Employee sync | ✅ 2971 employees |
| `face_processing.py` | Face embeddings | ✅ Ready |

## 🎉 **Simplified & Clean Structure**
- ❌ Removed video analysis system (not needed)
- ❌ Removed video recording features  
- ❌ Removed unnecessary folders
- ✅ Only live face recognition remains

---
**Total Files**: 6 core Python files + 1 HTML template
**Focus**: Live Face Recognition Only! 🎯