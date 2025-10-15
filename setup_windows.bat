@echo off
echo ClipABit Audio Search - Windows Setup Script
echo ============================================
echo.

echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found! Please install Python 3.8+ from python.org
    pause
    exit /b 1
)

echo Python found! Creating virtual environment...
python -m venv audio_search_venv

echo Activating virtual environment...
call audio_search_venv\Scripts\activate.bat

echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ============================================ 
echo Setup complete! 
echo.
echo To run the dual pipeline audio search system:
echo 1. Open Command Prompt/PowerShell
echo 2. Navigate to this folder
echo 3. Run: audio_search_venv\Scripts\activate
echo 4. Run: audio_search_venv\Scripts\python.exe -m streamlit run audio_search.py --server.port 8527
echo.
echo The app will open at http://localhost:8527
echo ============================================
pause