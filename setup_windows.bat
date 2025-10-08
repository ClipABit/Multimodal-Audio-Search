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
echo To run the app:
echo 1. Open Command Prompt/PowerShell
echo 2. Navigate to this folder
echo 3. Run: audio_search_venv\Scripts\activate
echo 4. Run: streamlit run streamlit_app.py
echo.
echo The app will open at http://localhost:8501
echo ============================================
pause