@echo off
:: 1. Launch the backend server in its own persistent window
start "AI Backend Server" cmd /k "call venv\Scripts\activate && python main.py"

:: 2. Wait 2 seconds to let the server spin up, then open your interface
timeout /t 2 >nul
start "" "index.html"