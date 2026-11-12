@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    py -3.12 -m venv .venv
    if errorlevel 1 goto erreur
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto erreur
".venv\Scripts\python.exe" -m streamlit run app.py
if errorlevel 1 goto erreur
exit /b 0
:erreur
echo Consultez le message et le README pour corriger l'installation.
pause
exit /b 1
