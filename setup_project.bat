@echo off
REM ================================
REM Setup para Proyecto de Reconocimiento de Patrones
REM ================================

echo Creando entorno virtual (rp-env)...
python -m venv rp-env

echo Activando entorno virtual...
call rp-env\Scripts\activate.bat

echo Instalando requerimientos...
pip install --upgrade pip
pip install -r requirements.txt

echo ==================================
echo Listo! El entorno rp-env esta configurado.
echo Para usarlo en el futuro: 
echo     call rp-env\Scripts\activate.bat
echo ==================================
