# Proyecto - Detección y Clasificación de Fresas 🍓

Este proyecto implementa un sistema de procesamiento de imágenes para detectar fresas en imágenes y clasificarlas como **maduras** (rojas) o **no maduras** (verdes/hojas).  
Incluye una interfaz gráfica en PyQt5 para procesar imágenes individuales o carpetas completas en modo batch.

---

## 🚀 Instalación

1. Clona el repositorio:
   ```bash
   git clone https://github.com/EnriqueEsc/RecPat_2025_Fresas.git
   cd RecPat_2025_Fresas
   ```

2. Ejecuta el script de instalación (Windows):
   ```bash
   setup_project.bat
   ```

   Esto:
   - Crea un entorno virtual `rp-env/`
   - Activa el entorno
   - Instala dependencias desde `requirements.txt`

3. Para usar el entorno más adelante:
   ```bash
   call rp-env\Scripts\activate.bat
   ```

---

## 🖥️ Uso

### GUI para **una sola imagen**
Ejecutar:
```bash
python ImageProcessing/pyqt_imgproc_gui.py
```

### GUI para **procesamiento por carpetas**
Ejecutar:
```bash
python ImageProcessing/pyqt_batch_gui.py
```

---

## 🧩 Funcionamiento de la GUI

### Modo **una sola imagen**
1. Cargar imagen (`Seleccionar imagen`).
2. Escribir o cargar un archivo de **tratamiento** (script de pasos de procesamiento).
3. Pulsar **Iniciar tratamiento (single)**.
4. La imagen procesada se muestra en el panel derecho y se pueden ver variables intermedias.

### Modo **batch (carpetas)**
1. Seleccionar carpeta con imágenes (`Seleccionar carpeta`).
2. Opcional: elegir carpeta de salida (por defecto crea `outputs/` dentro de la carpeta seleccionada).
3. Escribir o cargar un **script de tratamiento**.
4. Pulsar **Procesar carpeta (batch)**.
   - El sistema ejecuta el script sobre cada imagen.
   - Los resultados se guardan en subcarpetas dentro de `outputs/`:
     - `ripe/` → fresas maduras
     - `unripe/` → fresas verdes/no maduras
     - `overlay/` → imágenes con detección superpuesta
     - `other/` → cualquier otro resultado
5. Durante el proceso, se muestra el progreso en el log.

### Scripts de tratamiento
Los scripts son secuencias de comandos con pasos de procesamiento (ejemplo: suavizado, conversión a HSV, segmentación de colores, conteo de componentes, etc.).  
Ejemplo de script incluido: `detectar_fresas_para_GUI_corrected.txt`.

---

## 📂 Estructura del Proyecto

```
Proyecto-RP/
│
├── ImageProcessing/
│   ├── pyqt_imgproc_gui.py   # GUI para una imagen
│   ├── pyqt_batch_gui.py     # GUI para carpetas
│   ├── script_parser.py      # Motor de parser de scripts
│   ├── strawberry_commands.py # Comandos extra para fresas
│   └── ...
│
├── requirements.txt
├── setup_project.bat
└── README.md
```

---

## ⚙️ Requisitos

- Python 3.8 o superior
- Librerías: PyQt5, OpenCV, NumPy, scikit-image, scikit-learn (instaladas con `requirements.txt`)
