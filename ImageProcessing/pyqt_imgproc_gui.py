# pyqt_scriptparser_gui_threaded.py
"""
GUI PyQt5 con ejecución de scripts en QThread (no bloqueante).
Requisitos: proyecto_imageproc.py (o image_processing.py), script_parser.py y strawberry_commands.py
"""

import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QTextEdit, QFileDialog,
    QVBoxLayout, QHBoxLayout, QMessageBox, QListWidget, QSizePolicy, QSplitter
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import numpy as np

# importar módulos (intento ambos nombres por si tu archivo se llama distinto)
try:
    from proyecto_imageproc import Proyecto
except Exception:
    try:
        from image_processing import Proyecto
    except Exception as e:
        raise ImportError("No se encontró el módulo Proyecto (proyecto_imageproc.py o image_processing.py). Error: " + str(e))

try:
    from script_parser import ScriptParser
except Exception as e:
    raise ImportError("No se encontró script_parser.py. Error: " + str(e))

# strawberry_commands contiene register_strawberry_commands(parser)
try:
    from strawberry_commands import register_strawberry_commands
except Exception:
    # no fatal: el usuario puede no querer comandos fresa
    register_strawberry_commands = None


# ---------- util: numpy -> QPixmap ----------
def numpy_to_qpixmap(img: np.ndarray) -> QPixmap:
    if img is None:
        return QPixmap()
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    h, w = arr.shape[:2]
    if arr.ndim == 2:
        qimg = QImage(arr.data, w, h, w, QImage.Format_Grayscale8)
        return QPixmap.fromImage(qimg)
    elif arr.ndim == 3 and arr.shape[2] == 3:
        # si arr viene en BGR (OpenCV), convertir a RGB para Qt
        rgb = arr[:, :, ::-1].copy()
        bytes_per_line = 3 * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)
    else:
        gray = np.mean(arr, axis=2).astype(np.uint8)
        qimg = QImage(gray.data, w, h, w, QImage.Format_Grayscale8)
        return QPixmap.fromImage(qimg)


# ---------- Worker QThread ----------
class ScriptWorker(QThread):
    # progress: (res_object, status_str) ; res_object puede ser numpy image, number, etc.
    progress = pyqtSignal(object, str)
    # finished: (ok: bool, result: object)
    finished = pyqtSignal(bool, object)

    def __init__(self, parser: ScriptParser, proyecto, steps):
        super().__init__()
        self.parser = parser
        self.proyecto = proyecto
        self.steps = steps
        self._abort = False

    def run(self):
        # callback que se ejecuta dentro del hilo worker; emite signal para la UI
        def upd(res, status):
            # puedes filtrar aquí si quieres reducir la frecuencia
            self.progress.emit(res, status)

        try:
            ok, result = self.parser.execute(self.steps, proyecto=self.proyecto, update_callback=upd)
            self.finished.emit(ok, result)
        except Exception as e:
            self.finished.emit(False, f"Exception en worker: {e}")

    def abort(self):
        # Hook: parser.execute no conoce cancelación; sería necesario modificar parser para chequear flag.
        self._abort = True


# ---------- Main Window ----------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ImageProc - Script Parser GUI (threaded)")
        self.resize(1100, 700)

        self.proj = Proyecto()
        self.parser = ScriptParser()
        # registrar comandos fresa si existe el módulo
        if register_strawberry_commands is not None:
            register_strawberry_commands(self.parser)

        self.worker = None  # referencia al worker activo

        # UI layout (igual que antes)
        splitter = QSplitter(Qt.Horizontal)
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_widget.setLayout(left_layout)

        btns_row = QHBoxLayout()
        self.btn_load_image = QPushButton("Seleccionar imagen")
        self.btn_load_image.clicked.connect(self.load_image)
        btns_row.addWidget(self.btn_load_image)

        self.btn_load_script = QPushButton("Seleccionar tratamiento (.txt)")
        self.btn_load_script.clicked.connect(self.load_script)
        btns_row.addWidget(self.btn_load_script)

        left_layout.addLayout(btns_row)

        self.script_edit = QTextEdit()
        self.script_edit.setPlaceholderText("# Escribe o carga el archivo de tratamiento aquí...")
        left_layout.addWidget(self.script_edit, stretch=3)

        bottom_controls = QHBoxLayout()
        self.btn_run = QPushButton("Iniciar tratamiento")
        self.btn_run.clicked.connect(self.run_script)
        bottom_controls.addWidget(self.btn_run)

        self.btn_cancel = QPushButton("Cancelar")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel_script)
        bottom_controls.addWidget(self.btn_cancel)

        self.btn_save = QPushButton("Guardar imagen procesada")
        self.btn_save.clicked.connect(self.save_image)
        bottom_controls.addWidget(self.btn_save)

        left_layout.addLayout(bottom_controls)

        self.log_list = QListWidget()
        left_layout.addWidget(self.log_list, stretch=1)

        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_widget.setLayout(right_layout)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setStyleSheet("background-color: #2b2b2b; border: 1px solid #222;")
        right_layout.addWidget(self.image_label, stretch=7)

        quick_row = QHBoxLayout()
        self.btn_show_original = QPushButton("Mostrar original")
        self.btn_show_original.clicked.connect(self.show_original)
        quick_row.addWidget(self.btn_show_original)
        self.btn_refresh = QPushButton("Refrescar vista")
        self.btn_refresh.clicked.connect(self.refresh_view)
        quick_row.addWidget(self.btn_refresh)
        right_layout.addLayout(quick_row)

        self.vars_list = QListWidget()
        right_layout.addWidget(self.vars_list, stretch=3)

        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self.setCentralWidget(splitter)

        self.current_image_path = None
        self.original_image = None
        self.script_path = None

    # ---------- Slots ----------
    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Seleccionar imagen", "", "Imágenes (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;Todos (*)")
        if not path:
            return
        try:
            self.proj.load(path, as_gray=False)
            self.current_image_path = path
            self.original_image = self.proj.get_processed().copy() if self.proj.get_processed() is not None else None
            self.log("Imagen cargada: " + os.path.basename(path))
            self.refresh_view()
        except Exception as e:
            QMessageBox.critical(self, "Error carga imagen", str(e))

    def load_script(self):
        path, _ = QFileDialog.getOpenFileName(self, "Seleccionar script de tratamiento", "", "Text files (*.txt *.script);;All files (*)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            self.script_edit.setPlainText(text)
            self.script_path = path
            self.log("Script cargado: " + os.path.basename(path))
        except Exception as e:
            QMessageBox.critical(self, "Error carga script", str(e))

    def run_script(self):
        if self.proj.get_processed() is None:
            QMessageBox.warning(self, "Sin imagen", "Primero selecciona una imagen para procesar.")
            return
        script_text = self.script_edit.toPlainText()
        if not script_text.strip():
            QMessageBox.warning(self, "Sin script", "Escribe o carga un archivo de tratamiento.")
            return

        # parsear
        try:
            steps = self.parser.parse(script_text)
        except Exception as e:
            QMessageBox.critical(self, "Error parseo", f"No se pudo parsear el script: {e}")
            return

        # bloquear botones
        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.log_list.clear()
        self.vars_list.clear()

        # crear worker
        self.worker = ScriptWorker(self.parser, self.proj, steps)
        self.worker.progress.connect(self.on_worker_progress)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()
        self.log("Worker iniciado...")

    def cancel_script(self):
        if self.worker is not None:
            # nota: parser.execute no soporta cancelación por defecto. Esto solo marca un flag.
            # Para cancelación real habría que instrumentar parser.execute para chequearlo.
            self.worker.abort()
            self.log("Solicitud de cancelación enviada (no garantizada).")
            self.btn_cancel.setEnabled(False)

    # signals handlers
    def on_worker_progress(self, res, status):
        # res puede ser cualquier cosa; preferimos mostrar la imagen actual del proyecto
        try:
            img = self.proj.get_processed()
            pix = numpy_to_qpixmap(img)
            if not pix.isNull():
                self.image_label.setPixmap(pix.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception:
            pass
        self.log(status)

    def on_worker_finished(self, ok, result):
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        if not ok:
            QMessageBox.critical(self, "Error ejecución", str(result))
            self.log("Error: " + str(result))
            return

        out = result  # {"final_image":..., "variables": {...}}
        varsdict = out.get("variables", {})
        self.vars_list.clear()
        for k, v in varsdict.items():
            display = k
            try:
                if hasattr(v, "shape"):
                    display += f" : image {getattr(v, 'shape', '')}"
                else:
                    display += f" : {type(v).__name__} = {str(v)}"
            except Exception:
                display += f" : {type(v).__name__}"
            self.vars_list.addItem(display)

        self.log("Tratamiento finalizado.")
        self.refresh_view()
        # liberar worker
        self.worker = None

    def save_image(self):
        img = self.proj.get_processed()
        if img is None:
            QMessageBox.warning(self, "Sin imagen", "No hay imagen procesada para guardar.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Guardar imagen procesada", "", "PNG (*.png);;JPEG (*.jpg *.jpeg);;TIFF (*.tif *.tiff)")
        if not path:
            return
        try:
            import cv2
            tosave = np.asarray(img)
            ok = cv2.imwrite(path, tosave)
            if not ok:
                raise IOError("cv2.imwrite devolvió False")
            self.log("Guardado: " + os.path.basename(path))
            QMessageBox.information(self, "Guardado", "Imagen guardada correctamente.")
        except Exception as e:
            QMessageBox.critical(self, "Error guardado", str(e))

    def show_original(self):
        if self.original_image is None:
            QMessageBox.information(self, "Original no disponible", "No hay imagen original cargada.")
            return
        pix = numpy_to_qpixmap(self.original_image)
        if not pix.isNull():
            self.image_label.setPixmap(pix.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.log("Mostrando imagen original (no procesada).")

    def refresh_view(self):
        img = self.proj.get_processed()
        if img is None:
            self.image_label.clear()
            return
        pix = numpy_to_qpixmap(img)
        if not pix.isNull():
            self.image_label.setPixmap(pix.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def log(self, text: str):
        self.log_list.addItem(text)
        self.log_list.scrollToBottom()


# ---------- run ----------
def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
