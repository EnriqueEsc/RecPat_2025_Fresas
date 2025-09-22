# pyqt_batch_gui.py
"""
PyQt5 GUI con ejecución por lote (batch) para procesar carpetas de imágenes.
Guarda outputs organizados en subcarpetas: ripe, unripe, overlay, other.
Genera CSVs: summary.csv (por imagen) e instances.csv (por objeto detectado).

Requisitos:
 - PyQt5, numpy, opencv-python, scikit-image, scipy, scikit-learn (según Proyecto)
 - pandas (recomendado; si no está se usa csv stdlib)
 - script_parser.py
 - strawberry_commands.py (opcional pero recomendado)
 - proyecto_imageproc.py o image_processing.py (implementación de Proyecto)

Ejecutar: python pyqt_batch_gui.py
"""

import sys
import os
import math
import time
import glob
import csv
from typing import Dict, List
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QTextEdit, QFileDialog,
    QVBoxLayout, QHBoxLayout, QMessageBox, QListWidget, QSizePolicy, QSplitter, QLineEdit
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import numpy as np
import cv2

# intentar importar Proyecto con nombres comunes
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
    register_strawberry_commands = None

# intentar importar pandas (opcional)
try:
    import pandas as pd
    _HAS_PANDAS = True
except Exception:
    _HAS_PANDAS = False

# ---------------- helper: feature extraction ----------------
def region_props_from_mask(image_bgr: np.ndarray, mask: np.ndarray, label: str, stem: str,
                           out_dirs: Dict[str,str], instance_start: int = 0,
                           save_masks: bool = True, save_crops: bool = True) -> List[Dict]:
    """
    Extrae propiedades de cada contorno en `mask` (uint8 0/255).
    Guarda máscara y crop si save_masks/save_crops True.
    Devuelve lista de dicts con propiedades por instancia.
    """
    props = []
    if mask is None or mask.max() == 0:
        return props

    # encontrar contornos
    mask_copy = mask.copy().astype(np.uint8)
    contours, _ = cv2.findContours(mask_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    inst_id = instance_start
    hsv_img = None
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area <= 0:
            continue
        inst_id += 1
        x,y,w,h = cv2.boundingRect(cnt)
        perimeter = float(cv2.arcLength(cnt, True))
        M = cv2.moments(cnt)
        if M.get('m00', 0) != 0:
            cx = float(M['m10']/M['m00'])
            cy = float(M['m01']/M['m00'])
        else:
            cx, cy = x + w/2.0, y + h/2.0

        hull = cv2.convexHull(cnt)
        hull_area = float(cv2.contourArea(hull)) if hull is not None and len(hull) > 2 else area
        solidity = float(area / hull_area) if hull_area > 0 else 0.0

        eccentricity = 0.0
        if cnt.shape[0] >= 5:
            try:
                (xc,yc),(MA,ma),angle = cv2.fitEllipse(cnt)
                a = max(MA, ma) / 2.0
                b = min(MA, ma) / 2.0
                if a > 0:
                    eccentricity = math.sqrt(max(0.0, 1.0 - (b*b)/(a*a)))
            except Exception:
                eccentricity = 0.0

        extent = float(area / (w*h)) if (w*h) > 0 else 0.0
        hu = cv2.HuMoments(M).flatten().tolist()
        hu = [float(hv) for hv in hu]

        # mean HSV
        if hsv_img is None:
            hsv_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        mask_roi = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(mask_roi, [cnt], -1, 255, -1)
        mean_vals = cv2.mean(hsv_img, mask=mask_roi)  # (h,s,v,a)
        mean_h, mean_s, mean_v = float(mean_vals[0]), float(mean_vals[1]), float(mean_vals[2])

        mask_path = ""
        crop_path = ""
        if save_masks:
            sub = out_dirs.get('masks', '.')
            os.makedirs(sub, exist_ok=True)
            mask_filename = f"{stem}_{label}_inst{inst_id}_mask.png"
            mask_path = os.path.join(sub, mask_filename)
            # save mask as 0/255 single channel
            cv2.imwrite(mask_path, mask_roi)

        if save_crops:
            subc = out_dirs.get('crops', '.')
            os.makedirs(subc, exist_ok=True)
            crop_img = image_bgr[y:y+h, x:x+w].copy()
            mask_crop = mask_roi[y:y+h, x:x+w]
            # apply mask to crop: keep background black
            if crop_img.size != 0:
                crop_img[mask_crop==0] = 0
            crop_filename = f"{stem}_{label}_inst{inst_id}_crop.png"
            crop_path = os.path.join(subc, crop_filename)
            cv2.imwrite(crop_path, crop_img)

        row = {
            "filename": stem,
            "instance_id": int(inst_id),
            "label": label,
            "area": float(area),
            "perimeter": float(perimeter),
            "centroid_x": float(cx),
            "centroid_y": float(cy),
            "bbox_x": int(x),
            "bbox_y": int(y),
            "bbox_w": int(w),
            "bbox_h": int(h),
            "mean_h": mean_h, "mean_s": mean_s, "mean_v": mean_v,
            "solidity": float(solidity),
            "eccentricity": float(eccentricity),
            "extent": float(extent),
            "hu1": hu[0], "hu2": hu[1], "hu3": hu[2], "hu4": hu[3], "hu5": hu[4], "hu6": hu[5], "hu7": hu[6],
            "mask_path": mask_path,
            "crop_path": crop_path,
            "contour_len": int(cnt.shape[0])
        }
        props.append(row)
    return props

def make_summary_row(fname: str, image_bgr: np.ndarray, props_ripe: List[Dict], props_unripe: List[Dict], overlay_relpath: str):
    def stats_from_props(props):
        if not props:
            return {"count":0, "total_area":0.0, "mean_area":0.0, "max_area":0.0, "min_area":0.0, "mean_h":None, "mean_s":None, "mean_v":None}
        areas = np.array([p["area"] for p in props], dtype=float)
        mean_h = float(np.mean([p["mean_h"] for p in props])) if any(p["mean_h"] is not None for p in props) else None
        mean_s = float(np.mean([p["mean_s"] for p in props])) if any(p["mean_s"] is not None for p in props) else None
        mean_v = float(np.mean([p["mean_v"] for p in props])) if any(p["mean_v"] is not None for p in props) else None
        return {"count": len(props), "total_area": float(areas.sum()), "mean_area": float(areas.mean()), "max_area": float(areas.max()), "min_area": float(areas.min()), "mean_h": mean_h, "mean_s": mean_s, "mean_v": mean_v}

    r = stats_from_props(props_ripe)
    u = stats_from_props(props_unripe)
    h, w = image_bgr.shape[:2]
    return {
        "filename": fname,
        "width": int(w),
        "height": int(h),
        "n_ripe": int(r["count"]),
        "n_unripe": int(u["count"]),
        "total_area_ripe": r["total_area"],
        "total_area_unripe": u["total_area"],
        "mean_area_ripe": r["mean_area"],
        "mean_area_unripe": u["mean_area"],
        "max_area_ripe": r["max_area"],
        "min_area_ripe": r["min_area"],
        "mean_h_ripe": r["mean_h"],
        "mean_s_ripe": r["mean_s"],
        "mean_v_ripe": r["mean_v"],
        "mean_h_unripe": u["mean_h"],
        "mean_s_unripe": u["mean_s"],
        "mean_v_unripe": u["mean_v"],
        "overlay_path": overlay_relpath,
        "timestamp": datetime.utcnow().isoformat()
    }

def write_summary_and_instances_csv(summary_rows: List[Dict], instance_rows: List[Dict], out_folder: str):
    os.makedirs(out_folder, exist_ok=True)
    summary_csv = os.path.join(out_folder, "summary.csv")
    instances_csv = os.path.join(out_folder, "instances.csv")
    if _HAS_PANDAS:
        try:
            pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
            pd.DataFrame(instance_rows).to_csv(instances_csv, index=False)
            return summary_csv, instances_csv
        except Exception:
            pass
    # fallback to csv module
    if summary_rows:
        keys = list(summary_rows[0].keys())
        with open(summary_csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            for r in summary_rows:
                writer.writerow(r)
    else:
        # write empty file with standard headers
        with open(summary_csv, "w", newline="", encoding="utf-8") as fh:
            fh.write("filename,width,height,n_ripe,n_unripe\n")
    if instance_rows:
        keys = list(instance_rows[0].keys())
        with open(instances_csv, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            for r in instance_rows:
                writer.writerow(r)
    else:
        with open(instances_csv, "w", newline="", encoding="utf-8") as fh:
            fh.write("filename,instance_id,label,area\n")
    return summary_csv, instances_csv

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
        rgb = arr[:, :, ::-1].copy()
        bytes_per_line = 3 * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)
    else:
        gray = np.mean(arr, axis=2).astype(np.uint8)
        qimg = QImage(gray.data, w, h, w, QImage.Format_Grayscale8)
        return QPixmap.fromImage(qimg)

# ---------- Batch worker ----------
class BatchWorker(QThread):
    progress = pyqtSignal(str)        # mensaje de progreso
    image_update = pyqtSignal(object) # numpy image to show
    finished = pyqtSignal(bool, object)  # ok, info

    def __init__(self, parser: ScriptParser, script_steps, input_files, out_root, proyecto_cls):
        super().__init__()
        self.parser = parser
        self.steps = script_steps
        self.files = input_files
        self.out_root = out_root
        self.ProyectoClass = proyecto_cls
        self._abort = False

    def run(self):
        # output subdirs
        dirs = {
            "ripe": os.path.join(self.out_root, "ripe"),
            "unripe": os.path.join(self.out_root, "unripe"),
            "overlay": os.path.join(self.out_root, "overlay"),
            "other": os.path.join(self.out_root, "other"),
            "masks": os.path.join(self.out_root, "masks"),
            "crops": os.path.join(self.out_root, "crops"),
        }
        for d in dirs.values():
            os.makedirs(d, exist_ok=True)

        total = len(self.files)
        results = {"processed": 0, "errors": []}

        # for CSV
        summary_rows = []
        instance_rows = []

        # Keep original Save/SaveText
        orig_save = self.parser.registry.get("Save")
        orig_savetext = self.parser.registry.get("SaveText")

        for idx, fpath in enumerate(self.files, start=1):
            if self._abort:
                self.progress.emit("Cancelado por el usuario.")
                break
            fname = os.path.basename(fpath)
            stem = os.path.splitext(fname)[0]
            self.progress.emit(f"[{idx}/{total}] Procesando: {fname}")

            proj = self.ProyectoClass()
            try:
                proj.load(fpath, as_gray=False)
            except Exception as e:
                results["errors"].append((fpath, str(e)))
                self.progress.emit(f"  ERROR al cargar {fname}: {e}")
                continue

            # classify subfolder based on output filename
            def classify_subfolder(output_filename: str) -> str:
                name = output_filename.lower()
                if "ripe" in name or "red" in name:
                    return dirs["ripe"]
                if "unripe" in name or "green" in name:
                    return dirs["unripe"]
                if "overlay" in name or "over" in name:
                    return dirs["overlay"]
                return dirs["other"]

            # override Save / SaveText to route files into appropriate subfolders
            def save_override(proj_arg, image_or_var, out_name):
                try:
                    sub = classify_subfolder(str(out_name))
                    out_filename = f"{stem}_{os.path.basename(str(out_name))}"
                    out_path = os.path.join(sub, out_filename)
                    img = np.asarray(image_or_var)
                    if img.dtype != np.uint8:
                        if np.issubdtype(img.dtype, np.floating):
                            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
                        else:
                            img = np.clip(img, 0, 255).astype(np.uint8)
                    ok = cv2.imwrite(out_path, img)
                    if not ok:
                        raise IOError(f"cv2.imwrite devolvió False al escribir {out_path}")
                    self.progress.emit(f"    Guardado -> {os.path.relpath(out_path, self.out_root)}")
                    return out_path
                except Exception as e:
                    self.progress.emit(f"    ERROR guardando {out_name}: {e}")
                    raise

            def savetext_override(proj_arg, path_name, text):
                sub = classify_subfolder(str(path_name))
                out_filename = f"{stem}_{os.path.basename(str(path_name))}"
                out_path = os.path.join(sub, out_filename)
                try:
                    with open(out_path, "w", encoding="utf-8") as fh:
                        fh.write(str(text))
                    self.progress.emit(f"    Guardado texto -> {os.path.relpath(out_path, self.out_root)}")
                    return out_path
                except Exception as e:
                    self.progress.emit(f"    ERROR guardando texto {path_name}: {e}")
                    raise

            # Monkeypatch parser Save/SaveText
            self.parser.register_command("Save", save_override)
            self.parser.register_command("SaveText", savetext_override)

            # Execute script
            try:
                ok, info = self.parser.execute(self.steps, proyecto=proj, update_callback=lambda res, status: self._update_cb(proj, res, status))
                if not ok:
                    results["errors"].append((fpath, str(info)))
                    self.progress.emit(f"  ERROR en ejecución: {info}")
                else:
                    results["processed"] += 1
            except Exception as e:
                results["errors"].append((fpath, str(e)))
                self.progress.emit(f"  EXCEPCIÓN: {e}")
            finally:
                # restore Save/SaveText to original (if any)
                if orig_save:
                    self.parser.register_command("Save", orig_save)
                else:
                    # remove if we added one and there was none
                    self.parser.registry.pop("Save", None)
                if orig_savetext:
                    self.parser.register_command("SaveText", orig_savetext)
                else:
                    self.parser.registry.pop("SaveText", None)

            # ---------------- SAFE extraction of masks/overlays from parser variables ----------------
            vars_out = {}
            if isinstance(info, dict):
                vars_out = info.get("variables", {})

            # helper: decide if v is a usable image/array
            def _is_valid_image_like(v):
                if v is None:
                    return False
                try:
                    arr = np.asarray(v)
                except Exception:
                    return False
                if arr.size == 0:
                    return False
                if arr.ndim >= 2:
                    return True
                return False

            def find_var_by_candidates(cands):
                for c in cands:
                    if c in vars_out:
                        v = vars_out[c]
                        if _is_valid_image_like(v):
                            return np.asarray(v)
                return None

            # attempt to find masks/overlay using multiple candidate names (safe, avoids boolean ambig)
            ripe_mask = find_var_by_candidates(["mask_ripe_big", "mask_ripe_clean", "mask_ripe", "mask_r1", "mask_r2", "mask_r"])
            green_mask = find_var_by_candidates(["mask_green_big", "mask_green_clean", "mask_green", "mask_g", "mask_green1"])
            combined_overlay = find_var_by_candidates(["combined_overlay", "overlay", "combined", "combined_overlay_img", "overlay_final"])

            # If overlay not provided by script, fallback to proyecto processed image
            if combined_overlay is None:
                try:
                    proc_img = proj.get_processed()
                    if _is_valid_image_like(proc_img):
                        combined_overlay = np.asarray(proc_img)
                except Exception:
                    combined_overlay = None
            # ---------------- end safe extraction ----------------

            # save overlay if available, else use processed image already saved in fallback below
            overlay_relpath = ""
            if combined_overlay is not None:
                overlay_path = os.path.join(dirs["overlay"], f"{stem}_overlay.png")
                try:
                    ov = np.asarray(combined_overlay)
                    if ov.dtype != np.uint8:
                        if np.issubdtype(ov.dtype, np.floating):
                            ov = (np.clip(ov, 0, 1) * 255).astype(np.uint8)
                        else:
                            ov = np.clip(ov, 0, 255).astype(np.uint8)
                    cv2.imwrite(overlay_path, ov)
                    overlay_relpath = os.path.relpath(overlay_path, self.out_root)
                    self.progress.emit(f"    Overlay guardado -> {overlay_relpath}")
                except Exception as e:
                    self.progress.emit(f"    ERROR guardando overlay: {e}")
            else:
                # use processed image
                try:
                    proc = proj.get_processed()
                    if proc is not None:
                        outp = os.path.join(dirs["overlay"], f"{stem}_overlay.png")
                        imgp = np.asarray(proc)
                        if imgp.dtype != np.uint8:
                            if np.issubdtype(imgp.dtype, np.floating):
                                imgp = (np.clip(imgp, 0, 1) * 255).astype(np.uint8)
                            else:
                                imgp = np.clip(imgp, 0, 255).astype(np.uint8)
                        cv2.imwrite(outp, imgp)
                        overlay_relpath = os.path.relpath(outp, self.out_root)
                    else:
                        overlay_relpath = ""
                except Exception:
                    overlay_relpath = ""

            # extract instance props and summary
            img_bgr = proj.get_processed()
            if img_bgr is None:
                img_bgr = cv2.imread(fpath, cv2.IMREAD_COLOR)

            props_ripe = region_props_from_mask(img_bgr, ripe_mask, 'ripe', stem, {'masks': dirs['masks'], 'crops': dirs['crops']}, instance_start=0) if (ripe_mask is not None) else []
            props_unripe = region_props_from_mask(img_bgr, green_mask, 'unripe', stem, {'masks': dirs['masks'], 'crops': dirs['crops']}, instance_start=len(props_ripe)) if (green_mask is not None) else []

            # add to global instances list
            instance_rows.extend(props_ripe)
            instance_rows.extend(props_unripe)

            # make summary
            summary_row = make_summary_row(fname, img_bgr, props_ripe, props_unripe, overlay_relpath)
            summary_rows.append(summary_row)

            # emit preview
            try:
                self.image_update.emit(img_bgr)
            except Exception:
                pass

        # write CSVs at end
        try:
            summary_csv, instances_csv = write_summary_and_instances_csv(summary_rows, instance_rows, self.out_root)
            results['summary_csv'] = summary_csv
            results['instances_csv'] = instances_csv
            self.progress.emit(f"CSV guardados: {os.path.relpath(summary_csv, self.out_root)}, {os.path.relpath(instances_csv, self.out_root)}")
        except Exception as e:
            results['errors'].append(("csv", str(e)))
            self.progress.emit(f"ERROR guardando CSVs: {e}")

        self.finished.emit(True, results)

    def _update_cb(self, proyecto, res, status):
        # Called from parser.execute inside worker thread
        self.progress.emit(status)
        try:
            img = proyecto.get_processed()
            self.image_update.emit(img)
        except Exception:
            pass

    def abort(self):
        self._abort = True

# ---------- Main Window ----------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ImageProc - Batch Script GUI (with CSV & crops)")
        self.resize(1200, 750)

        self.parser = ScriptParser()
        # register strawberry commands if available
        if register_strawberry_commands is not None:
            register_strawberry_commands(self.parser)

        splitter = QSplitter(Qt.Horizontal)
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_widget.setLayout(left_layout)

        # Top row buttons
        top_row = QHBoxLayout()
        self.btn_load_image = QPushButton("Seleccionar imagen")
        self.btn_load_image.clicked.connect(self.load_image)
        top_row.addWidget(self.btn_load_image)

        self.btn_select_folder = QPushButton("Seleccionar carpeta")
        self.btn_select_folder.clicked.connect(self.select_folder)
        top_row.addWidget(self.btn_select_folder)

        self.out_line = QLineEdit()
        self.out_line.setPlaceholderText("Carpeta de salida (opcional). Por defecto: <input>/outputs/")
        top_row.addWidget(self.out_line)

        self.btn_choose_out = QPushButton("Elegir salida")
        self.btn_choose_out.clicked.connect(self.choose_output_folder)
        top_row.addWidget(self.btn_choose_out)

        left_layout.addLayout(top_row)

        # script editor
        self.script_edit = QTextEdit()
        self.script_edit.setPlaceholderText("# Pega aquí tu script (ej: detectar_fresas_para_GUI_corrected.txt)")
        left_layout.addWidget(self.script_edit, stretch=3)

        # control buttons
        control_row = QHBoxLayout()
        self.btn_run = QPushButton("Iniciar tratamiento (single)")
        self.btn_run.clicked.connect(self.run_script_single)
        control_row.addWidget(self.btn_run)

        self.btn_process_folder = QPushButton("Procesar carpeta (batch)")
        self.btn_process_folder.clicked.connect(self.process_folder)
        control_row.addWidget(self.btn_process_folder)

        self.btn_cancel = QPushButton("Cancelar")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel_batch)
        control_row.addWidget(self.btn_cancel)

        left_layout.addLayout(control_row)

        # logs
        self.log_list = QListWidget()
        left_layout.addWidget(self.log_list, stretch=2)

        # right panel: image preview and variables
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_widget.setLayout(right_layout)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setStyleSheet("background-color: #2b2b2b; border: 1px solid #222;")
        right_layout.addWidget(self.image_label, stretch=7)

        vars_row = QHBoxLayout()
        self.btn_show_original = QPushButton("Mostrar original")
        self.btn_show_original.clicked.connect(self.show_original)
        vars_row.addWidget(self.btn_show_original)
        self.btn_refresh = QPushButton("Refrescar vista")
        self.btn_refresh.clicked.connect(self.refresh_view)
        vars_row.addWidget(self.btn_refresh)
        right_layout.addLayout(vars_row)

        self.vars_list = QListWidget()
        right_layout.addWidget(self.vars_list, stretch=3)

        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        # estado
        self.input_folder = None
        self.output_folder = None
        self.input_files = []
        self.current_image = None
        self.proj = Proyecto()  # proyecto de uso local
        self.batch_worker = None

    # ---------- helpers ----------
    def log(self, text: str):
        self.log_list.addItem(text)
        self.log_list.scrollToBottom()

    def numpy_preview(self, img):
        try:
            pix = numpy_to_qpixmap(img)
            if not pix.isNull():
                self.image_label.setPixmap(pix.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception:
            pass

    # ---------- slots ----------
    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Seleccionar imagen", "", "Imágenes (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;Todos (*)")
        if not path:
            return
        try:
            self.proj.load(path, as_gray=False)
            self.current_image = path
            self.log("Imagen cargada: " + os.path.basename(path))
            self.numpy_preview(self.proj.get_processed())
        except Exception as e:
            QMessageBox.critical(self, "Error carga imagen", str(e))

    def select_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta de imágenes")
        if not path:
            return
        self.input_folder = path
        exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
        files = []
        for e in exts:
            files.extend(glob.glob(os.path.join(path, e)))
        files = sorted(files)
        self.input_files = files
        self.log(f"Carpeta seleccionada: {path} ({len(files)} imágenes)")
        default_out = os.path.join(path, "outputs")
        self.out_line.setText(default_out)
        self.output_folder = default_out

    def choose_output_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Seleccionar carpeta de salida")
        if path:
            self.output_folder = path
            self.out_line.setText(path)
            self.log(f"Carpeta de salida establecida: {path}")

    def run_script_single(self):
        if self.proj.get_processed() is None:
            QMessageBox.warning(self, "Sin imagen", "Primero selecciona una imagen para procesar.")
            return
        script_text = self.script_edit.toPlainText()
        if not script_text.strip():
            QMessageBox.warning(self, "Sin script", "Escribe o carga un archivo de tratamiento.")
            return
        try:
            steps = self.parser.parse(script_text)
        except Exception as e:
            QMessageBox.critical(self, "Error parseo", f"No se pudo parsear el script: {e}")
            return
        self.log("Ejecutando script (single)...")
        ok, result = self.parser.execute(steps, proyecto=self.proj, update_callback=lambda r, s: (self.log(s), self.numpy_preview(self.proj.get_processed())))
        if not ok:
            QMessageBox.critical(self, "Error ejecución", str(result))
            self.log("Error: " + str(result))
            return
        self.log("Single-run finalizado.")
        self.refresh_view()

    def process_folder(self):
        if not self.input_folder or not self.input_files:
            QMessageBox.warning(self, "Sin carpeta", "Selecciona primero una carpeta con imágenes.")
            return
        script_text = self.script_edit.toPlainText()
        if not script_text.strip():
            QMessageBox.warning(self, "Sin script", "Escribe o carga un archivo de tratamiento.")
            return
        try:
            steps = self.parser.parse(script_text)
        except Exception as e:
            QMessageBox.critical(self, "Error parseo", f"No se pudo parsear el script: {e}")
            return

        out_root = self.out_line.text().strip() or os.path.join(self.input_folder, "outputs")
        os.makedirs(out_root, exist_ok=True)

        # disable buttons
        self.btn_process_folder.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.btn_run.setEnabled(False)

        # start worker
        self.batch_worker = BatchWorker(self.parser, steps, self.input_files, out_root, Proyecto)
        self.batch_worker.progress.connect(lambda s: self.log(s))
        self.batch_worker.image_update.connect(lambda img: self.numpy_preview(img))
        self.batch_worker.finished.connect(self.on_batch_finished)
        self.batch_worker.start()
        self.log("Batch iniciado...")

    def cancel_batch(self):
        if self.batch_worker:
            self.batch_worker.abort()
            self.log("Solicitud de cancelación enviada; esperando a que termine la imagen actual...")

    def on_batch_finished(self, ok, info):
        self.btn_process_folder.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.btn_run.setEnabled(True)
        self.batch_worker = None
        if not ok:
            self.log(f"Batch finalizado con errores: {info}")
            QMessageBox.warning(self, "Batch terminado", "Terminado con errores. Revisa el log.")
        else:
            self.log(f"Batch completado. Resumen: {info}")
            QMessageBox.information(self, "Batch terminado", "Procesamiento por carpeta completado. CSVs y outputs guardados en la carpeta de salida.")

    def show_original(self):
        if not self.current_image:
            QMessageBox.information(self, "Original no disponible", "No hay imagen original cargada.")
            return
        try:
            img = cv2.imread(self.current_image, cv2.IMREAD_COLOR)
            self.numpy_preview(img)
            self.log("Mostrando original: " + os.path.basename(self.current_image))
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def refresh_view(self):
        try:
            img = self.proj.get_processed()
            if img is not None:
                self.numpy_preview(img)
        except Exception:
            pass

# ---------- run ----------
def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
