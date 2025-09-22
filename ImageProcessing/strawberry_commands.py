# strawberry_commands.py
"""
Comandos auxiliares para detección de fresas (HSV, máscaras, overlay, etc.)
Registrar con: register_strawberry_commands(parser)
"""

import cv2
import numpy as np
import os
from typing import Any

def register_strawberry_commands(parser):
    # Helper para asegurar BGR uint8
    def _ensure_bgr(img: Any):
        arr = np.asarray(img)
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating):
                arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
        if arr.ndim == 2:
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        if arr.ndim == 3 and arr.shape[2] == 3:
            return arr
        # fallback (intentar convertir)
        return arr

    # ToHSV(): devuelve imagen HSV de la imagen actual del proyecto
    def cmd_ToHSV(proj):
        img = proj.get_processed()
        if img is None:
            raise RuntimeError("ToHSV: no hay imagen cargada en Proyecto.")
        bgr = _ensure_bgr(img)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        return hsv

    # MaskHSV(hsv, lower_tuple, upper_tuple): devuelve máscara uint8 (0/255)
    def cmd_MaskHSV(proj, hsv_img, lower, upper):
        hsv = np.asarray(hsv_img)
        low = np.array(lower, dtype=np.uint8)
        up = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, low, up)
        return mask

    # CombineMasks(m1, m2, ...): OR de máscaras (entrada: listas/arrays)
    def cmd_CombineMasks(proj, *masks):
        if not masks:
            raise RuntimeError("CombineMasks: se requieren máscaras.")
        res = np.zeros_like(np.asarray(masks[0]), dtype=np.uint8)
        for m in masks:
            mm = np.asarray(m, dtype=np.uint8)
            res = cv2.bitwise_or(res, mm)
        return res

    # CleanMask(mask, kernel=[[0,1,0]...], erode=1, dilate=1)
    def cmd_CleanMask(proj, mask, kernel=[[0,1,0],[1,1,1],[0,1,0]], erode=1, dilate=1):
        m = np.asarray(mask, dtype=np.uint8)
        ker = np.array(kernel, dtype=np.uint8)
        ker = (ker > 0).astype(np.uint8)
        if int(erode) > 0:
            m = cv2.erode(m, ker, iterations=int(erode))
        if int(dilate) > 0:
            m = cv2.dilate(m, ker, iterations=int(dilate))
        return m

    # CCFilter(mask, min_area): remueve componentes con area < min_area
    def cmd_CCFilter(proj, mask, min_area=100):
        m = np.asarray(mask, dtype=np.uint8)
        if m.max() == 0:
            return m
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(m, connectivity=8)
        out = np.zeros_like(m)
        for lab in range(1, num_labels):
            area = int(stats[lab, cv2.CC_STAT_AREA])
            if area >= int(min_area):
                out[labels == lab] = 255
        return out

    # OverlayMask(base_img, mask, color_BGR): dibuja mask con color (retorna BGR uint8 image)
    def cmd_OverlayMask(proj, base, mask, color=(0,0,255)):
        base = _ensure_bgr(base)
        mask_u8 = (np.asarray(mask) > 0).astype(np.uint8) * 255
        color_layer = np.zeros_like(base, dtype=np.uint8)
        color_layer[:, :] = tuple(int(c) for c in color)  # BGR
        alpha = 0.6
        blended = base.copy()
        mask_bool = mask_u8.astype(bool)
        # aplicar mezcla solo donde mask es True
        blended[mask_bool] = cv2.addWeighted(base, 1 - alpha, color_layer, alpha, 0)[mask_bool]
        return blended

    # CombineOverlays(img1, img2): combina sumando y remapeando (clamp)
    def cmd_CombineOverlays(proj, a, b):
        A = _ensure_bgr(a).astype(np.int16)
        B = _ensure_bgr(b).astype(np.int16)
        out = np.clip(A + B, 0, 255).astype(np.uint8)
        return out

    # CountComponents(mask): devuelve entero
    def cmd_CountComponents(proj, mask):
        m = np.asarray(mask, dtype=np.uint8)
        if m.max() == 0:
            return 0
        nlabels, labels = cv2.connectedComponents(m)
        return max(0, int(nlabels) - 1)

    # SaveText(path, text): guarda texto en fichero
    def cmd_SaveText(proj, path, text):
        p = str(path)
        with open(p, "w", encoding="utf-8") as f:
            f.write(str(text))
        return p

    # Registrar en el parser
    parser.register_command("ToHSV", lambda proj: cmd_ToHSV(proj))
    parser.register_command("MaskHSV", lambda proj, hsv, low, high: cmd_MaskHSV(proj, hsv, low, high))
    parser.register_command("CombineMasks", lambda proj, *masks: cmd_CombineMasks(proj, *masks))
    parser.register_command("CleanMask", lambda proj, mask, kernel=None, erode=1, dilate=1:
                            cmd_CleanMask(proj, mask, kernel if kernel is not None else [[0,1,0],[1,1,1],[0,1,0]], erode, dilate))
    parser.register_command("CCFilter", lambda proj, mask, min_area=100: cmd_CCFilter(proj, mask, min_area))
    parser.register_command("OverlayMask", lambda proj, base, mask, color=(0,0,255): cmd_OverlayMask(proj, base, mask, color))
    parser.register_command("CombineOverlays", lambda proj, a, b: cmd_CombineOverlays(proj, a, b))
    parser.register_command("CountComponents", lambda proj, mask: cmd_CountComponents(proj, mask))
    parser.register_command("SaveText", lambda proj, path, text: cmd_SaveText(proj, path, text))

    # devuelve true para confirmar (opcional)
    return True
