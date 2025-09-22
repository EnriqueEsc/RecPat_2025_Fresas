"""

Clase Proyecto que replica operaciones del código MATLAB que compartiste.
Diseño:
- Estado interno: self.image (uint8 0..255), self.last_image
- Métodos que modifican la imagen y la actualizan internamente.
- Cada método devuelve la imagen resultante además de actualizar el estado.
- Soporta imágenes grayscale y RGB; para operaciones que requieren gris,
  por defecto se convierten a gris (pero puede forzarse per_channel=True).
"""

from typing import Optional, Tuple, Any
import numpy as np
import cv2
from skimage import exposure, color, util, segmentation
from skimage.filters import roberts, prewitt, sobel
from skimage.segmentation import mark_boundaries
from scipy import ndimage as ndi
from scipy.stats import mode
from sklearn.cluster import KMeans

def _to_uint8(img: np.ndarray) -> np.ndarray:
    if img is None:
        return None
    img = np.asarray(img)
    if img.dtype == np.uint8:
        return img
    if np.issubdtype(img.dtype, np.floating):
        if img.max() <= 1.0:
            return (img * 255.0).round().astype(np.uint8)
    return np.clip(img, 0, 255).astype(np.uint8)

def _is_color(img: np.ndarray) -> bool:
    return img is not None and img.ndim == 3 and img.shape[2] == 3

def _apply_per_channel(img: np.ndarray, fn):
    """Aplica fn a cada canal independientemente y concatena."""
    chans = cv2.split(img)
    out_ch = []
    for c in chans:
        out_ch.append(fn(c))
    return cv2.merge(out_ch)

class Proyecto:
    def __init__(self, image: Optional[np.ndarray] = None, path: Optional[str] = None, as_gray: bool = False):
        """
        Inicializa con una imagen (array) o ruta.
        as_gray: si True, siempre carga/conserva en escala de grises.
        """
        self.image = None
        self.last_image = None
        if path is not None:
            self.load(path, as_gray=as_gray)
        elif image is not None:
            self.image = _to_uint8(image)
            if as_gray and _is_color(self.image):
                self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    # ---------- I/O ----------
    def load(self, path: str, as_gray: bool = True) -> np.ndarray:
        flag = cv2.IMREAD_GRAYSCALE if as_gray else cv2.IMREAD_COLOR
        img = cv2.imread(path, flag)
        img = _to_uint8(img)
        self.last_image = self.image
        self.image = img
        return self.image

    def Update(self, img: np.ndarray) -> dict:
        """Equivalente a tu Update de MATLAB: actualiza self.image y devuelve dict con key grayScaleImage."""
        self.last_image = self.image
        self.image = _to_uint8(img)
        return {"grayScaleImage": self.image}

    def get_processed(self) -> np.ndarray:
        """Devuelve la imagen actual (procesada)."""
        return self.image

    # ---------- Utilitarios ----------
    def getHistogram(self) -> dict:
        img = _to_uint8(self.image)
        if _is_color(img):
            # devuelve histograma por canal
            hists = [np.histogram(img[..., i].flatten(), bins=256, range=(0,255))[0] for i in range(3)]
            return {"imageHistogram": hists}
        else:
            hist = np.histogram(img.flatten(), bins=256, range=(0,255))[0]
            return {"imageHistogram": hist}

    # ---------- Cambio de profundidad ----------
    def changeDepth(self, bits: int, per_channel: bool = False) -> dict:
        """Reduce quantización a 'bits' niveles."""
        img = _to_uint8(self.image)
        levels = 2 ** bits if bits >= 1 else 1
        def quantize(channel):
            ch = channel.astype(np.float32)
            q = np.floor(ch / 256.0 * levels) * (255.0 / max(levels - 1, 1))
            return np.clip(q, 0, 255).astype(np.uint8)
        if _is_color(img) and per_channel:
            out = _apply_per_channel(img, quantize)
        else:
            if _is_color(img):
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            out = quantize(gray)
        self.last_image = self.image
        self.image = out
        return {"imageMap": out}

    def updateGrayScaleWithMap(self, new_map: np.ndarray) -> dict:
        """Símil a la llamada que usabas en MATLAB: actualiza grayscale con mapa dado."""
        self.last_image = self.image
        self.image = _to_uint8(new_map)
        return {"grayScaleImage": self.image}

    # ---------- Ecualización ----------
    def equalizeImage(self, per_channel: bool = False) -> dict:
        """Ecualiza histograma. Por defecto sobre gris; si per_channel=True aplica a cada canal."""
        img = _to_uint8(self.image)
        def eqch(ch):
            eq = exposure.equalize_hist(ch)  # float 0..1
            return (eq * 255.0).round().astype(np.uint8)
        if _is_color(img) and per_channel:
            out = _apply_per_channel(img, eqch)
        else:
            if _is_color(img):
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            out = eqch(gray)
        self.last_image = self.image
        self.image = out
        return {"equalizedImage": out}

    # ---------- Inversiones ----------
    def photoInvertImage(self) -> dict:
        out = (255 - _to_uint8(self.image)).astype(np.uint8)
        self.last_image = self.image
        self.image = out
        return {"invertedImage": out}

    def binaryInvert(self) -> dict:
        """Imbinariza con Otsu y luego complementa (como tus pasos imbinarize + imcomplement)."""
        img = _to_uint8(self.image)
        if _is_color(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        comp = cv2.bitwise_not(bin_img)
        self.last_image = self.image
        self.image = comp
        return {"binaryInverted": comp}

    # ---------- Aritmética ----------
    def AddToImage(self, other: np.ndarray) -> dict:
        a = _to_uint8(self.image)
        b = _to_uint8(other)
        b_resized = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)
        res = np.clip(a.astype(np.int16) + b_resized.astype(np.int16), 0, 255).astype(np.uint8)
        self.last_image = self.image
        self.image = res
        return {"addImage": res}

    def SusToImage(self, other: np.ndarray) -> dict:
        a = _to_uint8(self.image)
        b = _to_uint8(other)
        b_resized = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)
        res = np.clip(a.astype(np.int16) - b_resized.astype(np.int16), 0, 255).astype(np.uint8)
        self.last_image = self.image
        self.image = res
        return {"susImage": res}

    # ---------- Transformaciones geométricas ----------
    def rotate(self, angle: float) -> dict:
        img = _to_uint8(self.image)
        (h, w) = img.shape[:2]
        center = (w/2.0, h/2.0)
        M = cv2.getRotationMatrix2D(center, -angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        self.last_image = self.image
        self.image = rotated
        return {"rotatedImage": rotated}

    def mirror(self) -> dict:
        """Construye el bloque espejo como en tu MATLAB: [img2,img5; img,img3]."""
        img = _to_uint8(self.image)
        # adaptado para cualquier canal (si color, se aplica por channel)
        def make_mirror(ch):
            img2 = np.flipud(ch)      # flipdim(ch,1)
            img3 = np.fliplr(ch)      # flipdim(ch,2)
            img4 = np.flipud(ch)
            img5 = np.fliplr(img4)
            top = np.concatenate((img2, img5), axis=1)
            bottom = np.concatenate((ch, img3), axis=1)
            full = np.concatenate((top, bottom), axis=0)
            return full
        if _is_color(img):
            out = _apply_per_channel(img, make_mirror)
        else:
            out = make_mirror(img)
        self.last_image = self.image
        self.image = out
        return {"espejo": out}

    # ---------- Filtros ----------
    def gaussianFilter(self, repeats: int = 10, ksize: Tuple[int,int]=(5,5), per_channel: bool=False) -> dict:
        def gf(ch):
            out = ch.copy()
            for _ in range(repeats):
                out = cv2.GaussianBlur(out, ksize, 0)
            return out
        img = _to_uint8(self.image)
        if _is_color(img) and per_channel:
            out = _apply_per_channel(img, gf)
        elif _is_color(img) and not per_channel:
            # aplicar al color completo (funciona con cv2 GaussianBlur sobre 3-ch)
            out = img.copy()
            for _ in range(repeats):
                out = cv2.GaussianBlur(out, ksize, 0)
        else:
            out = gf(img)
        self.last_image = self.image
        self.image = out
        return {"gaussian": out}

    def laplacian4(self) -> dict:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        out = cv2.filter2D(_to_uint8(self.image), -1, kernel)
        self.last_image = self.image
        self.image = np.clip(out, 0, 255).astype(np.uint8)
        return {"lap4": self.image}

    def laplacian8(self) -> dict:
        kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]], dtype=np.float32)
        out = cv2.filter2D(_to_uint8(self.image), -1, kernel)
        self.last_image = self.image
        self.image = np.clip(out, 0, 255).astype(np.uint8)
        return {"lap8": self.image}

    def averageFilter(self, ksize: Tuple[int,int]=(3,3)) -> dict:
        kernel = np.ones(ksize, dtype=np.float32) / (ksize[0]*ksize[1])
        out = cv2.filter2D(_to_uint8(self.image), -1, kernel)
        self.last_image = self.image
        self.image = np.clip(out, 0, 255).astype(np.uint8)
        return {"avg": self.image}

    def modeFilter(self, size: int = 3, per_channel: bool = False) -> dict:
        def local_mode(block):
            m = mode(block, axis=None)
            return m.mode[0]
        img = _to_uint8(self.image)
        if _is_color(img) and per_channel:
            out = _apply_per_channel(img, lambda ch: ndi.generic_filter(ch, local_mode, size=size, mode='reflect').astype(np.uint8))
        else:
            if _is_color(img):
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            out = ndi.generic_filter(gray, local_mode, size=size, mode='reflect').astype(np.uint8)
        self.last_image = self.image
        self.image = out
        return {"mode": out}

    def medianFilter(self, size: int = 3) -> dict:
        img = _to_uint8(self.image)
        if _is_color(img):
            out = cv2.medianBlur(img, size)
        else:
            out = ndi.median_filter(img, size=size).astype(np.uint8)
        self.last_image = self.image
        self.image = out
        return {"median": out}

    def maxFilter(self, size: int = 3) -> dict:
        out = ndi.maximum_filter(_to_uint8(self.image), size=size).astype(np.uint8)
        self.last_image = self.image
        self.image = out
        return {"max": out}

    def minFilter(self, size: int = 3) -> dict:
        out = ndi.minimum_filter(_to_uint8(self.image), size=size).astype(np.uint8)
        self.last_image = self.image
        self.image = out
        return {"min": out}

    # ---------- Bordes ----------
    def roberts(self) -> dict:
        img = _to_uint8(self.image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if _is_color(img) else img
        out = roberts(gray)
        out_u8 = (out * 255).astype(np.uint8)
        self.last_image = self.image
        self.image = out_u8
        return {"roberts": out_u8}

    def prewitt(self) -> dict:
        img = _to_uint8(self.image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if _is_color(img) else img
        out = prewitt(gray)
        out_u8 = (np.abs(out) * 255 / (np.max(np.abs(out)) + 1e-12)).astype(np.uint8)
        self.last_image = self.image
        self.image = out_u8
        return {"prewitt": out_u8}

    def sobel(self) -> dict:
        img = _to_uint8(self.image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if _is_color(img) else img
        out = sobel(gray)
        out_u8 = (np.abs(out) * 255 / (np.max(np.abs(out)) + 1e-12)).astype(np.uint8)
        self.last_image = self.image
        self.image = out_u8
        return {"sobel": out_u8}

    # ---------- Morfología ----------
    def DilateImage(self, el3x3, iterations: int = 1) -> dict:
        kernel = np.array(el3x3, dtype=np.uint8)
        img = _to_uint8(self.image)
        if _is_color(img):
            chans = cv2.split(img)
            res_ch = [cv2.dilate(c, kernel, iterations=iterations) for c in chans]
            out = cv2.merge(res_ch)
        else:
            out = cv2.dilate(img, kernel, iterations=iterations)
        self.last_image = self.image
        self.image = out
        return {"dilImage": out}

    def ErodeImage(self, el3x3, iterations: int = 1) -> dict:
        kernel = np.array(el3x3, dtype=np.uint8)
        img = _to_uint8(self.image)
        if _is_color(img):
            chans = cv2.split(img)
            res_ch = [cv2.erode(c, kernel, iterations=iterations) for c in chans]
            out = cv2.merge(res_ch)
        else:
            out = cv2.erode(img, kernel, iterations=iterations)
        self.last_image = self.image
        self.image = out
        return {"erImage": out}

    # ---------- Segmentación ----------
    def imsegkmeans(self, n_clusters: int = 3, use_color: bool = False) -> dict:
        """
        Segmentación estilo imsegkmeans. Si use_color=True y la imagen es RGB se usa espacio RGB como características.
        Devuelve labels y overlay (RGB uint8).
        """
        img = _to_uint8(self.image)
        if _is_color(img) and use_color:
            feats = img.reshape(-1, 3).astype(np.float32)
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if _is_color(img) else img
            feats = gray.reshape(-1, 1).astype(np.float32)
        k = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(feats)
        labels = k.labels_.reshape(img.shape[0], img.shape[1])
        overlay = (color.label2rgb(labels, image=(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if _is_color(img) else img), bg_label=0) * 255).astype(np.uint8)
        # no modificamos self.image a menos que el usuario quiera; pero para compatibilidad lo pondremos como overlay
        self.last_image = self.image
        self.image = overlay
        return {"labels": labels, "overlay": overlay}

    def slic_superpixels(self, n_segments: int = 100) -> dict:
        img = _to_uint8(self.image)
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if not _is_color(img) else img
        segments = segmentation.slic(util.img_as_float(rgb), n_segments=n_segments, compactness=10, start_label=1)
        boundaries = mark_boundaries(rgb, segments)
        boundaries_u8 = (boundaries * 255).astype(np.uint8)
        self.last_image = self.image
        self.image = boundaries_u8
        return {"segments": segments, "boundaries": boundaries_u8}
