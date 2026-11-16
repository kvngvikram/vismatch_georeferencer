"""Rendering of canvas windows + vismatch worker thread."""
import contextlib
import os
import ctypes
import threading
import numpy as np
from qgis.core import QgsMapSettings, QgsMapRendererSequentialJob
from qgis.PyQt.QtCore import QSize, QThread, pyqtSignal
from qgis.PyQt.QtGui import QImage, QColor


class WindowGeoref:
    """Exact pixel <-> map relation of a rendered window.

    Built from the *actual* rendered image size and the visible extent, so it can't
    drift because of device-pixel-ratio, rounding of the output size, canvas widget
    frames or map rotation.  Pixel (0, 0) is the top-left *corner* of the first pixel.
    """

    def __init__(self, extent, width, height):
        self.xmin = extent.xMinimum()
        self.ymax = extent.yMaximum()
        self.sx = extent.width() / float(width)    # map units per pixel (x)
        self.sy = extent.height() / float(height)  # map units per pixel (y)
        self.width, self.height = width, height

    def to_map(self, px, py):
        """px, py in corner-based pixel units (float) -> (map_x, map_y)."""
        return self.xmin + px * self.sx, self.ymax - py * self.sy


def render_canvas_window(canvas, layer, max_side):
    """Render ONLY `layer` for the canvas' current extent at <= max_side pixels.

    Reading is done by QGIS/GDAL, so overviews are used and the full raster is
    never loaded into RAM. Returns (QImage, ndarray HxWx3 uint8, WindowGeoref).
    """
    s = QgsMapSettings(canvas.mapSettings())
    s.setLayers([layer])
    s.setRotation(0.0)              # keep the window axis-aligned
    s.setDevicePixelRatio(1.0)      # image size must equal the output size (HiDPI!)
    w, h = max(canvas.width(), 1), max(canvas.height(), 1)
    k = max_side / float(max(w, h))
    s.setOutputSize(QSize(max(64, int(round(w * k))), max(64, int(round(h * k)))))
    s.setBackgroundColor(QColor(0, 0, 0))
    ext = s.visibleExtent()         # extent that is really rendered into the image
    job = QgsMapRendererSequentialJob(s)
    job.start()
    job.waitForFinished()
    img = job.renderedImage().convertToFormat(QImage.Format.Format_RGB888)
    W, H, bpl = img.width(), img.height(), img.bytesPerLine()
    bits = img.constBits()
    if hasattr(bits, "setsize"):
        bits.setsize(H * bpl)
    arr = np.frombuffer(bits, np.uint8).reshape(H, bpl)[:, : W * 3].reshape(H, W, 3).copy()
    return img.copy(), arr, WindowGeoref(ext, W, H)


def _np_pts(x):
    if x is None:
        return np.zeros((0, 2))
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=float).reshape(-1, 2)


def _np_vec(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    try:
        return np.asarray(x, dtype=float).reshape(-1)
    except Exception:
        return None


def inlier_scores(res):
    """Matcher confidence ('matched_confidences' in the vismatch result) for every inlier.

    vismatch reports confidences for all matches, not only the RANSAC inliers, and does not
    return the inlier mask, so each inlier is looked up among the matches by its exact
    keypoint pair. Returns an array aligned with the inliers (NaN where unknown) or None
    if the model provides no usable scores."""
    mc = _np_vec(res.get("matched_confidences"))
    m0, m1 = _np_pts(res.get("matched_kpts0")), _np_pts(res.get("matched_kpts1"))
    i0, i1 = _np_pts(res.get("inlier_kpts0")), _np_pts(res.get("inlier_kpts1"))
    if mc is None or len(mc) == 0 or len(mc) != len(m0) or len(m0) != len(m1) or len(i0) != len(i1):
        return None
    look = {}
    for j, row in enumerate(np.ascontiguousarray(np.hstack([m0, m1]))):
        look.setdefault(row.tobytes(), j)
    conf = np.full(len(i0), np.nan)
    for i, row in enumerate(np.ascontiguousarray(np.hstack([i0, i1]))):
        j = look.get(row.tobytes())
        if j is not None:
            conf[i] = mc[j]
    return conf if np.isfinite(conf).any() else None


def import_vismatch():
    from vismatch import get_matcher
    return get_matcher


def list_models():
    try:
        from vismatch import available_models
        return sorted(list(available_models))
    except Exception:
        return []


def pick_device(name):
    if name != "auto":
        return name
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class MatchCancelled(Exception):
    pass


def weight_dirs():
    """Folders where model weights typically land (used to estimate download progress)."""
    dirs = []
    home = os.path.expanduser("~")
    torch_home = os.environ.get("TORCH_HOME") or os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.join(home, ".cache")), "torch")
    dirs.append(os.path.join(torch_home, "hub"))
    hf = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.join(home, ".cache", "huggingface")), "hub")
    dirs.append(hf)
    for name in ("vismatch",):
        with contextlib.suppress(Exception):
            mod = __import__(name)
            dirs.append(os.path.dirname(mod.__file__))
    return [d for d in dirs if os.path.isdir(d)]


def dir_size(path):
    total = 0
    stack = [path]
    while stack:
        p = stack.pop()
        try:
            with os.scandir(p) as it:
                for e in it:
                    try:
                        if e.is_dir(follow_symlinks=False):
                            if e.name != "__pycache__":
                                stack.append(e.path)
                        else:
                            total += e.stat(follow_symlinks=False).st_size
                    except OSError:
                        pass
        except OSError:
            pass
    return total


class MatchWorker(QThread):
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)
    status = pyqtSignal(str)

    def __init__(self, cache, model, device, kwargs, arr0, arr1, parent=None):
        super().__init__(parent)
        self.cache, self.model, self.device = cache, model, device
        self.kwargs = dict(kwargs)  # matcher constructor options (RANSAC settings)
        self.arr0, self.arr1 = arr0, arr1
        self.cancelled = False
        self._tid = None

    def cancel(self):
        """Best effort: python threads can't be killed, so we flag the job as
        cancelled (its results are dropped) and inject an exception that fires
        as soon as the thread is back in python code (e.g. between download chunks)."""
        self.cancelled = True
        if self._tid is not None:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_ulong(self._tid), ctypes.py_object(MatchCancelled))

    def _get_matcher(self):
        import torch  # noqa
        dev = pick_device(self.device)
        key = (self.model, dev, tuple(sorted(self.kwargs.items())))
        if key in self.cache:
            return self.cache[key]
        get_matcher = import_vismatch()
        self.status.emit("Loading / downloading weights for '%s'" % self.model)
        try:
            m = get_matcher(self.model, device=dev, **self.kwargs)
        except TypeError:  # model doesn't accept some of the optional kwargs
            m = get_matcher(self.model, device=dev)
        self.cache.clear()  # keep only one model in memory
        self.cache[key] = m
        return m

    def run(self):
        self._tid = threading.get_ident()
        try:
            self.status.emit("Importing torch / vismatch")
            import torch
            matcher = self._get_matcher()
            if self.cancelled:
                return
            self.status.emit("Matching")
            t0 = torch.from_numpy(self.arr0).permute(2, 0, 1).float() / 255.0
            t1 = torch.from_numpy(self.arr1).permute(2, 0, 1).float() / 255.0
            res = matcher(t0, t1)
            out = {
                "k0": _np_pts(res.get("inlier_kpts0")),
                "k1": _np_pts(res.get("inlier_kpts1")),
                "conf": inlier_scores(res),  # aligned with k0/k1, or None
            }
            if not self.cancelled:
                self.finished_ok.emit(out)
        except MatchCancelled:
            return
        except Exception as e:  # noqa
            if self.cancelled:
                return
            import traceback
            self.failed.emit("%s\n\n%s" % (e, traceback.format_exc()))
