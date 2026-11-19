import contextlib
import os
import time
import numpy as np
from qgis.core import (QgsCoordinateReferenceSystem,
                       QgsPointXY, QgsRectangle, QgsMapLayerProxyModel)
from qgis.gui import QgsMapCanvas, QgsMapToolPan, QgsMapLayerComboBox, QgsVertexMarker, QgsRubberBand, QgsMapTool
from qgis.PyQt.QtCore import Qt, QRectF, QPointF, QEvent
from qgis.PyQt.QtGui import QColor, QPainter, QPen, QCursor, QKeySequence
from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit, QPushButton,
    QSpinBox, QDoubleSpinBox, QComboBox, QSplitter, QWidget, QTableWidget,
    QTableWidgetItem, QFileDialog, QMessageBox, QCompleter, QAbstractItemView,
    QPlainTextEdit, QApplication, QProgressBar)
try:  # Qt6 (QGIS 4) moved QAction to QtGui
    from qgis.PyQt.QtGui import QAction
except ImportError:  # Qt5 (QGIS 3)
    from qgis.PyQt.QtWidgets import QAction

from qgis.PyQt.QtCore import QTimer
from .fit import residuals, make_fits, METHODS
from .matching import render_canvas_window, MatchWorker, list_models, weight_dirs, dir_size

try:
    from qgis.core import Qgis
    RASTER_FILTER = Qgis.LayerFilter.RasterLayer  # QGIS >= 3.32 and QGIS 4
except (ImportError, AttributeError):
    try:
        RASTER_FILTER = QgsMapLayerProxyModel.Filter.RasterLayer
    except AttributeError:
        RASTER_FILTER = QgsMapLayerProxyModel.RasterLayer

POINTS_HEADER = "mapX,mapY,sourceX,sourceY,enable,dX,dY,residual"


def px_to_map(win, x, y):
    """Matcher keypoint (pixel-centre convention: centre of first pixel = 0.0)
    -> map coordinates.  +0.5 converts to the corner-based convention of the window."""
    mx, my = win.to_map(float(x) + 0.5, float(y) + 0.5)
    return QgsPointXY(mx, my)


def spread_select(pts, k):
    """Farthest-point sampling so the kept GCPs are spatially well distributed."""
    n = len(pts)
    if n <= k:
        return np.arange(n)
    first = int(np.argmin(((pts - pts.mean(0)) ** 2).sum(1)))
    sel = [first]
    d = ((pts - pts[first]) ** 2).sum(1)
    for _ in range(k - 1):
        i = int(np.argmax(d))
        sel.append(i)
        d = np.minimum(d, ((pts - pts[i]) ** 2).sum(1))
    return np.array(sorted(sel))


class ClickPanTool(QgsMapToolPan):
    """Pan tool that also handles clicks on accepted GCP markers.

    A press that starts on a GCP marker (hit_test) is NOT passed to the pan tool, so it
    never recentres/pans the canvas - a click there only selects the GCP. Everywhere
    else it behaves like the normal pan tool."""

    def __init__(self, canvas, on_click, hit_test):
        super().__init__(canvas)
        self._on_click = on_click
        self._hit_test = hit_test
        self._press = None
        self._hit = False

    def canvasPressEvent(self, e):
        self._press = e.pos()
        self._hit = bool(self._hit_test(self.canvas(), e.pos()))
        if not self._hit:
            super().canvasPressEvent(e)

    def canvasMoveEvent(self, e):
        if not self._hit:
            super().canvasMoveEvent(e)

    def canvasReleaseEvent(self, e):
        click = (self._press is not None and
                 (e.pos() - self._press).manhattanLength() <= 4)
        if self._hit:
            self._hit = False
            if click:
                self._on_click(self.canvas(), e.pos())
            return
        super().canvasReleaseEvent(e)


class ManualPointTool(QgsMapTool):
    """Crosshair tool for placing manual GCPs. Left click only picks a point - it never
    recentres/pans the canvas (the pan tool does that on every click). Pan with the
    middle or right mouse button drag; the wheel still zooms at the cursor."""

    def __init__(self, canvas, on_click):
        super().__init__(canvas)
        self._on_click = on_click
        self._press = None
        self._panning = False
        self.setCursor(QCursor(Qt.CursorShape.CrossCursor))

    def canvasPressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._press = e.pos()
        elif e.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton):
            self._panning = True

    def canvasMoveEvent(self, e):
        if self._panning:
            self.canvas().panAction(e)

    def canvasReleaseEvent(self, e):
        if self._panning and e.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.RightButton):
            self._panning = False
            self.canvas().panActionEnd(e.pos())
        elif e.button() == Qt.MouseButton.LeftButton and self._press is not None:
            if (e.pos() - self._press).manhattanLength() <= 4:
                self._on_click(self.canvas(), e.pos())
            self._press = None


class MatchView(QWidget):
    """Side-by-side preview of the two windows with lines joining inlier keypoints.

    Mouse wheel = zoom at cursor, left-drag = pan, double-click = reset view."""

    MAX_ZOOM = 40.0

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(240)
        self.img0 = self.img1 = None
        self.k0 = self.k1 = np.zeros((0, 2))
        self.zoom = 1.0
        self.pan = QPointF(0.0, 0.0)   # screen-space offset of the zoomed content
        self._drag = None
        self.setToolTip("Wheel = zoom (at cursor), drag = pan, double-click = reset view")

    def set_data(self, img0, img1, k0, k1):
        self.img0, self.img1, self.k0, self.k1 = img0, img1, k0, k1
        self.reset_view()

    def reset_view(self):
        self.zoom = 1.0
        self.pan = QPointF(0.0, 0.0)
        self.update()

    # ------------------------------------------------------------ interaction
    @staticmethod
    def _pos(e):
        try:
            return QPointF(e.position())
        except AttributeError:  # Qt5
            return QPointF(e.pos())

    def wheelEvent(self, e):
        if self.img0 is None:
            return
        steps = e.angleDelta().y() / 120.0
        if steps == 0:
            return
        new_zoom = min(max(self.zoom * (1.25 ** steps), 1.0), self.MAX_ZOOM)
        f = new_zoom / self.zoom
        pos = self._pos(e)
        # keep the point under the cursor fixed while zooming
        self.pan = pos - (pos - self.pan) * f
        self.zoom = new_zoom
        if self.zoom <= 1.0:
            self.pan = QPointF(0.0, 0.0)
        self.update()
        e.accept()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self.img0 is not None:
            self._drag = (self._pos(e), QPointF(self.pan))
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, e):
        if self._drag is not None:
            start, pan0 = self._drag
            self.pan = pan0 + (self._pos(e) - start)
            self.update()

    def mouseReleaseEvent(self, e):
        self._drag = None
        self.unsetCursor()

    def mouseDoubleClickEvent(self, e):
        self.reset_view()

    # ---------------------------------------------------------------- painting
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(35, 35, 35))
        if self.img0 is None:
            p.setPen(QColor(200, 200, 200))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       "Match preview appears here (inlier keypoints joined by lines)")
            return
        gap = 12
        w0, h0, w1, h1 = self.img0.width(), self.img0.height(), self.img1.width(), self.img1.height()
        s = min((self.width() - gap) / float(w0 + w1), self.height() / float(max(h0, h1)))
        x1 = w0 * s + gap
        p.setClipRect(self.rect())
        p.translate(self.pan)
        p.scale(self.zoom, self.zoom)
        # smooth when zoomed out, crisp pixels when zoomed far in
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, self.zoom * s < 3.0)
        p.drawImage(QRectF(0, 0, w0 * s, h0 * s), self.img0)
        p.drawImage(QRectF(x1, 0, w1 * s, h1 * s), self.img1)
        r = 3.0 / self.zoom  # marker radius stays ~3 screen px at any zoom
        for i, (a, b) in enumerate(zip(self.k0, self.k1)):
            pen = QPen(QColor.fromHsv((i * 37) % 360, 230, 255), 1.3)
            pen.setCosmetic(True)  # constant on-screen width
            p.setPen(pen)
            # keypoints use pixel-centre convention -> +0.5 to draw on the pixel centre
            pa = QPointF((a[0] + 0.5) * s, (a[1] + 0.5) * s)
            pb = QPointF(x1 + (b[0] + 0.5) * s, (b[1] + 0.5) * s)
            p.drawLine(pa, pb)
            p.drawEllipse(pa, r, r)
            p.drawEllipse(pb, r, r)
        p.resetTransform()
        p.setPen(QColor(220, 220, 220))
        p.drawText(8, self.height() - 8, "zoom x%.1f  (wheel = zoom, drag = pan, double-click = reset)" % self.zoom)


class VisMatchDialog(QDialog):
    def __init__(self, iface, parent=None):
        super().__init__(parent)
        self.iface = iface
        self.setWindowTitle("VisMatch Georeferencer")
        self.resize(1400, 950)
        self.setWindowFlag(Qt.WindowType.WindowMaximizeButtonHint, True)

        self.gcps = []          # dicts: mx,my,col,row,region
        self.markers = []
        self.region_counter = 0
        self._manual = False
        self._fits = None
        self._sync_on = False
        self._syncing = False
        self._sync_mk = {}
        self._warping = False
        self._warp_cancel = False
        self._pending = {}
        self._pend_markers = []
        self.pending = None     # last match result (not yet accepted)
        self.matcher_cache = {}
        self.worker = None
        self._orphans = []      # cancelled workers still unwinding in the background
        self._stage = ""
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self._tick)
        self._build_ui()
        self._on_layers_changed()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        root = QVBoxLayout(self)

        top = QHBoxLayout()
        self.cb_base = QgsMapLayerComboBox()
        self.cb_base.setFilters(RASTER_FILTER)
        self.cb_move = QgsMapLayerComboBox()
        self.cb_move.setFilters(RASTER_FILTER)
        top.addWidget(QLabel("Basemap (georeferenced):"))
        top.addWidget(self.cb_base, 1)
        top.addWidget(QLabel("Raster to register:"))
        top.addWidget(self.cb_move, 1)
        self.sp_rot = QDoubleSpinBox()
        self.sp_rot.setRange(-180, 180)
        self.sp_rot.setSuffix("° rot")
        self.sp_rot.setToolTip("Rotate the right view so both windows are roughly aligned "
                               "(helps matchers that are not rotation invariant).")
        top.addWidget(self.sp_rot)
        root.addLayout(top)

        split_v = QSplitter(Qt.Orientation.Vertical)
        split_h = QSplitter(Qt.Orientation.Horizontal)
        self.canvas_base = self._make_canvas()
        self.canvas_move = self._make_canvas()
        split_h.addWidget(self._wrap("Basemap", self.canvas_base))
        split_h.addWidget(self._wrap("Raster to register (pixel space if not georeferenced)", self.canvas_move))
        split_v.addWidget(split_h)
        self.view = MatchView()
        split_v.addWidget(self.view)
        split_v.setStretchFactor(0, 3)
        split_v.setStretchFactor(1, 2)
        root.addWidget(split_v, 1)

        # --- matcher settings
        g = QGridLayout()
        self.ed_model = QLineEdit("superpoint-lightglue")
        models = list_models()
        if models:
            comp = QCompleter(models)
            comp.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
            comp.setFilterMode(Qt.MatchFlag.MatchContains)
            self.ed_model.setCompleter(comp)
        self.ed_model.setPlaceholderText("vismatch model name, e.g. superpoint-lightglue, loftr, roma ...")
        self.cb_dev = QComboBox()
        self.cb_dev.addItems(["auto", "cpu", "cuda", "mps"])
        self.sp_chunk = QSpinBox()
        self.sp_chunk.setRange(128, 8192)
        self.sp_chunk.setValue(1024)
        self.sp_chunk.setSingleStep(64)
        self.sp_chunk.setToolTip("Size (longest side, px) of the image of each zoomed view that is given to the\n"
                                 "matcher. Larger = more detail and more GCP precision, but more memory and time.\n"
                                 "It does not change what the canvases show.")
        self.sp_ransac = QDoubleSpinBox()
        self.sp_ransac.setRange(0.5, 50)
        self.sp_ransac.setValue(3.0)
        self.sp_maxg = QSpinBox()
        self.sp_maxg.setRange(3, 10000)
        self.sp_maxg.setValue(25)
        self.sp_maxg.setToolTip("Max GCPs kept per region (chosen from the inliers by 'GCP selection').")
        self.cb_sel = QComboBox()
        self.cb_sel.addItems(["Vismatch score", "Spatial spread"])
        self.cb_sel.setToolTip("How the kept GCPs are chosen when there are more inliers than the maximum:\n"
                               "Vismatch score = highest matcher confidence first (falls back to spatial\n"
                               "spread if the model gives no scores); Spatial spread = farthest-point sampling.")
        self.sp_mininl = QSpinBox()
        self.sp_mininl.setRange(3, 10000)
        self.sp_mininl.setValue(8)
        row = 0
        for lbl, w, c in [("Model:", self.ed_model, 0), ("Device:", self.cb_dev, 2),
                          ("Match size (px):", self.sp_chunk, 4)]:
            g.addWidget(QLabel(lbl), row, c)
            g.addWidget(w, row, c + 1)
        g.setColumnStretch(1, 3)
        row = 1
        for lbl, w, c in [("RANSAC thresh (px):", self.sp_ransac, 0), ("Max GCPs / region:", self.sp_maxg, 2),
                          ("Min inliers:", self.sp_mininl, 4), ("GCP selection:", self.cb_sel, 6)]:
            g.addWidget(QLabel(lbl), row, c)
            g.addWidget(w, row, c + 1)
        root.addLayout(g)

        btns = QHBoxLayout()
        self.btn_match = QPushButton("1. Match current windows")
        self.btn_accept = QPushButton("2. Accept → add GCPs")
        self.btn_accept.setEnabled(False)
        self.btn_discard = QPushButton("Discard")
        self.btn_discard.setEnabled(False)
        # Keyboard shortcuts: keys 1 / 2. A disabled button ignores its shortcut, and text boxes /
        # spin boxes keep the key for typing, so these only fire when you are not entering text.
        self.btn_match.setShortcut(QKeySequence("1"))
        self.btn_match.setToolTip("Shortcut: 1")
        self.btn_accept.setShortcut(QKeySequence("2"))
        self.btn_accept.setToolTip("Shortcut: 2")
        self.btn_undo = QPushButton("3. Delete last accepted")
        self.btn_undo.setShortcut(QKeySequence("3"))
        self.btn_undo.setToolTip("Shortcut: 3\nRemoves every GCP of the most recently accepted region (manual GCPs are kept).")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # busy indicator
        self.progress.setMaximumWidth(140)
        self.progress.setVisible(False)
        self.lbl_status = QLabel("Zoom both views to the same approximate area, then match.")
        btns.addWidget(self.btn_match)
        btns.addWidget(self.btn_accept)
        btns.addWidget(self.btn_undo)
        btns.addWidget(self.btn_discard)
        self.btn_manual = QPushButton("+ Manual GCP")
        self.btn_manual.setCheckable(True)
        self.btn_manual.setToolTip("Add a GCP by hand: click the point on the basemap, then the same point on the\n"
                                   "other image (zoom in first for accuracy). Manual GCPs get region 0.")
        self.btn_manual.toggled.connect(self._toggle_manual)
        btns.addWidget(self.btn_manual)
        self.btn_sync = QPushButton("Sync views")
        self.btn_sync.setCheckable(True)
        self.btn_sync.setToolTip("Keep both canvases on the same area using the selected fitting method, and show a\n"
                                 "second cursor on the other canvas at the corresponding location.\n"
                                 "Needs enough GCPs for the method (3 / 6 / 10 / 3). Polynomials can drift far\n"
                                 "outside the GCP area.")
        self.btn_sync.toggled.connect(self._toggle_sync)
        btns.addWidget(self.btn_sync)
        btns.addWidget(self.btn_cancel)
        btns.addWidget(self.progress)
        btns.addWidget(self.lbl_status, 1)
        root.addLayout(btns)

        # --- GCP table + output
        bottom = QHBoxLayout()
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(["#", "mapX", "mapY", "sourceX", "sourceY", "region",
                                              "res px", "res map"])
        self.table.setToolTip("Click a row to zoom both canvases to that GCP.\n"
                              "res px = residual in pixels of the raster being registered,\n"
                              "res map = residual in map units; both for the selected fitting method.")
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setMaximumHeight(170)
        bottom.addWidget(self.table, 3)

        out = QVBoxLayout()
        h = QHBoxLayout()
        self.ed_points = QLineEdit()
        self.ed_points.setPlaceholderText("GCP .points file (auto-saved on every accept)")
        b = QPushButton("…")
        b.clicked.connect(self._browse_points)
        h.addWidget(self.ed_points)
        h.addWidget(b)
        out.addLayout(h)
        h2 = QHBoxLayout()
        self.btn_del = QPushButton("Delete selected")
        self.btn_clear = QPushButton("Clear all")
        self.btn_save = QPushButton("Save .points")
        h2.addWidget(self.btn_del)
        h2.addWidget(self.btn_clear)
        h2.addWidget(self.btn_save)
        out.addLayout(h2)
        h3 = QHBoxLayout()
        self.btn_geo = QPushButton("Open QGIS Georeferencer")
        self.cb_order = QComboBox()
        self.cb_order.addItems(["Polynomial 1", "Polynomial 2", "Polynomial 3", "Thin plate spline"])
        self.btn_warp = QPushButton("Warp with GDAL…")
        h3.addWidget(self.btn_geo)
        h3.addWidget(self.cb_order)
        h3.addWidget(self.btn_warp)
        out.addLayout(h3)
        h4 = QHBoxLayout()
        h4.addWidget(QLabel("Source coords in .points:"))
        self.cb_src = QComboBox()
        self.cb_src.addItems(["Auto", "Pixel (col, -row)", "Layer map coords"])
        self.cb_src.setToolTip(
            "What the Georeferencer expects as source X/Y.\n"
            "Rasters that already carry a georeference (even an approximate one) are\n"
            "handled by the Georeferencer in their own map coordinates, un-referenced\n"
            "rasters in pixel units. 'Auto' detects this; override if points look shifted.")
        h4.addWidget(self.cb_src, 1)
        out.addLayout(h4)
        h5 = QHBoxLayout()
        h5.addWidget(QLabel("Residual line ×"))
        self.sp_resscale = QSpinBox()
        self.sp_resscale.setRange(1, 10000)
        self.sp_resscale.setValue(1)
        self.sp_resscale.setToolTip("1 = true scale. Residual lines on the canvases are drawn this many times\n"
                                    "longer than the real residual (raise it if they are too small to see).")
        h5.addWidget(self.sp_resscale)
        h5.addWidget(QLabel("Zoom to GCP (px across)"))
        self.sp_zoompx = QSpinBox()
        self.sp_zoompx.setRange(20, 5000)
        self.sp_zoompx.setValue(200)
        self.sp_zoompx.setToolTip("Clicking a GCP in the list shows about this many pixels of each\n"
                                  "raster across the width of its canvas.")
        h5.addWidget(self.sp_zoompx)
        self.lbl_rms = QLabel("RMSE: –")
        h5.addWidget(self.lbl_rms, 1)
        out.addLayout(h5)
        self.cb_order.currentIndexChanged.connect(lambda *_: self._refresh_gcps())
        self.sp_resscale.valueChanged.connect(lambda *_: self._refresh_gcps())
        self.table.itemSelectionChanged.connect(self._on_row_selected)
        self._sort = (0, True)
        hdr = self.table.horizontalHeader()
        hdr.setSectionsClickable(True)
        hdr.setSortIndicatorShown(True)
        hdr.setSortIndicator(0, Qt.SortOrder.AscendingOrder)
        hdr.sectionClicked.connect(self._on_sort)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(70)
        out.addWidget(self.log)
        bottom.addLayout(out, 2)
        root.addLayout(bottom)

        self.cb_base.layerChanged.connect(self._on_layers_changed)
        self.cb_move.layerChanged.connect(self._on_layers_changed)
        self.sp_rot.valueChanged.connect(lambda v: (self.canvas_move.setRotation(v), self.canvas_move.refresh()))
        self.btn_match.clicked.connect(self._run_match)
        self.btn_accept.clicked.connect(self._accept)
        self.btn_discard.clicked.connect(self._discard)
        self.btn_undo.clicked.connect(self._delete_last_region)
        self.btn_cancel.clicked.connect(self._cancel_match)
        self.btn_del.clicked.connect(lambda *_: self._delete_selected())
        self.btn_clear.clicked.connect(lambda *_: self._clear_all(True))
        self.btn_save.clicked.connect(lambda: self._save_points(True))
        self.btn_geo.clicked.connect(self._open_georeferencer)
        self.btn_warp.clicked.connect(self._warp)

    def _make_canvas(self):
        c = QgsMapCanvas()
        c.setCanvasColor(QColor(255, 255, 255))
        c.enableAntiAliasing(True)
        c.setMinimumSize(300, 250)
        c._pan = ClickPanTool(c, self._on_canvas_click, self._is_hit)
        c._manual_tool = ManualPointTool(c, self._on_canvas_click)
        c.setMapTool(c._pan)
        c.xyCoordinates.connect(lambda pt, cv=c: self._on_cursor(cv, pt))
        c.extentsChanged.connect(lambda cv=c: self._on_extent(cv))
        c.installEventFilter(self)
        c.viewport().installEventFilter(self)
        return c

    def _wrap(self, title, canvas):
        w = QWidget()
        L = QVBoxLayout(w)
        L.setContentsMargins(0, 0, 0, 0)
        L.addWidget(QLabel("<b>%s</b>  (wheel = zoom, drag = pan)" % title))
        L.addWidget(canvas, 1)
        return w

    def _log(self, msg):
        self.log.appendPlainText(msg)

    # ------------------------------------------------------------- layers
    def _setup_canvas(self, canvas, layer):
        if layer is None:
            canvas.setLayers([])
            return
        crs = layer.crs() if layer.crs().isValid() else QgsCoordinateReferenceSystem("EPSG:3857")
        canvas.setDestinationCrs(crs)
        canvas.setLayers([layer])
        canvas.setExtent(layer.extent())
        canvas.refresh()

    def _on_layers_changed(self, *_):
        self._clear_all(confirm=False)
        self._setup_canvas(self.canvas_base, self.cb_base.currentLayer())
        self._setup_canvas(self.canvas_move, self.cb_move.currentLayer())

    # ------------------------------------------------------------ matching
    def _run_match(self):
        base, mov = self.cb_base.currentLayer(), self.cb_move.currentLayer()
        if base is None or mov is None or base == mov:
            QMessageBox.warning(self, "VisMatch", "Choose two different raster layers.")
            return
        model = self.ed_model.text().strip()
        if not model:
            QMessageBox.warning(self, "VisMatch", "Type a vismatch model name.")
            return
        try:
            import vismatch  # noqa: F401
        except ImportError:
            QMessageBox.critical(self, "VisMatch",
                                 "The 'vismatch' package is not installed in QGIS' Python.\n"
                                 "Install it with: python -m pip install vismatch (see README).")
            return
        side = self.sp_chunk.value()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.img0, arr0, self.win0 = render_canvas_window(self.canvas_base, base, side)
            self.img1, arr1, self.win1 = render_canvas_window(self.canvas_move, mov, side)
        finally:
            QApplication.restoreOverrideCursor()
        self.btn_match.setEnabled(False)
        self.btn_accept.setEnabled(False)
        self._stage = "Starting"
        self._dirs = weight_dirs()
        self._t0 = time.time()
        self._base = self._last = sum(dir_size(p) for p in self._dirs)
        self._last_t = self._t0
        kwargs = {"ransac_reproj_thresh": self.sp_ransac.value()}
        self.worker = MatchWorker(self.matcher_cache, model, self.cb_dev.currentText(), kwargs, arr0, arr1, self)
        self.worker.finished_ok.connect(self._match_done)
        self.worker.failed.connect(self._match_failed)
        self.worker.status.connect(self._on_stage)
        self._busy(True)
        self.worker.start()

    def _busy(self, on):
        self.progress.setVisible(on)
        self.btn_cancel.setEnabled(on)
        self.btn_match.setEnabled(not on)
        if on:
            self.timer.start()
            self._tick()
        else:
            self.timer.stop()

    def _on_stage(self, s):
        self._stage = s
        self._tick()

    def _tick(self):
        now_t = time.time()
        size = sum(dir_size(p) for p in self._dirs)
        grown = (size - self._base) / 1e6
        rate = (size - self._last) / 1e6 / max(now_t - self._last_t, 1e-3)
        self._last, self._last_t = size, now_t
        txt = "%s — %ds elapsed" % (self._stage, now_t - self._t0)
        if grown > 0.5:
            txt += " — weights cache +%.1f MB (%.1f MB/s)" % (grown, max(rate, 0))
        elif "Loading" in self._stage:
            txt += " — no download detected (cached or loading from disk)"
        self.lbl_status.setText(txt)

    def _cancel_match(self):
        if self._warping:
            self._warp_cancel = True
            self._log("Cancelling warp...")
            self.lbl_status.setText("Cancelling warp...")
            return
        w = self.worker
        if w is None:
            return
        w.cancel()
        self._orphans.append(w)
        w.finished.connect(lambda w=w: self._orphans.remove(w) if w in self._orphans else None)
        self.worker = None
        self._busy(False)
        self.lbl_status.setText("Cancelled. (A download stuck in a blocking read stops at its next chunk; "
                                "partial files are discarded by the downloader.)")
        self._log("Matching cancelled by user")

    def _match_failed(self, msg):
        self._busy(False)
        self.lbl_status.setText("Matching failed – see log.")
        self._log(msg)

    def _match_done(self, res):
        self._busy(False)
        self.pending = res
        n = len(res["k0"])
        self.view.set_data(self.img0, self.img1, res["k0"], res["k1"])
        ok = n >= self.sp_mininl.value()
        self.btn_accept.setEnabled(ok)
        self.btn_discard.setEnabled(True)
        self.lbl_status.setText("%d inliers (%s)." % (n, "ready to accept" if ok else "too few – move/zoom and retry"))
        self._log("Match: %d inliers" % n)

    def _discard(self):
        self.pending = None
        self.view.set_data(None, None, np.zeros((0, 2)), np.zeros((0, 2)))
        self.btn_accept.setEnabled(False)
        self.btn_discard.setEnabled(False)

    # ----------------------------------------------------------------- GCPs
    def _accept(self, *_):
        try:
            self._accept_impl()
        except Exception:
            import traceback
            tb = traceback.format_exc()
            self._log(tb)
            QMessageBox.critical(self, "VisMatch – Accept failed", tb)

    def _accept_impl(self):
        if not self.pending:
            self._log("Accept pressed but there is no pending match.")
            return
        mov = self.cb_move.currentLayer()
        k0, k1 = self.pending["k0"], self.pending["k1"]
        kmax = self.sp_maxg.value()
        conf = self.pending.get("conf")
        if len(k0) <= kmax:
            sel = np.arange(len(k0))
        elif self.cb_sel.currentIndex() == 0 and conf is not None and len(conf) == len(k0):
            score = np.where(np.isfinite(conf), conf, -np.inf)
            sel = np.sort(np.argsort(-score, kind="stable")[:kmax])
            self._log("Kept top %d of %d inliers by vismatch score (kept %.3f - %.3f)"
                      % (len(sel), len(k0), float(conf[sel].min()), float(conf[sel].max())))
        else:
            if self.cb_sel.currentIndex() == 0:
                self._log("This model gives no match scores - using spatial spread instead.")
            sel = spread_select(k0, kmax)
        self.region_counter += 1
        for i in sel:
            p0 = px_to_map(self.win0, k0[i][0], k0[i][1])
            p1 = px_to_map(self.win1, k1[i][0], k1[i][1])
            col, row = self._px_from_mov(mov, p1)
            self.gcps.append({"mx": p0.x(), "my": p0.y(), "col": col, "row": row, "region": self.region_counter})
        self._log("Region %d: added %d GCPs (total %d)" % (self.region_counter, len(sel), len(self.gcps)))
        self._refresh_gcps()
        self._discard()
        if self.ed_points.text().strip():
            self._save_points(False)
        self.lbl_status.setText("Accepted %d GCPs (total %d). Pan both views to a new region and match again."
                                % (len(sel), len(self.gcps)))

    def _refresh_gcps(self):
        method = self.cb_order.currentIndex() if hasattr(self, "cb_order") else 0
        self._fits = make_fits(self.gcps, method) if hasattr(self, "cb_order") else None
        res = residuals(self.gcps, method, self._fits) if self._fits else None
        self._res = res
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.gcps))
        n = len(self.gcps)

        def key(i, col):  # numeric sort key of GCP i for table column col
            g = self.gcps[i]
            if col == 0:
                return i
            if col == 1:
                return g["mx"]
            if col == 2:
                return g["my"]
            if col == 3:
                return g["col"]
            if col == 4:
                return -g["row"]
            if col == 5:
                return g["region"]
            if res is None:
                return -1.0
            return float(np.hypot(*res[1][i])) if col == 6 else float(np.hypot(*res[0][i]))

        scol, sasc = getattr(self, "_sort", (0, True))
        self._order = sorted(range(n), key=lambda i: key(i, scol), reverse=not sasc)
        for r, i in enumerate(self._order):
            g = self.gcps[i]
            if res is not None:
                rpx = "%.2f" % float(np.hypot(*res[1][i]))
                rmap = "%.4g" % float(np.hypot(*res[0][i]))
            else:
                rpx = rmap = "–"
            for c, v in enumerate([i + 1, "%.6f" % g["mx"], "%.6f" % g["my"], "%.2f" % g["col"],
                                   "%.2f" % -g["row"], g["region"], rpx, rmap]):
                self.table.setItem(r, c, QTableWidgetItem(str(v)))
        self.table.blockSignals(False)
        self._sel = None
        if res is not None and len(self.gcps):
            rms = float(np.sqrt((res[1] ** 2).sum(1).mean()))
            self.lbl_rms.setText("RMSE: %.3f px  (%s, lines x%d)" % (
                rms, METHODS[self.cb_order.currentIndex()][0], self.sp_resscale.value()))
        else:
            need = METHODS[self.cb_order.currentIndex()][1] if hasattr(self, "cb_order") else 3
            self.lbl_rms.setText("RMSE: – (need %d GCPs)" % need)
        self._safe_markers()

    def _on_sort(self, col):
        """Click a column header to sort (click again to reverse)."""
        scol, sasc = self._sort
        sasc = (not sasc) if col == scol else True
        self._sort = (col, sasc)
        self.table.horizontalHeader().setSortIndicator(
            col, Qt.SortOrder.AscendingOrder if sasc else Qt.SortOrder.DescendingOrder)
        self._refresh_gcps()

    def _safe_markers(self):
        try:
            self._refresh_markers()
        except Exception as e:  # markers are cosmetic – never block GCP handling
            self._log("Could not draw GCP markers: %s" % e)

    def _mov_xy(self, g, mov, dcol=0.0, drow=0.0):
        """GCP (plus optional pixel offset) -> map coordinates on the raster-to-register canvas."""
        return self._mov_from_px(mov, g["col"] + dcol, g["row"] + drow)

    def _line(self, canvas, p0, p1):
        try:
            from qgis.core import Qgis
            gt = Qgis.GeometryType.Line
        except (ImportError, AttributeError):
            from qgis.core import QgsWkbTypes
            gt = QgsWkbTypes.LineGeometry
        rb = QgsRubberBand(canvas, gt)
        rb.setColor(QColor(255, 200, 0))
        rb.setWidth(2)
        rb.addPoint(p0, False)
        rb.addPoint(p1, True)
        return rb

    def _refresh_markers(self):
        for m in self.markers:
            with contextlib.suppress(Exception):
                m.hide()
                m.scene().removeItem(m)
        self.markers = []
        mov = self.cb_move.currentLayer()
        res = getattr(self, "_res", None)
        k = float(self.sp_resscale.value())
        sel = getattr(self, "_sel", None)
        for i, g in enumerate(self.gcps):
            pb = QgsPointXY(g["mx"], g["my"])
            self.markers.append(self._marker(self.canvas_base, pb))
            if res is not None:  # line from the GCP towards where the fit puts it (x scale)
                q = QgsPointXY(pb.x() + res[0][i][0] * k, pb.y() + res[0][i][1] * k)
                self.markers.append(self._line(self.canvas_base, pb, q))
            if mov is not None:
                pm = self._mov_xy(g, mov)
                self.markers.append(self._marker(self.canvas_move, pm))
                if res is not None:
                    q = self._mov_xy(g, mov, res[1][i][0] * k, res[1][i][1] * k)
                    self.markers.append(self._line(self.canvas_move, pm, q))
            if i == sel:  # highlight the selected GCP
                self.markers.append(self._marker(self.canvas_base, pb, QColor(0, 160, 255), 18, True))
                if mov is not None:
                    self.markers.append(self._marker(self.canvas_move, self._mov_xy(g, mov),
                                                     QColor(0, 160, 255), 18, True))

    # -------------------------------------------------------- zoom to a GCP
    # ------------------------------------------------- fitting-based sync
    def _px_from_mov(self, mov, pt):
        e = mov.extent()
        return ((pt.x() - e.xMinimum()) / e.width() * mov.width(),
                (e.yMaximum() - pt.y()) / e.height() * mov.height())

    def _mov_from_px(self, mov, col, row):
        e = mov.extent()
        return QgsPointXY(e.xMinimum() + col / mov.width() * e.width(),
                          e.yMaximum() - row / mov.height() * e.height())

    def _convert(self, pt, from_base):
        """Corresponding location on the other canvas through the current fit, or None."""
        mov = self.cb_move.currentLayer()
        if self._fits is None or mov is None:
            return None
        try:
            if from_base:
                col, row = self._fits[1](np.array([[pt.x(), pt.y()]]))[0]
                out = self._mov_from_px(mov, col, row)
            else:
                col, row = self._px_from_mov(mov, pt)
                mx, my = self._fits[0](np.array([[col, row]]))[0]
                out = QgsPointXY(mx, my)
        except Exception:
            return None
        return out if np.isfinite(out.x()) and np.isfinite(out.y()) else None

    def _toggle_sync(self, on):
        if on and self._fits is None:
            need = METHODS[self.cb_order.currentIndex()][1]
            self._log("Sync needs at least %d GCPs for %s." % (need, METHODS[self.cb_order.currentIndex()][0]))
            self.lbl_status.setText("Sync needs at least %d GCPs for the selected fitting method." % need)
            self.btn_sync.blockSignals(True)
            self.btn_sync.setChecked(False)
            self.btn_sync.blockSignals(False)
            return
        self._sync_on = bool(on)
        for c in (self.canvas_base, self.canvas_move):
            self._apply_cursor(c, False)
        if on:
            self._on_extent(self.canvas_base)  # initial alignment: raster follows the basemap
        else:
            self._hide_sync_cursors()

    def _hide_sync_cursors(self):
        for m in self._sync_mk.values():
            m.hide()

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.Type.Leave and self._sync_on:
            self._hide_sync_cursors()
        return False

    def _on_cursor(self, canvas, pt):
        # clickable-point feedback (not while dragging)
        if not self._manual and QApplication.mouseButtons() == Qt.MouseButton.NoButton:
            sp = canvas.getCoordinateTransform().transform(pt)
            self._apply_cursor(canvas, self._nearest_gcp(canvas, sp.x(), sp.y()) is not None)
        if not self._sync_on or self._fits is None:
            return
        from_base = canvas is self.canvas_base
        other = self.canvas_move if from_base else self.canvas_base
        q = self._convert(pt, from_base)
        mk = self._sync_mk.get(id(other))
        if mk is None:
            mk = QgsVertexMarker(other)
            mk.setColor(QColor(0, 120, 255))
            mk.setIconSize(22)
            mk.setPenWidth(2)
            try:
                mk.setIconType(QgsVertexMarker.IconType.ICON_X)
            except AttributeError:
                mk.setIconType(QgsVertexMarker.ICON_X)
            self._sync_mk[id(other)] = mk
        if q is None:
            mk.hide()
            return
        mk.setCenter(q)
        mk.show()

    def _on_extent(self, canvas):
        if not self._sync_on or self._syncing or self._fits is None:
            return
        from_base = canvas is self.canvas_base
        other = self.canvas_move if from_base else self.canvas_base
        e = canvas.extent()
        pts = [self._convert(QgsPointXY(x, y), from_base)
               for x, y in ((e.xMinimum(), e.yMinimum()), (e.xMaximum(), e.yMinimum()),
                            (e.xMaximum(), e.yMaximum()), (e.xMinimum(), e.yMaximum()))]
        if any(p is None for p in pts):
            return
        xs, ys = [p.x() for p in pts], [p.y() for p in pts]
        rect = QgsRectangle(min(xs), min(ys), max(xs), max(ys))
        if rect.width() <= 0 or rect.height() <= 0:
            return
        self._syncing = True
        try:
            other.setExtent(rect)
            other.refresh()
        finally:
            self._syncing = False

    # ------------------------------------------------------- manual GCPs
    def _toggle_manual(self, on):
        self._manual = bool(on)
        self._pending = {}
        self._draw_pending()
        for c in (self.canvas_base, self.canvas_move):  # crosshair tool without click-to-pan
            c.setMapTool(c._manual_tool if on else c._pan)
        if not on:
            for c in (self.canvas_base, self.canvas_move):
                self._apply_cursor(c, False)
        if on:
            self.lbl_status.setText("Manual GCP: click the point on the basemap, then the same point on the other "
                                    "image. Pan = middle/right-drag, zoom = wheel.")
        else:
            self.lbl_status.setText("Manual mode off.")

    def _draw_pending(self):
        for m in self._pend_markers:
            with contextlib.suppress(Exception):
                m.hide()
                m.scene().removeItem(m)
        self._pend_markers = []
        green = QColor(0, 170, 0)
        if "base" in self._pending:
            self._pend_markers.append(self._marker(self.canvas_base, self._pending["base"], green))
        if "move" in self._pending:
            self._pend_markers.append(self._marker(self.canvas_move, self._pending["move"], green))

    def _manual_click(self, canvas, pos):
        mov = self.cb_move.currentLayer()
        if self.cb_base.currentLayer() is None or mov is None:
            self._log("Select both layers before adding manual GCPs.")
            return
        pt = QgsPointXY(canvas.getCoordinateTransform().toMapCoordinates(pos.x(), pos.y()))
        self._pending["base" if canvas is self.canvas_base else "move"] = pt
        self._draw_pending()
        if "base" not in self._pending or "move" not in self._pending:
            self.lbl_status.setText("Manual GCP: now click the %s." % (
                "same point on the other image" if canvas is self.canvas_base else "point on the basemap"))
            return
        pb, pm = self._pending["base"], self._pending["move"]
        col, row = self._px_from_mov(mov, pm)
        self._pending = {}
        self._draw_pending()
        if not (0 <= col <= mov.width() and 0 <= row <= mov.height()):
            self._log("Manual GCP ignored: the click on the second image is outside the raster.")
            return
        self.gcps.append({"mx": pb.x(), "my": pb.y(), "col": col, "row": row, "region": 0})
        self._log("Manual GCP added (region 0, total %d)" % len(self.gcps))
        self.lbl_status.setText("Manual GCP added. Click the next point on the basemap (toggle + Manual GCP to stop).")
        self._refresh_gcps()
        if self.ed_points.text().strip():
            self._save_points(False)

    def _nearest_gcp(self, canvas, x, y, tol=14.0):
        """Index of the GCP whose marker on `canvas` is within tol screen px of (x, y), or None."""
        mov = self.cb_move.currentLayer()
        tr = canvas.getCoordinateTransform()
        best, best_d = None, tol
        for i, g in enumerate(self.gcps):
            if canvas is self.canvas_base:
                pt = QgsPointXY(g["mx"], g["my"])
            elif mov is not None:
                pt = self._mov_xy(g, mov)
            else:
                continue
            sp = tr.transform(pt)
            dist = ((sp.x() - x) ** 2 + (sp.y() - y) ** 2) ** 0.5
            if dist < best_d:
                best, best_d = i, dist
        return best

    def _is_hit(self, canvas, pos):
        return (not self._manual) and bool(self.gcps) and \
            self._nearest_gcp(canvas, pos.x(), pos.y()) is not None

    def _apply_cursor(self, canvas, hover):
        """Pointing hand over a clickable GCP, crosshair while Sync is on, else the open hand."""
        if self._manual:
            return  # manual tool has its own crosshair
        shape = (Qt.CursorShape.PointingHandCursor if hover else
                 Qt.CursorShape.CrossCursor if self._sync_on else Qt.CursorShape.OpenHandCursor)
        if canvas.cursor().shape() != shape:
            canvas.setCursor(QCursor(shape))

    def _on_canvas_click(self, canvas, pos):
        """Click on an accepted GCP marker -> select its row (without moving the canvases).
        In manual mode a click instead sets the base / moving point of a new manual GCP."""
        if self._manual:
            self._manual_click(canvas, pos)
            return
        if not self.gcps:
            return
        best = self._nearest_gcp(canvas, pos.x(), pos.y())
        if best is None or best not in getattr(self, "_order", []):
            return
        row = self._order.index(best)
        self._no_zoom = True
        try:
            self.table.selectRow(row)
            self.table.scrollToItem(self.table.item(row, 0))
        finally:
            self._no_zoom = False

    def _on_row_selected(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        order = getattr(self, "_order", [])
        if not rows or rows[0] >= len(order):
            return
        idx = order[rows[0]]
        self._sel = idx
        if getattr(self, "_no_zoom", False):  # selected by clicking a canvas: just highlight
            self._safe_markers()
            return
        g = self.gcps[idx]
        n = self.sp_zoompx.value()
        base, mov = self.cb_base.currentLayer(), self.cb_move.currentLayer()
        if base is not None:
            self._focus(self.canvas_base, QgsPointXY(g["mx"], g["my"]),
                        n * abs(base.rasterUnitsPerPixelX()))
        if mov is not None:
            self._focus(self.canvas_move, self._mov_xy(g, mov), n * abs(mov.rasterUnitsPerPixelX()))
        self._safe_markers()

    @staticmethod
    def _focus(canvas, c, width):
        if width <= 0:
            return
        w, h = max(canvas.width(), 1), max(canvas.height(), 1)
        hw = width / 2.0
        hh = hw * h / float(w)
        canvas.setExtent(QgsRectangle(c.x() - hw, c.y() - hh, c.x() + hw, c.y() + hh))
        canvas.refresh()

    def _marker(self, canvas, pt, color=None, size=9, circle=False):
        m = QgsVertexMarker(canvas)
        m.setCenter(pt)
        m.setColor(color or QColor(255, 0, 0))
        m.setIconSize(size)
        m.setPenWidth(2)
        try:
            icon = QgsVertexMarker.IconType.ICON_CIRCLE if circle else QgsVertexMarker.IconType.ICON_CROSS
        except AttributeError:
            icon = QgsVertexMarker.ICON_CIRCLE if circle else QgsVertexMarker.ICON_CROSS
        m.setIconType(icon)
        return m

    def _delete_selected(self):
        order = getattr(self, "_order", list(range(len(self.gcps))))
        rows = {i.row() for i in self.table.selectedIndexes()}
        for gi in sorted({order[r] for r in rows if r < len(order)}, reverse=True):
            del self.gcps[gi]
        self._refresh_gcps()
        if self.ed_points.text().strip():
            self._save_points(False)

    def _delete_last_region(self):
        """Remove all GCPs of the most recently accepted (detected) region, from table and canvases."""
        regions = [g["region"] for g in self.gcps if g["region"] > 0]  # region 0 = manual GCPs
        if not regions:
            self.lbl_status.setText("No accepted region to delete.")
            return
        last = max(regions)
        n = len(self.gcps)
        self.gcps = [g for g in self.gcps if g["region"] != last]
        # keep numbering continuous: the next accept reuses the freed region number
        self.region_counter = max([g["region"] for g in self.gcps] + [0])
        self._log("Deleted region %d: removed %d GCPs (total %d)" % (last, n - len(self.gcps), len(self.gcps)))
        self.lbl_status.setText("Deleted region %d (%d GCPs)." % (last, n - len(self.gcps)))
        self._refresh_gcps()
        if self.ed_points.text().strip():
            self._save_points(False)

    def _clear_all(self, confirm=True):
        if confirm and self.gcps and QMessageBox.question(self, "VisMatch", "Remove all GCPs?") != QMessageBox.StandardButton.Yes:
            return
        self.gcps = []
        self.region_counter = 0
        self._pending = {}
        if hasattr(self, "table"):
            self._draw_pending()
            self._refresh_gcps()

    # --------------------------------------------------------------- output
    def _browse_points(self):
        # Suggest "<raster to register>.points" (what the Georeferencer looks for) but let
        # the user confirm or change it in the dialog; nothing is written before that.
        cur = self.ed_points.text().strip()
        if cur:
            suggestion = cur
        else:
            mov = self.cb_move.currentLayer()
            suggestion = (mov.source().split("|")[0] + ".points") if mov is not None else ""
        p, _ = QFileDialog.getSaveFileName(self, "GCP file", suggestion, "GCP points (*.points)")
        if p:
            if not p.endswith(".points"):
                p += ".points"
                if os.path.exists(p) and QMessageBox.question(
                        self, "VisMatch", "%s already exists. Overwrite it?" % p) != QMessageBox.StandardButton.Yes:
                    return
            self.ed_points.setText(p)
            self._save_points(False)

    def _save_points(self, ask):
        path = self.ed_points.text().strip()
        if not path:
            if not ask:
                return
            self._browse_points()
            path = self.ed_points.text().strip()
            if not path:
                return
        base = self.cb_base.currentLayer()
        lines = ["#CRS: " + base.crs().toWkt(), POINTS_HEADER]
        mov = self.cb_move.currentLayer()
        use_layer = self._source_is_layer_coords(mov)
        for g in self.gcps:
            if use_layer:
                p = self._mov_from_px(mov, g["col"], g["row"])
                sx, sy = p.x(), p.y()
            else:
                sx, sy = g["col"], -g["row"]
            lines.append("%.10f,%.10f,%.10f,%.10f,1,0,0,0" % (g["mx"], g["my"], sx, sy))
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        self._log("Saved %d GCPs → %s" % (len(self.gcps), path))

    def _source_is_layer_coords(self, mov):
        """True if the Georeferencer works in the raster's own map coordinates
        (raster already has a geotransform/GCPs), False for plain pixel units."""
        mode = self.cb_src.currentIndex()
        if mode == 1:
            return False
        if mode == 2:
            return True
        if mov is None:
            return False
        try:
            from osgeo import gdal
            ds = gdal.Open(mov.source().split("|")[0])
            gt = ds.GetGeoTransform()
            has_gcps = ds.GetGCPCount() > 0
            ds = None
            return has_gcps or tuple(gt) not in ((0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
                                                 (0.0, 1.0, 0.0, 0.0, 0.0, -1.0))
        except Exception:
            return mov.crs().isValid()

    def _open_georeferencer(self):
        """The Georeferencer has no public API; save the points and trigger its menu action."""
        self._save_points(True)
        for a in self.iface.mainWindow().findChildren(QAction):
            if "georeferencer" in a.text().lower().replace("&", ""):
                a.trigger()
                break
        QMessageBox.information(
            self, "Georeferencer",
            "In the Georeferencer: open the raster to register, then use\n"
            "'Load GCP Points' and choose:\n\n%s\n\n(Set the target CRS to the basemap CRS.)"
            % (self.ed_points.text() or "<save the .points file first>"))

    def _warp(self):
        mov = self.cb_move.currentLayer()
        order = self.cb_order.currentIndex()  # 0..2 polynomial, 3 TPS
        need = [3, 6, 10, 3][order]
        if len(self.gcps) < need:
            QMessageBox.warning(self, "VisMatch", "Need at least %d GCPs for this transform." % need)
            return
        out, _ = QFileDialog.getSaveFileName(self, "Output GeoTIFF", "", "GeoTIFF (*.tif)")
        if not out:
            return
        from osgeo import gdal
        wkt = self.cb_base.currentLayer().crs().toWkt()
        src = mov.source().split("|")[0]
        gcps = [gdal.GCP(g["mx"], g["my"], 0, g["col"], g["row"]) for g in self.gcps]
        method = ("thin plate spline" if order == 3 else "polynomial order %d" % (order + 1))
        self._log("--- GDAL warp ---")
        self._log("Source : %s" % src)
        self._log("Output : %s" % out)
        self._log("GCPs   : %d   transform: %s   resampling: bilinear" % (len(gcps), method))
        self._log("Target CRS: %s" % self.cb_base.currentLayer().crs().authid())
        t0 = time.time()
        state = {"last": -1}

        def cb(frac, msg, data):
            pct = int(frac * 100)
            self.progress.setValue(pct)
            if pct // 10 > state["last"] // 10 or pct >= 100:
                self._log("  warping... %3d%%  (%.0f s)" % (pct, time.time() - t0))
            state["last"] = pct
            QApplication.processEvents()
            return 0 if self._warp_cancel else 1  # 0 aborts the GDAL operation

        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.lbl_status.setText("Warping with GDAL...")
        # Disable everything except Cancel so nothing re-enters while the event loop is pumped.
        self._warp_cancel = False
        self._warping = True
        locked = [w for cls in (QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, QTableWidget)
                  for w in self.findChildren(cls) if w is not self.btn_cancel and w.isEnabled()]
        for w in locked:
            w.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self._log("Attaching GCPs (VRT)...")
            vrt = gdal.Translate("/vsimem/vismatch_gcp.vrt", src, format="VRT", GCPs=gcps, outputSRS=wkt)
            self._log("Source size: %d x %d px, %d band(s)" % (vrt.RasterXSize, vrt.RasterYSize, vrt.RasterCount))
            kw = dict(format="GTiff", dstSRS=wkt, resampleAlg="bilinear",
                      creationOptions=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=IF_SAFER"],
                      callback=cb)
            if order == 3:
                kw["tps"] = True
            else:
                kw["polynomialOrder"] = order + 1
            self._log("Warping...")
            ds = gdal.Warp(out, vrt, **kw)
            if ds is None or self._warp_cancel:
                raise RuntimeError("cancelled" if self._warp_cancel else
                                   "gdal.Warp returned no dataset (see QGIS log / GDAL error)")
            self._log("Output size: %d x %d px" % (ds.RasterXSize, ds.RasterYSize))
            ds = None  # flush to disk
            vrt = None
        except Exception as e:  # noqa
            ds = vrt = None
            if self._warp_cancel:
                try:
                    if os.path.exists(out):
                        os.remove(out)  # discard the partial output
                except OSError:
                    pass
                self._log("Warp cancelled after %.1f s; partial output removed." % (time.time() - t0))
                self.lbl_status.setText("Warp cancelled.")
            else:
                self._log("Warp failed: %s" % e)
                self.lbl_status.setText("Warp failed - see log.")
            return
        finally:
            QApplication.restoreOverrideCursor()
            self._warping = False
            for w in locked:
                w.setEnabled(True)
            self.btn_cancel.setEnabled(False)
            self.progress.setRange(0, 0)
            self.progress.setVisible(False)
        self._log("Warp finished in %.1f s" % (time.time() - t0))
        self.lbl_status.setText("Warp finished.")
        self.iface.addRasterLayer(out, os.path.basename(out))
        self._log("Warped → %s" % out)
