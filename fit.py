"""Small numpy implementations of the Georeferencer transformation types, used to
compute per-GCP residuals for the currently selected fitting method."""
import numpy as np

# index of the combo box in the dialog -> (name, minimum number of GCPs)
METHODS = [("Polynomial 1", 3), ("Polynomial 2", 6), ("Polynomial 3", 10), ("Thin plate spline", 3)]


def min_points(method):
    return METHODS[method][1]


def _poly_terms(u, v, order):
    cols = []
    for i in range(order + 1):
        for j in range(order + 1 - i):
            cols.append((u ** i) * (v ** j))
    return np.stack(cols, axis=1)


def _tps_kernel(r2):
    # U(r) = r^2 log(r) = 0.5 * r2 * log(r2), U(0) = 0
    with np.errstate(divide="ignore", invalid="ignore"):
        k = 0.5 * r2 * np.log(r2)
    k[~np.isfinite(k)] = 0.0
    return k


class Fit:
    """Least-squares polynomial (order 1-3) or exact thin plate spline src(N,2) -> dst(N,2)."""

    def __init__(self, src, dst, method):
        src = np.asarray(src, float)
        dst = np.asarray(dst, float)
        self.method = method
        # normalise for numerical conditioning (map coordinates can be ~1e6 or tiny degrees)
        self.ms, self.md = src.mean(0), dst.mean(0)
        self.ss = max(float(np.ptp(src, axis=0).max()), 1e-12)
        self.sd = max(float(np.ptp(dst, axis=0).max()), 1e-12)
        s = (src - self.ms) / self.ss
        d = (dst - self.md) / self.sd
        if method <= 2:
            A = _poly_terms(s[:, 0], s[:, 1], method + 1)
            self.coef, *_ = np.linalg.lstsq(A, d, rcond=None)
        else:
            n = len(s)
            r2 = ((s[:, None, :] - s[None, :, :]) ** 2).sum(-1)
            K = _tps_kernel(r2)
            P = np.hstack([np.ones((n, 1)), s])
            L = np.zeros((n + 3, n + 3))
            L[:n, :n] = K
            L[:n, n:] = P
            L[n:, :n] = P.T
            Y = np.vstack([d, np.zeros((3, 2))])
            self.coef = np.linalg.pinv(L) @ Y
            self.ctrl = s

    def __call__(self, pts):
        pts = np.asarray(pts, float).reshape(-1, 2)
        s = (pts - self.ms) / self.ss
        if self.method <= 2:
            out = _poly_terms(s[:, 0], s[:, 1], self.method + 1) @ self.coef
        else:
            r2 = ((s[:, None, :] - self.ctrl[None, :, :]) ** 2).sum(-1)
            K = _tps_kernel(r2)
            P = np.hstack([np.ones((len(s), 1)), s])
            out = K @ self.coef[:len(self.ctrl)] + P @ self.coef[len(self.ctrl):]
        return out * self.sd + self.md


def make_fits(gcps, method):
    """Return (forward, reverse) Fit objects ((col,row)->map and map->(col,row)) or None."""
    if len(gcps) < min_points(method):
        return None
    src = np.array([[g["col"], g["row"]] for g in gcps], float)
    dst = np.array([[g["mx"], g["my"]] for g in gcps], float)
    try:
        return Fit(src, dst, method), Fit(dst, src, method)
    except Exception:
        return None


def residuals(gcps, method, fits=None):
    """Return (res_map (N,2), res_px (N,2)) or None if there are too few GCPs.

    res_map: fitted map position minus actual map position (map units), from the
             forward fit (col,row) -> (mapX,mapY).
    res_px:  fitted (col,row) minus actual (col,row), from the reverse fit
             (mapX,mapY) -> (col,row); in pixels of the raster being registered.
    A thin plate spline interpolates the GCPs, so its residuals are (numerically) zero.
    """
    fits = fits or make_fits(gcps, method)
    if fits is None:
        return None
    src = np.array([[g["col"], g["row"]] for g in gcps], float)
    dst = np.array([[g["mx"], g["my"]] for g in gcps], float)
    try:
        return fits[0](src) - dst, fits[1](dst) - src
    except Exception:
        return None
