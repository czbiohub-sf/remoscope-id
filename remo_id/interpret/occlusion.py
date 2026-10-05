"""Cell removal ("occlusion") fill model.

Replaces chosen cells of a brightfield frame with a plausible empty-slide background so
the change in a model's output can be attributed to those cells. The cell screen in
:mod:`remo_id.interpret.cell_occlusion` uses it for every removal.

Fill rule (identical for every experiment):
  1. empty-slide background per experiment = per-pixel median over a frame subset, using
     only frames where that pixel is outside every DILATE-px-dilated cell mask (pixels
     cell-free in < MIN_UNCOVERED frames: masked-Gaussian smoothing of the map)
     (:func:`build_bg_model`).
  2. per frame: robust gain/offset of that background to the frame's own cell-free
     pixels, plus a smooth local residual correction from nearby cell-free pixels.
  3. grain: unit-std correlated Gaussian noise with the lag-1 autocorrelation of the
     frame's high-pass background residual, scaled to NOISE_SCALE x its local std.
  4. removed cells (union closed with a CLOSE_RADIUS disk) dilated by DILATE px are
     replaced by the fill with a FEATHER-px linear alpha ramp; where kept cells touch
     the fill a Gaussian band (the frame's optical edge spread) blends the seam. Kept
     cells' pixels are otherwise never modified.

Inputs are plain arrays: a ``uint8 [H, W]`` frame and an integer label mask of the same
shape (0 = background, k > 0 = cell k).
"""
from __future__ import annotations

import cv2
import numpy as np

DILATE = 3
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * DILATE + 1, 2 * DILATE + 1))
FEATHER = 3          # px soft seam outside the filled region
MIN_UNCOVERED = 5    # frames a pixel must be cell-free in to trust its bg median
NOISE_SCALE = 0.5    # grain std = NOISE_SCALE x local HIGH-PASS background std
LOCAL_SIGMA = 10.0   # px, Gaussian sigma of the local residual correction
LOCAL_MIN_W = 0.25   # min Gaussian-weighted cell-free fraction before widening sigma
BG_DENOISE_SIGMA = 1.0   # px, light Gaussian on the bg model inside the fill
GRAIN_TILE = 64      # px tiles for the local high-pass grain amplitude
EDGE_SIGMA_DEFAULT = 1.5   # px optical edge spread if it cannot be estimated
CLOSE_RADIUS = 5     # px disk: morphological closing of the removed-cell union
CLOSE_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                         (2 * CLOSE_RADIUS + 1, 2 * CLOSE_RADIUS + 1))
_K3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def _dilate(m: np.ndarray) -> np.ndarray:
    return cv2.dilate(m.view(np.uint8), KERNEL).astype(bool)


def pick_area_matched(rng, labels, areas, target):
    """Random subset of ``labels`` whose cumulative area is closest to target (0 -> none)."""
    if target <= 0 or len(labels) == 0:
        return labels[:0]
    perm = rng.permutation(len(labels))
    cs = np.concatenate([[0.0], np.cumsum(areas[perm])])
    k = int(np.argmin(np.abs(cs - target)))
    return labels[perm[:k]]


# ── background model ────────────────────────────────────────────────────────

def experiment_background(imgs, uncov):
    """Per-pixel median over only the frames where the pixel is cell-free.

    imgs [n,H,W] uint8, uncov [n,H,W] bool (outside all DILATE-px-dilated cell masks).
    Pixels cell-free in < MIN_UNCOVERED frames are filled from a masked Gaussian
    (normalized-convolution) smoothing of the valid map.
    Returns (bg [H,W] float32, frac_filled_from_smooth).
    """
    n, H, W = imgs.shape
    bg = np.zeros((H, W), np.float32)
    cnt_all = np.zeros((H, W), np.int32)
    for r0 in range(0, H, 64):
        v = imgs[:, r0:r0 + 64].astype(np.uint16)
        u = uncov[:, r0:r0 + 64]
        v[~u] = 256                                   # sentinel sorts last
        v.sort(axis=0)
        cnt = u.sum(0)
        lo = np.take_along_axis(v, np.maximum(cnt - 1, 0)[None] // 2, 0)[0]
        hi = np.take_along_axis(v, (cnt // 2)[None], 0)[0].clip(max=255)
        bg[r0:r0 + 64] = (lo.astype(np.float32) + hi) / 2
        cnt_all[r0:r0 + 64] = cnt
    valid = cnt_all >= MIN_UNCOVERED
    frac = 1.0 - valid.mean()
    if frac > 0:
        sig = 15.0
        while True:
            num = cv2.GaussianBlur(np.where(valid, bg, 0).astype(np.float32), (0, 0), sig)
            den = cv2.GaussianBlur(valid.astype(np.float32), (0, 0), sig)
            ok = den > 1e-3
            if ok[~valid].all() or sig > 400:
                break
            sig *= 2
        sm = np.where(ok, num / np.maximum(den, 1e-6), np.median(bg[valid]))
        bg = np.where(valid, bg, sm).astype(np.float32)
    return bg, float(frac)


def build_bg_model(frames_u8: np.ndarray, masks: np.ndarray):
    """Empty-slide background for one experiment from its frames + label masks.

    frames_u8 uint8 [n,H,W], masks int [n,H,W] (the experiment's frame subset).
    Returns (bg float32 [H,W], frac_px_filled_from_smoothing)."""
    uncov = np.stack([cell_free(m) for m in masks])
    return experiment_background(frames_u8, uncov)


# ── per-frame fill model ────────────────────────────────────────────────────

def _kernel_rho_table():
    """lag-1 autocorrelation of white noise blurred by a discrete Gaussian of sigma s."""
    ss = np.linspace(0.3, 6.0, 300)
    rho = []
    for s in ss:
        k = cv2.getGaussianKernel(int(2 * np.ceil(4 * s) + 1), s).ravel()
        rho.append((k[:-1] * k[1:]).sum() / (k * k).sum())
    return ss, np.array(rho)


_SS, _RHO = _kernel_rho_table()


def trimmed_std(v):
    """Std within median +- 4 robust sd (IQR-based)."""
    v = np.asarray(v, np.float64)
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    w = 4 * max((q3 - q1) / 1.349, 0.5)
    return float(v[np.abs(v - med) <= w].std())


def _lag1_rho(r, m):
    """Mean horizontal/vertical lag-1 autocorrelation of residual r over mask m."""
    out = []
    for a, b, mm in ((r[:, :-1], r[:, 1:], m[:, :-1] & m[:, 1:]),
                     (r[:-1], r[1:], m[:-1] & m[1:])):
        x, y = a[mm], b[mm]
        if len(x) > 100:
            out.append(np.corrcoef(x, y)[0, 1])
    return float(np.mean(out)) if out else 0.0


def frame_noise_model(img, bg, uncov):
    """Robust per-frame fit of the experiment bg to this frame + residual texture.
    Returns (gain, offset, sigma, rho): fitted bg = gain*bg + offset; sigma =
    1.4826*MAD and rho = lag-1 autocorrelation of the cell-free residual."""
    f = img[uncov].astype(np.float32)
    b = bg[uncov]
    gain = float(np.median(f / np.maximum(b, 1.0)))
    offset = float(np.median(f - gain * b))
    res = img.astype(np.float32) - (gain * bg + offset)
    rv = res[uncov]
    sigma = float(1.4826 * np.median(np.abs(rv - np.median(rv))))
    clip = np.clip(res, -4 * sigma - 1e-3, 4 * sigma + 1e-3)   # robust to debris/halos
    rho = _lag1_rho(clip, uncov)
    return gain, offset, sigma, rho


def correlated_noise(rng, shape, sigma, rho):
    """Gaussian noise with std sigma and lag-1 autocorrelation ~rho."""
    z = rng.standard_normal(shape, dtype=np.float32)
    if rho > _RHO[0]:
        s = float(np.interp(min(rho, _RHO[-1]), _RHO, _SS))
        k = cv2.getGaussianKernel(int(2 * np.ceil(4 * s) + 1), s).astype(np.float32)
        z = cv2.sepFilter2D(z, -1, k, k, borderType=cv2.BORDER_REFLECT)
        z /= float((k.ravel() ** 2).sum())                # exact unit std of the blurred field
    return z * sigma


def local_residual(img, fit, uncov, sigma=LOCAL_SIGMA):
    """Smooth local correction of the frame-fitted bg from nearby cell-free pixels:
    multi-scale normalized convolution of the clipped cell-free residual, handing over
    softly to wider scales where the local cell-free weight is low."""
    r = img.astype(np.float32) - fit
    rv = r[uncov]
    if rv.size < 100:
        return np.zeros_like(r)
    med = float(np.median(rv))
    sd = max(trimmed_std(rv), 0.5)
    w = uncov.astype(np.float32)
    rw = np.where(uncov, np.clip(r, med - 3 * sd, med + 3 * sd), 0).astype(np.float32)
    ests, wts = [], []
    s = float(sigma)
    for _ in range(4):
        num = cv2.GaussianBlur(rw, (0, 0), s, borderType=cv2.BORDER_REFLECT)
        den = cv2.GaussianBlur(w, (0, 0), s, borderType=cv2.BORDER_REFLECT)
        ests.append(num / np.maximum(den, 1e-6))
        wts.append(np.clip(den / LOCAL_MIN_W, 0.0, 1.0))
        s *= 2
    out = ests[-1] * wts[-1]                 # widest scale, fades to 0 if still empty
    for e, a in zip(ests[-2::-1], wts[-2::-1]):
        out = a * e + (1.0 - a) * out
    return out


class FrameFill(tuple):
    """frame_fill_model() result: (gain, offset, sigma, rho) plus
      .base      float32 [H,W]  smooth fill = gain*bg + offset + local residual
      .grain_sd  float32 [H,W]  local std of the high-pass residual
      .grain_rho float          lag-1 autocorrelation of that high-pass residual
      .edge_sigma float         optical edge-spread sigma (px) of real cell edges"""
    def __new__(cls, gain, offset, sigma, rho, base, grain_sd, grain_rho, edge_sigma):
        obj = super().__new__(cls, (gain, offset, sigma, rho))
        obj.base = base
        obj.grain_sd = grain_sd
        obj.grain_rho = float(grain_rho)
        obj.edge_sigma = float(edge_sigma)
        return obj


def grain_model(img, base, uncov):
    """Local amplitude map + lag-1 autocorr of the high-pass residual img - base."""
    hp = img.astype(np.float32) - base
    H, W = hp.shape
    t = GRAIN_TILE
    ny, nx = int(np.ceil(H / t)), int(np.ceil(W / t))
    glob = trimmed_std(hp[uncov]) if uncov.sum() > 100 else 1.0
    grid = np.full((ny, nx), np.nan, np.float32)
    for a in range(ny):
        for b in range(nx):
            m = uncov[a * t:(a + 1) * t, b * t:(b + 1) * t]
            if m.sum() >= 200:
                grid[a, b] = trimmed_std(hp[a * t:(a + 1) * t, b * t:(b + 1) * t][m])
    grid = np.where(np.isnan(grid), glob, grid).astype(np.float32)
    grid = cv2.GaussianBlur(grid, (0, 0), 1.0, borderType=cv2.BORDER_REPLICATE)
    sd = cv2.resize(grid, (W, H), interpolation=cv2.INTER_LINEAR)
    sd = np.sqrt(np.maximum(sd ** 2 - 1.0 / 12.0, 0.0)).astype(np.float32)  # uint8 rounding re-adds it
    med = float(np.median(hp[uncov])) if uncov.any() else 0.0
    rho = _lag1_rho(np.clip(hp - med, -4 * glob, 4 * glob), uncov)
    return sd, rho


def edge_sigma_estimate(img, mask, grid=(0.6, 0.9, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0)):
    """Optical edge-spread sigma: the Gaussian blur of the binary cell mask that best
    (least squares, with free gain/offset) explains the image within 3 px of cell
    boundaries."""
    cells = (mask > 0).astype(np.float32)
    if cells.sum() < 200 or cells.mean() > 0.97:
        return EDGE_SIGMA_DEFAULT
    din = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 3)
    dout = cv2.distanceTransform((mask == 0).astype(np.uint8), cv2.DIST_L2, 3)
    band = (din <= 3) & (dout <= 3)
    y = img[band].astype(np.float64)
    best, bs = np.inf, EDGE_SIGMA_DEFAULT
    for sg in grid:
        x = cv2.GaussianBlur(cells, (0, 0), sg)[band].astype(np.float64)
        A = np.stack([x, np.ones_like(x)], 1)
        coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
        r = float(((A @ coef - y) ** 2).sum())
        if r < best:
            best, bs = r, sg
    return float(bs)


def frame_fill_model(img, bg, uncov, mask) -> FrameFill:
    """Everything occlude() needs for one frame; compute once, reuse per frame."""
    gain, offset, sigma, rho = frame_noise_model(img, bg, uncov)
    fit = gain * cv2.GaussianBlur(bg, (0, 0), BG_DENOISE_SIGMA) + offset   # drop bg-map pixel noise
    base = fit + local_residual(img, fit, uncov)
    gsd, grho = grain_model(img, base, uncov)
    return FrameFill(gain, offset, sigma, rho, base, gsd, grho, edge_sigma_estimate(img, mask))


def cell_free(mask: np.ndarray) -> np.ndarray:
    """Pixels outside every DILATE-px-dilated cell mask (bool [H,W])."""
    return ~_dilate(mask > 0)


# ── fill ────────────────────────────────────────────────────────────────────

def fill_core(mask, removed):
    """Pixels that get filled: removed cells (closed with a CLOSE_RADIUS disk so the
    thin gaps between neighbouring removed cells are covered), dilated DILATE px,
    minus kept cells."""
    keep = (mask > 0) & ~removed
    r = removed
    if r.any():
        r = cv2.morphologyEx(r.view(np.uint8), cv2.MORPH_CLOSE, CLOSE_KERNEL).astype(bool)
    return _dilate(r) & ~keep, keep


def composite(img, mask, removed, ff, unit_noise):
    """Fill `removed` (bool [H,W]) with the frame fill model `ff` (uint8 frame out):
    smooth ff.base with a FEATHER-px alpha outside the core, an optical-blur band at
    kept-cell seams, and grain x ff.grain_sd x NOISE_SCALE."""
    core, keep = fill_core(mask, removed)
    if not core.any():
        return img.copy()
    imgf = img.astype(np.float32)
    d_out = cv2.distanceTransform((~core).view(np.uint8), cv2.DIST_L2, 3)
    alpha = np.clip(1.0 - d_out / (FEATHER + 1), 0.0, 1.0)
    alpha[keep] = 0.0
    x = alpha * ff.base + (1.0 - alpha) * imgf
    w = 0.0
    seam = (cv2.dilate(core.view(np.uint8), _K3).astype(bool) & keep) | \
           (cv2.dilate(keep.view(np.uint8), _K3).astype(bool) & core)
    if seam.any():
        es = ff.edge_sigma
        d_s = cv2.distanceTransform((~seam).view(np.uint8), cv2.DIST_L2, 3)
        w = np.clip(1.0 - d_s / (2.0 * es), 0.0, 1.0).astype(np.float32)
        x = w * cv2.GaussianBlur(x, (0, 0), es) + (1.0 - w) * x
    x = x + unit_noise * ff.grain_sd * NOISE_SCALE * alpha * (1.0 - w)
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def occlude(frame_u8, mask_labels, remove_labels, bg_model, rng, noise_params=None):
    """Remove the cells `remove_labels` (mask label ids) from one frame.

    frame_u8 uint8 [H,W]; mask_labels int [H,W]; bg_model float32 [H,W]
    (:func:`build_bg_model`); rng np.random.Generator for the grain (seeded per call);
    noise_params: optional precomputed frame_fill_model(frame, bg, cell_free(mask), mask)
    for repeated calls on the same frame. Returns the occluded uint8 frame."""
    ff = noise_params if isinstance(noise_params, FrameFill) else \
        frame_fill_model(frame_u8, bg_model, cell_free(mask_labels), mask_labels)
    rlut = np.zeros(int(mask_labels.max()) + 1, bool)
    rl = np.asarray(remove_labels, dtype=np.int64)
    rlut[rl[(rl > 0) & (rl <= mask_labels.max())]] = True
    un = correlated_noise(rng, frame_u8.shape, 1.0, ff.grain_rho)
    return composite(frame_u8, mask_labels, rlut[mask_labels], ff, un)
