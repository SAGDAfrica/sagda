"""
Data validation utilities to compare synthetic vs. real datasets after
generation or augmentation.

This module provides a complementary suite of metrics covering:
- Marginal & joint fidelity (KS, Wasserstein-1, EMD-approx, correlation/covariance,
  pairwise mutual information drift, MMD-RBF, PCA/UMAP overlap).
- Discriminability (Classifier Two-Sample Test / C2ST AUC).
- Temporal structure (ACF similarity, spectral distance; optional seasonal amplitude/phase).
- Spatial structure (Moran's I, variogram shape) if `lat`/`lon` are present.
- Downstream utility (Train-on-Synthetic-Test-on-Real and vice-versa).
- Robustness & diversity (nearest-neighbor coverage, support overlap).
- Optional privacy checks (nearest-neighbor reidentification risk).

Notes
-----
- All metrics are computed on numeric columns only unless specified.
- Heavy computations accept `sample_size` to downsample inputs deterministically.
- UMAP overlap is optional and requires `umap-learn`; if unavailable, it is skipped.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Sequence, Tuple, List
import math

import numpy as np
import pandas as pd

from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# -------------------------
# Internal constants
# -------------------------
_DEFAULT_BINS = 30
_MIN_SAMPLES = 5
_TEMPORAL_LAGS = (1, 2, 3, 4, 7, 14, 28)  # indices; assume roughly uniform sampling


# -------------------------
# Helper metrics
# -------------------------
def _hist_emd(x: np.ndarray, y: np.ndarray, bins: int = _DEFAULT_BINS) -> float:
    """Approximate 1D Earth Mover's Distance via cumulative histogram differences."""
    if x.size < _MIN_SAMPLES or y.size < _MIN_SAMPLES:
        return float("nan")
    lo = np.nanmin([np.min(x), np.min(y)])
    hi = np.nanmax([np.max(x), np.max(y)])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return 0.0
    hx, _ = np.histogram(x, bins=bins, range=(lo, hi), density=True)
    hy, _ = np.histogram(y, bins=bins, range=(lo, hi), density=True)
    sx, sy = np.sum(hx), np.sum(hy)
    if sx <= 0 or sy <= 0:
        return 0.0
    cx = np.cumsum(hx) / sx
    cy = np.cumsum(hy) / sy
    return float(np.mean(np.abs(cx - cy)))


def _mmd_rbf(X: np.ndarray, Y: np.ndarray, gamma: float | None = None) -> float:
    """Maximum Mean Discrepancy with RBF kernel (median heuristic if gamma=None)."""
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X and Y must be 2D arrays.")
    if gamma is None:
        Z = np.vstack([X, Y])
        d2 = np.sum((Z[:, None, :] - Z[None, :, :]) ** 2, axis=2)
        med = np.median(d2[d2 > 0])
        gamma = 1.0 / (med + 1e-9)

    def k(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.exp(-gamma * np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2))

    Kxx = k(X, X).mean()
    Kyy = k(Y, Y).mean()
    Kxy = k(X, Y).mean()
    return float(Kxx + Kyy - 2.0 * Kxy)


def _safe_numeric_intersection(real_df: pd.DataFrame, synth_df: pd.DataFrame, cols: Optional[Sequence[str]]) -> List[str]:
    """Return a list of overlapping numeric columns, or the provided subset filtered to numeric/overlap."""
    if cols is None:
        numeric_real = [c for c in real_df.columns if pd.api.types.is_numeric_dtype(real_df[c])]
        numeric_synth = [c for c in synth_df.columns if pd.api.types.is_numeric_dtype(synth_df[c])]
        cols = [c for c in numeric_synth if c in numeric_real]
    return [c for c in cols if c in real_df.columns and c in synth_df.columns and pd.api.types.is_numeric_dtype(real_df[c]) and pd.api.types.is_numeric_dtype(synth_df[c])]


def _downsample(df: pd.DataFrame, n: Optional[int], seed: int = 0) -> pd.DataFrame:
    """Deterministically downsample to at most n rows."""
    if n is None or len(df) <= n:
        return df
    return df.sample(n=n, random_state=seed)


def _corr_diff(real: pd.DataFrame, synth: pd.DataFrame) -> Dict[str, float]:
    """Return Frobenius norm differences between correlation and covariance matrices (normalized)."""
    # Align columns
    cols = [c for c in real.columns if c in synth.columns]
    if len(cols) < 2:
        return {"corr_fro": float("nan"), "cov_fro": float("nan")}
    Rr = real[cols].corr().to_numpy(dtype=float)
    Rs = synth[cols].corr().to_numpy(dtype=float)
    Cr = real[cols].cov().to_numpy(dtype=float)
    Cs = synth[cols].cov().to_numpy(dtype=float)

    # Normalize by matrix size to keep magnitudes comparable across dimensionalities
    d = Rr.shape[0]
    corr_fro = float(np.linalg.norm(Rr - Rs, ord="fro") / d)
    cov_fro = float(np.linalg.norm(Cr - Cs, ord="fro") / d)
    return {"corr_fro": corr_fro, "cov_fro": cov_fro}


def _mi_matrix(df: pd.DataFrame, nbins: int = 10) -> np.ndarray:
    """Approximate pairwise mutual information (symmetric) via equal-frequency binning."""
    cols = df.columns
    # Discretize to ranks/quantile bins to stabilize MI for continuous features
    disc = pd.DataFrame(index=df.index)
    for c in cols:
        s = df[c].copy()
        try:
            disc[c] = pd.qcut(s.rank(method="average", pct=True), q=nbins, duplicates="drop")
        except Exception:
            disc[c] = pd.qcut(s.rank(method="average", pct=True), q=max(2, nbins//2), duplicates="drop")
    # Compute MI via contingency tables
    n = len(cols)
    MI = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            ct = pd.crosstab(disc[cols[i]], disc[cols[j]]).to_numpy(dtype=float)
            pxy = ct / np.maximum(ct.sum(), 1.0)
            px = pxy.sum(axis=1, keepdims=True)
            py = pxy.sum(axis=0, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = pxy / (px @ py)
                log_ratio = np.where(ratio > 0, np.log(ratio), 0.0)
            mi = float(np.nansum(pxy * log_ratio))
            MI[i, j] = MI[j, i] = mi
    return MI


def _mi_drift(real: pd.DataFrame, synth: pd.DataFrame) -> float:
    """Mean absolute difference between pairwise MI matrices (same columns, discretized)."""
    cols = [c for c in real.columns if c in synth.columns]
    if len(cols) < 2:
        return float("nan")
    R = _mi_matrix(real[cols])
    S = _mi_matrix(synth[cols])
    d = R.shape[0]
    return float(np.mean(np.abs(R - S)) / (np.log(d) + 1e-9))  # normalized by a loose log scale


def _pca_overlap(real: pd.DataFrame, synth: pd.DataFrame, n_components: int = 2) -> float:
    """Fraction of synth inside real's axis-aligned bounding box in PCA space."""
    if len(real) < _MIN_SAMPLES or len(synth) < _MIN_SAMPLES:
        return 0.0
    X = pd.concat([real.assign(_src=0), synth.assign(_src=1)], axis=0, ignore_index=True)
    Xn = X.drop(columns=["_src"]).to_numpy(dtype=float)
    src = X["_src"].to_numpy()
    n_comp = int(min(n_components, Xn.shape[1]))
    pca = PCA(n_components=n_comp, svd_solver="full")
    Z = pca.fit_transform(Xn)
    Zr = Z[src == 0]; Zs = Z[src == 1]
    lo = Zr.min(axis=0); hi = Zr.max(axis=0)
    if Zs.size == 0:
        return 0.0
    inside = np.all((Zs >= lo) & (Zs <= hi), axis=1).mean()
    return float(inside)


def _umap_overlap(real: pd.DataFrame, synth: pd.DataFrame, n_components: int = 2, n_neighbors: int = 15, min_dist: float = 0.1) -> Optional[float]:
    """Optional UMAP overlap (requires umap-learn). Returns None if unavailable."""
    try:
        import umap  # type: ignore
    except Exception:
        return None
    if len(real) < _MIN_SAMPLES or len(synth) < _MIN_SAMPLES:
        return None
    X = pd.concat([real.assign(_src=0), synth.assign(_src=1)], axis=0, ignore_index=True)
    Xn = X.drop(columns=["_src"]).to_numpy(dtype=float)
    src = X["_src"].to_numpy()
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, random_state=0)
    Z = reducer.fit_transform(Xn)
    Zr = Z[src == 0]; Zs = Z[src == 1]
    lo = Zr.min(axis=0); hi = Zr.max(axis=0)
    if Zs.size == 0:
        return None
    inside = np.all((Zs >= lo) & (Zs <= hi), axis=1).mean()
    return float(inside)


def _c2st_auc(real: pd.DataFrame, synth: pd.DataFrame, test_size: float = 0.3, seed: int = 0) -> float:
    """Classifier two-sample test (C2ST) AUC using a simple RandomForestClassifier."""
    X = pd.concat([real.assign(_y=0), synth.assign(_y=1)], axis=0, ignore_index=True)
    y = X.pop("_y").to_numpy()
    Xn = X.to_numpy(dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(Xn, y, test_size=test_size, random_state=seed, stratify=y)
    clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    clf.fit(X_train, y_train)
    p = clf.predict_proba(X_test)[:, 1]
    return float(roc_auc_score(y_test, p))  # 0.5 is indistinguishable; 1.0 means trivially separable


def _acf(arr: np.ndarray, lags: Sequence[int]) -> np.ndarray:
    """Compute autocorrelation at given integer lags for a 1D series (NaNs ignored)."""
    arr = arr.astype(float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.full(len(lags), np.nan)
    arr = arr - np.mean(arr)
    var = np.var(arr) + 1e-12
    out = []
    for L in lags:
        if L >= len(arr):
            out.append(np.nan)
        else:
            out.append(np.dot(arr[:-L], arr[L:]) / ((len(arr) - L) * var))
    return np.array(out)


def _acf_diff(real: pd.DataFrame, synth: pd.DataFrame, time_col: str, cols: Sequence[str], lags: Sequence[int]) -> float:
    """Mean absolute difference of ACF across columns and lags (lower is better)."""
    # Sort by time to maintain temporal order
    R = real.sort_values(time_col)
    S = synth.sort_values(time_col)
    diffs = []
    for c in cols:
        r = _acf(R[c].to_numpy(dtype=float), lags)
        s = _acf(S[c].to_numpy(dtype=float), lags)
        m = np.nanmean(np.abs(r - s))
        if np.isfinite(m):
            diffs.append(m)
    return float(np.mean(diffs)) if diffs else float("nan")


def _spectral_distance(real: pd.DataFrame, synth: pd.DataFrame, cols: Sequence[str]) -> float:
    """Average L1 distance between normalized periodograms (lower is better)."""
    def periodogram(x: np.ndarray) -> np.ndarray:
        x = x[np.isfinite(x)].astype(float)
        if x.size == 0:
            return np.array([np.nan])
        F = np.fft.rfft(x - x.mean())
        P = np.abs(F) ** 2
        if P.sum() == 0:
            return np.array([0.0])
        return P / P.sum()

    dists = []
    for c in cols:
        pr = periodogram(real[c].to_numpy(dtype=float))
        ps = periodogram(synth[c].to_numpy(dtype=float))
        # Align to shortest length
        m = min(len(pr), len(ps))
        if m == 0:
            continue
        d = np.nanmean(np.abs(pr[:m] - ps[:m]))
        if np.isfinite(d):
            dists.append(d)
    return float(np.mean(dists)) if dists else float("nan")


def _morans_I(lat: np.ndarray, lon: np.ndarray, v: np.ndarray, k: int = 8) -> float:
    """Compute Moran's I using k-NN weights (row-standardized)."""
    mask = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(v)
    lat, lon, v = lat[mask], lon[mask], v[mask]
    n = len(v)
    if n < max(_MIN_SAMPLES, k + 1):
        return float("nan")
    coords = np.c_[lat, lon]
    nbrs = NearestNeighbors(n_neighbors=min(k+1, n), algorithm="auto").fit(coords)
    dists, inds = nbrs.kneighbors(coords)  # includes self at index 0
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        neigh = [j for j in inds[i] if j != i]
        if not neigh:
            continue
        w = 1.0 / np.maximum(dists[i][1:], 1e-9)  # inverse distance
        w = w / (w.sum() + 1e-12)
        W[i, neigh] = w
    x = v - v.mean()
    num = 0.0
    for i in range(n):
        num += np.sum(W[i] * x[i] * x)
    denom = np.sum(x**2) + 1e-12
    I = (n / (W.sum() + 1e-12)) * (num / denom)
    return float(I)


def _variogram_curve(lat: np.ndarray, lon: np.ndarray, v: np.ndarray, nbins: int = 12, max_pairs: int = 20000, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Empirical semivariogram: distances (bin centers) and semivariances (subsampled up to max_pairs)."""
    rng = np.random.default_rng(seed)
    mask = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(v)
    lat, lon, v = lat[mask], lon[mask], v[mask]
    n = len(v)
    if n < _MIN_SAMPLES:
        return np.array([]), np.array([])
    # Randomly subsample pairs for efficiency
    idx = rng.choice(n, size=min(n, 2000), replace=False)
    coords = np.c_[lat[idx], lon[idx]]
    vals = v[idx]

    # Pairwise distances (euclidean in lat-lon degrees; acceptable for small regions)
    dmat = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
    # Upper triangle
    iu = np.triu_indices_from(dmat, k=1)
    d = dmat[iu]
    g = 0.5 * (vals[:, None] - vals[None, :])[iu] ** 2

    if d.size > max_pairs:
        take = rng.choice(d.size, size=max_pairs, replace=False)
        d = d[take]
        g = g[take]

    # Bin distances
    if d.size == 0:
        return np.array([]), np.array([])
    bins = np.quantile(d, np.linspace(0, 1, nbins + 1))
    # Avoid duplicate edges
    bins = np.unique(bins)
    if len(bins) <= 2:
        return np.array([]), np.array([])
    inds = np.digitize(d, bins) - 1
    centers = 0.5 * (bins[:-1] + bins[1:])
    semiv = np.zeros(len(centers), dtype=float)
    for b in range(len(centers)):
        mask = inds == b
        if np.any(mask):
            semiv[b] = np.mean(g[mask])
        else:
            semiv[b] = np.nan
    return centers, semiv


def _coverage_metrics(real: pd.DataFrame, synth: pd.DataFrame) -> Dict[str, float]:
    """
    Nearest-neighbor coverage metrics:
    - nn_med_r2s / nn_med_s2r: median NN distance (real→synth, synth→real)
    - coverage_at_q50: fraction of real points whose NN distance to synth
      is <= q-quantile of the real→real NN distances (q=0.5 by default).
    """
    def pairwise_nn(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(b)
        dists, _ = nbrs.kneighbors(a)
        return dists[:, 0]

    # Use standardized space for distance stability
    cols = [c for c in real.columns if c in synth.columns]
    scaler = StandardScaler().fit(pd.concat([real[cols], synth[cols]], axis=0).to_numpy(dtype=float))
    R = scaler.transform(real[cols].to_numpy(dtype=float))
    S = scaler.transform(synth[cols].to_numpy(dtype=float))

    d_r2s = pairwise_nn(R, S)
    d_s2r = pairwise_nn(S, R)
    # Baseline intrinsic scale: NN distances within real
    d_r2r = pairwise_nn(R, R)

    cov50 = float((d_r2s <= np.quantile(d_r2r, 0.5)).mean())
    metrics = {
        "nn_med_r2s": float(np.median(d_r2s)),
        "nn_med_s2r": float(np.median(d_s2r)),
        "coverage_at_q50": cov50,
    }
    return metrics


# -------------------------
# Public API
# -------------------------
def validate(
    synth_df: pd.DataFrame,
    real_df: pd.DataFrame,
    cols: Sequence[str] | None = None,
    pca_components: int = 2,
    time_col: Optional[str] = None,
    lat_col: str = "lat",
    lon_col: str = "lon",
    target_col: Optional[str] = None,
    sample_size: Optional[int] = 5000,
    season_period: Optional[int] = None,
    compute_privacy: bool = False,
) -> Dict[str, Any]:
    """
    Validate synthetic data against a real reference using a broad suite of metrics.

    Parameters
    ----------
    synth_df, real_df : pandas.DataFrame
        Synthetic and real (reference) data frames.
    cols : sequence of str or None, default=None
        Numeric columns to compare. If None, uses the overlapping numeric columns.
    pca_components : int, default=2
        Number of components for PCA/UMAP projections.
    time_col : str or None, default=None
        If provided, temporal metrics (ACF, spectral) are computed on `cols`.
        The data is assumed approximately uniformly sampled after sorting on `time_col`.
    lat_col, lon_col : str, default=("lat","lon")
        Column names for latitude/longitude; if present, spatial metrics are computed.
    target_col : str or None, default=None
        If provided, downstream utility (TSTR/TRTS) is computed using a simple
        RandomForestRegressor baseline.
    sample_size : int or None, default=5000
        Deterministic downsampling cap for heavy computations (MMD, C2ST, NN, PCA/UMAP).
    season_period : int or None, default=None
        Optional seasonal period (in samples) to compute first-harmonic amplitude/phase
        deltas. If None, this step is skipped.
    compute_privacy : bool, default=False
        If True, run a simple nearest-neighbor reidentification risk proxy.

    Returns
    -------
    dict
        A dictionary containing keys:
        - "marginals": {"ks": {...}, "wass": {...}, "emd": {...}}
        - "joint": {"corr_fro": float, "cov_fro": float, "mi_drift": float, "mmd_rbf": float}
        - "overlap": {"pca_overlap": float, "umap_overlap": float|None}
        - "c2st_auc": float
        - "temporal": {"acf_diff": float, "spectral_L1": float, "season_amp_phase": dict|None}
        - "spatial": {"morans_I_diff": dict, "variogram_L2": dict} (if lat/lon present)
        - "utility": {"tstr": {...}, "trts": {...}} (if target_col provided)
        - "robustness": {"nn_med_r2s": float, "nn_med_s2r": float, "coverage_at_q50": float}
        - "privacy": {"min_nn_synth_to_real": float, "pct_below_eps": float} (if compute_privacy)

    Raises
    ------
    ValueError
        If there are no overlapping numeric columns or pca_components < 1.
    """
    if pca_components < 1:
        raise ValueError("pca_components must be >= 1.")

    # Columns
    cols = _safe_numeric_intersection(real_df, synth_df, cols)
    if len(cols) == 0:
        raise ValueError("No overlapping numeric columns to compare. Provide `cols` explicitly or align schemas.")

    # Downsample for heavy blocks
    real = _downsample(real_df[cols].dropna(), sample_size, seed=0)
    synth = _downsample(synth_df[cols].dropna(), sample_size, seed=0)

    result: Dict[str, Any] = {}

    # ---- Marginals
    ks: Dict[str, float] = {}
    wass: Dict[str, float] = {}
    emd: Dict[str, float] = {}
    for c in cols:
        a = synth[c].to_numpy(dtype=float); a = a[np.isfinite(a)]
        b = real[c].to_numpy(dtype=float);  b = b[np.isfinite(b)]
        if a.size >= _MIN_SAMPLES and b.size >= _MIN_SAMPLES:
            ks[c] = float(ks_2samp(a, b, alternative="two-sided", mode="auto").statistic)
            wass[c] = float(wasserstein_distance(a, b))
            emd[c] = _hist_emd(a, b, bins=_DEFAULT_BINS)
        else:
            ks[c] = float("nan"); wass[c] = float("nan"); emd[c] = float("nan")
    result["marginals"] = {"ks": ks, "wass": wass, "emd": emd}

    # ---- Joint structure
    result["joint"] = {}
    result["joint"].update(_corr_diff(real, synth))
    try:
        result["joint"]["mi_drift"] = _mi_drift(real, synth)
    except Exception:
        result["joint"]["mi_drift"] = float("nan")

    # MMD (use shared rows; already downsampled)
    try:
        X = synth.to_numpy(dtype=float)
        Y = real.to_numpy(dtype=float)
        result["joint"]["mmd_rbf"] = _mmd_rbf(X, Y, gamma=None)
    except Exception:
        result["joint"]["mmd_rbf"] = float("nan")

    # ---- Overlap (PCA/UMAP)
    result["overlap"] = {
        "pca_overlap": _pca_overlap(real, synth, n_components=pca_components),
        "umap_overlap": _umap_overlap(real, synth, n_components=pca_components),
    }

    # ---- Discriminability (C2ST)
    try:
        result["c2st_auc"] = _c2st_auc(real, synth, test_size=0.3, seed=0)
    except Exception:
        result["c2st_auc"] = float("nan")

    # ---- Temporal structure (optional)
    temporal = {"acf_diff": None, "spectral_L1": None, "season_amp_phase": None}
    if time_col is not None and time_col in real_df.columns and time_col in synth_df.columns:
        # Work on the provided `cols` only; maintain full rows that contain all columns
        real_t = _downsample(real_df[[time_col] + list(cols)].dropna().sort_values(time_col), sample_size, seed=0)
        synth_t = _downsample(synth_df[[time_col] + list(cols)].dropna().sort_values(time_col), sample_size, seed=0)

        temporal["acf_diff"] = _acf_diff(real_t, synth_t, time_col, cols, _TEMPORAL_LAGS)
        temporal["spectral_L1"] = _spectral_distance(real_t, synth_t, cols)

        if season_period is not None and season_period > 1:
            # First-harmonic amplitude/phase delta averaged across columns
            def first_harm(x: np.ndarray, period: int) -> Tuple[float, float]:
                x = x[np.isfinite(x)]
                if x.size < period:
                    return np.nan, np.nan
                # DFT coefficient at fundamental frequency k=1
                k = 1
                n = len(x)
                # Normalize time axis to period multiples
                t = np.arange(n)
                omega = 2 * np.pi * k / period
                cos = np.cos(omega * t); sin = np.sin(omega * t)
                a = 2.0 / n * np.dot(x, cos)
                b = 2.0 / n * np.dot(x, sin)
                amp = np.sqrt(a*a + b*b)
                phase = np.arctan2(b, a)
                return amp, phase

            amps_phases = []
            for c in cols:
                ar = real_t[c].to_numpy(dtype=float)
                as_ = synth_t[c].to_numpy(dtype=float)
                amp_r, ph_r = first_harm(ar, season_period)
                amp_s, ph_s = first_harm(as_, season_period)
                if np.isfinite(amp_r) and np.isfinite(amp_s) and np.isfinite(ph_r) and np.isfinite(ph_s):
                    amps_phases.append({"amp_diff": float(abs(amp_r - amp_s)),
                                        "phase_diff": float(abs((ph_r - ph_s + np.pi) % (2*np.pi) - np.pi))})
            temporal["season_amp_phase"] = {
                "mean_amp_diff": float(np.mean([ap["amp_diff"] for ap in amps_phases])) if amps_phases else float("nan"),
                "mean_phase_diff": float(np.mean([ap["phase_diff"] for ap in amps_phases])) if amps_phases else float("nan"),
            }

    result["temporal"] = temporal

    # ---- Spatial structure (optional if lat/lon present)
    spatial = {"morans_I_diff": {}, "variogram_L2": {}}
    if lat_col in real_df.columns and lon_col in real_df.columns and lat_col in synth_df.columns and lon_col in synth_df.columns:
        # We'll compute on a small set of representative columns (up to 5 numeric cols)
        rep_cols = cols[:5]
        for c in rep_cols:
            try:
                Ir = _morans_I(real_df[lat_col].to_numpy(dtype=float),
                               real_df[lon_col].to_numpy(dtype=float),
                               real_df[c].to_numpy(dtype=float))
                Is = _morans_I(synth_df[lat_col].to_numpy(dtype=float),
                               synth_df[lon_col].to_numpy(dtype=float),
                               synth_df[c].to_numpy(dtype=float))
                spatial["morans_I_diff"][c] = float(abs(Ir - Is)) if np.isfinite(Ir) and np.isfinite(Is) else float("nan")
            except Exception:
                spatial["morans_I_diff"][c] = float("nan")

            try:
                dr, gr = _variogram_curve(real_df[lat_col].to_numpy(dtype=float),
                                          real_df[lon_col].to_numpy(dtype=float),
                                          real_df[c].to_numpy(dtype=float))
                ds, gs = _variogram_curve(synth_df[lat_col].to_numpy(dtype=float),
                                          synth_df[lon_col].to_numpy(dtype=float),
                                          synth_df[c].to_numpy(dtype=float))
                m = min(len(gr), len(gs))
                if m > 0:
                    spatial["variogram_L2"][c] = float(np.nanmean((gr[:m] - gs[:m]) ** 2))
                else:
                    spatial["variogram_L2"][c] = float("nan")
            except Exception:
                spatial["variogram_L2"][c] = float("nan")

    result["spatial"] = spatial

    # ---- Downstream utility (optional, needs target_col)
    utility = None
    if target_col is not None and target_col in real_df.columns and target_col in synth_df.columns:
        try:
            # Align feature set: numeric cols excluding target
            feat_cols = [c for c in cols if c != target_col and pd.api.types.is_numeric_dtype(real_df[c])]
            # Split sets
            Xr = real_df[feat_cols].dropna().to_numpy(dtype=float)
            yr = real_df[target_col].reindex(real_df[feat_cols].dropna().index).to_numpy(dtype=float)
            Xs = synth_df[feat_cols].dropna().to_numpy(dtype=float)
            ys = synth_df[target_col].reindex(synth_df[feat_cols].dropna().index).to_numpy(dtype=float)

            # Train-on-Synth, Test-on-Real
            reg = RandomForestRegressor(n_estimators=400, random_state=0, n_jobs=-1)
            reg.fit(Xs, ys)
            pred_r = reg.predict(Xr)
            def _metrics(y, yhat):
                mae = float(np.mean(np.abs(y - yhat)))
                rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
                mape = float(np.mean(np.abs((y - yhat) / (np.maximum(1e-9, np.abs(y))))) * 100.0)
                return {"MAE": mae, "RMSE": rmse, "MAPE": mape}

            tstr = _metrics(yr, pred_r)

            # Train-on-Real, Test-on-Synth
            reg2 = RandomForestRegressor(n_estimators=400, random_state=0, n_jobs=-1)
            reg2.fit(Xr, yr)
            pred_s = reg2.predict(Xs)
            trts = _metrics(ys, pred_s)

            utility = {"tstr": tstr, "trts": trts}
        except Exception:
            utility = None

    result["utility"] = utility

    # ---- Robustness & diversity (coverage)
    try:
        result["robustness"] = _coverage_metrics(real, synth)
    except Exception:
        result["robustness"] = {"nn_med_r2s": float("nan"), "nn_med_s2r": float("nan"), "coverage_at_q50": float("nan")}

    # ---- Privacy (optional)
    if compute_privacy:
        try:
            scaler = StandardScaler().fit(pd.concat([real, synth], axis=0).to_numpy(dtype=float))
            R = scaler.transform(real.to_numpy(dtype=float))
            S = scaler.transform(synth.to_numpy(dtype=float))
            nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto").fit(R)
            dists, _ = nbrs.kneighbors(S)
            min_dist = float(np.min(dists))
            # Flag small distances below an epsilon as potential memorization
            eps = float(np.quantile(dists, 0.01))
            pct_below = float((dists <= eps).mean())
            result["privacy"] = {"min_nn_synth_to_real": min_dist, "pct_below_eps": pct_below}
        except Exception:
            result["privacy"] = {"min_nn_synth_to_real": float("nan"), "pct_below_eps": float("nan")}

    return result

def report(
    res: Dict[str, Any],
    *,
    as_markdown: bool = False,
    thresholds: Optional[Dict[str, float]] = None,
    title: str = "Synthetic Data Validation – Summary",
) -> str:
    """
    Format a full validation summary from `validate(...)` output.

    Parameters
    ----------
    res : dict
        Result returned by `validate(...)`.
    as_markdown : bool, default=True
        If True, returns Markdown; otherwise, returns monospaced plain text (aligned).
    thresholds : dict or None
        Optional gates. Defaults are sensible; override per domain if needed.
    title : str
        Title shown at the top of the report.

    Returns
    -------
    str
        A Markdown or plain-text report.
    """
    thr = {
        # Fidelity
        "ks_median_max": 0.15,
        "mmd_max": 0.01,
        "pca_min": 0.90,
        "c2st_sep_max": 0.20,
        # Utility
        "util_rmse_ratio_max": 1.20,
        # Robustness
        "cov_q50_min": 0.50,
        # Privacy (in standardized space ideally)
        "privacy_min_nn_min": 0.02,
        "privacy_pct_eps_max": 0.05,
        # Temporal
        "acf_diff_max": 0.20,
        "spectral_L1_max": 0.10,
    }
    if thresholds:
        thr.update(thresholds)

    def _fmt(v, nd=3):
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            return "n/a"
        return f"{v:.{nd}f}"

    def _status(v, *, mode: str, gate: float | None):
        """
        mode='min'  → lower is better (e.g., KS, MMD)
        mode='max'  → higher is better (e.g., overlap, coverage)
        gate is the threshold to compare against (None → info only)
        """
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            return "⚪"
        if gate is None:
            return "⚪"
        ok = (v <= gate) if mode == "min" else (v >= gate)
        return "✅" if ok else "❌"

    # ---------- aggregate headline metrics ----------
    ks_vals = list(res["marginals"]["ks"].values())
    clean = [x for x in ks_vals if isinstance(x, (int, float)) and not math.isnan(x)]
    ks_median = (sorted(clean)[len(clean)//2] if clean else float("nan"))
    mmd = float(res["joint"]["mmd_rbf"])
    pca = float(res["overlap"]["pca_overlap"])
    auc = float(res["c2st_auc"])
    separability = abs(auc - 0.5) * 2.0

    util = res.get("utility")
    util_ratio = None
    if util and util.get("tstr") and util.get("trts"):
        r1 = util["tstr"]["RMSE"]; r2 = util["trts"]["RMSE"]
        util_ratio = (r1 / r2) if (r2 and r2 != 0) else float("nan")

    cov_q50 = float(res["robustness"]["coverage_at_q50"])
    nn_r2s = float(res["robustness"]["nn_med_r2s"])
    nn_s2r = float(res["robustness"]["nn_med_s2r"])

    priv = res.get("privacy", {})
    priv_min = priv.get("min_nn_synth_to_real", float("nan"))
    priv_pct = priv.get("pct_below_eps", float("nan"))

    acf_diff = res.get("temporal", {}).get("acf_diff", float("nan"))
    spec_L1 = res.get("temporal", {}).get("spectral_L1", float("nan"))
    season = res.get("temporal", {}).get("season_amp_phase", None)

    # ---------- headline lines ----------
    head = []
    head.append(f"- Fidelity: KS(med)={_fmt(ks_median)} {_status(ks_median, mode='min', gate=thr['ks_median_max'])}, "
                f"MMD={_fmt(mmd)} {_status(mmd, mode='min', gate=thr['mmd_max'])}, "
                f"Overlap(PCA)={_fmt(pca)} {_status(pca, mode='max', gate=thr['pca_min'])}")
    head.append(f"- Separability (C2ST): AUC={_fmt(auc)} → sep={_fmt(separability)} "
                f"{_status(separability, mode='min', gate=thr['c2st_sep_max'])}")
    if util_ratio is not None:
        head.append(f"- Utility: TSTR/TRTS RMSE ratio={_fmt(util_ratio)} "
                    f"{_status(util_ratio, mode='min', gate=thr['util_rmse_ratio_max'])}")
    head.append(f"- Coverage: cov@q50={_fmt(cov_q50)} {_status(cov_q50, mode='max', gate=thr['cov_q50_min'])} "
                f"| NN r→s={_fmt(nn_r2s)}; s→r={_fmt(nn_s2r)}")
    if (not math.isnan(priv_min)) or (not math.isnan(priv_pct)):
        head.append(f"- Privacy proxy: min NN(s→r)={_fmt(priv_min)} {_status(priv_min, mode='max', gate=thr['privacy_min_nn_min'])} "
                    f"| pct≤ε={_fmt(priv_pct)} {_status(priv_pct, mode='min', gate=thr['privacy_pct_eps_max'])}")

    # ---------- build tables (Markdown + Plain text) ----------
    # Marginals: per-feature KS/Wasserstein/EMD
    features = sorted(set(res["marginals"]["ks"].keys())
                      | set(res["marginals"]["wass"].keys())
                      | set(res["marginals"]["emd"].keys()))
    marg_rows = []
    for f in features:
        ks = res["marginals"]["ks"].get(f, float("nan"))
        ws = res["marginals"]["wass"].get(f, float("nan"))
        em = res["marginals"]["emd"].get(f, float("nan"))
        marg_rows.append((f, _fmt(ks), _fmt(ws), _fmt(em)))

    # Joint metrics
    joint_items = [
        ("corr_fro", _fmt(res["joint"].get("corr_fro"))),
        ("cov_fro", _fmt(res["joint"].get("cov_fro"))),
        ("mi_drift", _fmt(res["joint"].get("mi_drift"))),
        ("mmd_rbf", _fmt(res["joint"].get("mmd_rbf"))),
    ]
    # Overlap metrics
    overlap_items = [
        ("pca_overlap", _fmt(res["overlap"].get("pca_overlap"))),
        ("umap_overlap", _fmt(res["overlap"].get("umap_overlap"))),
    ]
    # Temporal metrics
    temporal_items = [
        ("acf_diff", _fmt(acf_diff)),
        ("spectral_L1", _fmt(spec_L1)),
    ]
    if isinstance(season, dict):
        temporal_items.extend([
            ("season_amp_diff", _fmt(season.get("mean_amp_diff"))),
            ("season_phase_diff", _fmt(season.get("mean_phase_diff"))),
        ])

    # Spatial metrics (per-feature)
    morans = res.get("spatial", {}).get("morans_I_diff", {}) or {}
    variog = res.get("spatial", {}).get("variogram_L2", {}) or {}
    spatial_features = sorted(set(morans.keys()) | set(variog.keys()))
    spatial_rows = []
    for f in spatial_features:
        spatial_rows.append((f, _fmt(morans.get(f)), _fmt(variog.get(f))))

    # Utility metrics
    util_rows = []
    if util and util.get("tstr") and util.get("trts"):
        util_rows = [
            ("TSTR", _fmt(util["tstr"].get("MAE")), _fmt(util["tstr"].get("RMSE")), _fmt(util["tstr"].get("MAPE"))),
            ("TRTS", _fmt(util["trts"].get("MAE")), _fmt(util["trts"].get("RMSE")), _fmt(util["trts"].get("MAPE"))),
        ]

    # Robustness & Privacy
    robustness_items = [
        ("nn_med_r2s", _fmt(nn_r2s)),
        ("nn_med_s2r", _fmt(nn_s2r)),
        ("coverage_at_q50", _fmt(cov_q50)),
    ]
    privacy_items = []
    if (not math.isnan(priv_min)) or (not math.isnan(priv_pct)):
        privacy_items = [
            ("min_nn_synth_to_real", _fmt(priv_min)),
            ("pct_below_eps", _fmt(priv_pct)),
        ]

    # ---- renderers ----
    def md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
        align = ["---", ":---:", ":---:", ":---:", ":---:"][:len(headers)]
        out = ["| " + " | ".join(headers) + " |",
               "| " + " | ".join(align) + " |"]
        for r in rows:
            out.append("| " + " | ".join(r) + " |")
        return "\n".join(out)

    def txt_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
        cols = list(zip(*([headers] + list(rows)))) if rows else [headers]
        widths = [max(len(str(x)) for x in col) for col in cols]
        def fmt_row(r):
            return "  ".join(str(v).ljust(w) for v, w in zip(r, widths))
        line = "  ".join("-" * w for w in widths)
        out = [fmt_row(headers), line]
        out += [fmt_row(r) for r in rows]
        return "\n".join(out)

    # ---- assemble report ----
    if as_markdown:
        parts = [f"# {title}", "\n## Headline\n", *head, "\n## Details\n"]
        parts.append("### Marginals (per feature)")
        parts.append(md_table(["Feature", "KS", "Wasserstein", "EMD"], marg_rows))

        parts.append("\n### Joint structure")
        parts.append(md_table(["Metric", "Value"], [(k, v) for k, v in joint_items]))

        parts.append("\n### Overlap")
        parts.append(md_table(["Metric", "Value"], [(k, v) for k, v in overlap_items]))

        parts.append("\n### Temporal")
        parts.append(md_table(["Metric", "Value"], [(k, v) for k, v in temporal_items]))

        parts.append("\n### Spatial (per feature)")
        if spatial_rows:
            parts.append(md_table(["Feature", "Moran's I Δ", "Variogram L2"], spatial_rows))
        else:
            parts.append("_No spatial metrics available (missing/constant lat-lon or too few samples)._")

        parts.append("\n### Utility")
        if util_rows:
            parts.append(md_table(["Fold", "MAE", "RMSE", "MAPE (%)"], util_rows))
        else:
            parts.append("_Utility not computed (no target_col or insufficient data)._")

        parts.append("\n### Robustness")
        parts.append(md_table(["Metric", "Value"], [(k, v) for k, v in robustness_items]))

        if privacy_items:
            parts.append("\n### Privacy (proxy)")
            parts.append(md_table(["Metric", "Value"], [(k, v) for k, v in privacy_items]))
        return "\n".join(parts)

    # Plain text version
    parts = [f"{title}\n", "HEADLINE"]
    parts += ["  " + h for h in head]
    gated_rows = [
        ("KS (median)",         _fmt(ks_median),        f"≤ {thr['ks_median_max']}",    _status(ks_median,  mode='min', gate=thr['ks_median_max'])),
        ("MMD (RBF)",           _fmt(mmd),              f"≤ {thr['mmd_max']}",          _status(mmd,        mode='min', gate=thr['mmd_max'])),
        ("PCA overlap",         _fmt(pca),              f"≥ {thr['pca_min']}",          _status(pca,        mode='max', gate=thr['pca_min'])),
        ("C2ST separability",   _fmt(separability),     f"≤ {thr['c2st_sep_max']}",     _status(separability, mode='min', gate=thr['c2st_sep_max'])),
        ("TSTR/TRTS RMSE",      _fmt(util_ratio),       f"≤ {thr['util_rmse_ratio_max']}", _status(util_ratio, mode='min', gate=thr['util_rmse_ratio_max']) if util_ratio is not None else "⚪"),
        ("Coverage @q50",       _fmt(cov_q50),          f"≥ {thr['cov_q50_min']}",      _status(cov_q50,    mode='max', gate=thr['cov_q50_min'])),
        ("Privacy: min NN s→r", _fmt(priv_min),         f"≥ {thr['privacy_min_nn_min']}", _status(priv_min,  mode='max', gate=thr['privacy_min_nn_min'])),
        ("Privacy: pct≤ε",      _fmt(priv_pct),         f"≤ {thr['privacy_pct_eps_max']}", _status(priv_pct, mode='min', gate=thr['privacy_pct_eps_max'])),
        ("Temporal: ACF Δ",     _fmt(acf_diff),         f"≤ {thr['acf_diff_max']}",     _status(acf_diff,   mode='min', gate=thr['acf_diff_max'])),
        ("Temporal: spectral",  _fmt(spec_L1),          f"≤ {thr['spectral_L1_max']}",  _status(spec_L1,    mode='min', gate=thr['spectral_L1_max'])),
    ]
    parts += ["", "GATED OVERVIEW"]
    parts += [txt_table(["Metric", "Value", "Gate", "Status"], gated_rows)]
    parts += ["", "DETAILS", "", "Marginals (per feature)"]
    parts += [txt_table(["Feature", "KS", "Wasserstein", "EMD"], marg_rows)]
    parts += ["", "Joint structure"]
    parts += [txt_table(["Metric", "Value"], [(k, v) for k, v in joint_items])]
    parts += ["", "Overlap"]
    parts += [txt_table(["Metric", "Value"], [(k, v) for k, v in overlap_items])]
    parts += ["", "Temporal"]
    parts += [txt_table(["Metric", "Value"], [(k, v) for k, v in temporal_items])]
    parts += ["", "Spatial (per feature)"]
    parts += [txt_table(["Feature", "Moran's I Δ", "Variogram L2"], spatial_rows)] if spatial_rows else ["  (none)"]
    parts += ["", "Utility"]
    parts += [txt_table(["Fold", "MAE", "RMSE", "MAPE (%)"], util_rows)] if util_rows else ["  (not computed)"]
    parts += ["", "Robustness"]
    parts += [txt_table(["Metric", "Value"], [(k, v) for k, v in robustness_items])]
    if privacy_items:
        parts += ["", "Privacy (proxy)"]
        parts += [txt_table(["Metric", "Value"], [(k, v) for k, v in privacy_items])]
    return "\n".join(parts)
