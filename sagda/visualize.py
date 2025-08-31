from __future__ import annotations

import math
from typing import Sequence, Tuple, Optional, Iterable, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


# -------------------------
# Utilities / validation
# -------------------------
def _assert_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

def _mask_finite(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones_like(arrays[0], dtype=bool)
    for a in arrays:
        mask &= np.isfinite(a)
    return mask

def _std_space(real: pd.DataFrame, synth: pd.DataFrame, cols: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler().fit(pd.concat([real[cols], synth[cols]], axis=0).to_numpy(dtype=float))
    return scaler.transform(real[cols].to_numpy(dtype=float)), scaler.transform(synth[cols].to_numpy(dtype=float))


# -------------------------
# 1) Marginals
# -------------------------
def plot_distributions(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    cols: Sequence[str],
    *,
    bins: int = 30,
    density: bool = True,
    alpha: float = 0.5,
    figsize: Tuple[int, int] | None = None,
) -> Tuple[Figure, list[Axes]]:
    """
    Overlayed histograms for selected features.

    Returns
    -------
    (fig, axes)
    """
    _assert_columns(real_df, cols)
    _assert_columns(synth_df, cols)

    n = len(cols)
    if figsize is None:
        figsize = (8, 3 * n)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=figsize)
    if n == 1:
        axes = [axes]  # type: ignore[assignment]
    for ax, c in zip(axes, cols):
        r = real_df[c].to_numpy(dtype=float)
        s = synth_df[c].to_numpy(dtype=float)
        r = r[np.isfinite(r)]; s = s[np.isfinite(s)]
        ax.hist(r, bins=bins, alpha=alpha, label="real", density=density)
        ax.hist(s, bins=bins, alpha=alpha, label="synthetic", density=density)
        ax.set_title(f"Distribution: {c}")
        ax.legend()
    fig.tight_layout()
    return fig, list(axes)


def plot_ecdf(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    cols: Sequence[str],
    *,
    figsize: Tuple[int, int] | None = None,
) -> Tuple[Figure, list[Axes]]:
    """
    Empirical CDF overlays for selected features (tail-sensitive).
    """
    _assert_columns(real_df, cols)
    _assert_columns(synth_df, cols)
    n = len(cols)
    if figsize is None:
        figsize = (8, 3 * n)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=figsize)
    if n == 1:
        axes = [axes]  # type: ignore[assignment]
    for ax, c in zip(axes, cols):
        for arr, lab in [(real_df[c].to_numpy(float), "real"),
                         (synth_df[c].to_numpy(float), "synthetic")]:
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue
            x = np.sort(arr)
            y = np.linspace(0, 1, len(x), endpoint=True)
            ax.step(x, y, where="post", label=lab)
        ax.set_title(f"ECDF: {c}")
        ax.set_ylabel("F(x)")
        ax.legend()
    fig.tight_layout()
    return fig, list(axes)


def plot_qq(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    cols: Sequence[str],
    *,
    q: int = 200,
    figsize: Tuple[int, int] | None = None,
) -> Tuple[Figure, list[Axes]]:
    """
    QQ plots: quantiles of synthetic vs real per feature.
    Ideal match lies on y=x.
    """
    _assert_columns(real_df, cols)
    _assert_columns(synth_df, cols)
    n = len(cols)
    if figsize is None:
        figsize = (7, 3 * n)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=figsize)
    if n == 1:
        axes = [axes]  # type: ignore[assignment]
    qs = np.linspace(0.01, 0.99, q)
    for ax, c in zip(axes, cols):
        r = real_df[c].to_numpy(float); r = r[np.isfinite(r)]
        s = synth_df[c].to_numpy(float); s = s[np.isfinite(s)]
        if r.size == 0 or s.size == 0:
            continue
        rq = np.quantile(r, qs)
        sq = np.quantile(s, qs)
        ax.scatter(rq, sq, s=10, alpha=0.7)
        lo = float(np.nanmin([rq.min(), sq.min()]))
        hi = float(np.nanmax([rq.max(), sq.max()]))
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
        ax.set_title(f"QQ: synth vs real – {c}")
        ax.set_xlabel("Real quantiles"); ax.set_ylabel("Synthetic quantiles")
    fig.tight_layout()
    return fig, list(axes)


# -------------------------
# 2) Joint structure
# -------------------------
def plot_corr_heatmaps(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    cols: Sequence[str],
    *,
    vmin: float = -1.0,
    vmax: float = 1.0,
    cmap: str = "coolwarm",
    annotate: bool = False,
    figsize: Tuple[int, int] = (15, 4.5),
) -> Tuple[Figure, list[Axes]]:
    """
    Correlation heatmaps: real, synthetic, and delta (synthetic - real).
    """
    _assert_columns(real_df, cols)
    _assert_columns(synth_df, cols)
    corr_r = real_df[cols].corr().to_numpy(dtype=float)
    corr_s = synth_df[cols].corr().to_numpy(dtype=float)
    delta = corr_s - corr_r

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    titles = ["Corr (real)", "Corr (synthetic)", "Δ Corr (synthetic - real)"]
    mats = [corr_r, corr_s, delta]
    vmins = [vmin, vmin, -1.0]
    vmaxs = [vmax, vmax, 1.0]

    for ax, M, t, lo, hi in zip(axes, mats, titles, vmins, vmaxs):
        im = ax.imshow(M, vmin=lo, vmax=hi, cmap=cmap)
        ax.set_title(t)
        ax.set_xticks(range(len(cols))); ax.set_yticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=90); ax.set_yticklabels(cols)
        if annotate:
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig, list(axes)


def plot_pairgrid(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    cols: Sequence[str],
    *,
    sample: Optional[int] = 3000,
    alpha_real: float = 0.4,
    alpha_synth: float = 0.4,
    s: float = 8.0,
    figsize: Optional[Tuple[int, int]] = None,
) -> Tuple[Figure, list[Axes]]:
    """
    Upper-triangle pairwise scatter overlay (real vs synthetic) for selected features.
    For large data, optionally subsample for speed.
    """
    _assert_columns(real_df, cols)
    _assert_columns(synth_df, cols)

    R = real_df[cols].dropna()
    S = synth_df[cols].dropna()
    if sample is not None:
        R = R.sample(n=min(sample, len(R)), random_state=0)
        S = S.sample(n=min(sample, len(S)), random_state=0)
    d = len(cols)
    if figsize is None:
        figsize = (3 * d, 3 * d)
    fig, axes = plt.subplots(d, d, figsize=figsize)
    for i in range(d):
        for j in range(d):
            ax = axes[i, j]
            if i == j:
                # Diagonal: hist overlay
                ax.hist(R.iloc[:, j], bins=30, alpha=0.5, density=True, label="real")
                ax.hist(S.iloc[:, j], bins=30, alpha=0.5, density=True, label="synthetic")
                if i == 0:
                    ax.legend(fontsize=7)
            elif i < j:
                # Upper triangle: overlay scatter
                ax.scatter(R.iloc[:, j], R.iloc[:, i], s=s, alpha=alpha_real, label="real")
                ax.scatter(S.iloc[:, j], S.iloc[:, i], s=s, alpha=alpha_synth, label="synthetic")
            else:
                ax.axis("off")
            if i == d - 1:
                ax.set_xlabel(cols[j], rotation=0)
            if j == 0 and i != 0:
                ax.set_ylabel(cols[i])
    fig.suptitle("Pairwise overlay (real vs synthetic)", y=0.995)
    fig.tight_layout()
    return fig, [ax for row in axes for ax in row]


# -------------------------
# 3) Overlap (PCA/UMAP)
# -------------------------
def plot_embedding_overlap(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    cols: Sequence[str],
    *,
    method: str = "pca",
    n_components: int = 2,
    show_box: bool = True,
    figsize: Tuple[int, int] = (7, 6),
) -> Tuple[Figure, Axes]:
    """
    2D embedding scatter (real vs synthetic) in PCA or UMAP space.

    Notes
    -----
    - UMAP requires `umap-learn`. Falls back to PCA if unavailable.
    """
    _assert_columns(real_df, cols)
    _assert_columns(synth_df, cols)

    X = pd.concat([real_df[cols].assign(_src=0), synth_df[cols].assign(_src=1)], ignore_index=True)
    src = X.pop("_src").to_numpy()
    Xn = X.to_numpy(dtype=float)
    Xn = Xn[np.all(np.isfinite(Xn), axis=1)]
    src = src[: len(Xn)]

    if method.lower() == "umap":
        try:
            import umap  # type: ignore
            reducer = umap.UMAP(n_components=n_components, n_neighbors=15, min_dist=0.1, random_state=0)
            Z = reducer.fit_transform(Xn)
        except Exception:
            method = "pca"  # fallback
    if method.lower() == "pca":
        n_comp = int(min(n_components, Xn.shape[1]))
        Z = PCA(n_components=n_comp, svd_solver="full").fit_transform(Xn)

    fig, ax = plt.subplots(figsize=figsize)
    Zr = Z[src == 0]; Zs = Z[src == 1]
    ax.scatter(Zr[:, 0], Zr[:, 1], s=12, alpha=0.6, label="real")
    ax.scatter(Zs[:, 0], Zs[:, 1], s=12, alpha=0.6, label="synthetic")
    ax.set_title(f"{method.upper()} overlap: real vs synthetic")
    ax.set_xlabel(f"{method.upper()}-1"); ax.set_ylabel(f"{method.upper()}-2")
    ax.legend()

    fig.tight_layout()
    return fig, ax


# -------------------------
# 4) Temporal
# -------------------------
def plot_timeseries(
    df: pd.DataFrame,
    cols: Sequence[str],
    time_col: str = "date",
    *,
    rolling: Optional[int] = None,
    figsize: Tuple[int, int] | None = None,
) -> Tuple[Figure, list[Axes]]:
    """
    Simple time-series lines; optional rolling window for smoothing (per column).
    """
    _assert_columns(df, [time_col] + list(cols))
    n = len(cols)
    if figsize is None:
        figsize = (10, 3 * n)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=figsize)
    if n == 1:
        axes = [axes]  # type: ignore[assignment]
    for ax, c in zip(axes, cols):
        ax.plot(df[time_col], df[c], label=c)
        if rolling and rolling > 1:
            ax.plot(df[time_col], df[c].rolling(rolling, min_periods=1).mean(), linewidth=2, alpha=0.7, label=f"{c} (roll{rolling})")
        ax.set_title(f"{c} over time")
        ax.set_xlabel(time_col)
        ax.set_ylabel(c)
        ax.legend()
    fig.tight_layout()
    return fig, list(axes)


def plot_mean_band_over_time(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    cols: Sequence[str],
    time_col: str = "date",
    *,
    window: int = 7,
    figsize: Tuple[int, int] | None = None,
) -> Tuple[Figure, list[Axes]]:
    """
    Rolling mean ± 1 std band for real vs synthetic (per column).
    """
    _assert_columns(real_df, [time_col] + list(cols))
    _assert_columns(synth_df, [time_col] + list(cols))
    n = len(cols)
    if figsize is None:
        figsize = (10, 3 * n)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=figsize)
    if n == 1:
        axes = [axes]  # type: ignore[assignment]
    R = real_df.sort_values(time_col)
    S = synth_df.sort_values(time_col)
    for ax, c in zip(axes, cols):
        r_m = R[c].rolling(window, min_periods=1).mean()
        r_s = R[c].rolling(window, min_periods=1).std()
        s_m = S[c].rolling(window, min_periods=1).mean()
        s_s = S[c].rolling(window, min_periods=1).std()

        ax.plot(R[time_col], r_m, label="real mean", linewidth=2)
        ax.fill_between(R[time_col], r_m - r_s, r_m + r_s, alpha=0.2, label="real ±1σ")

        ax.plot(S[time_col], s_m, label="synthetic mean", linewidth=2)
        ax.fill_between(S[time_col], s_m - s_s, s_m + s_s, alpha=0.2, label="synthetic ±1σ")

        ax.set_title(f"Rolling mean±band ({c})")
        ax.set_xlabel(time_col); ax.set_ylabel(c); ax.legend()
    fig.tight_layout()
    return fig, list(axes)


def plot_acf_compare(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    cols: Sequence[str],
    lags: Sequence[int] = (1, 2, 3, 4, 7, 14, 28),
    *,
    figsize: Tuple[int, int] | None = None,
) -> Tuple[Figure, list[Axes]]:
    """
    Bar plots of ACF at specified lags: real vs synthetic (per column).
    """
    _assert_columns(real_df, cols)
    _assert_columns(synth_df, cols)
    n = len(cols)
    if figsize is None:
        figsize = (10, 3 * n)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=figsize)
    if n == 1:
        axes = [axes]  # type: ignore[assignment]

    def _acf(arr: np.ndarray, lags: Sequence[int]) -> np.ndarray:
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return np.full(len(lags), np.nan)
        arr = arr - arr.mean()
        var = arr.var() + 1e-12
        out = []
        for L in lags:
            if L >= len(arr): out.append(np.nan)
            else:
                out.append(np.dot(arr[:-L], arr[L:]) / ((len(arr)-L)*var))
        return np.array(out)

    idx = np.arange(len(lags))
    width = 0.4
    for ax, c in zip(axes, cols):
        ar = real_df[c].to_numpy(float); as_ = synth_df[c].to_numpy(float)
        r = _acf(ar, lags); s = _acf(as_, lags)
        ax.bar(idx - width/2, r, width, label="real")
        ax.bar(idx + width/2, s, width, label="synthetic")
        ax.set_xticks(idx); ax.set_xticklabels([str(L) for L in lags])
        ax.set_title(f"ACF compare: {c}")
        ax.set_xlabel("lag"); ax.set_ylabel("autocorr"); ax.legend()
    fig.tight_layout()
    return fig, list(axes)


def plot_spectrum_compare(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    cols: Sequence[str],
    *,
    figsize: Tuple[int, int] | None = None,
) -> Tuple[Figure, list[Axes]]:
    """
    Normalized periodogram overlays for real vs synthetic (per column).
    """
    _assert_columns(real_df, cols)
    _assert_columns(synth_df, cols)
    n = len(cols)
    if figsize is None:
        figsize = (10, 3 * n)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=figsize)
    if n == 1:
        axes = [axes]  # type: ignore[assignment]

    def _spec(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = x[np.isfinite(x)].astype(float)
        if x.size < 4:
            return np.array([0.0]), np.array([0.0])
        F = np.fft.rfft(x - x.mean())
        P = np.abs(F)**2
        if P.sum() == 0: P = np.ones_like(P)
        P /= P.sum()
        f = np.linspace(0, 0.5, len(P))
        return f, P

    for ax, c in zip(axes, cols):
        fr, Pr = _spec(real_df[c].to_numpy(float))
        fs, Ps = _spec(synth_df[c].to_numpy(float))
        m = min(len(fr), len(fs))
        ax.plot(fr[:m], Pr[:m], label="real")
        ax.plot(fs[:m], Ps[:m], label="synthetic", linestyle="--")
        ax.set_title(f"Normalized spectrum: {c}")
        ax.set_xlabel("normalized frequency"); ax.set_ylabel("power"); ax.legend()
    fig.tight_layout()
    return fig, list(axes)


# -------------------------
# 5) Spatial
# -------------------------
def _variogram_curve(lat: np.ndarray, lon: np.ndarray, v: np.ndarray, nbins: int = 12, max_pairs: int = 20000, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    m = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(v)
    lat, lon, v = lat[m], lon[m], v[m]
    n = len(v)
    if n < 5:
        return np.array([]), np.array([])
    idx = rng.choice(n, size=min(n, 2000), replace=False)
    coords = np.c_[lat[idx], lon[idx]]
    vals = v[idx]
    dmat = np.sqrt(((coords[:, None, :] - coords[None, :, :])**2).sum(axis=2))
    iu = np.triu_indices_from(dmat, k=1)
    d = dmat[iu]
    g = 0.5 * (vals[:, None] - vals[None, :])[iu] ** 2
    if d.size > max_pairs:
        take = rng.choice(d.size, size=max_pairs, replace=False)
        d = d[take]; g = g[take]
    if d.size == 0:
        return np.array([]), np.array([])
    bins = np.quantile(d, np.linspace(0, 1, nbins + 1))
    bins = np.unique(bins)
    if len(bins) <= 2:
        return np.array([]), np.array([])
    inds = np.digitize(d, bins) - 1
    centers = 0.5 * (bins[:-1] + bins[1:])
    semiv = np.zeros(len(centers), dtype=float)
    for b in range(len(centers)):
        mask = inds == b
        semiv[b] = float(np.mean(g[mask])) if np.any(mask) else np.nan
    return centers, semiv


def plot_variogram(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    value_col: str,
    lat_col: str = "lat",
    lon_col: str = "lon",
    *,
    nbins: int = 12,
    max_pairs: int = 20000,
    figsize: Tuple[int, int] = (7, 5),
) -> Tuple[Figure, Axes]:
    """
    Empirical semivariogram curves: real vs synthetic.
    """
    _assert_columns(real_df, [lat_col, lon_col, value_col])
    _assert_columns(synth_df, [lat_col, lon_col, value_col])

    dr, gr = _variogram_curve(real_df[lat_col].to_numpy(float),
                              real_df[lon_col].to_numpy(float),
                              real_df[value_col].to_numpy(float),
                              nbins=nbins, max_pairs=max_pairs, seed=0)
    ds, gs = _variogram_curve(synth_df[lat_col].to_numpy(float),
                              synth_df[lon_col].to_numpy(float),
                              synth_df[value_col].to_numpy(float),
                              nbins=nbins, max_pairs=max_pairs, seed=0)

    fig, ax = plt.subplots(figsize=figsize)
    if len(dr) > 0:
        ax.plot(dr, gr, marker="o", label="real")
    if len(ds) > 0:
        ax.plot(ds, gs, marker="o", label="synthetic")
    ax.set_title(f"Variogram: {value_col}")
    ax.set_xlabel("distance (deg)"); ax.set_ylabel("semivariance"); ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_moran_scatter(
    df: pd.DataFrame,
    value_col: str,
    lat_col: str = "lat",
    lon_col: str = "lon",
    *,
    k: int = 8,
    figsize: Tuple[int, int] = (6, 5),
    title: Optional[str] = None,
) -> Tuple[Figure, Axes]:
    """
    Moran scatter: value vs spatially lagged value (k-NN weights).
    """
    _assert_columns(df, [lat_col, lon_col, value_col])
    lat = df[lat_col].to_numpy(float)
    lon = df[lon_col].to_numpy(float)
    v = df[value_col].to_numpy(float)
    mask = _mask_finite(lat, lon, v)
    lat, lon, v = lat[mask], lon[mask], v[mask]
    n = len(v)
    if n < max(5, k + 1):
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Insufficient points", ha="center", va="center")
        ax.axis("off")
        return fig, ax
    coords = np.c_[lat, lon]
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, n)).fit(coords)
    dists, inds = nbrs.kneighbors(coords)
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        neigh = [j for j in inds[i] if j != i]
        if not neigh: continue
        w = 1.0 / np.maximum(dists[i][1:], 1e-9)
        w = w / (w.sum() + 1e-12)
        W[i, neigh] = w
    v0 = v - v.mean()
    v_lag = (W @ v0)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(v0, v_lag, s=15, alpha=0.6)
    m = np.polyfit(v0, v_lag, deg=1)
    xs = np.linspace(v0.min(), v0.max(), 100)
    ax.plot(xs, m[0]*xs + m[1], linestyle="--", linewidth=1)
    ax.set_title(title or f"Moran scatter: {value_col} (k={k})")
    ax.set_xlabel(f"{value_col} (demeaned)")
    ax.set_ylabel(f"Spatial lag of {value_col}")
    fig.tight_layout()
    return fig, ax


# -------------------------
# 6) Utility (parity + errors)
# -------------------------
def plot_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    title: str = "Parity plot",
    s: float = 12.0,
    alpha: float = 0.6,
    figsize: Tuple[int, int] = (6, 5),
) -> Tuple[Figure, Axes]:
    """
    ŷ vs y parity scatter with y=x reference.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = _mask_finite(y_true, y_pred)
    y_true = y_true[m]; y_pred = y_pred[m]
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(y_true, y_pred, s=s, alpha=alpha)
    lo = float(np.nanmin([y_true.min(), y_pred.min()]))
    hi = float(np.nanmax([y_true.max(), y_pred.max()]))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("y (true)"); ax.set_ylabel("ŷ (pred)")
    fig.tight_layout()
    return fig, ax


def plot_parity_grid(
    tstr: Tuple[np.ndarray, np.ndarray] | None = None,
    trts: Tuple[np.ndarray, np.ndarray] | None = None,
    *,
    figsize: Tuple[int, int] = (12, 5),
) -> Tuple[Figure, list[Axes]]:
    """
    Two parity plots: TSTR and TRTS (if provided).
    """
    n = sum(x is not None for x in (tstr, trts))
    if n == 0:
        raise ValueError("Provide at least one of tstr or trts.")
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]  # type: ignore[assignment]
    idx = 0
    if tstr is not None:
        y, yhat = tstr
        ax = axes[idx]
        _ = plot_parity(y, yhat, title="Parity – TSTR", figsize=(figsize[0]//n, figsize[1]))
        ax_ = plt.gcf().axes[-1]  # last created
        axes[idx] = ax_
        idx += 1
    if trts is not None:
        y, yhat = trts
        ax = axes[idx]
        _ = plot_parity(y, yhat, title="Parity – TRTS", figsize=(figsize[0]//n, figsize[1]))
        ax_ = plt.gcf().axes[-1]
        axes[idx] = ax_
    fig.suptitle("Parity plots")
    fig.tight_layout()
    return fig, list(axes)


def plot_error_hist(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    bins: int = 30,
    figsize: Tuple[int, int] = (6, 4),
    title: str = "Prediction error histogram",
) -> Tuple[Figure, Axes]:
    """
    Histogram of residuals (y - ŷ).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = _mask_finite(y_true, y_pred)
    e = (y_true[m] - y_pred[m]).astype(float)
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(e, bins=bins, alpha=0.7)
    ax.axvline(0.0, linestyle="--", linewidth=1)
    ax.set_title(title); ax.set_xlabel("error (y - ŷ)"); ax.set_ylabel("count")
    fig.tight_layout()
    return fig, ax


# -------------------------
# 7) Robustness / Privacy (NN distances)
# -------------------------
def plot_nn_distance_cdf(
    real_df: Optional[pd.DataFrame] = None,
    synth_df: Optional[pd.DataFrame] = None,
    cols: Optional[Sequence[str]] = None,
    *,
    dists: Optional[Dict[str, np.ndarray]] = None,
    figsize: Tuple[int, int] = (7, 5),
    title: str = "NN distance CDFs",
) -> Tuple[Figure, Axes]:
    """
    CDFs of nearest-neighbor distances:
    - real→synthetic (r2s), synthetic→real (s2r), and optionally real→real (r2r)
    Either provide (real_df, synthetic, cols) to compute, or pass precomputed `dists`.
    """
    if dists is None:
        if real_df is None or synth_df is None or cols is None:
            raise ValueError("Provide (real_df, synth_df, cols) or precomputed dists.")
        _assert_columns(real_df, cols)
        _assert_columns(synth_df, cols)
        R, S = _std_space(real_df, synth_df, cols)
        nn_s = NearestNeighbors(n_neighbors=1).fit(S)
        r2s = nn_s.kneighbors(R)[0][:, 0]
        nn_r = NearestNeighbors(n_neighbors=1).fit(R)
        s2r = nn_r.kneighbors(S)[0][:, 0]
        r2r = nn_r.kneighbors(R)[0][:, 0]
    else:
        r2s = np.asarray(dists.get("r2s", []), dtype=float)
        s2r = np.asarray(dists.get("s2r", []), dtype=float)
        r2r = np.asarray(dists.get("r2r", []), dtype=float)

    def _cdf(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = x[np.isfinite(x)]
        if x.size == 0:
            return np.array([]), np.array([])
        x = np.sort(x)
        y = np.linspace(0, 1, len(x), endpoint=True)
        return x, y

    fig, ax = plt.subplots(figsize=figsize)
    for arr, lab in [(r2s, "real→synthetic"), (s2r, "synthetic→real"), (r2r, "real→real (baseline)")]:
        xs, ys = _cdf(arr)
        if len(xs) == 0: continue
        ax.step(xs, ys, where="post", label=lab)
    ax.set_title(title)
    ax.set_xlabel("NN distance (standardized space)"); ax.set_ylabel("CDF")
    ax.legend()
    fig.tight_layout()
    return fig, ax
