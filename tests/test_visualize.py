import unittest
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

from sagda.visualize import (
    plot_distributions,
    plot_ecdf,
    plot_qq,
    plot_corr_heatmaps,
    plot_pairgrid,
    plot_embedding_overlap,
    plot_timeseries,
    plot_mean_band_over_time,
    plot_acf_compare,
    plot_spectrum_compare,
    plot_variogram,
    plot_moran_scatter,
    plot_parity,
    plot_parity_grid,
    plot_error_hist,
    plot_nn_distance_cdf,
)

def _make_data(n=120, seed=0):
    """Deterministic small dataset with time, lat/lon, features, target."""
    rng = np.random.default_rng(seed)
    date = pd.date_range("2019-01-06", periods=n, freq="W")  # weekly
    lat_r = rng.uniform(29.0, 35.0, n)
    lon_r = rng.uniform(-11.0, -1.0, n)

    z = rng.normal(size=(n, 4))
    pH_r   = 6.6 + 0.30*z[:,0] - 0.10*z[:,1]
    OM_r   = 1.8 + 0.40*z[:,1] + 0.20*z[:,2]
    CEC_r  = 18.0 + 3.0*z[:,2] + 1.0*z[:,0]
    N_r    = np.clip(80.0 + 25.0*np.abs(z[:,3]), 0, 250)
    P_r    = np.clip(40.0 + 15.0*np.abs(0.5*z[:,3] + 0.5*z[:,1]), 0, 120)
    K_r    = np.clip(40.0 + 15.0*np.abs(0.3*z[:,3] + 0.7*z[:,2]), 0, 120)
    season = 0.4*np.sin(2*np.pi*np.arange(n)/52)
    y_r    = (800 + 80*np.sqrt(N_r) + 50*np.sqrt(P_r) + 35*np.sqrt(K_r)
              + 60*OM_r - 40*(pH_r - 6.6)**2 + 10*season + rng.normal(0, 40, size=n))

    real = pd.DataFrame({
        "date": date, "lat": lat_r, "lon": lon_r,
        "pH": pH_r, "OM_%": OM_r, "CEC_cmolkg": CEC_r,
        "N_rate": N_r, "P2O5_rate": P_r, "K2O_rate": K_r,
        "yield_kg_ha": y_r,
    })

    # Synthetic: small drifts + noise
    lat_s = np.clip(lat_r + rng.normal(0, 0.05, n), 28.8, 35.2)
    lon_s = np.clip(lon_r + rng.normal(0, 0.05, n), -11.2, -0.8)
    pH_s  = pH_r + 0.05 + rng.normal(0, 0.05, n)
    OM_s  = OM_r * rng.normal(1.02, 0.03, n)
    CEC_s = CEC_r + rng.normal(0, 0.3, n)
    N_s   = np.clip(N_r * 1.03 + rng.normal(0, 2.0, n), 0, 250)
    P_s   = np.clip(P_r * 1.04 + rng.normal(0, 1.0, n), 0, 120)
    K_s   = np.clip(K_r * 1.02 + rng.normal(0, 1.0, n), 0, 120)
    season_s = 0.4*np.sin(2*np.pi*np.arange(n)/52 + 0.1)
    y_s   = (790 + 80*np.sqrt(N_s) + 50*np.sqrt(P_s) + 35*np.sqrt(K_s)
             + 62*OM_s - 42*(pH_s - 6.6)**2 + 10*season_s + rng.normal(0, 48, size=n))

    synth = pd.DataFrame({
        "date": date, "lat": lat_s, "lon": lon_s,
        "pH": pH_s, "OM_%": OM_s, "CEC_cmolkg": CEC_s,
        "N_rate": N_s, "P2O5_rate": P_s, "K2O_rate": K_s,
        "yield_kg_ha": y_s,
    })
    return real, synth


class TestVisualizeAll(unittest.TestCase):
    def setUp(self):
        self.real, self.synth = _make_data()
        self.cols = ["pH","OM_%","CEC_cmolkg","N_rate","P2O5_rate","K2O_rate","yield_kg_ha"]

    def tearDown(self):
        plt.close("all")

    # ---- Marginals
    def test_plot_distributions(self):
        fig, axes = plot_distributions(self.real, self.synth, self.cols[:2])
        self.assertIsNotNone(fig); self.assertEqual(len(axes), 2)
        fig.canvas.draw()

    def test_plot_ecdf(self):
        fig, axes = plot_ecdf(self.real, self.synth, self.cols[:3])
        self.assertIsNotNone(fig); self.assertEqual(len(axes), 3)
        fig.canvas.draw()

    def test_plot_qq(self):
        fig, axes = plot_qq(self.real, self.synth, self.cols[:2])
        self.assertIsNotNone(fig); self.assertEqual(len(axes), 2)
        fig.canvas.draw()

    # ---- Joint
    def test_plot_corr_heatmaps(self):
        fig, axes = plot_corr_heatmaps(self.real, self.synth, self.cols[:5])
        self.assertEqual(len(axes), 3)
        for ax in axes: ax.figure.canvas.draw()

    def test_plot_pairgrid(self):
        fig, axes = plot_pairgrid(self.real, self.synth, self.cols[:3], sample=200)
        self.assertEqual(len(axes), 3*3)  # grid flattened
        fig.canvas.draw()

    # ---- Overlap
    def test_plot_embedding_overlap_pca(self):
        fig, ax = plot_embedding_overlap(self.real, self.synth, self.cols[:5], method="pca")
        self.assertIsNotNone(ax); fig.canvas.draw()

    def test_plot_embedding_overlap_umap(self):
        # If umap-learn isn't installed, function falls back to PCA (still valid)
        fig, ax = plot_embedding_overlap(self.real, self.synth, self.cols[:5], method="umap")
        self.assertIsNotNone(ax); fig.canvas.draw()

    # ---- Temporal
    def test_plot_timeseries(self):
        fig, axes = plot_timeseries(self.real, ["pH","OM_%"], time_col="date", rolling=5)
        self.assertEqual(len(axes), 2); fig.canvas.draw()

    def test_plot_mean_band_over_time(self):
        fig, axes = plot_mean_band_over_time(self.real, self.synth, ["pH","yield_kg_ha"], "date", window=5)
        self.assertEqual(len(axes), 2); fig.canvas.draw()

    def test_plot_acf_compare(self):
        fig, axes = plot_acf_compare(self.real, self.synth, ["pH","OM_%"])
        self.assertEqual(len(axes), 2); fig.canvas.draw()

    def test_plot_spectrum_compare(self):
        fig, axes = plot_spectrum_compare(self.real, self.synth, ["pH","OM_%"])
        self.assertEqual(len(axes), 2); fig.canvas.draw()

    # ---- Spatial
    def test_plot_variogram(self):
        fig, ax = plot_variogram(self.real, self.synth, "pH", lat_col="lat", lon_col="lon")
        self.assertIsNotNone(ax); fig.canvas.draw()

    def test_plot_moran_scatter(self):
        fig, ax = plot_moran_scatter(self.real, "pH", lat_col="lat", lon_col="lon", k=8)
        self.assertIsNotNone(ax); fig.canvas.draw()

    # ---- Utility
    def test_plot_parity_and_errors(self):
        # Fake predictions with slight noise
        y = self.real["yield_kg_ha"].to_numpy()
        yhat = y + np.random.default_rng(1).normal(0, 50, size=len(y))
        fig1, ax1 = plot_parity(y, yhat)
        fig2, ax2 = plot_error_hist(y, yhat)
        self.assertIsNotNone(ax1); self.assertIsNotNone(ax2)
        fig1.canvas.draw(); fig2.canvas.draw()

    def test_plot_parity_grid(self):
        y_r = self.real["yield_kg_ha"].to_numpy()
        y_s = self.synth["yield_kg_ha"].to_numpy()
        # Make synthetic predictions for both folds
        yhat_tstr = y_r + np.random.default_rng(2).normal(0, 60, size=len(y_r))
        yhat_trts = y_s + np.random.default_rng(3).normal(0, 60, size=len(y_s))
        fig, axes = plot_parity_grid(tstr=(y_r, yhat_tstr), trts=(y_s, yhat_trts))
        self.assertGreaterEqual(len(axes), 1)
        fig.canvas.draw()

    # ---- Robustness/Privacy
    def test_plot_nn_distance_cdf_compute(self):
        fig, ax = plot_nn_distance_cdf(self.real, self.synth, cols=self.cols[:5])
        self.assertIsNotNone(ax); fig.canvas.draw()

    def test_plot_nn_distance_cdf_precomputed(self):
        # Precompute simple arrays to exercise the branch
        r2s = np.linspace(0.05, 1.0, 50)
        s2r = np.linspace(0.06, 1.1, 60)
        r2r = np.linspace(0.03, 0.9, 40)
        fig, ax = plot_nn_distance_cdf(dists={"r2s": r2s, "s2r": s2r, "r2r": r2r})
        self.assertIsNotNone(ax); fig.canvas.draw()

    # ---- Input validation / errors
    def test_input_validation_errors(self):
        with self.assertRaises(ValueError):
            _ = plot_distributions(self.real, self.synth, cols=["NOT_A_COL"])
        with self.assertRaises(ValueError):
            _ = plot_timeseries(self.real, cols=["pH"], time_col="NOT_DATE")
        with self.assertRaises(ValueError):
            _ = plot_nn_distance_cdf()  # neither dfs nor dists provided


if __name__ == "__main__":
    unittest.main()
