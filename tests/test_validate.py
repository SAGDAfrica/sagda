import unittest
import numpy as np
import pandas as pd

from sagda.validate import validate, report


def make_datasets(n=80, seed=0):
    """Small but rich dataset with time, lat/lon, agronomic features, and yield."""
    rng = np.random.default_rng(seed)
    date = pd.date_range("2018-01-07", periods=n, freq="W")

    lat_real = rng.uniform(29.0, 35.0, size=n)
    lon_real = rng.uniform(-11.0, -1.0, size=n)

    z = rng.normal(size=(n, 4))
    pH_real   = 6.6 + 0.30*z[:,0] - 0.10*z[:,1]
    OM_real   = 1.8 + 0.40*z[:,1] + 0.20*z[:,2]
    CEC_real  = 18.0 + 3.0*z[:,2] + 1.0*z[:,0]
    N_rate    = np.clip(80.0 + 25.0*np.abs(z[:,3]), 0, 250)
    P2O5_rate = np.clip(40.0 + 15.0*np.abs(0.5*z[:,3] + 0.5*z[:,1]), 0, 120)
    K2O_rate  = np.clip(40.0 + 15.0*np.abs(0.3*z[:,3] + 0.7*z[:,2]), 0, 120)
    season    = 0.4*np.sin(2*np.pi*np.arange(n)/52)
    y_real    = (
        800 + 80*np.sqrt(N_rate) + 50*np.sqrt(P2O5_rate) + 35*np.sqrt(K2O_rate)
        + 60*OM_real - 40*(pH_real - 6.6)**2 + 10*season + rng.normal(0, 40, size=n)
    )

    real = pd.DataFrame({
        "date": date, "lat": lat_real, "lon": lon_real,
        "pH": pH_real, "OM_%": OM_real, "CEC_cmolkg": CEC_real,
        "N_rate": N_rate, "P2O5_rate": P2O5_rate, "K2O_rate": K2O_rate,
        "yield_kg_ha": y_real,
    })

    # Synthetic with small drifts + noise
    lat_syn = np.clip(lat_real + rng.normal(0, 0.05, size=n), 28.8, 35.2)
    lon_syn = np.clip(lon_real + rng.normal(0, 0.05, size=n), -11.2, -0.8)
    pH_syn  = pH_real + 0.05 + rng.normal(0, 0.05, size=n)
    OM_syn  = OM_real * rng.normal(1.02, 0.03, size=n)
    CEC_syn = CEC_real + rng.normal(0, 0.3, size=n)
    N_syn   = np.clip(N_rate * 1.03 + rng.normal(0, 2.0, size=n), 0, 250)
    P_syn   = np.clip(P2O5_rate * 1.04 + rng.normal(0, 1.0, size=n), 0, 120)
    K_syn   = np.clip(K2O_rate * 1.02 + rng.normal(0, 1.0, size=n), 0, 120)
    season_s= 0.4*np.sin(2*np.pi*np.arange(n)/52 + 0.1)
    y_syn   = (
        790 + 80*np.sqrt(N_syn) + 50*np.sqrt(P_syn) + 35*np.sqrt(K_syn)
        + 62*OM_syn - 42*(pH_syn - 6.6)**2 + 10*season_s + rng.normal(0, 48, size=n)
    )

    synth = pd.DataFrame({
        "date": date, "lat": lat_syn, "lon": lon_syn,
        "pH": pH_syn, "OM_%": OM_syn, "CEC_cmolkg": CEC_syn,
        "N_rate": N_syn, "P2O5_rate": P_syn, "K2O_rate": K_syn,
        "yield_kg_ha": y_syn,
    })
    return real, synth


class TestDataValidation(unittest.TestCase):

    def test_validate_full_pipeline(self):
        real, synth = make_datasets(n=80, seed=1)
        cols = ["pH","OM_%","CEC_cmolkg","N_rate","P2O5_rate","K2O_rate","yield_kg_ha"]

        res = validate(
            synth_df=synth,
            real_df=real,
            cols=cols,
            time_col="date",
            lat_col="lat",
            lon_col="lon",
            target_col="yield_kg_ha",
            pca_components=2,
            season_period=52,   # weekly → yearly, n=80 so defined
            sample_size=80,
            compute_privacy=True,
        )

        # Top-level keys
        for key in ["marginals", "joint", "overlap", "c2st_auc", "temporal",
                    "spatial", "utility", "robustness", "privacy"]:
            self.assertIn(key, res)

        # Marginals presence for a representative feature
        self.assertIn("ks", res["marginals"])
        self.assertIn("pH", res["marginals"]["ks"])
        self.assertIn("wass", res["marginals"])
        self.assertIn("emd", res["marginals"])

        # Joint structure values are numeric
        self.assertIn("mmd_rbf", res["joint"])
        self.assertTrue(np.isfinite(res["joint"]["mmd_rbf"]))

        # Overlap ranges
        self.assertIn("pca_overlap", res["overlap"])
        self.assertGreaterEqual(res["overlap"]["pca_overlap"], 0.0)
        self.assertLessEqual(res["overlap"]["pca_overlap"], 1.0)
        # UMAP may be None if umap-learn is not installed
        u = res["overlap"]["umap_overlap"]
        self.assertTrue(u is None or (0.0 <= u <= 1.0))

        # Discriminability in [0,1]
        self.assertGreaterEqual(res["c2st_auc"], 0.0)
        self.assertLessEqual(res["c2st_auc"], 1.0)

        # Temporal metrics exist, seasonals present (not NaN with our n/period)
        self.assertIn("acf_diff", res["temporal"])
        self.assertIn("spectral_L1", res["temporal"])
        self.assertIn("season_amp_phase", res["temporal"])
        self.assertIn("mean_amp_diff", res["temporal"]["season_amp_phase"])
        self.assertIn("mean_phase_diff", res["temporal"]["season_amp_phase"])

        # Spatial dicts contain at least one feature key
        self.assertIn("morans_I_diff", res["spatial"])
        self.assertIn("variogram_L2", res["spatial"])

        # Utility (TSTR/TRTS) present with metrics
        self.assertIsNotNone(res["utility"])
        for fold in ["tstr", "trts"]:
            for m in ["MAE", "RMSE", "MAPE"]:
                self.assertIn(m, res["utility"][fold])
                self.assertTrue(np.isfinite(res["utility"][fold][m]))

        # Robustness & privacy keys
        for k in ["nn_med_r2s", "nn_med_s2r", "coverage_at_q50"]:
            self.assertIn(k, res["robustness"])
        for k in ["min_nn_synth_to_real", "pct_below_eps"]:
            self.assertIn(k, res["privacy"])

    def test_report_formats(self):
        real, synth = make_datasets(n=60, seed=2)
        cols = ["pH","OM_%","CEC_cmolkg","N_rate","P2O5_rate","K2O_rate","yield_kg_ha"]
        res = validate(synth, real, cols=cols, time_col="date", lat_col="lat", lon_col="lon",
                       target_col="yield_kg_ha", pca_components=2, season_period=52,
                       sample_size=60, compute_privacy=True)
        md = report(res, as_markdown=True)
        txt = report(res, as_markdown=False)
        self.assertIsInstance(md, str)
        self.assertIsInstance(txt, str)
        self.assertIn("Synthetic Data Validation", md)
        self.assertIn("HEADLINE", txt)
        self.assertIn("Marginals", md)
        self.assertIn("Robustness", txt)

    def test_errors_and_edges(self):
        real, synth = make_datasets(n=20, seed=3)

        # pca_components must be >=1
        with self.assertRaises(ValueError):
            validate(synth, real, pca_components=0)

        # No overlapping numeric columns → ValueError
        r = pd.DataFrame({"a": ["x","y","z"], "b": ["u","v","w"]})
        s = pd.DataFrame({"c": [1,2,3], "d": [4,5,6]})
        with self.assertRaises(ValueError):
            validate(s, r)

        # Temporal/spatial skipped gracefully when columns absent
        res2 = validate(synth[["pH","OM_%","CEC_cmolkg"]], real[["pH","OM_%","CEC_cmolkg"]])
        self.assertIn("temporal", res2)
        self.assertIn("spatial", res2)
        # Privacy omitted when compute_privacy=False
        self.assertNotIn("privacy", res2)


if __name__ == "__main__":
    unittest.main()
