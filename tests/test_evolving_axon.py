import unittest
import numpy as np

from evolving_axon import GAConfig, low_rank_next_prediction, gate0_prediction, run_ga


class TestEvolvingAxon(unittest.TestCase):
    def test_predictor_recovers_linear_rotation(self):
        theta = 0.12
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        x = np.array([1.0, 0.0])
        hist = []
        for _ in range(24):
            hist.append(x.copy())
            x = rot @ x
        pred = low_rank_next_prediction(hist, rank=2, window=20)
        self.assertIsNotNone(pred)
        self.assertLess(np.linalg.norm(pred - x), 2e-2)

    def test_ga_improves(self):
        cfg = GAConfig(dim=6, population=48, generations=30, seed=3)
        rec = run_ga(cfg, guided=False)
        self.assertLess(-rec[-1].best_fitness, -rec[0].best_fitness)

    def test_gate0_returns_finite_errors(self):
        cfg = GAConfig(dim=6, population=48, generations=35, seed=4, predictor_min_history=8)
        out = gate0_prediction(cfg)
        self.assertGreater(out["comparisons"], 5)
        for v in out["mean_error"].values():
            self.assertTrue(np.isfinite(v))


if __name__ == "__main__":
    unittest.main()
