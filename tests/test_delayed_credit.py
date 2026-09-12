import unittest
import numpy as np

from delayed_credit import DelayedConfig, run_delayed, run_battery


class TestDelayedCredit(unittest.TestCase):
    def test_tagged_delay_runs(self):
        cfg = DelayedConfig(branches=6, dim=4, steps=180, seed=3)
        out = run_delayed(cfg, "tagged")
        self.assertTrue(np.isfinite(out["final_mean_score"]))
        self.assertGreater(len(out["score_curve"]), 5)

    def test_tagged_credit_beats_current_branch_in_smoke_case(self):
        cfg = DelayedConfig(branches=8, dim=5, steps=800, seed=5)
        tagged = run_delayed(cfg, "tagged")["final_mean_score"]
        current = run_delayed(cfg, "current_branch")["final_mean_score"]
        self.assertGreater(tagged, current)

    def test_battery_returns_all_modes(self):
        out = run_battery(seeds=2, steps=240)
        for mode in ["immediate", "tagged", "history", "wrong_tag", "current_branch"]:
            self.assertIn(mode, out["summary"])


if __name__ == "__main__":
    unittest.main()
