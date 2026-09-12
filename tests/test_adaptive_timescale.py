import math
import unittest

from adaptive_timescale import (
    AdaptiveTimescaleConfig,
    run_adaptive,
    run_battery,
    run_fixed_trace,
)


class AdaptiveTimescaleTests(unittest.TestCase):
    def test_adaptive_smoke_is_finite_and_normalized(self):
        cfg = AdaptiveTimescaleConfig(steps=240, seed=3)
        run = run_adaptive(cfg)
        self.assertTrue(math.isfinite(run["final_mean_score"]))
        self.assertAlmostEqual(sum(run["trace_fraction_before"]), 1.0, places=10)
        self.assertAlmostEqual(sum(run["trace_fraction_after"]), 1.0, places=10)
        self.assertEqual(len(run["trace_fraction_before"]), len(cfg.trace_decays))

    def test_fixed_trace_smoke(self):
        cfg = AdaptiveTimescaleConfig(steps=240, seed=4)
        run = run_fixed_trace(cfg, 0.70)
        self.assertTrue(math.isfinite(run["final_mean_score"]))
        self.assertGreater(len(run["score_curve"]), 0)

    def test_battery_has_matched_controls(self):
        receipt = run_battery(seeds=2, steps=240)
        self.assertEqual(receipt["seeds"], 2)
        self.assertEqual(set(receipt["fixed"]), {"0.2", "0.7", "0.95", "0.98"})
        for decay in receipt["trace_decays"]:
            self.assertIn(
                f"adaptive_beats_fixed_{decay}_seeds", receipt["comparisons"]
            )


if __name__ == "__main__":
    unittest.main()
