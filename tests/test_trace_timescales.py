import unittest

from trace_timescales import run_sweep


class TraceTimescaleTests(unittest.TestCase):
    def test_slow_world_prefers_slower_trace_in_smoke_map(self):
        result = run_sweep(
            seeds=2,
            steps=400,
            downstream_decays=(0.20, 0.90),
            eligibility_decays=(0.00, 0.70, 0.98),
        )
        self.assertTrue(result["best_trace_slows_monotonically"])
        best = result["best_by_downstream"]
        self.assertLessEqual(
            best[0]["best_eligibility_decay"],
            best[1]["best_eligibility_decay"],
        )
        self.assertGreater(
            best[1]["best_mean_final_score"],
            best[1]["fixed_070_score"],
        )


if __name__ == "__main__":
    unittest.main()
