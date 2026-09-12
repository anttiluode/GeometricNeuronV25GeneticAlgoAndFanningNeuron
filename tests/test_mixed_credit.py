import math
import unittest

from mixed_credit import MixedCreditConfig, run_mixed_credit


class MixedCreditTests(unittest.TestCase):
    def test_reproducible(self):
        cfg = MixedCreditConfig(seed=3, steps=120)
        a = run_mixed_credit(cfg, "eligibility")
        b = run_mixed_credit(cfg, "eligibility")
        self.assertEqual(a["final_mean_score"], b["final_mean_score"])
        self.assertEqual(a["final_branch_scores"], b["final_branch_scores"])

    def test_local_trace_beats_source_destroying_controls(self):
        cfg = MixedCreditConfig(seed=0, steps=800)
        eligibility = run_mixed_credit(cfg, "eligibility")["final_mean_score"]
        current = run_mixed_credit(cfg, "current_only")["final_mean_score"]
        shuffled = run_mixed_credit(cfg, "shuffled_eligibility")["final_mean_score"]
        pooled = run_mixed_credit(cfg, "pooled_eligibility")["final_mean_score"]
        self.assertGreater(eligibility, current)
        self.assertGreater(eligibility, shuffled)
        self.assertGreater(eligibility, pooled)

    def test_scores_are_finite(self):
        cfg = MixedCreditConfig(seed=1, steps=120)
        for mode in (
            "local_oracle",
            "immediate_global",
            "eligibility",
            "current_only",
            "shuffled_eligibility",
            "pooled_eligibility",
        ):
            run = run_mixed_credit(cfg, mode)
            self.assertTrue(math.isfinite(run["final_mean_score"]))
            self.assertTrue(math.isfinite(run["final_min_branch_score"]))


if __name__ == "__main__":
    unittest.main()
