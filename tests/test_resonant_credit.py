import unittest

from resonant_credit import ResonantConfig, run_resonant


class ResonantCreditTests(unittest.TestCase):
    def test_adaptive_identifies_branch_frequency_better_than_chance(self):
        run = run_resonant(
            ResonantConfig(seed=0, steps=1200, warmup_steps=250),
            "adaptive",
        )
        self.assertGreater(run["frequency_id_accuracy"], 0.50)

    def test_adaptive_beats_single_fixed_channel_in_smoke_run(self):
        cfg = ResonantConfig(seed=1, steps=1400, warmup_steps=250)
        adaptive = run_resonant(cfg, "adaptive")
        fixed = run_resonant(cfg, "fixed1")
        self.assertGreater(
            adaptive["final_mean_score"], fixed["final_mean_score"]
        )

    def test_oracle_is_strong_upper_bound(self):
        cfg = ResonantConfig(seed=2, steps=1200, warmup_steps=200)
        oracle = run_resonant(cfg, "oracle")
        shuffled = run_resonant(cfg, "shuffled")
        self.assertGreater(
            oracle["final_mean_score"], shuffled["final_mean_score"]
        )


if __name__ == "__main__":
    unittest.main()
