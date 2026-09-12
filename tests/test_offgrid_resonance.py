import unittest

from offgrid_resonance import OffGridConfig, run_offgrid


class OffGridResonanceTests(unittest.TestCase):
    def test_oracle_offgrid_frequency_beats_nearest_channel(self):
        cfg = OffGridConfig(seed=0, steps=1200)
        oracle = run_offgrid(cfg, "oracle")
        nearest = run_offgrid(cfg, "nearest")
        self.assertGreater(
            oracle["final_mean_score"], nearest["final_mean_score"]
        )

    def test_discrete_selector_beats_naive_softmix_in_smoke(self):
        cfg = OffGridConfig(seed=1, steps=1200)
        discrete = run_offgrid(cfg, "discrete")
        softmix = run_offgrid(cfg, "softmix")
        self.assertGreater(
            discrete["final_mean_score"], softmix["final_mean_score"]
        )

    def test_hidden_frequencies_are_off_candidate_grid(self):
        cfg = OffGridConfig(seed=2, steps=100)
        run = run_offgrid(cfg, "nearest")
        self.assertGreater(run["nearest_frequency_mae"], 0.0)


if __name__ == "__main__":
    unittest.main()
