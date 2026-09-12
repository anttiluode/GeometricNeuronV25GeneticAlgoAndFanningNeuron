import unittest

from adaptive_resonator import AdaptiveResonatorConfig, run_adaptive_resonator
from offgrid_resonance import OffGridConfig, run_offgrid


class AdaptiveResonatorTests(unittest.TestCase):
    def test_frequency_fan_reduces_offgrid_error(self):
        run = run_adaptive_resonator(
            AdaptiveResonatorConfig(
                seed=1,
                steps=2400,
                epoch_steps=400,
                predictor_settle_steps=80,
            )
        )
        self.assertLess(
            run["final_frequency_mae"], run["initial_frequency_mae"]
        )

    def test_adaptive_pole_beats_nearest_grid_in_smoke(self):
        adaptive = run_adaptive_resonator(
            AdaptiveResonatorConfig(
                seed=0,
                steps=2400,
                epoch_steps=400,
                predictor_settle_steps=80,
            )
        )
        nearest = run_offgrid(
            OffGridConfig(seed=0, steps=2400), "nearest"
        )
        self.assertGreater(
            adaptive["final_mean_score"], nearest["final_mean_score"]
        )


if __name__ == "__main__":
    unittest.main()
