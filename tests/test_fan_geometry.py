import unittest
import numpy as np

from fan_geometry import (
    FanConfig,
    branch_directions,
    branch_score,
    discover_fan_centers,
    gate_f0,
    run_matrix_fan,
)


class TestFanGeometry(unittest.TestCase):
    def test_discover_fan_centers_recovers_three_rays(self):
        rng = np.random.default_rng(4)
        true = branch_directions(3)
        samples = []
        for direction in true:
            for _ in range(30):
                x = direction + 0.08 * rng.normal(size=2)
                samples.append(x)
        found = discover_fan_centers(np.stack(samples), branches=3)
        self.assertIsNotNone(found)
        alignment = true @ found.T
        self.assertGreater(np.mean(np.max(alignment, axis=1)), 0.97)

    def test_branch_score_prefers_progress_on_ray(self):
        dirs = branch_directions(3)
        on_ray = np.array([[2.0, 0.0]])
        off_ray = np.array([[2.0, 0.7]])
        a, _ = branch_score(on_ray, dirs, 1.2)
        b, _ = branch_score(off_ray, dirs, 1.2)
        self.assertGreater(a[0], b[0])

    def test_matrix_search_preserves_all_three_branches(self):
        cfg = FanConfig(seed=2, generations=25, lineages=18)
        out = run_matrix_fan(cfg, mode="isotropic")
        self.assertEqual(out["final_branch_coverage"], 3)

    def test_gate_f0_history_is_prospectively_informative(self):
        out = gate_f0(seed=3, generations=35)
        self.assertGreater(out["comparisons"], 10)
        self.assertGreater(
            out["history_fan_alignment"],
            out["shuffled_history_alignment"],
        )


if __name__ == "__main__":
    unittest.main()
