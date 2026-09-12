import unittest
import numpy as np

from nested_fan import NestedConfig, branch_score, run_nested, run_battery


class TestNestedFan(unittest.TestCase):
    def test_branch_score_prefers_target_direction(self):
        target = np.array([[1.0, 0.0]])
        aligned = branch_score(np.array([[1.0, 0.0]]), target, 0.2)[0]
        orthogonal = branch_score(np.array([[0.0, 1.0]]), target, 0.2)[0]
        self.assertGreater(aligned, orthogonal)

    def test_all_modes_use_same_score_probe_budget(self):
        cfg = NestedConfig(branches=6, dim=4, cycles=5, eval_budget=12, replay_contexts=2, seed=2)
        modes = [
            "flat",
            "flat_active_matched",
            "flat_replay",
            "local",
            "history",
            "shuffled_history",
        ]
        probes = [run_nested(cfg, mode)["score_probes"] for mode in modes]
        self.assertEqual(len(set(probes)), 1)

    def test_local_separation_protects_routes_in_smoke_case(self):
        cfg = NestedConfig(branches=8, dim=5, cycles=12, eval_budget=16, seed=4)
        flat = run_nested(cfg, "flat")["final_mean_score"]
        local = run_nested(cfg, "local")["final_mean_score"]
        self.assertGreater(local, flat)

    def test_history_battery_is_finite(self):
        out = run_battery(seeds=2, cycles=6)
        value = out["summary"]["history"]["mean_final_score"]
        self.assertTrue(np.isfinite(value))
        self.assertEqual(out["seeds"], 2)


if __name__ == "__main__":
    unittest.main()
