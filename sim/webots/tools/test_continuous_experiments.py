import json
import math
import unittest

from compare_continuous import block_counts, qa
from experiments import CLEARANCE, RouteReference, obstacle_distance
from make_continuous_world import COMPLEX_ADDITIONS
from run_continuous_experiments import CONTROLLERS


class ContinuousExperimentsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = CONTROLLERS / "continuous_qlearning"
        cls.layout = json.loads((cls.folder / "eval_result_best.json").read_text(encoding="utf-8"))

    def test_archived_blocks_use_episode_numbers(self):
        for algorithm, expected, partial in (("qlearning", 2007, 7), ("dqn", 1436, 186)):
            data = qa.load_log(CONTROLLERS / f"continuous_{algorithm}" / "episode_log_best.csv")
            counts = block_counts(data, 9)
            sizes = counts.sum(axis=0)
            self.assertEqual(expected, sizes.sum())
            self.assertEqual(partial, sizes[sizes > 0][-1])
            occupied = counts[:, sizes > 0] / sizes[sizes > 0]
            qa.np.testing.assert_allclose(occupied.sum(axis=0), 1.0)

    def test_density_works_with_and_without_collisions(self):
        data = qa.load_log(self.folder / "episode_log_best.csv")
        density, count = qa.collision_density(data, self.layout)
        self.assertEqual(1305, count)
        self.assertGreater(density.max(), 0)
        self.assertTrue(density.mask.any())
        data["out"][:] = "goal"
        empty, count = qa.collision_density(data, self.layout)
        self.assertEqual(0, count)
        self.assertEqual(0, empty.max())

    def test_route_connects_both_layouts(self):
        actions = ((10, 10), (-4, 4), (4, -4))
        durations = []
        for extra in ((), COMPLEX_ADDITIONS):
            layout = {**self.layout, "obstacles": [*self.layout["obstacles"], *extra]}
            route = RouteReference(layout, layout["arena"], layout["start"], layout["target_r"], actions, 0.0145, 0.0746)
            duration, path = route.from_yaw(math.pi / 4)
            durations.append(duration)
            self.assertLessEqual(math.dist(path[-1], layout["target"]), layout["target_r"])
            x, y = qa.np.array(path).T
            for obstacle in layout["obstacles"]:
                self.assertTrue((obstacle_distance(x, y, obstacle) >= CLEARANCE - 1e-10).all())
            for hx, hy, radius in layout["hazards"]:
                self.assertTrue((qa.np.hypot(x - hx, y - hy) > radius).all())
        self.assertGreater(durations[1], durations[0])


if __name__ == "__main__":
    unittest.main()
