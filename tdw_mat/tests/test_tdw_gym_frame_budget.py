import importlib.util
import pathlib
import unittest


TDW_GYM_PATH = (
    pathlib.Path(__file__).resolve().parents[1] / "tdw-gym" / "frame_budget.py"
)
SPEC = importlib.util.spec_from_file_location("frame_budget", TDW_GYM_PATH)
FRAME_BUDGET = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FRAME_BUDGET)


class FrameBudgetTests(unittest.TestCase):
    def test_next_frame_is_allowed_inside_budget(self):
        self.assertFalse(FRAME_BUDGET.frame_budget_exhausted(2998, 0, 3000))

    def test_stops_before_requesting_frame_beyond_budget(self):
        self.assertTrue(FRAME_BUDGET.frame_budget_exhausted(2998, 2, 3000))

    def test_exact_limit_is_exhausted(self):
        self.assertTrue(FRAME_BUDGET.frame_budget_exhausted(3000, 0, 3000))


if __name__ == "__main__":
    unittest.main()
