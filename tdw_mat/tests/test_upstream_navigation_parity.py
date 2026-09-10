import ast
import unittest
import zipfile
from pathlib import Path


TEST_FILE = Path(__file__).resolve()
TDW_MAT_ROOT = TEST_FILE.parents[1]
PRISTINE_ZIP = next(
    parent / "CoELA-master.zip"
    for parent in TEST_FILE.parents
    if (parent / "CoELA-master.zip").is_file()
)


def _class_methods(source, class_name):
    tree = ast.parse(source)
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == class_name)
    return {
        node.name: ast.dump(node, include_attributes=False)
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


class UpstreamNavigationParityTests(unittest.TestCase):
    CASES = {
        "tdw-gym/lm_agent.py": (
            "lm_agent",
            ("reach_target_pos", "move", "gotoroom", "goexplore",
             "gograsp", "goput"),
        ),
        "tdw-gym/agent_memory.py": (
            "AgentMemory",
            ("__init__", "update", "dep2map", "get_angle",
             "find_shortest_path", "have_wall", "move_to_pos"),
        ),
    }

    @classmethod
    def setUpClass(cls):
        cls.current = {}
        cls.pristine = {}
        with zipfile.ZipFile(PRISTINE_ZIP) as archive:
            for relative, (class_name, _) in cls.CASES.items():
                current_source = (TDW_MAT_ROOT / relative).read_text(
                    encoding="utf-8")
                pristine_source = archive.read(
                    f"CoELA-master/tdw_mat/{relative}").decode("utf-8")
                cls.current[relative] = _class_methods(
                    current_source, class_name)
                cls.pristine[relative] = _class_methods(
                    pristine_source, class_name)

    def test_selected_navigation_methods_match_pristine_ast(self):
        for relative, (_, method_names) in self.CASES.items():
            for method_name in method_names:
                with self.subTest(file=relative, method=method_name):
                    self.assertEqual(
                        self.current[relative][method_name],
                        self.pristine[relative][method_name],
                    )

    def test_no_collision_feedback_overlay_symbols_remain(self):
        sources = "\n".join(
            (TDW_MAT_ROOT / relative).read_text(encoding="utf-8")
            for relative in self.CASES
        )
        for forbidden in (
                "TDW_MAT_FIX_NAV_COLLISION",
                "blocked_corridor_map",
                "record_blocked_corridor",
                "_consume_forward_move_result",
                "_guard_repeated_failed_forward"):
            with self.subTest(symbol=forbidden):
                self.assertNotIn(forbidden, sources)


if __name__ == "__main__":
    unittest.main()
