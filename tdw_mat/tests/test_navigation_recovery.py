import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


TDW_MAT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TDW_MAT_ROOT))
sys.path.insert(0, str(TDW_MAT_ROOT / "tdw-gym"))

from agent_memory import AgentMemory
import lm_agent as lm_agent_module
from lm_agent import lm_agent
from scout_agent import ScoutAgent
from tdw_gym import TDW
from tdw.replicant.action_status import ActionStatus


class _NullLogger:
    def info(self, *args, **kwargs):
        pass


def _legacy_get_pc(obs, color):
    depth = obs["depth"].copy()
    for i in range(len(obs["seg_mask"])):
        for j in range(len(obs["seg_mask"][0])):
            if (obs["seg_mask"][i][j] != color).any():
                depth[i][j] = 1e9
    fov = obs["FOV"]
    width, height = depth.shape
    cx = width / 2.0
    cy = height / 2.0
    fx = cx / np.tan(math.radians(fov / 2.0))
    fy = cy / np.tan(math.radians(fov / 2.0))
    x_index = np.linspace(0, width - 1, width)
    y_index = np.linspace(0, height - 1, height)
    xx, yy = np.meshgrid(x_index, y_index)
    xx = (xx - cx) / fx * depth
    yy = (yy - cy) / fy * depth
    index = np.where((depth > 0) & (depth < 10))
    xx = xx[index].copy().reshape(-1)
    yy = yy[index].copy().reshape(-1)
    depth = depth[index].copy().reshape(-1)
    point_cloud = np.stack((xx, yy, depth, np.ones_like(xx))).reshape(4, -1)
    inverse = np.linalg.inv(np.array(obs["camera_matrix"]).reshape((4, 4)))
    inverse = np.dot(inverse, np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1],
    ]))
    return np.dot(inverse, point_cloud)[:3]


def _published_color2id(seg_mask, agent_color, agent_id, color2id):
    """Reference the exact assignment order in upstream AgentMemory."""

    result = np.full(seg_mask.shape[:2], -100, dtype=np.int64)
    for i in range(seg_mask.shape[0]):
        for j in range(seg_mask.shape[1]):
            color = tuple(seg_mask[i, j])
            if np.array_equal(seg_mask[i, j], agent_color):
                result[i, j] = agent_id
            if color in color2id:
                result[i, j] = color2id[color]
    return result


class NavigationAlignmentTests(unittest.TestCase):
    def make_memory(self):
        memory = AgentMemory(
            agent_id=0,
            agent_color=np.array([255, 0, 0]),
            gt_mask=True,
            gt_behavior=True,
            map_size=(30, 30),
            scene_bounds={
                "x_min": 0.0,
                "x_max": 3.625,
                "z_min": 0.0,
                "z_max": 3.625,
            },
        )
        memory.obs = {"current_frames": 10}
        memory.known_map.fill(1)
        return memory

    def make_lm_agent(self, agent_id=0, role=None):
        args = SimpleNamespace(
            source="test",
            lm_id="test",
            prompt_template_path="unused.csv",
            communication=False,
            cot=False,
            embodiments=["replicant", "box"],
        )
        with patch.object(lm_agent_module, "LLM", return_value=SimpleNamespace()):
            return lm_agent(
                agent_id, _NullLogger(), 3000, args, agent_role=role)

    def test_human_has_no_collision_recovery_state(self):
        agent = self.make_lm_agent(agent_id=0, role="human")
        for attribute in (
                "fix_nav_collision", "_issued_forward_pose",
                "_blocked_forward_signature"):
            self.assertFalse(hasattr(agent, attribute), attribute)
        for method in (
                "_consume_forward_move_result",
                "_guard_repeated_failed_forward"):
            self.assertFalse(hasattr(lm_agent, method), method)

    def test_scout_uses_the_same_published_navigation_methods(self):
        self.assertTrue(issubclass(ScoutAgent, lm_agent))
        self.assertIs(ScoutAgent.move, lm_agent.move)
        self.assertIs(ScoutAgent.gotoroom, lm_agent.gotoroom)
        self.assertIs(ScoutAgent.goexplore, lm_agent.goexplore)
        self.assertIs(ScoutAgent.reach_target_pos, lm_agent.reach_target_pos)

    def test_agent_memory_has_no_failed_corridor_overlay(self):
        memory = self.make_memory()
        self.assertFalse(hasattr(memory, "blocked_corridor_map"))
        self.assertFalse(hasattr(memory, "last_blocked_corridor"))
        self.assertFalse(hasattr(AgentMemory, "record_blocked_corridor"))
        position = np.array([1.25, 0.0, 1.25])
        goal = np.array([1.25, 0.0, 2.50])
        path, _ = memory.find_shortest_path(position, goal)
        self.assertEqual(tuple(path[1]), (10, 11))

    def test_environment_preserves_raw_action_diagnostics_without_consuming_them(self):
        action = SimpleNamespace(
            status=ActionStatus.detected_obstacle,
            obstacle_ids=(9, 3, 9),
            hit_wall=True,
        )
        self.assertEqual(
            TDW._action_diagnostics(action),
            ("detected_obstacle", 9, (3, 9), True),
        )
        replicant_action = SimpleNamespace(status=ActionStatus.collision)
        self.assertEqual(
            TDW._action_diagnostics(replicant_action),
            ("collision", 8, (), False),
        )

    def test_vectorized_point_cloud_is_numerically_equivalent(self):
        color = np.array([11, 22, 33])
        seg_mask = np.zeros((4, 4, 3), dtype=np.uint8)
        seg_mask[0, :3] = color
        seg_mask[1, 1:4] = color
        seg_mask[3, 2] = color
        obs = {
            "depth": np.array([
                [0.5, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
                [0.0, 1.5, 2.5, 3.5],
            ], dtype=np.float32),
            "seg_mask": seg_mask,
            "FOV": 90,
            "camera_matrix": np.eye(4, dtype=np.float32),
        }
        agent = lm_agent.__new__(lm_agent)
        agent.obs = obs
        np.testing.assert_allclose(
            agent.get_pc(color), _legacy_get_pc(obs, color), rtol=0, atol=0)

    def test_packed_color_lookup_matches_published_assignment_order(self):
        memory = self.make_memory()
        memory.agent_color = np.array([250, 1, 2])
        memory.color2id = {
            (0, 0, 0): 10,
            (11, 22, 33): 20,
            (255, 255, 255): 30,
            tuple(memory.agent_color): 999,
        }
        seg_mask = np.array([
            [[0, 0, 0], [11, 22, 33], [7, 8, 9], [250, 1, 2]],
            [[255, 255, 255], [0, 0, 0], [11, 22, 33], [-1, -1, -1]],
            [[7, 8, 9], [7, 8, 9], [250, 1, 2], [255, 255, 255]],
        ], dtype=np.int32)
        expected = _published_color2id(
            seg_mask, memory.agent_color, memory.agent_id, memory.color2id)
        np.testing.assert_array_equal(
            memory.color2id_fc_vectorized(seg_mask), expected)

    def test_dep2map_is_not_cached_across_published_calls(self):
        memory = self.make_memory()
        memory.position = np.array([1.0, 0.0, 1.0])
        memory.forward = np.array([0.0, 0.0, 1.0])
        memory.obs = {
            "current_frames": 10,
            "depth": np.ones((4, 4), dtype=np.float32),
            "seg_mask": np.zeros((4, 4, 3), dtype=np.uint8),
            "FOV": 90,
            "camera_matrix": np.eye(4, dtype=np.float32),
        }
        calls = 0
        original_conv2d = memory.conv2d

        def counted_conv2d(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original_conv2d(*args, **kwargs)

        memory.conv2d = counted_conv2d
        memory.dep2map()
        first_call_count = calls
        memory.dep2map()
        self.assertGreater(first_call_count, 0)
        self.assertEqual(calls, 2 * first_call_count)

    def test_update_preserves_published_pose_assignment_order(self):
        memory = self.make_memory()
        old_position = np.array([0.5, 0.0, 0.5])
        old_forward = np.array([0.0, 0.0, 1.0])
        memory.position = old_position.copy()
        memory.forward = old_forward.copy()
        observed_positions = []
        memory.ignore_logic = lambda *args, **kwargs: None
        memory.get_object_list = lambda: None
        memory.dep2map = lambda: (
            observed_positions.append(memory.position.copy()) or
            np.zeros(memory.map_size, dtype=np.int32))
        obs = {
            "current_frames": 11,
            "agent": np.array([1.0, 0.0, 1.0, 1.0, 0.0, 0.0]),
            "previous_action": {},
            "previous_status": {},
            "oppo_held_objects": [{"type": None}, {"type": None}],
        }
        memory.update(obs)
        np.testing.assert_array_equal(observed_positions[0], old_position)
        np.testing.assert_array_equal(memory.position, obs["agent"][:3])
        np.testing.assert_array_equal(memory.forward, obs["agent"][3:])


if __name__ == "__main__":
    unittest.main()
