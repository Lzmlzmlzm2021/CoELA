import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np


TDW_MAT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TDW_MAT_ROOT))
sys.path.insert(0, str(TDW_MAT_ROOT / "tdw-gym"))

import lm_agent as lm_agent_module
from lm_agent import lm_agent


class _NullLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


class V4ExecutorIsolationTests(unittest.TestCase):
    def make_agent(self, protocol="PeerConsultV4"):
        args = SimpleNamespace(
            source="test",
            lm_id="test",
            prompt_template_path="unused.csv",
            communication=True,
            cot=False,
            embodiments=["replicant", "replicant"],
        )
        planner = SimpleNamespace(
            peer_consult_protocol=protocol,
            peer_decision_card=None,
            pending_perception_events=[],
            reset=Mock(),
        )
        with patch.object(lm_agent_module, "LLM", return_value=planner):
            agent = lm_agent(0, _NullLogger(), 3000, args)
        return agent

    def test_v4_is_episode_isolated_but_not_a_v34_or_v35_protocol(self):
        agent = self.make_agent()

        self.assertTrue(agent._uses_v4_protocol())
        self.assertTrue(agent._uses_episode_isolation_protocol())
        self.assertFalse(agent._uses_v34_protocol())
        self.assertFalse(agent._uses_v35_protocol())

    def test_legacy_protocol_capabilities_are_unchanged(self):
        v34 = self.make_agent("PeerConsultV3.4")
        v35 = self.make_agent("PeerConsultV3.5")
        legacy = self.make_agent("PeerConsultV3")

        self.assertTrue(v34._uses_v34_protocol())
        self.assertFalse(v34._uses_episode_isolation_protocol())
        self.assertTrue(v35._uses_v34_protocol())
        self.assertTrue(v35._uses_v35_protocol())
        self.assertTrue(v35._uses_episode_isolation_protocol())
        self.assertFalse(legacy._uses_v34_protocol())
        self.assertFalse(legacy._uses_v35_protocol())
        self.assertFalse(legacy._uses_episode_isolation_protocol())

    def test_v4_room_navigation_forces_original_executor(self):
        agent = self.make_agent()
        # Defense in depth: even a stale/incorrect coordinator flag must not
        # opt V4 into a legacy V3.x executor.
        agent.stable_task_protocol = True
        agent.gotoroom = Mock(return_value={"executor": "original"})
        agent.gotoroom_v33 = Mock(return_value={"executor": "v33"})
        agent.gotoroom_v34 = Mock(return_value={"executor": "v34"})

        action = agent._execute_room_navigation_plan()

        self.assertEqual(action, {"executor": "original"})
        agent.gotoroom.assert_called_once_with()
        agent.gotoroom_v33.assert_not_called()
        agent.gotoroom_v34.assert_not_called()

    def test_v4_room_exploration_forces_original_executor(self):
        agent = self.make_agent()
        agent.coverage_exploration = True
        agent.goexplore = Mock(return_value={"executor": "original"})
        agent.goexplore_v33 = Mock(return_value={"executor": "waypoint"})

        action = agent._execute_room_exploration_plan()

        self.assertEqual(action, {"executor": "original"})
        agent.goexplore.assert_called_once_with()
        agent.goexplore_v33.assert_not_called()

    def test_v35_executor_dispatch_remains_unchanged(self):
        agent = self.make_agent("PeerConsultV3.5")
        agent.stable_task_protocol = True
        agent.coverage_exploration = True
        agent.gotoroom = Mock(return_value={"executor": "original"})
        agent.gotoroom_v33 = Mock(return_value={"executor": "v33"})
        agent.gotoroom_v34 = Mock(return_value={"executor": "v34"})
        agent.goexplore = Mock(return_value={"executor": "original"})
        agent.goexplore_v33 = Mock(return_value={"executor": "waypoint"})

        self.assertEqual(
            agent._execute_room_navigation_plan(), {"executor": "v34"})
        self.assertEqual(
            agent._execute_room_exploration_plan(), {"executor": "waypoint"})
        agent.gotoroom_v34.assert_called_once_with()
        agent.gotoroom.assert_not_called()
        agent.goexplore_v33.assert_called_once_with()
        agent.goexplore.assert_not_called()

    def test_v33_and_legacy_navigation_dispatch_remain_unchanged(self):
        v33 = self.make_agent("PeerConsultV3.3")
        v33.stable_task_protocol = True
        v33.gotoroom = Mock(return_value={"executor": "original"})
        v33.gotoroom_v33 = Mock(return_value={"executor": "v33"})
        v33.gotoroom_v34 = Mock(return_value={"executor": "v34"})
        self.assertEqual(
            v33._execute_room_navigation_plan(), {"executor": "v33"})
        v33.gotoroom_v33.assert_called_once_with()
        v33.gotoroom.assert_not_called()
        v33.gotoroom_v34.assert_not_called()

        legacy = self.make_agent("PeerConsultV3")
        legacy.gotoroom = Mock(return_value={"executor": "original"})
        legacy.gotoroom_v33 = Mock(return_value={"executor": "v33"})
        legacy.gotoroom_v34 = Mock(return_value={"executor": "v34"})
        self.assertEqual(
            legacy._execute_room_navigation_plan(), {"executor": "original"})
        legacy.gotoroom.assert_called_once_with()
        legacy.gotoroom_v33.assert_not_called()
        legacy.gotoroom_v34.assert_not_called()

    def test_v4_does_not_inject_v34_shared_bed_adapter_state(self):
        agent = self.make_agent()
        agent.object_list = {0: [], 1: [], 2: []}
        agent.object_per_room = {
            "<Bedroom> (1000)": {0: [], 1: [], 2: []}}
        agent.LLM.shared_delivery_target = Mock(return_value={
            "id": 99,
            "name": "bed",
            "position": [1.0, 0.0, 2.0],
            "room": "<Bedroom> (1000)",
        })

        agent._inject_v34_shared_bed()

        self.assertEqual(agent.object_list[2], [])
        agent.LLM.shared_delivery_target.assert_not_called()

    def test_v4_opt_in_shared_target_uses_original_executor_shape(self):
        agent = self.make_agent()
        agent.object_list = {0: [], 1: [], 2: []}
        agent.object_per_room = {
            "<Bedroom> (1000)": {0: [], 1: [], 2: []}}
        agent.env_api = {
            "belongs_to_which_room": Mock(
                return_value="<Bedroom> (1000)")}
        agent.LLM.shared_delivery_target = Mock(return_value={
            "id": 99,
            "name": "bed",
            "position": [1.0, 0.0, 2.0],
            "room": "<Bedroom> (1000)",
        })

        with patch.dict(
                os.environ,
                {"TDW_MAT_V4_SHARED_DELIVERY_TARGET": "1"}):
            agent._inject_v34_shared_bed()

        self.assertEqual(agent.object_list[2][0]["id"], 99)
        self.assertEqual(
            agent.object_list[2][0]["knowledge_source"],
            "shared_memory_board")

    def test_v4_reset_clears_episode_scoped_planner_views(self):
        agent = self.make_agent()
        agent.object_per_room = {
            "<OldRoom> (9000)": {0: [{"id": 7}], 1: [], 2: []}}
        agent.target_pos = np.asarray([9.0, 0.0, 9.0])
        agent.visible_objects = [{"id": 7}]
        agent.pre_action = {"type": 0}
        agent.local_step = 41
        agent.explore_count = 9
        agent.pending_perception_events = [{"object_id": 7}]
        agent.LLM.peer_decision_card = {"active_task": {"task_id": "old"}}
        agent.LLM.pending_perception_events = [{"object_id": 7}]
        agent.LLM.current_room = "<OldRoom> (9000)"
        agent.LLM.rooms_explored = {"<OldRoom> (9000)": "all"}
        agent.LLM.object_list = {0: [{"id": 7}]}
        agent.LLM.holding_objects = [{"id": 7}]
        agent.LLM.obj_per_room = agent.object_per_room
        agent.LLM.allow_message_this_turn = False

        output_dir = TDW_MAT_ROOT / "results" / "v4_test_run" / "17"
        obs = {"agent": np.asarray([0.0, 0.0, 0.0,
                                    0.0, 0.0, 1.0])}
        env_api = {
            "belongs_to_which_room": lambda position: "<Bedroom> (1000)"}
        with patch.object(
                lm_agent_module, "AgentMemory",
                return_value=SimpleNamespace()):
            agent.reset(
                obs=obs,
                goal_objects={"apple": 1},
                output_dir=str(output_dir),
                env_api=env_api,
                rooms_name=["<Bedroom> (1000)"],
                agent_color=np.asarray([255, 0, 0]),
                agent_id=0,
                gt_mask=True,
                save_img=False,
            )

        self.assertEqual(agent.object_per_room, {})
        self.assertIsNone(agent.target_pos)
        self.assertIsNone(agent.visible_objects)
        self.assertIsNone(agent.pre_action)
        self.assertEqual(agent.local_step, 0)
        self.assertEqual(agent.explore_count, 0)
        self.assertEqual(agent.pending_perception_events, [])
        self.assertIsNone(agent.LLM.peer_decision_card)
        self.assertEqual(agent.LLM.pending_perception_events, [])
        self.assertIsNone(agent.LLM.current_room)
        self.assertIsNone(agent.LLM.rooms_explored)
        self.assertIsNone(agent.LLM.object_list)
        self.assertIsNone(agent.LLM.holding_objects)
        self.assertIsNone(agent.LLM.obj_per_room)
        self.assertTrue(agent.LLM.allow_message_this_turn)
        self.assertEqual(
            agent.episode_provenance,
            os.path.normcase(os.path.abspath(output_dir)),
        )
        agent.LLM.reset.assert_called_once_with(
            ["<Bedroom> (1000)"], {"apple": 1})


if __name__ == "__main__":
    unittest.main()
