"""The V4 shared target must use the original CoELA skills safely."""

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tdw-gym")]

import lm_agent as lm_agent_module
from lm_agent import lm_agent


class V4ExecutorIsolationTests(unittest.TestCase):
    def make_agent(self, protocol="PeerConsultV4"):
        args = SimpleNamespace(
            source="test", lm_id="test", prompt_template_path="unused.csv",
            communication=True, cot=False,
            embodiments=["replicant", "replicant"],
        )
        planner = SimpleNamespace(
            peer_consult_protocol=protocol, peer_decision_card=None, reset=Mock(),
        )
        with patch.object(lm_agent_module, "LLM", return_value=planner):
            agent = lm_agent(0, Mock(), 3000, args)
        agent.object_list = {0: [], 1: [], 2: []}
        agent.object_per_room = {"<Bedroom> (1000)": {0: [], 1: [], 2: []}}
        agent.episode_provenance = "current-episode"
        agent._peer_consult_episode_epoch = 1
        agent.env_api = {
            "belongs_to_which_room": lambda position: "<Bedroom> (1000)"
        }
        planner.shared_delivery_target = Mock(return_value={
            "id": 99, "name": "bed", "position": [1.0, 0.0, 2.0],
            "room": "<Bedroom> (1000)", "episode_epoch": 1,
        })
        return agent

    def test_v4_and_baseline_use_original_navigation_and_exploration(self):
        for protocol in ("PeerConsultV4", None):
            with self.subTest(protocol=protocol):
                agent = self.make_agent(protocol)
                agent.gotoroom = Mock(return_value={"executor": "original-nav"})
                agent.goexplore = Mock(return_value={"executor": "original-search"})
                self.assertEqual(
                    agent._execute_room_navigation_plan(),
                    {"executor": "original-nav"},
                )
                self.assertEqual(
                    agent._execute_room_exploration_plan(),
                    {"executor": "original-search"},
                )
                agent.gotoroom.assert_called_once_with()
                agent.goexplore.assert_called_once_with()

    def test_v4_shares_bed_by_default_but_baseline_does_not(self):
        v4 = self.make_agent()
        v4._inject_shared_bed()
        self.assertEqual(v4.object_list[2][0]["id"], 99)
        self.assertEqual(
            v4.object_list[2][0]["knowledge_source"], "shared_memory_board"
        )
        baseline = self.make_agent(None)
        baseline._inject_shared_bed()
        self.assertEqual(baseline.object_list[2], [])
        baseline.LLM.shared_delivery_target.assert_not_called()

    def test_local_bed_overrides_shared_record_and_old_epoch_is_rejected(self):
        agent = self.make_agent()
        agent._inject_shared_bed()
        local = {"id": 100, "type": 2, "name": "bed", "position": [3, 0, 4]}
        agent.object_list[2].append(local)
        agent._inject_shared_bed()
        self.assertEqual(agent.object_list[2], [local])

        stale = self.make_agent()
        stale.LLM.shared_delivery_target.return_value["episode_epoch"] = 0
        stale._inject_shared_bed()
        self.assertEqual(stale.object_list[2], [])

    def test_reset_clears_episode_memory_and_decision_card(self):
        agent = self.make_agent()
        agent.object_per_room["<OldRoom> (9000)"] = {0: [{"id": 7}], 1: [], 2: []}
        agent.target_pos = np.asarray([9.0, 0.0, 9.0])
        agent.visible_objects = [{"id": 7}]
        agent.pre_action = {"type": 0}
        agent.local_step = 41
        agent.LLM.peer_decision_card = {"active_task": {"task_id": "old"}}
        obs = {"agent": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])}
        output = ROOT / "results" / "v4_test_run" / "17"
        with patch.object(lm_agent_module, "AgentMemory", return_value=SimpleNamespace()):
            agent.reset(
                obs=obs, goal_objects={"apple": 1}, output_dir=str(output),
                env_api=agent.env_api, rooms_name=["<Bedroom> (1000)"],
                agent_color=np.asarray([255, 0, 0]), agent_id=0,
                gt_mask=True, save_img=False,
            )
        self.assertEqual(agent.object_per_room, {})
        self.assertIsNone(agent.target_pos)
        self.assertIsNone(agent.visible_objects)
        self.assertIsNone(agent.pre_action)
        self.assertEqual(agent.local_step, 0)
        self.assertIsNone(agent.LLM.peer_decision_card)
        self.assertEqual(
            agent.episode_provenance,
            os.path.normcase(os.path.abspath(output)),
        )


if __name__ == "__main__":
    unittest.main()
