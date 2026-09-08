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


class EpisodeIsolationTests(unittest.TestCase):
    def make_agent(self, protocol="PeerConsultV3.5"):
        args = SimpleNamespace(
            source="test",
            lm_id="test",
            prompt_template_path="unused.csv",
            communication=False,
            cot=False,
            embodiments=["replicant", "replicant"],
        )
        llm = SimpleNamespace(
            peer_consult_protocol=protocol,
            peer_decision_card=None,
            pending_perception_events=[],
            reset=Mock(),
        )
        with patch.object(lm_agent_module, "LLM", return_value=llm):
            agent = lm_agent(0, _NullLogger(), 3000, args)
        return agent

    def test_v35_reset_clears_episode_scoped_views_and_cursors(self):
        agent = self.make_agent()
        agent.object_per_room = {
            "<OldRoom> (9000)": {0: [{"id": 7}], 1: [], 2: []}}
        agent.target_pos = np.asarray([9.0, 0.0, 9.0])
        agent.visible_objects = [{"id": 7}]
        agent.pre_action = {"type": 0}
        agent.local_step = 41
        agent.pending_perception_events = [{"object_id": 7}]
        agent.LLM.peer_decision_card = {"shared_bed": {"id": 99}}
        agent.LLM.pending_perception_events = [{"object_id": 7}]
        agent.LLM.current_room = "<OldRoom> (9000)"
        agent.LLM.rooms_explored = {"<OldRoom> (9000)": "all"}
        agent.LLM.object_list = {2: [{"id": 99}]}
        agent.LLM.holding_objects = [{"id": 7}]
        agent.LLM.obj_per_room = agent.object_per_room
        agent.LLM.allow_message_this_turn = False

        output_dir = TDW_MAT_ROOT / "results" / "test_run" / "17"
        obs = {"agent": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])}
        env_api = {"belongs_to_which_room": lambda position:
                   "<Bedroom> (1000)"}
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

    def test_v35_shared_bed_is_stamped_and_local_bed_overrides_it(self):
        agent = self.make_agent()
        room = "<Bedroom> (1000)"
        agent.episode_provenance = "episode-current"
        agent.object_list = {0: [], 1: [], 2: []}
        agent.object_per_room = {room: {0: [], 1: [], 2: []}}
        agent.env_api = {"belongs_to_which_room": lambda position: room}
        agent.LLM.shared_delivery_target = lambda: {
            "id": 90,
            "name": "bed",
            "type": 2,
            "position": [4.0, 0.0, 7.0],
            "room": room,
        }

        agent._inject_v34_shared_bed()

        self.assertEqual(len(agent.object_list[2]), 1)
        injected = agent.object_list[2][0]
        self.assertEqual(injected["episode_provenance"],
                         "episode-current")
        self.assertEqual(injected["knowledge_source"],
                         "shared_memory_board")

        local_bed = {
            "id": 91,
            "name": "bed",
            "type": 2,
            "position": np.asarray([1.0, 0.0, 2.0]),
        }
        agent.object_list[2].append(local_bed)
        agent.object_per_room[room][2].append(local_bed)
        agent._inject_v34_shared_bed()

        self.assertEqual(agent.object_list[2], [local_bed])
        self.assertEqual(agent.object_per_room[room][2], [local_bed])

    def test_v35_rejects_provenance_bearing_stale_shared_bed(self):
        agent = self.make_agent()
        room = "<Bedroom> (1000)"
        agent.episode_provenance = "episode-current"
        agent.object_list = {0: [], 1: [], 2: []}
        agent.object_per_room = {room: {0: [], 1: [], 2: []}}
        agent.env_api = {"belongs_to_which_room": lambda position: room}
        agent.LLM.shared_delivery_target = lambda: {
            "id": 90,
            "name": "bed",
            "type": 2,
            "position": [4.0, 0.0, 7.0],
            "room": room,
            "episode_provenance": "episode-old",
        }

        agent._inject_v34_shared_bed()

        self.assertEqual(agent.object_list[2], [])
        self.assertEqual(agent.object_per_room[room][2], [])


if __name__ == "__main__":
    unittest.main()
