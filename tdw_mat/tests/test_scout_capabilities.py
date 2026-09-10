import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


TDW_MAT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TDW_MAT_ROOT))
sys.path.insert(0, str(TDW_MAT_ROOT / "tdw-gym"))

from LLM.LLM import (LLM, action_allowed_for_role,
                     plan_allowed_for_role)
from lm_agent import _resolve_team_roles
from scout_agent import scout_agent


def empty_hand():
    return {
        "id": None,
        "type": None,
        "name": None,
        "contained": [None, None, None],
        "contained_name": [None, None, None],
    }


class ScoutCapabilityTests(unittest.TestCase):
    def make_planner(self, role):
        planner = LLM.__new__(LLM)
        planner.agent_role = role
        planner.communication = True
        planner.holding_objects = [empty_hand(), empty_hand()]
        planner.object_list = {
            0: [{"name": "apple", "id": 10, "type": 0}],
            1: [{"name": "bowl", "id": 11, "type": 1}],
            2: [{"name": "bed", "id": 12, "type": 2}],
        }
        planner.rooms = ["Kitchen", "Bedroom"]
        planner.current_room = "Kitchen"
        planner.rooms_explored = {"Kitchen": "part"}
        return planner

    def make_runnable_planner(self, first_output):
        planner = self.make_planner("human")
        planner.allow_message_this_turn = True
        planner.peer_consult_protocol = "PeerConsultV3.3"
        planner.peer_decision_card = {
            "task_queue": [{
                "task_id": "entity:10",
                "kind": "deliver_goal_object",
                "status": "pending",
                "object_id": 10,
            }],
            "actionable_containers": [],
            "delivery_advisory": {
                "communication_events": ["new_target_evidence"],
            },
        }
        planner.prompt_template = (
            "$GOAL$ $PROGRESS$ $ACTION_HISTORY$ $DIALOGUE_HISTORY$ "
            "$AVAILABLE_ACTIONS$")
        planner.generator_prompt_template = (
            "$GOAL$ $PROGRESS$ $ACTION_HISTORY$ $DIALOGUE_HISTORY$")
        planner.goal_desc = "Transport one apple."
        planner.agent_name = "Alice"
        planner.opponent_role = "human"
        planner.fix_lm_communication = True
        planner.cot = False
        planner.chat = False
        planner.sampling_params = {}
        planner.total_cost = 0
        planner.debug = False
        planner.single = False
        planner.progress2text = lambda *args: "progress"
        calls = []

        def generator(prompt, _sampling):
            calls.append(prompt)
            output = (first_output if len(calls) == 1 else
                      "New apple (10) found in Kitchen.")
            return [output], 0

        planner.generator = generator
        return planner, calls

    def test_scout_plan_candidates_have_no_manipulation(self):
        planner = self.make_planner("scout")
        _, _, plans = planner.get_available_plans("Apple found in Kitchen")

        self.assertIn("send a message: Apple found in Kitchen", plans)
        self.assertIn("go to Bedroom", plans)
        self.assertIn("explore current room Kitchen", plans)
        self.assertIn("wait", plans)
        self.assertTrue(all(plan_allowed_for_role("scout", p) for p in plans))
        self.assertFalse(any("grasp" in p for p in plans))
        self.assertFalse(any(p.startswith("put") for p in plans))
        self.assertFalse(any(p.startswith("transport") for p in plans))

    def test_human_keeps_original_manipulation_candidates(self):
        planner = self.make_planner("human")
        _, _, plans = planner.get_available_plans(None)

        self.assertIn("go grasp target object <apple> (10)", plans)
        self.assertIn("go grasp container <bowl> (11)", plans)
        self.assertNotIn("wait", plans)

    def test_v33_planner_decides_before_message_is_generated(self):
        planner = self.make_planner("human")
        planner.peer_decision_card = {
            "task_queue": [{
                "task_id": "entity:10",
                "kind": "deliver_goal_object",
                "status": "pending",
                "object_id": 10,
            }],
            "actionable_containers": [],
        }
        _, _, plans = planner.get_available_plans(include_message=True)

        self.assertEqual(plans[0], "send a message")
        self.assertIn("go grasp target object <apple> (10)", plans)
        self.assertNotIn("go to Bedroom", plans)
        self.assertNotIn("explore current room Kitchen", plans)
        self.assertFalse(any(plan.startswith("send a message:")
                             for plan in plans))

    def test_v33_physical_choice_does_not_call_message_generator(self):
        planner, calls = self.make_runnable_planner(
            "go grasp target object <apple> (10)")
        plan, info = planner.run(
            10, "Kitchen", {"Kitchen": "part"},
            [empty_hand(), empty_hand()], [], planner.object_list, {},
            ["go to Kitchen at initial step"], [], [], None)

        self.assertEqual(plan,
                         "go grasp target object <apple> (10)")
        self.assertEqual(len(calls), 1)
        self.assertNotIn("prompt_comm", info)

    def test_v33_message_content_is_generated_after_message_choice(self):
        planner, calls = self.make_runnable_planner("send a message")
        plan, info = planner.run(
            10, "Kitchen", {"Kitchen": "part"},
            [empty_hand(), empty_hand()], [], planner.object_list, {},
            ["go to Kitchen at initial step"], [], [], None)

        self.assertTrue(plan.startswith("send a message:"))
        self.assertEqual(len(calls), 2)
        self.assertIn("new_target_evidence", info["prompt_comm"])

    def test_v34_shared_bed_keeps_transport_live_without_private_bed(self):
        planner = self.make_planner("human")
        apple = {"name": "apple", "id": 10, "type": 0,
                 "contained": [None, None, None],
                 "contained_name": [None, None, None]}
        planner.holding_objects = [apple, empty_hand()]
        planner.object_list[2] = []
        planner.peer_decision_card = {
            "self": {"payload": [{"id": 10, "name": "apple"}]},
            "peer": {"payload": []},
            "task_queue": [{
                "task_id": "entity:10",
                "kind": "deliver_goal_object",
                "status": "carried",
                "object_id": 10,
            }],
            "actionable_containers": [],
            "delivery_advisory": {
                "bed_object_id": 99,
                "bed_position": [4.0, 0.0, 7.0],
                "bed_knowledge_source": "shared_memory_board",
            },
        }

        _, count, plans = planner.get_available_plans()

        self.assertGreater(count, 0)
        self.assertIn("transport objects I'm holding to the bed", plans)
        self.assertNotIn(
            "go grasp target object <apple> (10)", plans)

    def test_v34_full_payload_without_bed_never_has_empty_action_space(self):
        planner = self.make_planner("human")
        apple = {"name": "apple", "id": 10, "type": 0,
                 "contained": [None, None, None],
                 "contained_name": [None, None, None]}
        banana = {"name": "banana", "id": 13, "type": 0,
                  "contained": [None, None, None],
                  "contained_name": [None, None, None]}
        planner.holding_objects = [apple, banana]
        planner.object_list[2] = []
        planner.peer_decision_card = {
            "self": {"payload": [
                {"id": 10, "name": "apple"},
                {"id": 13, "name": "banana"},
            ]},
            "peer": {"payload": []},
            "task_queue": [],
            "actionable_containers": [],
            "delivery_advisory": {},
        }

        _, count, plans = planner.get_available_plans()

        self.assertGreater(count, 0)
        self.assertTrue(any(
            plan.startswith("go to ") or plan.startswith("explore ")
            for plan in plans))

    def test_scout_action_filter_rejects_manipulation_primitives(self):
        for action_type in (0, 1, 2, 6, 8, "ongoing"):
            self.assertTrue(action_allowed_for_role(
                "scout", {"type": action_type}))
        for action_type in (3, 4, 5):
            self.assertFalse(action_allowed_for_role(
                "scout", {"type": action_type}))

    def test_roles_are_resolved_from_embodiments(self):
        args = SimpleNamespace(embodiments=["replicant", "box"])
        self.assertEqual(
            _resolve_team_roles(args, 0), ("human", "scout"))
        self.assertEqual(
            _resolve_team_roles(args, 1), ("scout", "human"))
        self.assertEqual(
            _resolve_team_roles(SimpleNamespace(), 1), ("human", "human"))

    def test_scout_methods_refuse_direct_manipulation_calls(self):
        agent = scout_agent.__new__(scout_agent)
        with self.assertRaises(PermissionError):
            agent.gograsp()
        with self.assertRaises(PermissionError):
            agent.putin()
        with self.assertRaises(PermissionError):
            agent.goput()

    def test_role_prompt_is_a_two_row_communication_template(self):
        prompt_path = TDW_MAT_ROOT / "LLM" / "prompt_human_scout.csv"
        prompts = pd.read_csv(prompt_path)["prompt"].tolist()

        self.assertEqual(len(prompts), 2)
        for prompt in prompts:
            self.assertIn("$AGENT_ROLE$", prompt)
            self.assertIn("$OPPO_ROLE$", prompt)
            self.assertIn("Box Scout", prompt)
            self.assertIn("no hands or gripper", prompt)
        self.assertIn("$AVAILABLE_ACTIONS$", prompts[0])
        self.assertIn("$DIALOGUE_HISTORY$", prompts[1])

    def test_role_prompt_makes_scout_factual_and_human_delivery_first(self):
        prompt_path = TDW_MAT_ROOT / "LLM" / "prompt_human_scout.csv"
        planner_prompt, message_prompt = pd.read_csv(prompt_path)["prompt"].tolist()

        self.assertIn("factual reports, not instructions", planner_prompt)
        self.assertIn("both hand slots are occupied", planner_prompt)
        self.assertIn("If transport is available, choose it immediately", planner_prompt)
        self.assertIn("go to that reported room to confirm the bed", planner_prompt)
        self.assertIn("explore the current room only to find and confirm the bed",
                      planner_prompt)
        self.assertIn("resume that delivery route", planner_prompt)
        self.assertIn("Do not recommend, advise, prioritize, assign", planner_prompt)
        self.assertIn("repeat the bed's room in every factual report", planner_prompt)

        self.assertIn("final planning decision", message_prompt)
        self.assertIn("report only directly observed", message_prompt)
        self.assertIn("Include every newly observed or corrected target",
                      message_prompt)
        self.assertIn("do not omit a new target", message_prompt)
        self.assertIn("Do not recommend, advise, prioritize, assign", message_prompt)
        self.assertIn("include the bed's room in every message", message_prompt)
        self.assertIn("do not ask the Scout for advice or priorities", message_prompt)

        for prompt in (planner_prompt, message_prompt):
            self.assertNotIn("gives advice", prompt)
            self.assertNotIn("suggest where you should search next", prompt)


if __name__ == "__main__":
    unittest.main()
