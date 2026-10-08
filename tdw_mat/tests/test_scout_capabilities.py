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
