import sys
import unittest
from pathlib import Path


TDW_MAT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TDW_MAT_ROOT))

from LLM.LLM import LLM, _v4_coordination_event_reference


def empty_hand():
    return {
        "id": None,
        "type": None,
        "name": None,
        "contained": [None, None, None],
        "contained_name": [None, None, None],
    }


class V4CandidateCompletenessTests(unittest.TestCase):
    def make_planner(self, protocol="PeerConsultV4"):
        planner = LLM.__new__(LLM)
        planner.agent_role = "human"
        planner.peer_consult_protocol = protocol
        planner.communication = False
        planner.holding_objects = [empty_hand(), empty_hand()]
        # The active target intentionally appears second so continuity is
        # observable without adding any policy score.
        planner.object_list = {
            0: [
                {"name": "banana", "id": 11, "type": 0},
                {"name": "apple", "id": 10, "type": 0},
            ],
            1: [],
            2: [{"name": "bed", "id": 12, "type": 2}],
        }
        planner.rooms = ["Kitchen", "Bedroom"]
        planner.current_room = "Kitchen"
        planner.rooms_explored = {"Kitchen": "part"}
        planner.peer_decision_card = {
            "task_queue": [
                {
                    "task_id": "entity:11",
                    "kind": "deliver_goal_object",
                    "status": "pending",
                    "object_id": 11,
                },
                {
                    "task_id": "entity:10",
                    "kind": "deliver_goal_object",
                    "status": "in_progress",
                    "object_id": 10,
                },
            ],
            "actionable_containers": [],
        }
        return planner

    def make_runnable_planner(self):
        planner = self.make_planner()
        planner.communication = True
        planner.allow_message_this_turn = True
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
            return ["go grasp target object <apple> (10)"], 0

        planner.generator = generator
        return planner, calls

    def test_v4_known_target_does_not_hide_legal_exploration(self):
        planner = self.make_planner()

        _, _, plans = planner.get_available_plans()

        self.assertIn("go grasp target object <apple> (10)", plans)
        self.assertIn("go to Bedroom", plans)
        self.assertIn("explore current room Kitchen", plans)

    def test_v4_shared_card_is_not_a_private_object_whitelist(self):
        planner = self.make_planner()
        planner.peer_decision_card["task_queue"] = [
            planner.peer_decision_card["task_queue"][0]]
        planner.object_list[1] = [
            {"name": "bowl", "id": 20, "type": 1}]

        _, _, plans = planner.get_available_plans()

        # Apple and bowl are private local observations absent from the shared
        # card.  V4 keeps them local and selectable; publication is separate.
        self.assertIn("go grasp target object <apple> (10)", plans)
        self.assertIn("go grasp container <bowl> (20)", plans)

    def test_v4_message_is_a_planning_candidate_not_a_prefetch_call(self):
        planner, calls = self.make_runnable_planner()

        plan, info = planner.run(
            10, "Kitchen", {"Kitchen": "part"},
            [empty_hand(), empty_hand()], [], planner.object_list, {},
            ["go to Kitchen at initial step"], [], [], None)

        self.assertEqual(plan,
                         "go grasp target object <apple> (10)")
        self.assertEqual(len(calls), 1)
        self.assertNotIn("prompt_comm", info)

    def test_v4_message_generation_accepts_only_public_event_reference(self):
        self.assertEqual(
            _v4_coordination_event_reference(
                "coordination_event:7", [6, 7]),
            "coordination_event:7")
        self.assertIsNone(_v4_coordination_event_reference(
            "The apple is 2.1 meters away at [1, 0, 4].", [7]))
        self.assertIsNone(_v4_coordination_event_reference(
            "coordination_event:99", [7]))

    def test_v35_action_space_is_unchanged(self):
        planner = self.make_planner("PeerConsultV3.5")

        _, _, plans = planner.get_available_plans()

        self.assertIn("go grasp target object <apple> (10)", plans)
        self.assertNotIn("go to Bedroom", plans)
        self.assertNotIn("explore current room Kitchen", plans)

    def test_v35_card_remains_a_shared_object_whitelist(self):
        planner = self.make_planner("PeerConsultV3.5")
        planner.peer_decision_card["task_queue"] = [
            planner.peer_decision_card["task_queue"][0]]

        _, _, plans = planner.get_available_plans()

        self.assertIn("go grasp target object <banana> (11)", plans)
        self.assertNotIn("go grasp target object <apple> (10)", plans)

    def test_active_task_gets_non_binding_continuity_order(self):
        planner = self.make_planner()
        planner.peer_decision_card["active_task"] = {
            "task_id": "entity:10",
            "kind": "deliver_goal_object",
            "status": "in_progress",
            "object_id": 10,
        }

        _, _, plans = planner.get_available_plans()

        self.assertEqual(plans[0],
                         "go grasp target object <apple> (10)")
        self.assertIn("go grasp target object <banana> (11)", plans)
        self.assertIn("go to Bedroom", plans)

    def test_suspended_task_persists_without_continuity_bonus(self):
        planner = self.make_planner()
        planner.peer_decision_card["active_task"] = {
            "task_id": "entity:10",
            "kind": "deliver_goal_object",
            "status": "suspended",
            "object_id": 10,
        }

        _, _, plans = planner.get_available_plans()

        self.assertEqual(plans[0],
                         "go grasp target object <banana> (11)")
        self.assertIn("go grasp target object <apple> (10)", plans)

    def test_loop_guard_removes_only_continuity_not_candidate(self):
        planner = self.make_planner()
        planner.peer_decision_card["active_task"] = {
            "task_id": "entity:10",
            "kind": "deliver_goal_object",
            "status": "in_progress",
            "object_id": 10,
        }
        planner.peer_decision_card["planning_loop_guard"] = {
            "task_id": "entity:10",
            "action_identity": "go grasp target object <apple> (10)",
            "reason": "repeated_replan_without_progress",
            "applies_to_next_planning_boundary": True,
        }

        _, _, plans = planner.get_available_plans()

        self.assertEqual(plans[0],
                         "go grasp target object <banana> (11)")
        self.assertIn("go grasp target object <apple> (10)", plans)
        self.assertIn("go to Bedroom", plans)

    def test_active_exploration_can_keep_continuity_with_targets_visible(self):
        planner = self.make_planner()
        planner.peer_decision_card["active_task"] = {
            "task_id": "room:Kitchen",
            "kind": "explore_room",
            "status": "in_progress",
            "room": "Kitchen",
        }

        _, _, plans = planner.get_available_plans()

        self.assertEqual(plans[0], "explore current room Kitchen")
        self.assertIn("go grasp target object <apple> (10)", plans)
        self.assertIn("go to Bedroom", plans)


if __name__ == "__main__":
    unittest.main()
