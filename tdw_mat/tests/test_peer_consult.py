import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


TDW_MAT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TDW_MAT_ROOT / "tdw-gym"))

from peer_consult import (PROTOCOL_V31, PROTOCOL_V32, PROTOCOL_V33,
                          PROTOCOL_V34, PROTOCOL_V35,
                          TDWPeerConsultCoordinator,
                          TDWSharedBlackboard)


def empty_hand():
    return {
        "id": None,
        "type": None,
        "name": None,
        "contained": [None, None, None],
        "contained_name": [None, None, None],
    }


def state(agent_id, *, held=None, visible=None, status=1, valid=True,
          frame=10, action_id=0, action_terminal=True):
    return {
        "agent": [float(agent_id * 4), 0.0, 0.0, 0.0, 0.0, 1.0],
        "held_objects": held or [empty_hand(), empty_hand()],
        "visible_objects": (visible or []) + [{"id": None}] * 8,
        "status": status,
        "valid": valid,
        "action_status": "success" if valid else "cannot_grasp",
        "action_id": action_id,
        "action_type": "3",
        "action_terminal": action_terminal,
        "action_completed_frame": frame if action_terminal else -1,
        "current_frames": frame,
    }


class FakeAgent:
    def __init__(self, plan, action):
        self.plan = plan
        self.next_action = action
        self.satisfied = []
        self.dialogue_history = []
        self.object_list = {0: [], 1: [], 2: []}
        self.object_per_room = {}
        self.current_room = "<Kitchen> (1000)"
        self.rooms_name = ["<Kitchen> (1000)", "<Office> (2000)"]
        self.rooms_explored = {}
        self.target_pos = None
        self.fix_lm_satisfied = False

    def act(self, _state):
        return dict(self.next_action)


class GateAwareAgent(FakeAgent):
    def __init__(self, plan, action):
        super().__init__(plan, action)
        self.LLM = SimpleNamespace(
            communication=True,
            allow_message_this_turn=True,
        )
        self.observed_message_gates = []

    def act(self, state):
        self.observed_message_gates.append(
            self.LLM.allow_message_this_turn)
        return super().act(state)


class PeerConsultTests(unittest.TestCase):
    def test_nested_container_ownership_preserves_right_arm(self):
        agents = [FakeAgent(None, {"type": 8}),
                  FakeAgent(None, {"type": 8})]
        board = TDWSharedBlackboard({"apple": 1})
        basket = {
            "id": 20,
            "type": 1,
            "name": "basket",
            "contained": [10, None, None],
            "contained_name": ["apple", None, None],
        }
        states = {
            "0": state(0, held=[empty_hand(), basket]),
            "1": state(1),
        }
        board.observe(states, agents)
        self.assertEqual(board.physical_owners[20]["arm"], "right")
        self.assertEqual(board.physical_owners[10]["carrier"], "container")
        self.assertEqual(board.physical_owners[10]["container_id"], 20)
        self.assertIn(0, board.payload_states)

    def test_duplicate_object_claim_is_resolved_before_execution(self):
        agents = [
            FakeAgent("go grasp target object <apple> (10)", {"type": 0}),
            FakeAgent("go grasp target object <apple> (10)", {"type": 0}),
        ]
        apple = {"id": 10, "type": 0, "name": "apple"}
        states = {
            "0": state(0, visible=[apple]),
            "1": state(1, visible=[apple]),
        }
        with tempfile.TemporaryDirectory() as output_dir:
            team = TDWPeerConsultCoordinator(
                agents, logger=None, output_dir=output_dir)
            team.reset({"apple": 1})
            actions = team.act(states)
        self.assertEqual(team.board.claims[10]["agent"], 0)
        self.assertEqual(actions["0"], {"type": 0})
        self.assertEqual(actions["1"], {"type": 2})
        self.assertIsNone(agents[1].plan)
        self.assertEqual(team.board.reviews[-1]["trigger"],
                         "duplicate_claim")

    def test_ongoing_action_is_not_replaced(self):
        agents = [
            FakeAgent("go grasp target object <apple> (10)", {"type": 0}),
            FakeAgent("go grasp target object <apple> (10)",
                      {"type": "ongoing"}),
        ]
        apple = {"id": 10, "type": 0, "name": "apple"}
        states = {
            "0": state(0, visible=[apple]),
            "1": state(1, visible=[apple], status=0,
                       action_terminal=False),
        }
        with tempfile.TemporaryDirectory() as output_dir:
            team = TDWPeerConsultCoordinator(
                agents, logger=None, output_dir=output_dir)
            team.reset({"apple": 1})
            actions = team.act(states)
        self.assertEqual(actions["1"], {"type": "ongoing"})

    def test_full_payload_clears_stale_acquisition_without_contract(self):
        agents = [FakeAgent("go grasp target object <banana> (12)",
                            {"type": 0}),
                  FakeAgent(None, {"type": 8})]
        agents[0].object_list[2] = [{"id": 99, "type": 2,
                                    "name": "bed"}]
        agents[0].target_pos = [7.0, 0.0, -3.0]
        apple = {
            "id": 10,
            "type": 0,
            "name": "apple",
            "contained": [None, None, None],
            "contained_name": [None, None, None],
        }
        banana = {
            "id": 11,
            "type": 0,
            "name": "banana",
            "contained": [None, None, None],
            "contained_name": [None, None, None],
        }
        states = {"0": state(0, held=[apple, banana]),
                  "1": state(1)}
        with tempfile.TemporaryDirectory() as output_dir:
            team = TDWPeerConsultCoordinator(
                agents, logger=None, output_dir=output_dir)
            team.reset({"apple": 1})
            team.act(states)
        self.assertIsNone(agents[0].plan)
        self.assertIsNone(agents[0].target_pos)
        self.assertEqual(team.board.payload_states[0]["kind"],
                         "active_payload")

    def test_partial_container_keeps_collecting(self):
        agents = [FakeAgent("go grasp target object <apple> (10)",
                            {"type": 0}),
                  FakeAgent(None, {"type": 8})]
        agents[0].object_list[2] = [{"id": 99, "type": 2,
                                    "name": "bed"}]
        basket = {
            "id": 20,
            "type": 1,
            "name": "basket",
            "contained": [11, None, None],
            "contained_name": ["banana", None, None],
        }
        apple = {"id": 10, "type": 0, "name": "apple"}
        states = {
            "0": state(0, held=[basket, empty_hand()], visible=[apple]),
            "1": state(1),
        }
        with tempfile.TemporaryDirectory() as output_dir:
            team = TDWPeerConsultCoordinator(
                agents, logger=None, output_dir=output_dir)
            team.reset({"apple": 1, "banana": 1})
            actions = team.act(states)
        self.assertEqual(actions["0"], {"type": 0})
        self.assertEqual(agents[0].plan,
                         "go grasp target object <apple> (10)")

    def test_partial_payload_is_delivered_when_deadline_risk_dominates(self):
        agents = [FakeAgent("go grasp target object <apple> (10)",
                            {"type": 0}),
                  FakeAgent(None, {"type": 8})]
        agents[0].object_list[2] = [{"id": 99, "type": 2,
                                    "name": "bed",
                                    "position": [0.0, 0.0, 1.0]}]
        basket = {
            "id": 20,
            "type": 1,
            "name": "basket",
            "contained": [11, None, None],
            "contained_name": ["banana", None, None],
        }
        states = {
            "0": state(0, held=[basket, empty_hand()], frame=2850),
            "1": state(1, frame=2850),
        }
        with tempfile.TemporaryDirectory() as output_dir:
            team = TDWPeerConsultCoordinator(
                agents, logger=None, output_dir=output_dir,
                max_frames=3000)
            team.reset({"apple": 1, "banana": 1})
            team.act(states)
        self.assertEqual(agents[0].plan,
                         "transport objects I'm holding to the bed")

    def test_target_and_container_trigger_fresh_transport_decision(self):
        agents = [FakeAgent("go grasp target object <banana> (11)",
                            {"type": 0}),
                  FakeAgent(None, {"type": 8})]
        basket = {
            "id": 20,
            "type": 1,
            "name": "basket",
            "contained": [None, None, None],
            "contained_name": [None, None, None],
        }
        apple = {
            "id": 10,
            "type": 0,
            "name": "apple",
            "contained": [None, None, None],
            "contained_name": [None, None, None],
        }
        states = {"0": state(0, held=[basket, apple]),
                  "1": state(1)}
        with tempfile.TemporaryDirectory() as output_dir:
            team = TDWPeerConsultCoordinator(
                agents, logger=None, output_dir=output_dir)
            team.reset({"apple": 1, "banana": 1})
            team.act(states)
        self.assertIsNone(agents[0].plan)

    def test_container_resource_is_not_rejected_by_goal_quota(self):
        agents = [
            FakeAgent("go grasp container <tea_tray> (20)", {"type": 0}),
            FakeAgent(None, {"type": 8}),
        ]
        tray = {"id": 20, "type": 1, "name": "tea_tray",
                "position": [1.0, 0.0, 0.0]}
        states = {"0": state(0, visible=[tray]), "1": state(1)}
        with tempfile.TemporaryDirectory() as output_dir:
            team = TDWPeerConsultCoordinator(
                agents, logger=None, output_dir=output_dir)
            team.reset({"apple": 1})
            actions = team.act(states)
        self.assertEqual(actions["0"], {"type": 0})
        self.assertEqual(team.board.claims[20]["kind"],
                         "container_resource")
        self.assertFalse(any(review.get("trigger") == "goal_quota"
                             for review in team.board.reviews))

    def test_decision_card_exposes_container_opportunity(self):
        agents = [FakeAgent(None, {"type": 8}),
                  FakeAgent(None, {"type": 8})]
        tray = {"id": 20, "type": 1, "name": "tea_tray",
                "position": [1.0, 0.0, 0.0]}
        apple = {"id": 10, "type": 0, "name": "apple",
                 "position": [2.0, 0.0, 0.0]}
        agents[0].object_per_room = {
            "<Kitchen> (1000)": {0: [apple], 1: [tray], 2: []},
        }
        states = {"0": state(0, visible=[tray, apple]), "1": state(1)}
        with tempfile.TemporaryDirectory() as output_dir:
            team = TDWPeerConsultCoordinator(
                agents, logger=None, output_dir=output_dir)
            team.reset({"apple": 1})
            team.act(states)
        card = team.board.decision_view(
            0, states["0"], agents,
            team._delivery_advisory(0, states["0"]))
        self.assertEqual(card["actionable_containers"][0]["id"], 20)
        self.assertEqual(
            card["actionable_containers"][0]["same_room_useful_targets"],
            1)

    def test_duplicate_room_assignment_is_replanned(self):
        agents = [FakeAgent("go to <Kitchen> (1000)", {"type": 0}),
                  FakeAgent("go to <Kitchen> (1000)", {"type": 0})]
        states = {"0": state(0), "1": state(1)}
        with tempfile.TemporaryDirectory() as output_dir:
            team = TDWPeerConsultCoordinator(
                agents, logger=None, output_dir=output_dir)
            team.reset({"apple": 1})
            actions = team.act(states)
        self.assertEqual(team.board.room_claims[1000]["agent"], 0)
        self.assertEqual(actions["1"], {"type": 2})
        self.assertEqual(team.board.reviews[-1]["trigger"],
                         "room_assignment")

    def test_v31_room_assignment_requires_persistent_conflict(self):
        agents = [FakeAgent("go to <Kitchen> (1000)", {"type": 0}),
                  FakeAgent("go to <Kitchen> (1000)", {"type": 0})]
        with tempfile.TemporaryDirectory() as output_dir:
            team = TDWPeerConsultCoordinator(
                agents, logger=None, output_dir=output_dir,
                protocol_version=PROTOCOL_V31)
            team.reset({"apple": 1})
            first = team.act({"0": state(0, frame=10),
                              "1": state(1, frame=10)})
            second = team.act({"0": state(0, frame=130),
                               "1": state(1, frame=130)})
        self.assertEqual(first["1"], {"type": 0})
        self.assertEqual(second["1"], {"type": 2})
        self.assertTrue(any(
            review["trigger"] == "room_assignment"
            for review in team.board.reviews))

    def test_same_room_is_allowed_for_distinct_useful_targets(self):
        agents = [FakeAgent("go to <Kitchen> (1000)", {"type": 0}),
                  FakeAgent("go to <Kitchen> (1000)", {"type": 0})]
        apple = {"id": 10, "type": 0, "name": "apple",
                 "room": "<Kitchen> (1000)"}
        banana = {"id": 11, "type": 0, "name": "banana",
                  "room": "<Kitchen> (1000)"}
        agents[0].object_per_room = {
            "<Kitchen> (1000)": {0: [apple, banana]},
        }
        states = {"0": state(0), "1": state(1)}
        with tempfile.TemporaryDirectory() as output_dir:
            team = TDWPeerConsultCoordinator(
                agents, logger=None, output_dir=output_dir)
            team.reset({"apple": 1, "banana": 1})
            actions = team.act(states)
        self.assertEqual(actions["0"], {"type": 0})
        self.assertEqual(actions["1"], {"type": 0})
        self.assertNotIn(1000, team.board.room_claims)

    def test_remaining_type_quota_counts_physical_payload(self):
        agents = [FakeAgent(None, {"type": 8}),
                  FakeAgent("go grasp target object <apple> (11)",
                            {"type": 0})]
        held_apple = {
            "id": 10, "type": 0, "name": "apple",
            "contained": [None, None, None],
            "contained_name": [None, None, None],
        }
        other_apple = {"id": 11, "type": 0, "name": "apple",
                       "position": [5.0, 0.0, 0.0]}
        states = {
            "0": state(0, held=[held_apple, empty_hand()]),
            "1": state(1, visible=[other_apple]),
        }
        with tempfile.TemporaryDirectory() as output_dir:
            team = TDWPeerConsultCoordinator(
                agents, logger=None, output_dir=output_dir)
            team.reset({"apple": 1})
            actions = team.act(states)
        self.assertEqual(actions["1"], {"type": 2})
        self.assertEqual(team.board.reviews[-1]["trigger"], "goal_quota")

    def test_navigation_stall_clears_cached_plan(self):
        agents = [FakeAgent("go to <Kitchen> (1000)", {"type": 0}),
                  FakeAgent(None, {"type": 8})]
        with tempfile.TemporaryDirectory() as output_dir:
            team = TDWPeerConsultCoordinator(
                agents, logger=None, output_dir=output_dir)
            team.reset({"apple": 1})
            team.act({"0": state(0, frame=10), "1": state(1, frame=10)})
            actions = team.act({"0": state(0, frame=140),
                                "1": state(1, frame=140)})
        self.assertEqual(actions["0"], {"type": 1})
        self.assertIsNone(agents[0].plan)
        self.assertEqual(team.board.reviews[-1]["trigger"],
                         "navigation_recovery")

    def test_ongoing_acquisition_keeps_target_claim(self):
        agents = [
            FakeAgent("go grasp target object <apple> (10)",
                      {"type": "ongoing"}),
            FakeAgent(None, {"type": 8}),
        ]
        apple = {"id": 10, "type": 0, "name": "apple"}
        states = {
            "0": state(0, visible=[apple], status=0,
                       action_terminal=False),
            "1": state(1),
        }
        with tempfile.TemporaryDirectory() as output_dir:
            team = TDWPeerConsultCoordinator(
                agents, logger=None, output_dir=output_dir)
            team.reset({"apple": 1})
            team.act(states)
        self.assertEqual(team.board.current_intents[0]["stage"], "acquire")
        self.assertEqual(team.board.current_intents[0]["execution"],
                         "ongoing")
        self.assertEqual(team.board.claims[10]["agent"], 0)

    def test_cooled_target_switches_to_local_alternative(self):
        agents = [
            FakeAgent("go grasp target object <apple> (10)", {"type": 0}),
            FakeAgent(None, {"type": 8}),
        ]
        apple = {"id": 10, "type": 0, "name": "apple",
                 "position": [4.0, 0.0, 0.0]}
        banana = {"id": 11, "type": 0, "name": "banana",
                  "position": [1.0, 0.0, 0.0]}
        agents[0].object_list[0] = [apple, banana]
        agents[0].object_per_room = {
            "<Kitchen> (1000)": {0: [apple, banana], 1: [], 2: []},
        }
        states = {
            "0": state(0, visible=[apple, banana], frame=20),
            "1": state(1, frame=20),
        }
        with tempfile.TemporaryDirectory() as output_dir:
            team = TDWPeerConsultCoordinator(
                agents, logger=None, output_dir=output_dir)
            team.reset({"apple": 1, "banana": 1})
            team.board.frame = 10
            team.board.cooldown_target(0, 10, 360, "test stall")
            actions = team.act(states)
        self.assertEqual(actions["0"], {"type": 1})
        self.assertEqual(
            agents[0].plan,
            "go grasp target object <banana> (11)")
        self.assertEqual(team.board.reviews[-1]["trigger"],
                         "recovery_cooldown")
        self.assertIn("0:10", team.board.public_view()[
            "recovery_cooldowns"])

    def test_false_ownership_message_is_reconciled(self):
        agents = [FakeAgent(None, {
            "type": 6,
            "message": "I'm holding <basket> (20); put it inside the plate.",
        }), FakeAgent(None, {"type": 8})]
        basket = {
            "id": 20,
            "type": 1,
            "name": "basket",
            "contained": [None, None, None],
            "contained_name": [None, None, None],
        }
        states = {"0": state(0),
                  "1": state(1, held=[basket, empty_hand()])}
        with tempfile.TemporaryDirectory() as output_dir:
            team = TDWPeerConsultCoordinator(
                agents, logger=None, output_dir=output_dir)
            team.reset({"apple": 1})
            actions = team.act(states)
        self.assertTrue(actions["0"]["message"].startswith(
            "Grounded update:"))
        self.assertEqual(team.board.reviews[-1]["trigger"],
                         "message_reconciliation")

    def test_v31_routine_message_is_suppressed_without_an_event(self):
        agents = [FakeAgent(None, {
            "type": 6,
            "message": "I am still exploring the Kitchen.",
        }), FakeAgent(None, {"type": 8})]
        with tempfile.TemporaryDirectory() as output_dir:
            team = TDWPeerConsultCoordinator(
                agents, logger=None, output_dir=output_dir,
                protocol_version=PROTOCOL_V31)
            team.reset({"apple": 1})
            actions = team.act({"0": state(0), "1": state(1)})
        self.assertEqual(actions["0"], {"type": 8, "delay": 1})
        self.assertEqual(team.board.reviews[-1]["trigger"],
                         "communication_gate")

    def test_v31_closes_only_outbound_slot_and_keeps_dialogue_enabled(self):
        agents = [GateAwareAgent(None, {"type": 8}),
                  GateAwareAgent(None, {"type": 8})]
        with tempfile.TemporaryDirectory() as output_dir:
            team = TDWPeerConsultCoordinator(
                agents, logger=None, output_dir=output_dir,
                protocol_version=PROTOCOL_V31)
            team.reset({"apple": 1})
            team.act({"0": state(0), "1": state(1)})
        self.assertEqual(agents[0].observed_message_gates, [False])
        self.assertTrue(agents[0].LLM.allow_message_this_turn)
        self.assertTrue(agents[0].LLM.communication)

    def test_v31_new_target_event_opens_one_message_slot(self):
        agents = [FakeAgent(None, {"type": 8}),
                  FakeAgent(None, {"type": 8})]
        apple = {"id": 10, "type": 0, "name": "apple"}
        with tempfile.TemporaryDirectory() as output_dir:
            team = TDWPeerConsultCoordinator(
                agents, logger=None, output_dir=output_dir,
                protocol_version=PROTOCOL_V31)
            team.reset({"apple": 1})
            team.act({"0": state(0, frame=10),
                      "1": state(1, frame=10)})
            agents[0].next_action = {
                "type": 6,
                "message": "New apple found in the Kitchen.",
            }
            actions = team.act({
                "0": state(0, visible=[apple], frame=40),
                "1": state(1, frame=40),
            })
        self.assertEqual(actions["0"]["type"], 6)
        self.assertIn("new_target_evidence",
                      team._communication_events_this_step[0])

    def test_v31_delivery_priority_banks_payload_before_hard_deadline(self):
        agents = [FakeAgent("go grasp target object <apple> (10)",
                            {"type": 0}),
                  FakeAgent(None, {"type": 8})]
        agents[0].object_list[2] = [{
            "id": 99, "type": 2, "name": "bed",
            "position": [0.0, 0.0, 1.0],
        }]
        basket = {
            "id": 20,
            "type": 1,
            "name": "basket",
            "contained": [11, None, None],
            "contained_name": ["banana", None, None],
        }
        states = {
            "0": state(0, held=[basket, empty_hand()], frame=1950),
            "1": state(1, frame=1950),
        }
        with tempfile.TemporaryDirectory() as output_dir:
            team = TDWPeerConsultCoordinator(
                agents, logger=None, output_dir=output_dir,
                max_frames=3000, protocol_version=PROTOCOL_V31)
            team.reset({"apple": 1, "banana": 1})
            team.act(states)
            advisory = team._delivery_advisory(0, states["0"])
        self.assertFalse(advisory["force_delivery"])
        self.assertTrue(advisory["priority_delivery"])
        self.assertEqual(agents[0].plan,
                         "transport objects I'm holding to the bed")

    def test_v32_reconciles_planner_satisfied_with_evaluator(self):
        agents = [FakeAgent(None, {"type": 8}),
                  FakeAgent(None, {"type": 8})]
        agents[0].satisfied = [10, 20]
        agents[1].satisfied = [20]
        with tempfile.TemporaryDirectory() as output_dir:
            team = TDWPeerConsultCoordinator(
                agents, logger=None, output_dir=output_dir,
                protocol_version=PROTOCOL_V32)
            team.reset({"apple": 1})
            team.act(
                {"0": state(0), "1": state(1)},
                delivered_objects={10: "apple"})
        self.assertEqual(agents[0].satisfied, [10])
        self.assertEqual(agents[1].satisfied, [10])
        self.assertTrue(agents[0].authoritative_satisfied)
        self.assertEqual(
            team.board.reviews[-1]["trigger"],
            "delivery_reconciliation")

    def test_v32_rejects_pickup_that_cannot_be_banked(self):
        apple = {
            "id": 10, "type": 0, "name": "apple",
            "position": [10.0, 0.0, 0.0],
        }
        agents = [FakeAgent("go grasp target object <apple> (10)",
                            {"type": 0}),
                  FakeAgent(None, {"type": 8})]
        agents[0].current_room = "<Bedroom> (1000)"
        agents[0].object_list[0] = [apple]
        agents[0].object_list[2] = [{
            "id": 99, "type": 2, "name": "bed",
            "position": [0.0, 0.0, 1.0],
        }]
        with tempfile.TemporaryDirectory() as output_dir:
            team = TDWPeerConsultCoordinator(
                agents, logger=None, output_dir=output_dir,
                max_frames=3000, protocol_version=PROTOCOL_V32)
            team.reset({"apple": 1})
            actions = team.act({
                "0": state(0, visible=[apple], frame=2800),
                "1": state(1, frame=2800),
            })
        self.assertEqual(actions["0"], {"type": 1})
        self.assertIsNone(agents[0].plan)
        self.assertEqual(team.board.reviews[-1]["trigger"],
                         "late_acquisition")

    def test_v32_commits_priority_payload_inside_drop_range(self):
        apple = {
            "id": 10, "type": 0, "name": "apple",
            "contained": [None, None, None],
            "contained_name": [None, None, None],
        }
        agents = [FakeAgent("transport objects I'm holding to the bed",
                            {"type": 1}),
                  FakeAgent(None, {"type": 8})]
        agents[0].current_room = "<Bedroom> (1000)"
        agents[0].object_list[2] = [{
            "id": 99, "type": 2, "name": "bed",
            "position": [0.0, 0.0, 1.0],
        }]
        with tempfile.TemporaryDirectory() as output_dir:
            team = TDWPeerConsultCoordinator(
                agents, logger=None, output_dir=output_dir,
                max_frames=3000, protocol_version=PROTOCOL_V32)
            team.reset({"apple": 1})
            actions = team.act({
                "0": state(0, held=[apple, empty_hand()], frame=2500),
                "1": state(1, frame=2500),
            })
        self.assertEqual(actions["0"], {"type": 5, "arm": "left"})
        self.assertEqual(team.board.reviews[-1]["trigger"],
                         "delivery_commit")

    def test_v33_keeps_interrupted_goal_task_suspended(self):
        apple = {"id": 10, "type": 0, "name": "apple",
                 "position": [1.0, 0.0, 0.0]}
        banana = {"id": 11, "type": 0, "name": "banana",
                  "position": [2.0, 0.0, 0.0]}
        agents = [FakeAgent(
            "go grasp target object <apple> (10)", {"type": 0}),
            FakeAgent(None, {"type": 8})]
        with tempfile.TemporaryDirectory() as output_dir:
            team = TDWPeerConsultCoordinator(
                agents, logger=None, output_dir=output_dir,
                protocol_version=PROTOCOL_V33)
            team.reset({"apple": 1, "banana": 1})
            team.act({
                "0": state(0, visible=[apple, banana], frame=10),
                "1": state(1, frame=10),
            })
            agents[0].plan = "go grasp target object <banana> (11)"
            team.act({
                "0": state(0, visible=[apple, banana], frame=30),
                "1": state(1, frame=30),
            })
        self.assertEqual(
            team.board.tasks["entity:10"]["status"], "suspended")
        self.assertEqual(team.board.active_tasks[0], "entity:11")
        self.assertIn(
            "entity:10",
            [row["task_id"] for row in
             team.board.task_queue(0, state(0, frame=30))])

    def test_v33_review_does_not_hard_code_an_alternative(self):
        apple = {"id": 10, "type": 0, "name": "apple",
                 "position": [4.0, 0.0, 0.0]}
        banana = {"id": 11, "type": 0, "name": "banana",
                  "position": [1.0, 0.0, 0.0]}
        agents = [FakeAgent(
            "go grasp target object <apple> (10)", {"type": 0}),
            FakeAgent(None, {"type": 8})]
        agents[0].object_list[0] = [apple, banana]
        with tempfile.TemporaryDirectory() as output_dir:
            team = TDWPeerConsultCoordinator(
                agents, logger=None, output_dir=output_dir,
                protocol_version=PROTOCOL_V33)
            team.reset({"apple": 1, "banana": 1})
            team.board.frame = 10
            team.board.cooldown_target(0, 10, 360, "test stall")
            actions = team.act({
                "0": state(0, visible=[apple, banana], frame=20),
                "1": state(1, frame=20),
            })
        self.assertEqual(actions["0"], {"type": 1})
        self.assertIsNone(agents[0].plan)
        self.assertIsNone(team.board.active_tasks[0])
        self.assertEqual(team.board.tasks["entity:10"]["status"],
                         "blocked")

    def test_v33_late_target_rule_does_not_reject_container(self):
        tray = {"id": 20, "type": 1, "name": "tea_tray",
                "position": [8.0, 0.0, 0.0]}
        agents = [FakeAgent(
            "go grasp container <tea_tray> (20)", {"type": 0}),
            FakeAgent(None, {"type": 8})]
        agents[0].object_list[1] = [tray]
        agents[0].object_list[2] = [{
            "id": 99, "type": 2, "name": "bed",
            "position": [0.0, 0.0, 1.0],
        }]
        with tempfile.TemporaryDirectory() as output_dir:
            team = TDWPeerConsultCoordinator(
                agents, logger=None, output_dir=output_dir,
                max_frames=3000, protocol_version=PROTOCOL_V33)
            team.reset({"apple": 1})
            actions = team.act({
                "0": state(0, visible=[tray], frame=2900),
                "1": state(1, frame=2900),
            })
        self.assertEqual(actions["0"], {"type": 0})
        self.assertFalse(any(
            review["trigger"] == "late_acquisition"
            for review in team.board.reviews))

    def test_v35_inherits_target_only_late_acquisition_governance(self):
        apple = {
            "id": 10, "type": 0, "name": "apple",
            "position": [10.0, 0.0, 0.0],
        }
        tray = {
            "id": 20, "type": 1, "name": "tea_tray",
            "position": [8.0, 0.0, 0.0],
        }
        for entity, plan, expect_rejected in (
                (apple, "go grasp target object <apple> (10)", True),
                (tray, "go grasp container <tea_tray> (20)", False)):
            with self.subTest(entity=entity["name"]):
                agents = [FakeAgent(plan, {"type": 0}),
                          FakeAgent(None, {"type": 8})]
                agents[0].current_room = "<Bedroom> (1000)"
                with tempfile.TemporaryDirectory() as output_dir:
                    team = TDWPeerConsultCoordinator(
                        agents, logger=None, output_dir=output_dir,
                        max_frames=3000, protocol_version=PROTOCOL_V35)
                    team.reset({"apple": 1})
                    agents[0].object_list[entity["type"]] = [entity]
                    agents[0].object_list[2] = [{
                        "id": 99, "type": 2, "name": "bed",
                        "position": [0.0, 0.0, 1.0],
                    }]
                    team.act({
                        "0": state(0, visible=[entity], frame=2900),
                        "1": state(1, frame=2900),
                    })
                rejected = any(
                    review["trigger"] == "late_acquisition"
                    for review in team.board.reviews)
                self.assertEqual(rejected, expect_rejected)

    def test_v33_does_not_force_drop_at_v32_distance(self):
        apple = {
            "id": 10, "type": 0, "name": "apple",
            "contained": [None, None, None],
            "contained_name": [None, None, None],
        }
        agents = [FakeAgent("transport objects I'm holding to the bed",
                            {"type": 1}),
                  FakeAgent(None, {"type": 8})]
        agents[0].current_room = "<Bedroom> (1000)"
        agents[0].object_list[2] = [{
            "id": 99, "type": 2, "name": "bed",
            "position": [0.0, 0.0, 1.0],
        }]
        with tempfile.TemporaryDirectory() as output_dir:
            team = TDWPeerConsultCoordinator(
                agents, logger=None, output_dir=output_dir,
                max_frames=3000, protocol_version=PROTOCOL_V33)
            team.reset({"apple": 1})
            actions = team.act({
                "0": state(0, held=[apple, empty_hand()], frame=2500),
                "1": state(1, frame=2500),
            })
        self.assertEqual(actions["0"], {"type": 1})
        self.assertFalse(any(
            review["trigger"] == "delivery_commit"
            for review in team.board.reviews))

    def test_v33_uncredited_drop_returns_object_task_to_queue(self):
        apple = {
            "id": 10, "type": 0, "name": "apple",
            "contained": [None, None, None],
            "contained_name": [None, None, None],
        }
        agents = [FakeAgent("transport objects I'm holding to the bed",
                            {"type": 5, "arm": "left"}),
                  FakeAgent(None, {"type": 8})]
        agents[0].object_list[2] = [{
            "id": 99, "type": 2, "name": "bed",
            "position": [0.0, 0.0, 1.0],
        }]
        with tempfile.TemporaryDirectory() as output_dir:
            team = TDWPeerConsultCoordinator(
                agents, logger=None, output_dir=output_dir,
                protocol_version=PROTOCOL_V33)
            team.reset({"apple": 1})
            team.act({
                "0": state(0, held=[apple, empty_hand()], frame=100),
                "1": state(1, frame=100),
            })
            outcome0 = state(0, frame=120)
            outcome0["action_type"] = "5"
            team.observe_outcome(
                {"0": outcome0, "1": state(1, frame=120)},
                (0, 1, False), {}, delivered_objects={})
        self.assertEqual(team.board.tasks["entity:10"]["status"],
                         "pending")
        self.assertEqual(team.board.delivery_failure_counts[10], 1)
        self.assertEqual(team.board.reviews[-1]["trigger"],
                         "delivery_confirmation")

    def test_v34_delivered_child_cannot_resurrect_with_parked_container(self):
        apple = {
            "id": 10, "type": 0, "name": "apple",
            "contained": [None, None, None],
            "contained_name": [None, None, None],
        }
        tray = {
            "id": 20, "type": 1, "name": "tea_tray",
            "contained": [10, None, None],
            "contained_name": ["apple", None, None],
        }
        agents = [FakeAgent(None, {"type": 8}),
                  FakeAgent(None, {"type": 8})]
        board = TDWSharedBlackboard(
            {"apple": 1}, semantic_state_v34=True)

        board.observe({
            "0": state(0, held=[tray, empty_hand()], frame=100),
            "1": state(1, frame=100),
        }, agents)
        board.observe({
            "0": state(0, frame=120),
            "1": state(1, frame=120),
        }, agents, delivered_objects={10: "apple"})
        # Re-grasping the physical tray can make TDW report the credited child
        # again.  Evaluator truth must remain terminal in planner state.
        board.observe({
            "0": state(0, held=[tray, empty_hand()], frame=140),
            "1": state(1, frame=140),
        }, agents, delivered_objects={10: "apple"})

        self.assertTrue(board.container_parked_at_bed(20))
        self.assertNotIn(10, board.physical_owners)
        self.assertEqual(board.container_contents[20], [])
        self.assertEqual(board._payload_view(0), [{
            "id": 20,
            "name": "tea_tray",
            "carrier": "hand",
            "container_id": None,
        }])
        card = board.decision_view(
            1, state(1, frame=140), agents,
            {"remaining_frames": 2860})
        self.assertNotIn(
            20, [item["id"] for item in card["actionable_containers"]])

    def test_v34_shared_bed_is_available_to_agent_without_private_bed(self):
        agents = [FakeAgent(None, {"type": 8}),
                  FakeAgent(None, {"type": 8})]
        bed = {
            "id": 99, "type": 2, "name": "bed",
            "position": [4.0, 0.0, 7.0],
        }
        agents[1].object_list[2] = [bed]
        with tempfile.TemporaryDirectory() as output_dir:
            team = TDWPeerConsultCoordinator(
                agents, logger=None, output_dir=output_dir,
                protocol_version=PROTOCOL_V34)
            team.reset({"apple": 1})
            team.board.observe({
                "0": state(0, frame=100),
                "1": state(1, visible=[bed], frame=100),
            }, agents)
            advisory = team._delivery_advisory(
                0, state(0, frame=100))

        self.assertTrue(advisory["bed_known"])
        self.assertEqual(advisory["bed_object_id"], 99)
        self.assertEqual(advisory["bed_position"], [4.0, 0.0, 7.0])
        self.assertEqual(
            advisory["bed_knowledge_source"], "shared_memory_board")

    def test_v34_rejects_regrasp_of_own_contained_payload(self):
        tray = {
            "id": 20, "type": 1, "name": "tea_tray",
            "contained": [10, None, None],
            "contained_name": ["apple", None, None],
        }
        agents = [FakeAgent(
            "go grasp target object <apple> (10)", {"type": 0}),
            FakeAgent(None, {"type": 8})]
        with tempfile.TemporaryDirectory() as output_dir:
            team = TDWPeerConsultCoordinator(
                agents, logger=None, output_dir=output_dir,
                protocol_version=PROTOCOL_V34)
            team.reset({"apple": 1})
            actions = team.act({
                "0": state(0, held=[tray, empty_hand()], frame=100),
                "1": state(1, frame=100),
            })

        self.assertEqual(actions["0"], {"type": 1})
        self.assertIsNone(agents[0].plan)
        self.assertEqual(team.board.reviews[-1]["trigger"], "ownership")

    def test_v35_reset_isolates_episode_entity_caches(self):
        stale_bed = {
            "id": 99, "type": 2, "name": "bed",
            "position": [9.0, 0.0, 9.0],
        }
        agents = [FakeAgent(None, {"type": 8}),
                  FakeAgent(None, {"type": 8})]
        agents[0].object_list[2] = [stale_bed]
        agents[0].object_per_room = {
            "<Office> (2000)": {0: [], 1: [], 2: [stale_bed]},
        }
        agents[0].target_pos = [9.0, 0.0, 9.0]
        with tempfile.TemporaryDirectory() as output_dir:
            team = TDWPeerConsultCoordinator(
                agents, logger=None, output_dir=output_dir,
                protocol_version=PROTOCOL_V35)
            team.reset({"apple": 1})
            self.assertEqual(agents[0].object_per_room, {})
            self.assertEqual(agents[0].object_list[2], [])
            self.assertIsNone(agents[0].target_pos)
            team.act({"0": state(0, frame=0),
                      "1": state(1, frame=0)})

        self.assertEqual(team.episode_epoch, 1)
        self.assertEqual(team.board.episode_epoch, 1)
        self.assertNotIn(99, team.board.known_entities)

    def test_v35_drop_ticket_is_arm_scoped_and_gets_confirmation_grace(self):
        apple = {
            "id": 10, "type": 0, "name": "apple",
            "contained": [None, None, None],
            "contained_name": [None, None, None],
        }
        orange = {
            "id": 11, "type": 0, "name": "orange",
            "contained": [None, None, None],
            "contained_name": [None, None, None],
        }
        tray = {
            "id": 20, "type": 1, "name": "tea_tray",
            "contained": [10, None, None],
            "contained_name": ["apple", None, None],
        }
        bed = {
            "id": 99, "type": 2, "name": "bed",
            "position": [0.0, 0.0, 1.0],
        }
        agents = [FakeAgent("transport objects I'm holding to the bed",
                            {"type": 5, "arm": "left"}),
                  FakeAgent(None, {"type": 8})]
        with tempfile.TemporaryDirectory() as output_dir:
            team = TDWPeerConsultCoordinator(
                agents, logger=None, output_dir=output_dir,
                protocol_version=PROTOCOL_V35)
            team.reset({"apple": 1, "orange": 1})
            agents[0].object_list[2] = [bed]
            team.act({
                "0": state(0, held=[tray, orange], visible=[bed], frame=100),
                "1": state(1, frame=100),
            })
            attempt = team.pending_delivery_attempts[0]
            self.assertEqual(attempt["payload_ids"], [10])
            self.assertEqual(attempt["container_ids"], [20])
            self.assertNotIn(11, attempt["payload_ids"])

            outcome0 = state(0, held=[empty_hand(), orange], frame=120)
            outcome0["action_type"] = "5"
            team.observe_outcome(
                {"0": outcome0, "1": state(1, frame=120)},
                (0, 2, False), {}, delivered_objects={})

            self.assertEqual(
                team.pending_delivery_attempts[0]["phase"],
                "confirmation_grace")
            self.assertEqual(team.board.delivery_failure_counts[10], 0)
            self.assertTrue(team.board.container_delivery_pending(20))
            self.assertEqual(
                team.board.tasks["entity:10"]["status"], "blocked")

            # Evaluator credit arriving during the settling window closes the
            # ticket without recording a false failure or reviving the tray.
            later0 = state(0, held=[empty_hand(), orange], frame=140)
            later0["action_type"] = "8"
            team.observe_outcome(
                {"0": later0, "1": state(1, frame=140)},
                (1, 2, False), {}, delivered_objects={10: "apple"})

        self.assertNotIn(0, team.pending_delivery_attempts)
        self.assertEqual(team.board.delivery_failure_counts[10], 0)
        self.assertTrue(team.board.container_parked_at_bed(20))

    def test_v35_expired_confirmation_cools_target_and_invalidates_bed(self):
        apple = {
            "id": 10, "type": 0, "name": "apple",
            "contained": [None, None, None],
            "contained_name": [None, None, None],
        }
        stale_bed = {
            "id": 99, "type": 2, "name": "bed",
            "position": [9.0, 0.0, 9.0],
        }
        agents = [FakeAgent("transport objects I'm holding to the bed",
                            {"type": 5, "arm": "left"}),
                  FakeAgent(None, {"type": 8})]
        agents[0].object_info = {99: dict(stale_bed)}
        with tempfile.TemporaryDirectory() as output_dir:
            team = TDWPeerConsultCoordinator(
                agents, logger=None, output_dir=output_dir,
                protocol_version=PROTOCOL_V35)
            team.reset({"apple": 1})
            agents[0].object_list[2] = [stale_bed]
            agents[0].object_info = {99: dict(stale_bed)}
            team.act({
                "0": state(0, held=[apple, empty_hand()], frame=100),
                "1": state(1, frame=100),
            })
            terminal0 = state(0, frame=120)
            terminal0["action_type"] = "5"
            team.observe_outcome(
                {"0": terminal0, "1": state(1, frame=120)},
                (0, 1, False), {}, delivered_objects={})

            expired0 = state(0, frame=181)
            expired0["action_type"] = "8"
            team.observe_outcome(
                {"0": expired0, "1": state(1, frame=181)},
                (0, 1, False), {}, delivered_objects={})

        self.assertNotIn(0, team.pending_delivery_attempts)
        self.assertEqual(team.board.delivery_failure_counts[10], 1)
        self.assertEqual(
            team.invalid_bed_hypotheses[0]["bed_object_id"], 99)
        self.assertTrue(team.board.target_on_cooldown(0, 10))
        self.assertTrue(team.board.target_on_cooldown(1, 10))
        self.assertEqual(team.board.tasks["entity:10"]["status"], "blocked")
        self.assertNotIn(99, agents[0].object_info)
        self.assertEqual(agents[0].object_list[2], [])
        self.assertTrue(any(
            review["trigger"] == "delivery_confirmation" and
            review["verdict"] == "revise"
            for review in team.board.reviews))

    def test_memory_board_publishes_symbolic_room_and_position(self):
        agents = [FakeAgent(None, {"type": 8}),
                  FakeAgent(None, {"type": 8})]
        agents[0].object_per_room = {
            "<Kitchen> (1000)": {
                0: [{"id": 10, "type": 0, "name": "apple",
                     "position": [1.0, 0.0, 2.0]}],
            },
        }
        board = TDWSharedBlackboard({"apple": 1})
        board.observe({"0": state(0), "1": state(1)}, agents)
        entity = board.public_view()["known_entities"][10]
        self.assertEqual(entity["room"], "<Kitchen> (1000)")
        self.assertEqual(entity["position"], [1.0, 0.0, 2.0])

    def test_decision_card_is_compact_and_keeps_agent_dialogue(self):
        agents = [FakeAgent(None, {"type": 8}),
                  FakeAgent(None, {"type": 8})]
        targets = [
            {"id": index, "type": 0, "name": "apple",
             "position": [float(index), 0.0, 0.0]}
            for index in range(10, 22)
        ]
        agents[0].object_per_room = {
            "<Kitchen> (1000)": {0: targets},
        }
        states = {"0": state(0), "1": state(1)}
        states["0"]["messages"] = [
            None, "Can you check the Office for the remaining apple?"]
        states["1"]["messages"] = states["0"]["messages"]
        with tempfile.TemporaryDirectory() as output_dir:
            team = TDWPeerConsultCoordinator(
                agents, logger=None, output_dir=output_dir)
            team.reset({"apple": 3})
            team.act(states)
        card = team.board.decision_view(
            0, states["0"], agents,
            team._delivery_advisory(0, states["0"]))
        self.assertLessEqual(len(card["actionable_targets"]), 5)
        self.assertIn("check the Office",
                      card["recent_agent_dialogue"][-1]["message"])
        prompt = team.board.prompt_summary(
            0, states["0"], agents,
            team._delivery_advisory(0, states["0"]))
        self.assertNotIn('"known_entities"', prompt)
        self.assertNotIn("keep collecting while", prompt)

    def test_room_summary_exposes_team_coverage_without_a_room_contract(self):
        agents = [FakeAgent(None, {"type": 8}),
                  FakeAgent(None, {"type": 8})]
        agents[0].rooms_explored["<Kitchen> (1000)"] = "all"
        agents[1].current_room = "<Office> (2000)"
        states = {"0": state(0), "1": state(1)}
        with tempfile.TemporaryDirectory() as output_dir:
            team = TDWPeerConsultCoordinator(
                agents, logger=None, output_dir=output_dir)
            team.reset({"apple": 1})
            team.act(states)
        card = team.board.decision_view(
            0, states["0"], agents,
            team._delivery_advisory(0, states["0"]))
        rooms = {item["room"]: item for item in card["room_summary"]}
        self.assertEqual(rooms["<Kitchen> (1000)"]["coverage"], "all")
        self.assertEqual(rooms["<Office> (2000)"]["current_agents"], [1])
        self.assertNotIn("obligations", team.board.public_view())

    def test_terminal_action_ticket_becomes_execution_evidence(self):
        agents = [FakeAgent(None, {"type": 8}),
                  FakeAgent(None, {"type": 8})]
        board = TDWSharedBlackboard({"apple": 1})
        states = {
            "0": state(0, action_id=3, action_terminal=True, frame=42),
            "1": state(1),
        }
        board.observe(states, agents)
        board.observe(states, agents)
        self.assertEqual(len(board.execution_evidence), 1)
        self.assertEqual(board.execution_evidence[0]["action_id"], 3)

    def test_delivery_ledger_ignores_agent_self_report(self):
        agents = [FakeAgent(None, {"type": 8}),
                  FakeAgent(None, {"type": 8})]
        agents[0].satisfied = [10]
        board = TDWSharedBlackboard({"apple": 1})
        apple = {"id": 10, "type": 0, "name": "apple"}
        states = {"0": state(0, visible=[apple]), "1": state(1)}

        board.observe(states, agents)
        self.assertEqual(board.goal_ledger.delivered_counts()["apple"], 0)

        board.observe(states, agents, delivered_objects={10: "apple"})
        self.assertEqual(board.goal_ledger.delivered_counts()["apple"], 1)

    def test_coordinator_enables_corrected_lm_bookkeeping(self):
        agents = [FakeAgent(None, {"type": 8}),
                  FakeAgent(None, {"type": 8})]
        with tempfile.TemporaryDirectory() as output_dir:
            TDWPeerConsultCoordinator(
                agents, logger=None, output_dir=output_dir)
        self.assertTrue(agents[0].fix_lm_satisfied)
        self.assertTrue(agents[1].fix_lm_satisfied)


if __name__ == "__main__":
    unittest.main()
