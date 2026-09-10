import json
import sys
import tempfile
import unittest
from pathlib import Path


TDW_MAT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TDW_MAT_ROOT / "tdw-gym"))

from peer_consult import (PROTOCOL_V4, TDWPeerConsultCoordinator,
                          TDWSharedBlackboard, _intent_from_plan)


def empty_hand():
    return {
        "id": None,
        "type": None,
        "name": None,
        "contained": [None, None, None],
        "contained_name": [None, None, None],
    }


def state(agent_id, *, held=None, visible=None, frame=10, action_id=0,
          action_status="success", valid=True, action_terminal=True,
          status=1):
    return {
        "agent": [float(agent_id * 4), 0.0, 0.0, 0.0, 0.0, 1.0],
        "held_objects": held or [empty_hand(), empty_hand()],
        "visible_objects": (visible or []) + [{"id": None}] * 8,
        "status": status,
        "valid": valid,
        "action_status": action_status,
        "action_id": action_id,
        "action_type": "0",
        "action_terminal": action_terminal,
        "action_completed_frame": frame if action_terminal else -1,
        "current_frames": frame,
    }


class FakeAgent:
    def __init__(self, plan=None, action=None):
        self.plan = plan
        self.next_action = action or {"type": 8}
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


class BoundaryAgent(FakeAgent):
    def __init__(self, plan=None, action=None):
        super().__init__(plan, action)
        self.action_history = []
        self.append_planning_boundary = False

    def act(self, state):
        if self.append_planning_boundary:
            self.action_history.append(
                f"{self.plan or 'idle'} at step {state['current_frames']}")
        return super().act(state)


class ConsumingBoundaryAgent(BoundaryAgent):
    def act(self, state):
        selected_plan = self.plan
        action = super().act(state)
        self._peer_consult_selected_plan = selected_plan
        self.plan = None
        return action


class PeerConsultV4LoopGuardTests(unittest.TestCase):
    def make_team(self, agents=None):
        agents = agents or [FakeAgent(), FakeAgent()]
        output = tempfile.TemporaryDirectory()
        self.addCleanup(output.cleanup)
        team = TDWPeerConsultCoordinator(
            agents, logger=None, output_dir=output.name,
            protocol_version=PROTOCOL_V4)
        team.reset({"apple": 1})
        return team

    @staticmethod
    def descriptor(task_id="entity:10", identity="action:test"):
        return {
            "task_id": task_id,
            "action_identity": identity,
            "stage": "acquire",
        }

    def test_two_no_progress_outcomes_arm_one_boundary_guard(self):
        team = self.make_team()
        task = team.board._ensure_task(
            "entity:10", "deliver_goal_object", 100,
            object_id=10, name="apple")
        task.update(status="in_progress", owner=0, last_owner=0)
        team.board.active_tasks[0] = "entity:10"
        descriptor = self.descriptor()

        team._v4_note_no_progress(0, descriptor, "review_replan", 0)
        self.assertNotIn(0, team.board.planning_loop_guards)
        team._v4_note_no_progress(0, descriptor, "review_replan", 0)

        guard = team.board.planning_loop_guards[0]
        self.assertEqual(guard["action_identity"], "action:test")
        self.assertTrue(guard["applies_to_next_planning_boundary"])
        self.assertEqual(team.board.tasks["entity:10"]["status"],
                         "suspended")
        self.assertIsNone(team.board.tasks["entity:10"]["blocked_until"])
        self.assertIsNone(team.board.active_tasks[0])
        self.assertTrue(any(
            event["event"] == "planning_loop_guard_armed"
            for event in team.board.coordination_events))

    def test_terminal_low_level_success_without_task_progress_counts(self):
        team = self.make_team()
        descriptor = self.descriptor()
        for index in (1, 2):
            pending = dict(descriptor)
            pending["progress_version_at_start"] = 0
            team.v4_pending_boundaries[0] = pending
            team.board.execution_evidence.append({
                "evidence_id": f"success-{index}",
                "agent": 0,
                "valid": True,
                "status": "success",
            })
            team._v4_consume_execution_evidence()

        self.assertEqual(
            team.v4_loop_state[0]["consecutive_no_progress"], 2)
        self.assertIn(0, team.board.planning_loop_guards)
        outcomes = [
            event.get("outcome") for event in
            team.board.coordination_events
            if event["event"] == "planning_replan"
        ]
        self.assertEqual(outcomes, [
            "execution_completed_without_task_progress",
            "execution_completed_without_task_progress",
        ])

    def test_task_progress_heartbeat_resets_repeat_history(self):
        team = self.make_team()
        descriptor = self.descriptor()
        team._v4_note_no_progress(0, descriptor, "execution_replan", 0)
        self.assertEqual(
            team.v4_loop_state[0]["consecutive_no_progress"], 1)

        pending = dict(descriptor)
        pending["progress_version_at_start"] = 0
        team.v4_pending_boundaries[0] = pending
        team.board.note_task_progress(
            "entity:10", 0, "physical_ownership_acquired")
        team.board.execution_evidence.append({
            "evidence_id": "progressed-success",
            "agent": 0,
            "valid": True,
            "status": "success",
        })
        team._v4_consume_execution_evidence()

        self.assertEqual(
            team.v4_loop_state[0]["consecutive_no_progress"], 0)
        self.assertNotIn(0, team.board.planning_loop_guards)

    def test_progress_between_failed_attempts_restarts_count(self):
        team = self.make_team()
        descriptor = self.descriptor()
        team._v4_note_no_progress(0, descriptor, "execution_replan", 0)
        team.board.note_task_progress(
            "entity:10", 0, "adapter_task_progress")
        team._v4_note_no_progress(0, descriptor, "execution_replan", 1)

        self.assertEqual(
            team.v4_loop_state[0]["consecutive_no_progress"], 1)
        self.assertEqual(
            team.v4_loop_state[0]["last_no_progress_version"], 1)
        self.assertNotIn(0, team.board.planning_loop_guards)

    def test_physical_evaluator_and_room_completion_emit_progress(self):
        agents = [FakeAgent(), FakeAgent()]
        board = TDWSharedBlackboard(
            {"apple": 1}, mechanism_v4=True, episode_epoch=1)
        apple = {"id": 10, "type": 0, "name": "apple"}
        held_apple = dict(apple, contained=[None, None, None],
                          contained_name=[None, None, None])
        board.observe({
            "0": state(0, held=[held_apple, empty_hand()], visible=[apple]),
            "1": state(1),
        }, agents)
        self.assertGreater(board.task_progress_versions["entity:10"], 0)

        before = board.task_progress_versions["entity:10"]
        board.observe({"0": state(0, frame=20),
                       "1": state(1, frame=20)}, agents,
                      delivered_objects={10: "apple"})
        self.assertGreater(board.task_progress_versions["entity:10"], before)

        agents[0].rooms_explored["<Kitchen> (1000)"] = "all"
        board.observe({"0": state(0, frame=30),
                       "1": state(1, frame=30)}, agents,
                      delivered_objects={10: "apple"})
        self.assertGreater(
            board.task_progress_versions["room:<Kitchen> (1000)"], 0)

    def test_guard_ignores_ongoing_and_is_consumed_by_next_boundary(self):
        team = self.make_team()
        same_intent = {
            "stage": "acquire", "target_id": 10,
            "plan": "go grasp target object <apple> (10)",
        }
        same_action = {"type": 0}
        descriptor = team._v4_boundary_descriptor(
            0, same_intent, same_action)
        team.board.planning_loop_guards[0] = {
            "task_id": descriptor["task_id"],
            "action_identity": descriptor["action_identity"],
        }

        actions = {"0": {"type": "ongoing"}, "1": {"type": 8}}
        intents = {0: same_intent,
                   1: {"stage": "idle", "target_id": None, "plan": None}}
        team.v4_planning_boundaries_this_step = {0, 1}
        team._v4_apply_pending_guards(actions, intents)
        self.assertEqual(actions["0"], {"type": "ongoing"})
        self.assertIn(0, team.board.planning_loop_guards)

        different = {
            "stage": "explore", "target_id": 2000,
            "plan": "go to <Office> (2000)",
        }
        actions = {"0": {"type": 0}, "1": {"type": 8}}
        team.v4_planning_boundaries_this_step = {0, 1}
        team._v4_apply_pending_guards(
            actions, {0: different, 1: intents[1]})
        self.assertEqual(actions["0"], {"type": 0})
        self.assertNotIn(0, team.board.planning_loop_guards)

        team.board.planning_loop_guards[0] = {
            "task_id": descriptor["task_id"],
            "action_identity": descriptor["action_identity"],
        }
        team._revised_this_step = set()
        actions = {"0": dict(same_action), "1": {"type": 8}}
        team.v4_planning_boundaries_this_step = {0, 1}
        team._v4_apply_pending_guards(actions, intents)
        self.assertEqual(actions["0"], {"type": 1})
        self.assertNotIn(0, team.board.planning_loop_guards)

        # It was a one-boundary guard, not a cooldown: the same proposal is
        # legal again at a later boundary.
        team._revised_this_step = set()
        actions = {"0": dict(same_action), "1": {"type": 8}}
        team.v4_planning_boundaries_this_step = {0, 1}
        team._v4_apply_pending_guards(actions, intents)
        self.assertEqual(actions["0"], same_action)

    def test_executor_steps_under_same_plan_are_not_boundaries(self):
        agent = BoundaryAgent(
            "go grasp target object <apple> (10)", {"type": 0})
        agents = [agent, BoundaryAgent(None, {"type": 8})]
        team = self.make_team(agents)
        apple = {"id": 10, "type": 0, "name": "apple"}
        initial = {"0": state(0, visible=[apple]), "1": state(1)}

        agent.append_planning_boundary = True
        team.act(initial)
        self.assertIn(0, team.v4_pending_boundaries)
        terminal = {
            "0": state(0, visible=[apple], frame=20, action_id=1),
            "1": state(1, frame=20),
        }
        team.observe_outcome(terminal, (0, 1, False), {})
        self.assertEqual(
            team.v4_loop_state[0]["consecutive_no_progress"], 1)

        # The original executor continues the same high-level plan but does
        # not append action_history, so these terminal low-level steps are not
        # planning attempts and cannot arm the guard.
        agent.append_planning_boundary = False
        team.act(terminal)
        self.assertNotIn(0, team.v4_pending_boundaries)
        self.assertEqual(
            team.v4_loop_state[0]["consecutive_no_progress"], 1)
        self.assertNotIn(0, team.board.planning_loop_guards)

    def test_two_real_action_history_replans_arm_guard(self):
        agent = BoundaryAgent(
            "go grasp target object <apple> (10)", {"type": 0})
        agents = [agent, BoundaryAgent(None, {"type": 8})]
        team = self.make_team(agents)
        apple = {"id": 10, "type": 0, "name": "apple"}

        agent.append_planning_boundary = True
        team.act({"0": state(0, visible=[apple]), "1": state(1)})
        team.observe_outcome({
            "0": state(0, visible=[apple], frame=20, action_id=1,
                       action_status="cannot_grasp", valid=False),
            "1": state(1, frame=20),
        }, (0, 1, False), {})

        # A genuine second LLM_plan append for the exact same task/action.
        agent.plan = "go grasp target object <apple> (10)"
        team.act({
            "0": state(0, visible=[apple], frame=30, action_id=1,
                       action_status="cannot_grasp", valid=False),
            "1": state(1, frame=30),
        })
        team.observe_outcome({
            "0": state(0, visible=[apple], frame=40, action_id=2,
                       action_status="cannot_grasp", valid=False),
            "1": state(1, frame=40),
        }, (0, 1, False), {})
        self.assertIn(0, team.board.planning_loop_guards)

    def test_consumed_plan_keeps_selected_boundary_identity(self):
        agent = ConsumingBoundaryAgent(
            "go grasp target object <apple> (10)", {"type": 3})
        agents = [agent, BoundaryAgent(None, {"type": 8})]
        team = self.make_team(agents)
        agent.append_planning_boundary = True
        apple = {"id": 10, "type": 0, "name": "apple"}

        team.act({"0": state(0, visible=[apple]), "1": state(1)})

        intent = team.board.current_intents[0]
        self.assertIsNone(agent.plan)
        self.assertEqual(intent["stage"], "acquire")
        self.assertEqual(intent["target_id"], 10)
        self.assertEqual(intent["plan"],
                         "go grasp target object <apple> (10)")
        self.assertEqual(
            team.v4_pending_boundaries[0]["task_id"], "entity:10")

    def test_untasked_guard_does_not_suspend_or_release_old_task(self):
        team = self.make_team()
        task = team.board._ensure_task(
            "entity:10", "deliver_goal_object", 100,
            object_id=10, name="apple")
        task.update(status="in_progress", owner=0, last_owner=0)
        team.board.active_tasks[0] = "entity:10"
        team.board.claim(10, 0, "accepted_acquire_intent")
        descriptor = self.descriptor(task_id=None,
                                     identity="action:message")

        team._v4_note_no_progress(0, descriptor, "review_replan", 0)
        team._v4_note_no_progress(0, descriptor, "review_replan", 0)
        self.assertEqual(team.board.active_tasks[0], "entity:10")
        self.assertIn(10, team.board.claims)

        team.board.current_intents = {
            0: {"stage": "communicate", "target_id": None,
                "plan": "send a message: retry"},
        }
        team.v4_planning_boundaries_this_step = {0}
        team._revised_this_step = set()
        actions = {"0": {"type": 6, "message": "retry"},
                   "1": {"type": 8}}
        exact = team._v4_boundary_descriptor(
            0, team.board.current_intents[0], actions["0"])
        team.board.planning_loop_guards[0].update({
            "task_id": exact["task_id"],
            "action_identity": exact["action_identity"],
        })
        team._v4_apply_pending_guards(
            actions, {0: team.board.current_intents[0]})
        self.assertEqual(team.board.active_tasks[0], "entity:10")
        self.assertIn(10, team.board.claims)

    def test_v4_task_switch_and_idle_release_abandoned_object_claim(self):
        board = TDWSharedBlackboard(
            {"apple": 2}, mechanism_v4=True, episode_epoch=1)
        for object_id in (10, 11):
            board.known_entities[object_id] = {
                "id": object_id, "type": 0, "name": "apple"}
            board._ensure_task(
                f"entity:{object_id}", "deliver_goal_object", 100,
                object_id=object_id, name="apple")
        board.tasks["entity:10"].update(
            status="in_progress", owner=0, last_owner=0)
        board.active_tasks[0] = "entity:10"
        board.claim(10, 0, "accepted_acquire_intent")

        board.sync_tasks({
            0: {"stage": "acquire", "target_id": 11,
                "plan": "go grasp target object <apple> (11)"},
        }, planning_boundary_agents={0})
        self.assertNotIn(10, board.claims)
        self.assertEqual(board.active_tasks[0], "entity:11")

        board.claim(11, 0, "accepted_acquire_intent")
        board.sync_tasks({
            0: {"stage": "idle", "target_id": None, "plan": None},
        }, planning_boundary_agents={0})
        self.assertNotIn(11, board.claims)
        self.assertIsNone(board.active_tasks[0])
        self.assertEqual(board.tasks["entity:11"]["status"], "suspended")

    def test_v4_abandoned_room_reservation_is_atomically_reassigned(self):
        team = self.make_team()
        team.board.claim_room(
            2000, 0, "go to <Office> (2000)")
        intents = {
            0: {"stage": "acquire", "target_id": 10,
                "plan": "go grasp target object <apple> (10)"},
            1: {"stage": "explore", "target_id": 2000,
                "plan": "go to <Office> (2000)"},
        }
        team.board.current_intents = intents
        team._revised_this_step = set()
        actions = {"0": {"type": 0}, "1": {"type": 0}}

        team._govern_room_reservations_v4(actions, intents)

        self.assertEqual(team.board.room_claims[2000]["agent"], 1)
        self.assertEqual(actions["1"], {"type": 0})
        self.assertTrue(any(
            event.get("reason") == "reservation_owner_abandoned_intent"
            for event in team.board.coordination_events))

    def test_v4_card_does_not_share_uncommitted_peer_discovery(self):
        agents = [FakeAgent(), FakeAgent()]
        team = self.make_team(agents)
        peer_only_apple = {
            "id": 10, "type": 0, "name": "apple",
            "position": [8.0, 0.0, 1.0],
        }
        states = {
            "0": state(0),
            "1": state(1, visible=[peer_only_apple]),
        }
        team.board.observe(states, agents)
        card = team.board.decision_view(
            0, states["0"], agents, team._v4_advisory(states["0"]))

        self.assertNotIn("actionable_targets", card)
        self.assertNotIn("actionable_containers", card)
        self.assertFalse(any(
            row.get("object_id") == 10 for row in card["task_queue"]))
        serialized = str(card)
        for forbidden in (
                "distance_m", "confidence", "estimated_delivery_frames",
                "container_used_slots", "bed_distance_m"):
            self.assertNotIn(forbidden, serialized)
        self.assertFalse(agents[0].stable_task_protocol)
        self.assertFalse(agents[0].coverage_exploration)

    def test_v4_factual_branch_does_not_call_policy_governors(self):
        agents = [
            FakeAgent("go grasp target object <apple> (10)", {"type": 0}),
            FakeAgent(None, {"type": 8}),
        ]
        team = self.make_team(agents)
        apple = {"id": 10, "type": 0, "name": "apple"}
        states = {"0": state(0, visible=[apple]), "1": state(1)}

        def forbidden(*_args, **_kwargs):
            raise AssertionError("V4 called a policy-heavy governor")

        for name in (
                "_delivery_advisory", "_prepare_transport_decision",
                "_communication_events",
                "_govern_goal_type_quotas", "_govern_late_acquisitions",
                "_govern_navigation", "_govern_delivery_commit"):
            setattr(team, name, forbidden)

        actions = team.act(states)
        self.assertEqual(actions["0"], {"type": 0})

    def test_v4_invalid_message_is_blocked_without_grounded_rewrite(self):
        team = self.make_team()
        team.board.known_entities[10] = {
            "id": 10, "type": 0, "name": "apple"}
        team.board.physical_owners[10] = {
            "agent": 1, "carrier": "hand", "arm": "left"}

        def forbidden(*_args, **_kwargs):
            raise AssertionError("V4 synthesized a grounded message")

        team._grounded_message = forbidden
        actions = {
            "0": {"type": 6, "message": "I am holding apple (10)."},
            "1": {"type": 8},
        }
        team._reconcile_messages_v4(actions)
        self.assertEqual(actions["0"], {"type": 8, "delay": 1})
        self.assertNotIn("message", actions["0"])

    def test_v4_raw_geometry_message_is_never_forwarded(self):
        team = self.make_team()
        actions = {
            "0": {
                "type": 6,
                "message": ("Object 10 is at [8.2, 0.0, 1.4], distance "
                            "3.1, confidence 0.92."),
            },
            "1": {"type": 8},
        }

        team._reconcile_messages_v4(actions)

        self.assertEqual(actions["0"], {"type": 8, "delay": 1})
        self.assertTrue(any(
            review.get("trigger") == "message_event_reference"
            for review in team.board.reviews))

    def test_v4_selected_public_event_is_canonically_rendered(self):
        team = self.make_team()
        event = team.board.emit_coordination_event(
            "task_progress", 0, "entity:10",
            outcome="physical_ownership_acquired", progress_version=1)
        actions = {
            "0": {
                "type": 6,
                "message": f"coordination_event:{event['event_id']}",
            },
            "1": {"type": 8},
        }

        team._reconcile_messages_v4(actions)

        self.assertEqual(actions["0"]["type"], 6)
        payload = json.loads(actions["0"]["message"])
        self.assertEqual(payload, {
            "coordination_event": {
                "event_id": event["event_id"],
                "event": "task_progress",
                "agent": 0,
                "task_id": "entity:10",
                "outcome": "physical_ownership_acquired",
                "progress_version": 1,
            },
        })
        serialized = actions["0"]["message"]
        for forbidden in ("position", "distance", "confidence"):
            self.assertNotIn(forbidden, serialized)

    def test_v4_drop_ticket_is_arm_scoped_and_terminal_failure_clears(self):
        team = self.make_team()
        apple = {
            "id": 10, "type": 0, "name": "apple",
            "contained": [None, None, None],
            "contained_name": [None, None, None],
        }
        banana = dict(apple, id=11, name="banana")
        team.board.goal_ledger.required["banana"] = 1
        held_states = {
            "0": state(0, held=[apple, banana]),
            "1": state(1),
        }
        team.board.observe(held_states, team.agents)
        team._record_v4_delivery_attempts({
            "0": {"type": 5, "arm": "left"}, "1": {"type": 8}})
        self.assertEqual(
            team.pending_delivery_attempts[0]["payload_ids"], [10])

        # Terminal but still owned means factual release failure; the ticket
        # is consumed rather than leaking into a later drop.
        team._reconcile_v4_delivery_attempts(held_states)
        self.assertNotIn(0, team.pending_delivery_attempts)
        self.assertTrue(any(
            event.get("outcome") == "physical_release_failed"
            for event in team.board.coordination_events))

        basket = {
            "id": 20, "type": 1, "name": "basket",
            "contained": [12, None, None],
            "contained_name": ["apple", None, None],
        }
        team.board.observe({
            "0": state(0, held=[basket, banana], frame=30),
            "1": state(1, frame=30),
        }, team.agents)
        team._record_v4_delivery_attempts({
            "0": {"type": 5, "arm": "left"}, "1": {"type": 8}})
        self.assertEqual(
            team.pending_delivery_attempts[0]["payload_ids"], [12])

    def test_coordination_events_are_structured_and_observation_free(self):
        board = TDWSharedBlackboard(
            {"apple": 1}, mechanism_v4=True, episode_epoch=1)
        board.frame = 12
        board.known_entities[10] = {
            "id": 10, "type": 0, "name": "apple",
            "position": [99.0, 0.0, 99.0],
        }
        board.claim(10, 0, "accepted_acquire_intent")
        board.release_agent_claims(0, reason="planner_release")

        self.assertEqual(
            [event["event"] for event in board.coordination_events],
            ["claim_acquired", "claim_released"])
        serialized = str(board.coordination_events)
        self.assertNotIn("position", serialized)
        self.assertNotIn("confidence", serialized)
        self.assertNotIn("distance", serialized)

    def test_v4_episode_log_has_reproducible_run_provenance(self):
        output = tempfile.TemporaryDirectory()
        self.addCleanup(output.cleanup)
        team = TDWPeerConsultCoordinator(
            [FakeAgent(), FakeAgent()], logger=None,
            output_dir=output.name, protocol_version=PROTOCOL_V4)

        team.reset({"apple": 1}, episode_id=7)

        root_log = Path(output.name) / "peer_consult.jsonl"
        episode_log = Path(output.name) / "7" / "peer_consult.jsonl"
        root_record = json.loads(root_log.read_text(
            encoding="utf-8").splitlines()[-1])
        episode_record = json.loads(episode_log.read_text(
            encoding="utf-8").splitlines()[-1])

        self.assertEqual(root_record, episode_record)
        self.assertEqual(root_record["kind"], "reset")
        self.assertEqual(root_record["schema_version"],
                         "peer_consult_v4.0")
        self.assertEqual(root_record["protocol"], PROTOCOL_V4)
        self.assertEqual(root_record["episode_id"], 7)
        self.assertEqual(root_record["episode_epoch"], 1)
        self.assertEqual(root_record["protocol_step"], 0)
        self.assertTrue(root_record["run_instance_id"])


if __name__ == "__main__":
    unittest.main()
