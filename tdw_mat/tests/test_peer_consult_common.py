"""Simulator-free integration tests of the common TDW transaction boundary."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tdw-gym"))
from peer_consult_common import CommonTDWPeerConsultCoordinator
from test_peer_consult_v4_loop_guard import BoundaryAgent, state


class CommonTDWTests(unittest.TestCase):
    def make_team(self, plans, actions=None):
        agents = [BoundaryAgent(plan, action) for plan, action in zip(
            plans, actions or [{"type": 0}, {"type": 0}])]
        for agent in agents:
            agent.append_planning_boundary = True
        output = tempfile.TemporaryDirectory()
        self.addCleanup(output.cleanup)
        team = CommonTDWPeerConsultCoordinator(agents, None, output.name)
        team.reset({"apple": 1}, episode_id=1)
        return team

    @staticmethod
    def states(frame=10, action_id=0, **kwargs):
        return {str(agent): state(agent, frame=frame, action_id=action_id, **kwargs)
                for agent in range(2)}

    def test_same_object_loser_cannot_mutate_winner(self):
        team = self.make_team(["go grasp target object <apple> (10)"] * 2)
        actions = team.act(self.states())
        winner = next(agent for agent in range(2) if actions[str(agent)]["type"] == 0)
        loser = 1 - winner
        attempt_id = team.current_attempts[winner]
        self.assertEqual(team.core.attempts[attempt_id]["status"], "running")
        self.assertEqual(team.core.commitments[winner]["task_id"], "entity:10")
        self.assertNotIn(loser, team.core.commitments)
        self.assertEqual(team.core.leases["object:10"][0]["agent_id"], winner)
        self.assertEqual(actions[str(loser)], {"type": 8, "delay": 1})
        self.assertFalse(any(event["event"] == "task_failure"
                             for event in team.board.coordination_events))

    def test_physical_acquisition_and_evaluator_delivery_are_scoped_progress(self):
        held = {"id": 10, "type": 0, "name": "apple",
                "contained": [None, None, None], "contained_name": [None, None, None]}
        team = self.make_team(["go grasp target object <apple> (10)", "wait"],
                              [{"type": 3, "object": 10}, {"type": 8}])
        team.act(self.states())
        acquisition = team.current_attempts[0]
        team.agents[0].plan = "transport objects I'm holding to the bed"
        team.agents[0].next_action = {"type": 5, "arm": "left"}
        states = self.states(20, 1)
        states["0"]["held_objects"][0] = held
        team.act(states)
        self.assertEqual(team.core.attempts[acquisition]["status"], "succeeded")
        self.assertTrue(team.core.attempts[acquisition]["progress"])
        self.assertEqual(team.core.tasks["entity:10"]["progress_version"], 1)
        delivery = team.current_attempts[0]
        team.agents[0].plan, team.agents[0].next_action = "wait", {"type": 8}
        team.act(self.states(30, 2), delivered_objects={10: "apple"})
        self.assertEqual(team.core.attempts[delivery]["status"], "succeeded")
        self.assertTrue(team.core.attempts[delivery]["progress"])
        self.assertEqual(team.core.tasks["delivery:0"]["progress_version"], 1)

    def test_room_overlap_and_scored_object_reacquisition_are_allowed(self):
        team = self.make_team(["go to <Kitchen> (1000)"] * 2)
        actions = team.act(self.states())
        self.assertEqual([actions[str(i)]["type"] for i in range(2)], [0, 0])
        team.agents[0].plan = "go grasp target object <apple> (10)"
        actions = team.act(self.states(20, 1), delivered_objects={10: "apple"})
        self.assertEqual(actions["0"]["type"], 0)
        self.assertEqual(team.board.goal_ledger.delivered, {10: "apple"})

    def test_native_continuation_keeps_attempt_and_lease_past_legacy_ttl(self):
        team = self.make_team(["go grasp target object <apple> (10)", "wait"],
                              [{"type": 0}, {"type": 8}])
        team.act(self.states())
        original = team.current_attempts[0]
        team.agents[0].append_planning_boundary = False
        team.agents[0].next_action = {"type": "ongoing"}
        team.agents[1].plan = "go grasp target object <apple> (10)"
        team.agents[1].next_action = {"type": 0}
        states = self.states(1000, 1)
        states["0"] = state(0, frame=1000, action_id=1, action_terminal=False, status=0)
        actions = team.act(states)
        self.assertEqual(team.current_attempts[0], original)
        self.assertEqual(team.core.leases["object:10"][0]["attempt_id"], original)
        self.assertEqual(actions["1"]["type"], 8)
        team.agents[0].next_action = {"type": 2}
        team.act(self.states(1001, 1))
        self.assertEqual(team.current_attempts[0], original)
        self.assertEqual(team.core.attempts[original]["status"], "running")
        self.assertFalse(team.core.attempts[original]["evidence"][-1]["terminal"])

    def test_wait_preserves_task_and_explicit_release_removes_it(self):
        team = self.make_team(["go to <Kitchen> (1000)", "wait"])
        team.act(self.states())
        original_task = team.core.commitments[0]["task_id"]
        team.agents[0].plan = "wait"
        team.agents[0].next_action = {"type": 8}
        team.act(self.states(20, 1))
        self.assertEqual(team.core.commitments[0]["task_id"], original_task)
        self.assertEqual(team.core.commitments[0]["status"], "waiting")
        team.agents[0].plan = "release current task"
        team.act(self.states(30, 2))
        self.assertNotIn(0, team.core.commitments)
        self.assertNotIn(0, team.commitments)

    def test_repeated_failure_is_context_not_a_veto(self):
        team = self.make_team(["go grasp target object <apple> (10)", "wait"],
                              [{"type": 3, "object": 10}, {"type": 8}])
        for index in range(4):
            actions = team.act(self.states(10 * (index + 1), index,
                                           valid=False, action_status="cannot_reach"))
            self.assertEqual(actions["0"]["type"], 3)
        memory = team.core.context(0)["task_progress"]
        self.assertGreaterEqual(memory[0]["executed_failures"], 2)
        self.assertTrue(memory[0]["retry_is_allowed"])
        self.assertEqual(team.board.planning_loop_guards, {})

    def test_unbound_native_result_and_previous_episode_do_not_attach(self):
        team = self.make_team(["go to <Kitchen> (1000)", "wait"])
        team.act(self.states(action_id=7))
        attempt_id = team.current_attempts[0]
        self.assertEqual(team.core.attempts[attempt_id]["evidence"], [])
        team.board.execution_evidence.append({
            "evidence_id": "unrelated", "agent": 0, "action_id": 99,
            "status": "success", "valid": True})
        team._v4_consume_execution_evidence()
        self.assertEqual(team.core.attempts[attempt_id]["evidence"], [])
        team.reset({"apple": 1}, episode_id=2)
        self.assertEqual(team.native_bindings, {})
        self.assertNotIn(attempt_id, team.core.attempts)

    def test_cooperation_requires_peer_acceptance_and_keeps_readiness_unknown(self):
        proposal = {"kind": "propose", "participants": [0, 1],
                    "description": "Meet to coordinate transport"}
        team = self.make_team(["send a message", "wait"], [
            {"type": 6, "message": "coordination_intent:" + json.dumps(proposal)},
            {"type": 8}])
        actions = team.act(self.states())
        self.assertEqual(actions["0"]["type"], 6)
        intent = team.core.context(1)["coordination"][0]
        self.assertEqual(intent["status"], "proposed")
        self.assertNotIn(1, intent["responses"])
        team.agents[0].plan, team.agents[0].next_action = "wait", {"type": 8}
        team.agents[1].plan = "send a message"
        team.agents[1].next_action = {"type": 6, "message": "coordination_intent:" + json.dumps(
            {"kind": "accept", "intent_id": intent["intent_id"]})}
        team.act(self.states(20, 1))
        intent = team.core.context(0)["coordination"][0]
        self.assertEqual(intent["status"], "accepted")
        self.assertEqual(intent["physical_readiness"], "unknown")

    def test_same_batch_closed_message_is_rejected_before_execution(self):
        team = self.make_team(["send a message", "send a message"])
        intent = team.core.coordinate(0, {
            "kind": "propose", "participants": [0, 1], "description": "Meet"})
        for agent_id, kind in enumerate(("cancel", "accept")):
            team.agents[agent_id].next_action = {
                "type": 6, "message": "coordination_intent:" + json.dumps(
                    {"kind": kind, "intent_id": intent["intent_id"]})}
        actions = team.act(self.states())
        self.assertEqual(actions["0"]["type"], 6)
        self.assertEqual(actions["1"]["type"], 8)
        attempt = team.core.context(1)["own_attempts"][-1]
        self.assertEqual(attempt["status"], "rejected_before_start")
        self.assertEqual(attempt["evidence"], [])
        self.assertNotIn(1, team.core.commitments)

    def test_transported_release_does_not_resurrect_the_previous_work(self):
        team = self.make_team(["go grasp target object <apple> (10)", "wait"])
        team.act(self.states())
        team.agents[0].plan = "send a message"
        team.agents[0].next_action = {
            "type": 6, "message": 'coordination_intent:{"kind":"release_work"}'}
        team.act(self.states(20, 1))
        self.assertEqual(team.core.commitments[0]["status"], "release_requested")
        team.agents[0].plan, team.agents[0].next_action = "wait", {"type": 8}
        team.act(self.states(30, 2))
        self.assertEqual(team.commitments[0]["task_id"], "control:agent:0")
        self.assertEqual(team.core.commitments[0]["task_id"], "control:agent:0")
        self.assertNotIn("object:10", team.core.leases)


if __name__ == "__main__":
    unittest.main()
