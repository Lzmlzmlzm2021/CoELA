"""The common policy keeps strategic decisions in the planner's action domain."""
import json
import unittest

import numpy as np

import test_v4_candidate_completeness as candidates
import test_v4_executor_isolation as executor


class CommonLLMPolicyTests(unittest.TestCase):
    def test_ambiguous_output_cannot_randomly_select_an_action(self):
        planner = candidates.V4CandidateCompletenessTests().make_planner()
        planner.peer_consult_policy = "llm"
        plans = ["wait", "go to Kitchen"]
        self.assertEqual(planner.parse_answer(plans, "option B")[0], "go to Kitchen")
        self.assertIsNone(planner.parse_answer(plans, "unclear")[0])
        self.assertIsNone(planner.parse_answer(plans, "wait or go to Kitchen")[0])

    def test_reexplore_wait_release_and_natural_candidate_order(self):
        planner = candidates.V4CandidateCompletenessTests().make_planner()
        planner.peer_consult_policy = "llm"
        planner.rooms_explored = {"Kitchen": "all"}
        planner.peer_decision_card["active_task"] = {
            "task_id": "entity:10", "status": "in_progress", "object_id": 10}
        _, _, plans = planner.get_available_plans()
        self.assertIn("explore current room Kitchen", plans)
        self.assertIn("wait", plans)
        self.assertIn("release current task", plans)
        self.assertLess(plans.index("go grasp target object <banana> (11)"),
                        plans.index("go grasp target object <apple> (10)"))

    def test_model_can_generate_a_cooperation_proposal(self):
        planner, _ = candidates.V4CandidateCompletenessTests().make_runnable_planner()
        planner.peer_consult_policy = "llm"
        proposal = {"kind": "propose", "participants": [0, 1],
                    "description": "Meet in the kitchen"}
        outputs = iter(["send a message", "coordination_intent:" + json.dumps(proposal)])
        planner.generator = lambda *_: ([next(outputs)], 0)
        plan, info = planner.run(
            10, "Kitchen", {"Kitchen": "all"},
            [candidates.empty_hand(), candidates.empty_hand()], [],
            planner.object_list, {}, ["go to Kitchen at initial step"], [], [], None)
        self.assertTrue(plan.startswith("send a message: coordination_intent:"))
        self.assertTrue(info["communication_guard"]["accepted"])
        self.assertIn("choose your own task", info["prompt_plan_stage_2"])

    def test_scored_id_remains_in_physically_available_local_candidates(self):
        agent = executor.V4ExecutorIsolationTests().make_agent()
        agent.peer_consult_policy = "llm"
        agent.rooms_name = ["Kitchen"]
        agent.object_map = np.array([[1]])
        agent.id_map = np.array([[10]])
        agent.satisfied = [10]
        agent.holding_objects_id = []
        agent.oppo_holding_objects_id = []
        agent.with_oppo = []
        agent.object_info = {10: {"id": 10, "name": "apple", "position": [0, 0, 0]}}
        agent.env_api = {"belongs_to_which_room": lambda _: "Kitchen"}
        agent.get_object_list()
        self.assertEqual([obj["id"] for obj in agent.object_list[0]], [10])
        self.assertEqual(agent.satisfied, [10])
        agent.peer_consult_policy = "legacy"
        agent.get_object_list()
        self.assertEqual(agent.object_list[0], [])


if __name__ == "__main__":
    unittest.main()
