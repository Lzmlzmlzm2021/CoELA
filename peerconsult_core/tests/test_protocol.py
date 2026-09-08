import unittest

from peerconsult_core import CoordinationCore


def proposal(agent, task="move:x", action="pick:x", resources=("object:x",), **extra):
    return dict(agent_id=agent, task_id=task, action_id=action, resources=list(resources), **extra)


class ProtocolContracts(unittest.TestCase):
    def setUp(self):
        self.core = CoordinationCore("episode", [0, 1, 2])

    def begin(self, agent=0, **kwargs):
        decision = self.core.review([proposal(agent, **kwargs)])[agent]
        self.assertEqual(decision["status"], "accepted")
        self.core.start(decision["attempt_id"])
        return decision["attempt_id"]

    def finish(self, attempt, outcome="succeeded", **kwargs):
        return self.core.record_outcome(attempt, "terminal:" + attempt, outcome, **kwargs)

    def test_loser_does_not_mutate_winner_task_or_commitment(self):
        result = self.core.review([proposal(1), proposal(0)])
        self.assertEqual(result[0]["status"], "accepted")
        self.assertEqual(result[1]["status"], "deferred")
        self.assertNotIn(1, self.core.commitments)
        self.assertEqual(self.core.commitments[0]["attempt_id"], result[0]["attempt_id"])
        self.assertEqual(self.core.tasks["move:x"]["attempt_ids"], [result[0]["attempt_id"]])
        self.assertFalse(self.finish(result[1]["attempt_id"], "failed"))
        self.assertEqual(self.core.leases["object:x"][0]["agent_id"], 0)

    def test_active_lease_survives_peer_decisions_and_rejections(self):
        attempt = self.begin()
        for i in range(12):
            result = self.core.review([proposal(1)])
            self.assertEqual(result[1]["status"], "deferred")
        result = self.core.review([proposal(0, legal=False, reason="bad_input")])
        self.assertEqual(result[0]["status"], "rejected")
        self.assertEqual(self.core.leases["object:x"][0]["attempt_id"], attempt)

    def test_cancel_request_keeps_lease_until_confirmation(self):
        attempt = self.begin()
        self.core.request_cancel(attempt)
        self.assertIn("object:x", self.core.leases)
        with self.assertRaises(ValueError):
            self.core.release_commitment(0)
        self.assertTrue(self.finish(attempt, "cancelled"))
        self.assertNotIn("object:x", self.core.leases)
        self.core.release_commitment(0)

    def test_subaction_completion_is_not_attempt_completion(self):
        attempt = self.begin()
        self.core.record_outcome(attempt, "turn:1", "succeeded", scope="native_action")
        self.assertEqual(self.core.attempts[attempt]["status"], "running")
        self.assertIn("object:x", self.core.leases)

    def test_evidence_is_scoped_deduplicated_and_reset_safe(self):
        attempt = self.begin()
        self.assertFalse(self.core.record_outcome(attempt, "wrong", "failed", agent_id=1))
        self.assertFalse(self.core.record_outcome(attempt, "wrong", "failed", episode_id="other"))
        self.assertTrue(self.finish(attempt, progress=True))
        self.assertFalse(self.finish(attempt, progress=True))
        self.assertEqual(self.core.tasks["move:x"]["progress_version"], 1)
        self.core.reset("episode")
        next_attempt = self.begin()
        self.assertNotEqual(attempt, next_attempt)
        self.assertFalse(self.finish(attempt, "failed"))
        self.assertEqual(self.core.attempts[next_attempt]["status"], "running")

    def test_repeated_failures_and_historical_success_do_not_veto(self):
        for result in ["failed", "failed", "failed", "succeeded", "succeeded"]:
            attempt = self.begin()
            self.finish(attempt, result, progress=result == "succeeded")
        history = self.core.context(0)["task_progress"][0]
        self.assertEqual(history["executed_failures"], 3)
        self.assertEqual(history["verified_no_progress"], 3)
        self.assertTrue(history["retry_is_allowed"])

    def test_unrelated_agent_progress_does_not_erase_failure_history(self):
        attempt = self.begin()
        self.finish(attempt, "failed", progress=False)
        other = self.begin(1, task="light:y", action="on:y", resources=("object:y",))
        self.finish(other, progress=True)
        own = self.core.context(0)["task_progress"][0]
        self.assertEqual(own["progress_version"], 0)
        self.assertEqual(own["verified_no_progress"], 1)

    def test_unknown_is_not_failure_or_verified_no_progress(self):
        attempt = self.begin()
        self.finish(attempt, "completed_unverified")
        history = self.core.context(0)["task_progress"][0]
        self.assertEqual(history["executed_failures"], 0)
        self.assertEqual(history["verified_no_progress"], 0)
        self.assertEqual(history["unknown_outcomes"], 1)
        self.assertFalse(self.core.leases)

    def test_deferred_has_no_lease_or_execution_failure(self):
        result = self.core.review([proposal(0), proposal(1, resources=("object:y",))], capacity=1)
        self.assertEqual(result[1]["status"], "deferred")
        self.assertNotIn("object:y", self.core.leases)
        self.assertFalse(self.finish(result[1]["attempt_id"], "failed"))

    def test_resource_bundle_is_all_or_nothing(self):
        self.begin(resources=("object:y",))
        result = self.core.review([proposal(1, resources=("object:x", "object:y"))])
        self.assertEqual(result[1]["status"], "deferred")
        self.assertNotIn("object:x", self.core.leases)

    def test_fair_capacity_is_independent_of_proposal_order_and_priority(self):
        winners = []
        for _ in range(6):
            result = self.core.review([
                proposal(2, resources=(), priority=999),
                proposal(1, resources=(), priority=10),
                proposal(0, resources=(), priority=0)], capacity=1)
            winner = next(a for a, d in result.items() if d["status"] == "accepted")
            winners.append(winner)
            self.core.start(result[winner]["attempt_id"])
            self.finish(result[winner]["attempt_id"])
        self.assertEqual(winners, [0, 1, 2, 0, 1, 2])

    def test_physical_owner_is_checked_before_new_competition(self):
        result = self.core.review([proposal(0), proposal(1)], occupied={"object:x": 1})
        self.assertEqual(result[0]["status"], "deferred")
        self.assertEqual(result[1]["status"], "accepted")

    def test_shared_capacity_and_exclusive_mode(self):
        core = CoordinationCore("shared", [0, 1, 2], {"surface": 2})
        requests = [proposal(a, resources=[{"key": "surface", "mode": "shared"}]) for a in [0, 1]]
        result = core.review(requests)
        self.assertTrue(all(d["status"] == "accepted" for d in result.values()))
        denied = core.review([proposal(2, resources=("surface",))])
        self.assertEqual(denied[2]["status"], "deferred")

    def test_wait_preserves_commitment_and_explicit_release_is_separate(self):
        attempt = self.begin()
        self.finish(attempt)
        self.core.set_commitment(0, "move:x", "waiting", waiting_for="peer reply")
        self.assertEqual(self.core.context(0)["own_commitment"]["status"], "waiting")
        self.core.release_commitment(0)
        self.assertIsNone(self.core.context(0)["own_commitment"])

    def test_cooperation_cannot_impersonate_peer_acceptance_or_readiness(self):
        intent = self.core.coordinate(0, {"kind": "propose", "participants": [0, 1],
                                          "description": "meet for handoff"})
        iid = intent["intent_id"]
        self.assertEqual(intent["status"], "proposed")
        self.assertNotIn(1, intent["responses"])
        with self.assertRaises(ValueError):
            self.core.coordinate(2, {"kind": "accept", "intent_id": iid})
        with self.assertRaises(ValueError):
            self.core.coordinate(1, {"kind": "ready", "intent_id": iid})
        self.core.coordinate(1, {"kind": "accept", "intent_id": iid})
        intent = self.core.coordinate(1, {"kind": "ready", "intent_id": iid})
        self.assertEqual(intent["status"], "accepted")
        self.assertEqual(intent["physical_readiness"], "unknown")
        self.assertEqual(self.core.context(2)["coordination"], [])

    def test_planner_context_is_private_and_snapshot_isolated(self):
        self.core.record_fact("where:x", 0, "room:a")
        before = self.core.context(0)
        self.assertFalse(self.core.context(1)["facts"])
        self.core.record_fact("where:x", 0, "room:b", visible_to=[0, 1])
        after = self.core.context(0)
        self.assertGreater(after["snapshot_version"], before["snapshot_version"])
        self.assertEqual(before["facts"][0]["value"], "room:a")
        after["facts"][0]["value"] = "invented"
        self.assertEqual(self.core.context(0)["facts"][0]["value"], "room:b")

    def test_monotonic_credit_and_reversible_observations_are_distinct(self):
        self.core.record_fact("credit:x", 0, True, monotonic=True)
        self.core.record_fact("credit:x", 0, True)  # repeated observation cannot downgrade authority
        self.core.record_fact("on:x:table", 0, True)
        self.core.record_fact("on:x:table", 0, False)
        with self.assertRaises(ValueError):
            self.core.record_fact("credit:x", 0, False)
        self.begin()  # credit is not an action veto

    def test_batch_shape_error_does_not_partially_admit(self):
        with self.assertRaises(ValueError):
            self.core.review([proposal(0), proposal(1, resources=[{"key": "x", "units": -1}])])
        self.assertFalse(self.core.leases)
        self.assertFalse(self.core.attempts)

    def test_progress_evidence_history_is_bounded_only_in_context(self):
        attempt = self.begin()
        for n in range(20):
            self.core.record_outcome(attempt, "native:{}".format(n), "succeeded",
                                     scope="native_action", facts={"action_id": n})
        view = self.core.context(0)["own_attempts"][0]
        self.assertEqual(view["evidence_total"], 20)
        self.assertEqual(len(view["evidence"]), 6)
        self.assertEqual(len(self.core.attempts[attempt]["evidence"]), 20)

    def test_withdraw_deferred_does_not_release_work_or_peer_execution(self):
        self.core.set_commitment(1, "old_work", "waiting")
        winner = self.begin(0)
        loser = self.core.review([proposal(1)])[1]["attempt_id"]
        self.assertEqual(self.core.withdraw_deferred(1), 1)
        self.assertEqual(self.core.attempts[loser]["status"], "withdrawn")
        self.assertEqual(self.core.context(1)["own_commitment"]["task_id"], "old_work")
        self.assertEqual(self.core.leases["object:x"][0]["attempt_id"], winner)

    def test_revalidation_before_start_is_not_execution_failure(self):
        self.core.set_commitment(0, "old_work", "waiting")
        attempt = self.core.review([proposal(0)])[0]["attempt_id"]
        self.assertTrue(self.core.reject_before_start(attempt, "message state changed"))
        self.assertFalse(self.core.leases)
        self.assertEqual(self.core.context(0)["own_commitment"]["task_id"], "old_work")
        self.assertEqual(self.core.context(0)["task_progress"][0]["executed_failures"], 0)
        active = self.begin()
        self.assertFalse(self.core.reject_before_start(active, "cannot veto running"))
        self.assertIn("object:x", self.core.leases)

    def test_same_native_event_cannot_be_rebound_to_another_attempt(self):
        first = self.begin()
        self.core.record_outcome(first, "native-result:1", "succeeded")
        second = self.begin()
        self.assertFalse(self.core.record_outcome(second, "native-result:1", "failed"))
        self.assertEqual(self.core.attempts[second]["status"], "running")
        self.assertIn("object:x", self.core.leases)

    def test_explicit_work_release_waits_for_active_execution(self):
        attempt = self.begin()
        request = self.core.coordinate(0, {"kind": "release_work"})
        self.assertEqual(request["status"], "release_requested")
        self.assertTrue(self.core.context(0)["work_release_requested"])
        self.assertIn("object:x", self.core.leases)
        self.finish(attempt, "completed_unverified")
        self.assertIsNone(self.core.context(0)["own_commitment"])
        self.assertFalse(self.core.leases)

    def test_explicit_idle_work_release_is_immediate(self):
        self.core.set_commitment(0, "old_work", "waiting")
        request = self.core.coordinate(0, {"kind": "release_work"})
        self.assertEqual(request["status"], "released")
        self.assertIsNone(self.core.context(0)["own_commitment"])


if __name__ == "__main__":
    unittest.main()
