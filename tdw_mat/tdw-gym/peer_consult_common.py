"""TDW adapter for the shared, LLM-owned PeerConsult V4 policy.

The historical coordinator remains available for matched baseline runs.  This
adapter reuses its observation and native executor plumbing, but all execution
permission and attempt state belongs to the vendored ``peerconsult_core``.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from peerconsult_core import CoordinationCore
from peer_consult import (PROTOCOL_V4, TDWPeerConsultCoordinator,
                          TDWSharedBlackboard)


class CommonTDWBlackboard(TDWSharedBlackboard):
    """Environment facts plus a read-only projection of committed core state."""

    def sync_tasks(self, intents, planning_boundary_agents=None):
        # Base act() calls this before review. Proposals must never mutate
        # another agent's accepted work; the adapter commits after review.
        return

    def _decision_view_v4(self, agent_id, state, agents, delivery_advisory):
        card = super()._decision_view_v4(
            agent_id, state, agents, delivery_advisory)
        card.update({
            "common_core": self.coordinator.core.context(agent_id),
            "policy": "llm",
            "planning_loop_guard": None,
            "task_queue": [],
            "active_task": copy.deepcopy(
                self.coordinator.commitments.get(agent_id)),
            "room_coverage": copy.deepcopy(
                getattr(agents[agent_id], "rooms_explored", {})),
            "scored_objects": dict(self.goal_ledger.delivered),
            "coordination_guidance": [
                "Choose retry, recovery, exploration, task switch and waiting yourself.",
                "Prior failures and coverage are evidence; repeating a legal action is allowed.",
                "Scored objects stay scored; moving them again does not add the same credit.",
                "Rooms are shared spaces. A teammate's search intent is not a navigation lock.",
                "Wait keeps your commitment; release current task explicitly withdraws it.",
                "Coordination proposals and ready statements are agent claims, not world facts.",
            ],
        })
        card["peer"]["commitment"] = copy.deepcopy(
            self.coordinator.commitments.get(1 - agent_id))
        return card


class CommonTDWPeerConsultCoordinator(TDWPeerConsultCoordinator):
    """Concurrent TDW execution with common atomic review and native tickets."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("protocol_version", PROTOCOL_V4)
        super().__init__(*args, **kwargs)
        if not self.mechanism_v4:
            raise ValueError("the common adapter requires PeerConsultV4")
        for agent in self.agents:
            agent.peer_consult_policy = "llm"
            if getattr(agent, "LLM", None) is not None:
                agent.LLM.peer_consult_policy = "llm"

    def reset(self, goal_description, episode_id=None):
        super().reset(goal_description, episode_id)
        self.core = CoordinationCore(
            f"{self.run_instance_id}:{self.episode_epoch}:{episode_id}",
            list(range(len(self.agents))))
        self.board = CommonTDWBlackboard(
            goal_description, mechanism_v4=True,
            episode_epoch=self.episode_epoch)
        self.board.coordinator = self
        self.board.v4_shared_delivery_target_enabled = (
            self.v4_shared_delivery_target_enabled)
        self.current_attempts = {}
        self.attempt_details = {}
        self.native_bindings = {}
        self.commitments = {}
        self.native_outcomes = {}
        self._common_seen_evidence = set()

    def _v4_consume_execution_evidence(self):
        for evidence in self.board.execution_evidence:
            event_id = evidence["evidence_id"]
            if event_id in self._common_seen_evidence:
                continue
            self._common_seen_evidence.add(event_id)
            key = (int(evidence["agent"]), int(evidence["action_id"]))
            attempt_id = self.native_bindings.get(key)
            if attempt_id is None:
                # Initial, stale, or validator-replacement steps are not
                # evidence for whichever high-level plan happens to be current.
                continue
            success = evidence.get("valid", True) and evidence.get(
                "status") in ("success", "still_dropping")
            self.native_outcomes[attempt_id] = "succeeded" if success else "failed"
            self.core.record_outcome(
                attempt_id, event_id, self.native_outcomes[attempt_id],
                terminal=False, scope="native_action", agent_id=key[0],
                facts=copy.deepcopy(evidence))

    def _finish_previous(self, agent_id):
        attempt_id = self.current_attempts.pop(agent_id, None)
        if attempt_id is None:
            return
        detail = self.attempt_details[attempt_id]
        progressed = any(
            event.get("event") == "task_progress" and
            event.get("agent") == agent_id and
            event.get("task_id") == detail["task_id"] and
            event.get("progress_version", 0) > detail["progress_version"]
            for event in self.board.coordination_events)
        # A completed turn/navigation subaction never proves the whole goal.
        outcome = ("succeeded" if progressed else "failed"
                   if self.native_outcomes.get(attempt_id) == "failed"
                   else "completed_unverified")
        self.core.record_outcome(
            attempt_id, f"boundary:{self.protocol_step}:{agent_id}", outcome,
            terminal=True, progress=True if progressed else None, agent_id=agent_id,
            facts={"source": "native_executor_replanned", "task_id": detail["task_id"]})
        if self.core.context(agent_id)["own_commitment"] is None:
            # A transported release_work may have waited for this execution
            # to terminate. Never resurrect it from the adapter projection.
            self.commitments.pop(agent_id, None)
            self.board.active_tasks[agent_id] = None

    def _v4_finalize_planning_boundaries(self, actions, intents, review_start):
        # The common transaction already allocated attempts and bound each
        # native action ID. The legacy next-action loop guard is not invoked.
        return

    def _v4_apply_pending_guards(self, actions, intents):
        return

    def _review_and_govern(self, states, actions, intents):
        proposals = []
        proposal_by_agent = {}
        messages = {}
        for agent_id, intent in intents.items():
            action = actions[str(agent_id)]
            if action.get("type") == "ongoing":
                continue
            boundary = agent_id in self.v4_planning_boundaries_this_step
            if boundary:
                self._finish_previous(agent_id)
            attempt_id = self.current_attempts.get(agent_id)
            plan = str(intent.get("plan") or "")
            task_id = self.board._task_id_from_intent(agent_id, intent)
            previous = self.commitments.get(agent_id) or {}
            if task_id is None:
                task_id = previous.get("task_id")
            task_id = task_id or f"control:agent:{agent_id}"
            resources = []
            if intent.get("stage") == "acquire" and intent.get("target_id") is not None:
                resources.append(f"object:{intent['target_id']}")
            # Manipulation of held items also declares the physical objects;
            # the occupied map below already excludes competition by peers.
            if action.get("type") in (4, 5) or intent.get("stage") in ("deliver", "load_container"):
                resources.extend(f"object:{object_id}" for object_id, owner in
                                 self.board.physical_owners.items()
                                 if owner.get("agent") == agent_id)
            legal, reason = True, ""
            parse_error = getattr(self.agents[agent_id], "_peer_consult_validation_error", None)
            if parse_error:
                legal, reason = False, str(parse_error)
            target = intent.get("target_id")
            if intent.get("stage") == "acquire":
                if target in self.board.physical_owners:
                    legal, reason = False, "object is physically held"
                elif len(self._held(states[str(agent_id)])) >= 2:
                    legal, reason = False, "no free hand for acquisition"
            if action.get("type") == 6:
                try:
                    message = action.get("message", "")
                    if message.startswith("coordination_intent:"):
                        payload = json.loads(message.split(":", 1)[1])
                        if not isinstance(payload, dict):
                            raise ValueError("coordination payload must be an object")
                        # Validate against the frozen pre-transaction state;
                        # validation cannot publish a message or accept on a
                        # teammate's behalf. Actual publication follows review.
                        copy.deepcopy(self.core).coordinate(agent_id, payload)
                        messages[agent_id] = payload
                    else:
                        event_id = self._v4_coordination_event_id(message)
                        event = next((e for e in self.board.coordination_events[-12:]
                                      if e.get("event_id") == event_id and
                                      e.get("event") in self.V4_COMMUNICATION_EVENT_TYPES), None)
                        if event is None:
                            raise ValueError("message must reference a public event or coordination intent")
                        actions[str(agent_id)] = {"type": 6, "message": self._v4_render_coordination_event(event)}
                except (ValueError, TypeError, AttributeError) as error:
                    legal, reason = False, str(error)
            descriptor = self._v4_boundary_descriptor(agent_id, intent, action) or {}
            proposal = {
                "agent_id": agent_id, "task_id": task_id,
                "action_id": descriptor.get("action_identity", plan),
                "resources": resources, "legal": legal, "reason": reason,
            }
            if attempt_id is not None:
                # One high-level attempt can emit turns, moves and a grasp.
                # Its immutable action identity and resource set are those
                # admitted at the real planner boundary, not a subaction ID.
                original = self.attempt_details[attempt_id]["proposal"]
                for key in ("task_id", "action_id", "resources"):
                    proposal[key] = copy.deepcopy(original[key])
                proposal["attempt_id"] = attempt_id
            proposals.append(proposal)
            proposal_by_agent[agent_id] = proposal

        occupied = {f"object:{object_id}": owner["agent"]
                    for object_id, owner in self.board.physical_owners.items()}
        decisions = self.core.review(proposals, capacity=None, occupied=occupied)
        for agent_id, proposal in proposal_by_agent.items():
            decision = decisions[agent_id]
            attempt_id = decision["attempt_id"]
            accepted = decision["status"] in ("accepted", "running")
            if accepted and agent_id in messages:
                try:
                    result = self.core.coordinate(agent_id, messages[agent_id])
                    actions[str(agent_id)] = {
                        "type": 6,
                        "message": json.dumps({"coordination_intent": result}, ensure_ascii=False),
                    }
                except (ValueError, KeyError, TypeError) as error:
                    self.core.reject_before_start(attempt_id, str(error))
                    decision = {**decision, "status": "rejected", "reason": str(error)}
                    accepted = False
            if not accepted:
                self._clear_plan(self.agents[agent_id])
                actions[str(agent_id)] = {"type": 8, "delay": 1}
                self._revised_this_step.add(agent_id)
                self.board.reviews.append({
                    "frame": self.board.frame, "proposer": agent_id,
                    "trigger": "common_core", "verdict": decision["status"],
                    "reason": decision["reason"], "attempt_id": attempt_id,
                    "replacement": dict(actions[str(agent_id)]),
                })
                # No task, commitment, lease or evidence belonging to another
                # agent is touched by a rejected/deferred proposal.
                continue
            if decision["status"] == "accepted":
                self.core.start(attempt_id)
            self.current_attempts[agent_id] = attempt_id
            self.attempt_details.setdefault(attempt_id, {
                "task_id": proposal["task_id"],
                "progress_version": self.board.task_progress_versions.get(proposal["task_id"], 0),
                "proposal": copy.deepcopy(proposal),
            })
            plan = str(intents[agent_id].get("plan") or "")
            task_id = proposal["task_id"]
            if plan == "release current task":
                self.core.record_outcome(
                    attempt_id, f"release:{self.protocol_step}:{agent_id}",
                    "completed_unverified", terminal=True, agent_id=agent_id)
                self.current_attempts.pop(agent_id, None)
                self.core.release_commitment(agent_id)
                self.commitments.pop(agent_id, None)
                self.board.active_tasks[agent_id] = None
            elif task_id is not None:
                status = "waiting" if plan in ("wait", "[wait]") else "active"
                if messages.get(agent_id, {}).get("kind") == "release_work":
                    status = "release_requested"
                else:
                    self.core.set_commitment(agent_id, task_id, status=status)
                previous = self.commitments.get(agent_id)
                self.commitments[agent_id] = {"task_id": task_id, "agent_id": agent_id, "status": status}
                self.board.active_tasks[agent_id] = task_id
                if previous != self.commitments[agent_id]:
                    self.board.emit_coordination_event(
                        "task_commitment", agent_id, task_id, status=status)
            native_id = int(states[str(agent_id)].get("action_id", 0)) + 1
            if agent_id in self.current_attempts:
                self.native_bindings[(agent_id, native_id)] = attempt_id
        self._log({"kind": "common_transaction", "protocol_step": self.protocol_step,
                   "decisions": decisions, "actions": copy.deepcopy(actions)})

    def finalize(self, evaluator_progress):
        super().finalize(evaluator_progress)
        self._log({
            "kind": "common_final", "evaluator_progress": evaluator_progress,
            "agent_contexts": {agent_id: self.core.context(agent_id)
                               for agent_id in range(len(self.agents))},
            "execution_at_cutoff": copy.deepcopy(self.current_attempts),
        })
