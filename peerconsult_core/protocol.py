"""LLM chooses work; this module owns execution consistency, never task utility.

Adapters supply factual legality, native resource demands and scoped evidence.
There are no benchmark names, recovery actions, retry thresholds, room locks,
goal-completion vetoes, or simulator imports in this module.
"""

from copy import deepcopy
from uuid import uuid4


ACTIVE = frozenset(("accepted", "running", "cancelling"))
TERMINAL = frozenset(("succeeded", "failed", "cancelled", "completed_unverified"))
PRINCIPLES = (
    "Choose the next action yourself from the native legal action domain.",
    "Past success is history, not a prohibition on a new attempt.",
    "Retry, explore again, wait, change work, or request help using the evidence.",
    "Deferred means not executed; running and unknown do not mean failure.",
    "A cooperation proposal is not another agent's accepted commitment.",
    "Declared readiness is not verified physical readiness.",
)


class CoordinationCore:
    """Episode-local single-writer store with atomic batch admission.

    review returns accepted only for proposals scheduled in this transaction.
    Accepted/running/cancelling attempts hold leases until scoped terminal
    evidence arrives. A lease never expires because another agent replans.
    The owning runner must call start before dispatching an accepted attempt.
    Native cancellation is requested separately and confirmed through evidence.
    """

    def __init__(self, episode_id=None, agent_ids=(), resource_capacities=None):
        self.agent_ids = list(agent_ids)
        if len(set(self.agent_ids)) != len(self.agent_ids):
            raise ValueError("duplicate agent identity")
        self.resource_capacities = dict(resource_capacities or {})
        if any(type(n) is not int or n < 1 for n in self.resource_capacities.values()):
            raise ValueError("resource capacities must be positive integers")
        self.reset(episode_id)

    def reset(self, episode_id=None):
        self.episode_id = str(episode_id if episode_id is not None else uuid4().hex)
        # Prevent stale tickets aliasing after reset even if the caller reuses an ID.
        self._epoch = uuid4().hex
        self.tick = 0
        self.revision = 0
        self._counter = 0
        self._cursor = 0
        self.attempts = {}
        self.tasks = {}
        self.commitments = {}
        self.leases = {}
        self.coordination = {}
        self.facts = {}
        self.trace = []
        self._seen_events = set()
        self._event_attempts = {}
        self.work_release_requests = {}
        self._wait_age = {agent: 0 for agent in self.agent_ids}

    def _agent(self, agent):
        if agent not in self.agent_ids:
            if self.agent_ids:
                raise ValueError("unknown agent identity")
            raise ValueError("declare agent_ids before using the protocol")

    def _id(self, prefix):
        self._counter += 1
        return "{}:{}:{}:{}".format(self.episode_id, self._epoch, prefix, self._counter)

    @staticmethod
    def _resources(resources):
        result = []
        keys = set()
        for raw in resources or ():
            item = {"key": raw} if isinstance(raw, str) else dict(raw)
            key = item.get("key", item.get("resource_id"))
            mode = item.get("mode", "exclusive")
            mode = {"read": "shared", "write": "exclusive"}.get(mode, mode)
            units = item.get("units", item.get("capacity_units", 1))
            if not isinstance(key, str) or not key or key in keys:
                raise ValueError("resource keys must be nonempty and unique per proposal")
            if mode not in ("exclusive", "shared") or type(units) is not int or units < 1:
                raise ValueError("invalid resource mode or units")
            keys.add(key)
            result.append({"key": key, "mode": mode, "units": units})
        return result

    def _active_for(self, agent):
        return [a for a in self.attempts.values()
                if a["agent_id"] == agent and a["status"] in ACTIVE]

    def _resource_conflict(self, attempt, occupied):
        for request in attempt["resources"]:
            key = request["key"]
            owner = occupied.get(key)
            owners = owner if isinstance(owner, (list, tuple, set, frozenset)) else [owner]
            if owner is not None and any(x != attempt["agent_id"] for x in owners):
                return "physically_owned"
            held = self.leases.get(key, [])
            if held and (request["mode"] == "exclusive" or
                         any(x["mode"] == "exclusive" for x in held)):
                return "resource_busy"
            limit = self.resource_capacities.get(key, 1)
            if request["units"] + sum(x["units"] for x in held) > limit:
                return "resource_capacity"
        return None

    def review(self, proposals, capacity=None, occupied=None):
        """Resolve a frozen batch without ever selecting substitute actions.

        Proposal fields: agent_id, task_id, action_id, resources, legal, reason;
        attempt_id may reference an existing deferred or active attempt.
        One proposal per agent per batch. occupied is adapter-verified physical
        holding, NOT an extra source of planner-visible world observations.
        """
        proposals = [dict(p) for p in proposals]
        if capacity is not None and (type(capacity) is not int or capacity < 0):
            raise ValueError("execution capacity must be a nonnegative integer or None")
        agents = [p["agent_id"] for p in proposals]
        if len(set(agents)) != len(agents):
            raise ValueError("one proposal per agent per batch")
        # Validate the entire input shape before mutating any protocol state.
        for p in proposals:
            self._agent(p["agent_id"])
            if not isinstance(p.get("task_id"), str) or not p["task_id"]:
                raise ValueError("task_id must be a nonempty adapter-normalized identity")
            if not isinstance(p.get("action_id"), str) or not p["action_id"]:
                raise ValueError("action_id must be a nonempty adapter-normalized identity")
            p["resources"] = self._resources(p.get("resources", ()))
        self.tick += 1
        occupied = dict(occupied or {})
        decisions = {}
        pending = []
        for p in proposals:
            agent = p["agent_id"]
            existing_id = p.get("attempt_id")
            if existing_id is not None:
                old = self.attempts.get(existing_id)
                if (old is None or old["agent_id"] != agent or
                        old["task_id"] != p["task_id"] or
                        old["action_id"] != p["action_id"] or
                        old["resources"] != p["resources"] or
                        old["status"] not in ACTIVE.union(("deferred",))):
                    decisions[agent] = {"status": "rejected", "reason": "invalid_attempt",
                                        "attempt_id": existing_id}
                    continue
                if old["status"] in ACTIVE:
                    # Revalidation cannot silently cancel an already running action.
                    decisions[agent] = {"status": "running", "reason": "continuation",
                                        "attempt_id": existing_id}
                    continue
                attempt = old
            else:
                for old in self.attempts.values():
                    if old["agent_id"] == agent and old["status"] == "deferred":
                        old["status"] = "withdrawn"
                attempt = {
                    "episode_id": self.episode_id, "attempt_id": self._id("attempt"),
                    "agent_id": agent, "task_id": p["task_id"], "action_id": p["action_id"],
                    "resources": deepcopy(p["resources"]), "status": "proposed",
                    "boundary_id": p.get("boundary_id", self.tick),
                    "created_at": self.tick, "evidence": [], "progress": None,
                }
                self.attempts[attempt["attempt_id"]] = attempt
            if not p.get("legal", True):
                attempt["status"] = "rejected_before_start"
                attempt["reason"] = str(p.get("reason") or "invalid_native_action")
                decisions[agent] = {"status": "rejected", "reason": attempt["reason"],
                                    "attempt_id": attempt["attempt_id"]}
            elif self._active_for(agent):
                attempt["status"] = "rejected_before_start"
                attempt["reason"] = "agent_busy"
                decisions[agent] = {"status": "rejected", "reason": "agent_busy",
                                    "attempt_id": attempt["attempt_id"]}
            else:
                pending.append(attempt)
        rank = {agent: i for i, agent in enumerate(self.agent_ids)}
        pending.sort(key=lambda a: (-self._wait_age[a["agent_id"]],
                                   (rank[a["agent_id"]] - self._cursor) % len(rank)))
        used = len({a["agent_id"] for a in self.attempts.values() if a["status"] in ACTIVE})
        for attempt in pending:
            agent = attempt["agent_id"]
            reason = self._resource_conflict(attempt, occupied)
            if reason is None and capacity is not None and used >= capacity:
                reason = "execution_capacity"
            if reason:
                attempt["status"] = "deferred"
                attempt["reason"] = reason
                self._wait_age[agent] += 1
                decisions[agent] = {"status": "deferred", "reason": reason,
                                    "attempt_id": attempt["attempt_id"]}
                continue
            attempt["status"] = "accepted"
            attempt.pop("reason", None)
            for req in attempt["resources"]:
                self.leases.setdefault(req["key"], []).append(dict(
                    req, attempt_id=attempt["attempt_id"], agent_id=agent))
            task = self.tasks.setdefault(attempt["task_id"], {
                "task_id": attempt["task_id"], "progress_version": 0, "attempt_ids": []})
            task["attempt_ids"].append(attempt["attempt_id"])
            attempt["_previous_commitment"] = deepcopy(self.commitments.get(agent))
            self.set_commitment(agent, attempt["task_id"], "active")
            self.commitments[agent]["attempt_id"] = attempt["attempt_id"]
            self._wait_age[agent] = 0
            self._cursor = (rank[agent] + 1) % len(rank)
            used += 1
            decisions[agent] = {"status": "accepted", "reason": "scheduled",
                                "attempt_id": attempt["attempt_id"]}
        self.trace.append({"kind": "review", "tick": self.tick,
                           "decisions": deepcopy(decisions)})
        self.revision += 1
        return deepcopy(decisions)

    def start(self, attempt_id):
        attempt = self.attempts[attempt_id]
        if attempt["status"] not in ("accepted", "running"):
            raise ValueError("only a scheduled attempt can start")
        attempt["status"] = "running"
        self.revision += 1
        return deepcopy(attempt)

    def request_cancel(self, attempt_id):
        attempt = self.attempts[attempt_id]
        if attempt["status"] not in ACTIVE:
            return False
        attempt["status"] = "cancelling"
        self.revision += 1
        return True

    def reject_before_start(self, attempt_id, reason):
        """Rollback admission if factual pre-dispatch revalidation fails.

        This is not an execution failure. It cannot cancel a running attempt or
        release a peer's lease. Preserve the actor's prior work commitment.
        """
        attempt = self.attempts.get(attempt_id)
        if attempt is None or attempt["status"] != "accepted":
            return False
        attempt["status"] = "rejected_before_start"
        attempt["reason"] = str(reason)
        self._release_leases(attempt_id)
        agent = attempt["agent_id"]
        if self.commitments.get(agent, {}).get("attempt_id") == attempt_id:
            old = attempt.get("_previous_commitment")
            if old is None:
                self.commitments.pop(agent, None)
            else:
                self.commitments[agent] = deepcopy(old)
        if self.work_release_requests.get(agent) == attempt_id:
            self.work_release_requests.pop(agent, None)
            self.commitments.pop(agent, None)
        self.revision += 1
        self.trace.append({"kind": "rejected_before_start", "attempt_id": attempt_id,
                           "reason": str(reason), "tick": self.tick})
        return True

    def _release_leases(self, attempt_id):
        for resource in list(self.leases):
            self.leases[resource] = [lease for lease in self.leases[resource]
                                     if lease["attempt_id"] != attempt_id]
            if not self.leases[resource]:
                del self.leases[resource]

    def record_outcome(self, attempt_id, event_id, outcome, terminal=True,
                       progress=None, scope="attempt", agent_id=None,
                       episode_id=None, facts=None):
        """Consume evidence once. False denotes stale/duplicate/unstarted evidence.

        Native subaction evidence never ends a high-level attempt. progress is
        task-related True/False/None, where None means not known. Arbitrary
        observation change belongs in facts, not implicitly in progress.
        """
        if progress is not None and type(progress) is not bool:
            raise ValueError("progress must be True, False, or unknown (None)")
        attempt = self.attempts.get(attempt_id)
        if (attempt is None or attempt["status"] not in ACTIVE or
                (episode_id is not None and str(episode_id) != self.episode_id) or
                (agent_id is not None and agent_id != attempt["agent_id"])):
            return False
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("evidence requires a stable event_id")
        key = (attempt_id, event_id)
        if key in self._seen_events:
            return False
        source_event = (attempt["agent_id"], scope, event_id)
        bound_attempt = self._event_attempts.get(source_event)
        if bound_attempt is not None and bound_attempt != attempt_id:
            return False
        if terminal and scope == "attempt" and outcome not in TERMINAL:
            raise ValueError("terminal attempt requires a known terminal outcome")
        if scope == "attempt" and outcome == "cancelled" and not terminal:
            raise ValueError("cancellation acknowledgement must be terminal")
        self._seen_events.add(key)
        self._event_attempts[source_event] = attempt_id
        event = {"event_id": event_id, "attempt_id": attempt_id, "scope": scope,
                 "outcome": outcome, "terminal": bool(terminal and scope == "attempt"),
                 "progress": progress, "facts": deepcopy(facts), "tick": self.tick}
        attempt["evidence"].append(event)
        if progress is True:
            self.tasks[attempt["task_id"]]["progress_version"] += 1
            attempt["progress"] = True
        elif progress is False and attempt["progress"] is None:
            attempt["progress"] = False
        if event["terminal"]:
            attempt["status"] = outcome
            self._release_leases(attempt_id)
            commitment = self.commitments.get(attempt["agent_id"])
            if commitment and commitment.get("attempt_id") == attempt_id:
                commitment["status"] = "ready_to_plan"
            if self.work_release_requests.get(attempt["agent_id"]) == attempt_id:
                self.work_release_requests.pop(attempt["agent_id"], None)
                self.commitments.pop(attempt["agent_id"], None)
        self.trace.append({"kind": "evidence", "event": deepcopy(event)})
        self.revision += 1
        return True

    def set_commitment(self, agent_id, task_id, status="active", waiting_for=None):
        self._agent(agent_id)
        if status not in ("active", "waiting", "released"):
            raise ValueError("invalid commitment status")
        if status == "released":
            if self._active_for(agent_id):
                raise ValueError("confirm execution termination before releasing commitment")
            self.commitments.pop(agent_id, None)
            self.revision += 1
            return
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("a commitment needs a task identity")
        if any(a["task_id"] != task_id for a in self._active_for(agent_id)):
            raise ValueError("cannot replace the commitment of an active execution")
        old = self.commitments.get(agent_id, {})
        result = {"agent_id": agent_id, "task_id": task_id, "status": status,
                  "waiting_for": deepcopy(waiting_for)}
        if old.get("task_id") == task_id and old.get("attempt_id"):
            result["attempt_id"] = old["attempt_id"]
        self.commitments[agent_id] = result
        self.revision += 1

    def release_commitment(self, agent_id):
        self.set_commitment(agent_id, None, "released")

    def withdraw_deferred(self, agent_id):
        """Abstaining withdraws unstarted proposals, not accepted work or leases."""
        self._agent(agent_id)
        count = 0
        for attempt in self.attempts.values():
            if attempt["agent_id"] == agent_id and attempt["status"] == "deferred":
                attempt["status"] = "withdrawn"
                count += 1
        if count:
            self.revision += 1
            self.trace.append({"kind": "withdraw_deferred", "agent_id": agent_id,
                               "count": count, "tick": self.tick})
        return count

    def coordinate(self, actor, payload):
        """Record an explicitly transported model-selected collaboration intent.

        Adapters call this only after their declared communication transport is
        admitted/charged. This method never dispatches a native action or makes
        a participant's acceptance on behalf of another participant.
        """
        self._agent(actor)
        payload = deepcopy(dict(payload))
        kind = payload.get("kind")
        if kind == "release_work":
            self.withdraw_deferred(actor)
            active = self._active_for(actor)
            intent = {"intent_id": self._id("work_control"), "kind": kind,
                      "agent_id": actor, "status": "released"}
            if active:
                attempt_id = active[0]["attempt_id"]
                self.work_release_requests[actor] = attempt_id
                self.commitments[actor]["status"] = "release_requested"
                intent.update(status="release_requested", after_attempt_id=attempt_id)
            else:
                self.release_commitment(actor)
            self.trace.append({"kind": "work_control", "actor": actor,
                               "payload": payload, "result": deepcopy(intent), "tick": self.tick})
            self.revision += 1
            return deepcopy(intent)
        if kind == "propose":
            participants = list(payload.get("participants", ()))
            if actor not in participants:
                participants.insert(0, actor)
            if len(participants) < 2 or len(set(participants)) != len(participants):
                raise ValueError("a proposal needs distinct cooperating participants")
            for participant in participants:
                self._agent(participant)
            intent_id = self._id("coordination")
            intent = {"intent_id": intent_id, "sender": actor,
                      "participants": participants, "status": "proposed",
                      "task_id": payload.get("task_id"),
                      "description": payload.get("description", ""),
                      "conditions": payload.get("conditions", []),
                      "location": payload.get("location"),
                      "responses": {actor: "accept"}, "ready_declarations": [],
                      "physical_readiness": "unknown"}
            self.coordination[intent_id] = intent
        elif kind in ("accept", "decline", "ready", "cancel"):
            intent_id = payload.get("intent_id")
            intent = self.coordination.get(intent_id)
            if intent is None or actor not in intent["participants"]:
                raise ValueError("unknown intent or actor is not a participant")
            if intent["status"] in ("cancelled", "declined"):
                raise ValueError("closed cooperation intent")
            if kind == "cancel":
                intent["status"] = "cancelled"
                intent["cancelled_by"] = actor
            elif kind == "ready":
                if intent["responses"].get(actor) != "accept":
                    raise ValueError("accept your role before declaring readiness")
                if actor not in intent["ready_declarations"]:
                    intent["ready_declarations"].append(actor)
            else:
                intent["responses"][actor] = kind
                if kind == "decline":
                    intent["status"] = "declined"
                elif all(intent["responses"].get(p) == "accept"
                         for p in intent["participants"]):
                    intent["status"] = "accepted"
        else:
            raise ValueError("unknown coordination intent kind")
        self.trace.append({"kind": "coordination", "actor": actor,
                           "payload": payload, "intent_id": intent_id, "tick": self.tick})
        self.revision += 1
        return deepcopy(intent)

    def record_fact(self, fact_id, observer, value, visible_to=None, monotonic=False):
        """Store adapter-authorized observations; never infer a global evaluator fact."""
        self._agent(observer)
        viewers = [observer] if visible_to is None else list(visible_to)
        for viewer in viewers:
            self._agent(viewer)
        if not isinstance(fact_id, str) or not fact_id:
            raise ValueError("fact_id must be a stable nonempty identity")
        # Different observers may disagree; do not overwrite one source with another.
        key = (observer, fact_id)
        previous = self.facts.get(key)
        if previous and previous["monotonic"] and previous["value"] != value:
            raise ValueError("a monotonic fact cannot be retracted")
        self.facts[key] = {"fact_id": fact_id, "observer": observer,
                               "value": deepcopy(value), "visible_to": viewers,
                               "monotonic": bool(monotonic or (previous and previous["monotonic"])),
                               "observed_at": self.tick}
        self.revision += 1

    def context(self, agent_id, history_limit=8):
        """Return an isolated planner view, not the full internal audit trace."""
        self._agent(agent_id)
        own = [a for a in self.attempts.values() if a["agent_id"] == agent_id]
        active = [a for a in own if a["status"] in ACTIVE]
        past = [a for a in own if a["status"] not in ACTIVE]
        own_tasks = {a["task_id"] for a in own}
        history = []
        for task_id in sorted(own_tasks):
            related = [a for a in own if a["task_id"] == task_id]
            history.append({
                "task_id": task_id,
                "progress_version": self.tasks.get(task_id, {}).get("progress_version", 0),
                "executed_failures": sum(a["status"] == "failed" for a in related),
                "verified_no_progress": sum(a["status"] in TERMINAL and
                                             a["progress"] is False for a in related),
                "unknown_outcomes": sum(a["status"] == "completed_unverified" for a in related),
                "rejected_before_start": sum(a["status"] == "rejected_before_start" for a in related),
                "retry_is_allowed": True,
                "retry_requires_native_legality": True,
            })
        attempt_views = deepcopy(active + (past[-history_limit:] if history_limit > 0 else []))
        for attempt in attempt_views:
            attempt.pop("_previous_commitment", None)
            attempt["evidence_total"] = len(attempt["evidence"])
            attempt["evidence"] = attempt["evidence"][-6:]
        result = {
            "protocol_version": "0.1.0", "episode_id": self.episode_id,
            "snapshot_version": self.revision,
            "own_commitment": self.commitments.get(agent_id),
            "work_release_requested": agent_id in self.work_release_requests,
            "own_attempts": attempt_views,
            "task_progress": history,
            "peer_commitments": [c for a, c in self.commitments.items() if a != agent_id],
            "resource_occupancy": [lease for leases in self.leases.values() for lease in leases],
            "coordination": [c for c in self.coordination.values() if agent_id in c["participants"]],
            "facts": [f for f in self.facts.values() if agent_id in f["visible_to"]],
            "decision_principles": list(PRINCIPLES),
        }
        return deepcopy(result)
