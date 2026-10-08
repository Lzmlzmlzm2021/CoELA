"""PeerConsult V4 coordination for two TDW-MAT CoELA agents.

The shared board publishes structured task facts and a bed delivery target.
Each agent keeps its private perception and original skill executor.
"""
from __future__ import annotations
from collections import Counter
import copy
import hashlib
import json
import os
import re
import uuid
from typing import Any, Dict, Iterable, List, Optional, Tuple
import numpy as np

BOARD_PREFIX = "[Memory Board] "
PROTOCOL_V4 = "PeerConsultV4"
SUPPORTED_PROTOCOLS = (PROTOCOL_V4,)
PROTOCOL_VERSION = PROTOCOL_V4
ENTITY_CATEGORIES = {
    0: "goal_object",
    1: "container_resource",
    2: "goal_location",
    3: "agent",
}


def _entity_category(entity_type: Any) -> str:
    """Separate task goals from reusable transport resources."""
    return ENTITY_CATEGORIES.get(entity_type, "unknown")


def _jsonable(value: Any) -> Any:
    """Return a compact JSON-safe representation for protocol logging."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for (key, item) in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "name"):
        return value.name
    return value


def _real_objects(items: Iterable[dict]) -> List[dict]:
    return [item for item in items if item and item.get("id") is not None]


def _plan_target_id(plan: Optional[str]) -> Optional[int]:
    if not plan:
        return None
    matches = re.findall("\\((\\d+)\\)", plan)
    return int(matches[-1]) if matches else None


def _intent_from_plan(
    agent_id: int, plan: Optional[str], action: dict, state: dict
) -> dict:
    """Convert a CoELA plan into the short-lived public intent schema."""
    if plan is None:
        stage = "idle"
    elif plan.startswith("go grasp"):
        stage = "acquire"
    elif plan.startswith("put"):
        stage = "load_container"
    elif plan.startswith("transport"):
        stage = "deliver"
    elif plan.startswith(("go to", "explore")):
        stage = "explore"
    elif plan.startswith("send a message"):
        stage = "communicate"
    else:
        stage = "act"
    return {
        "agent": agent_id,
        "stage": stage,
        "target_id": _plan_target_id(plan),
        "plan": plan,
        "action": copy.deepcopy(action),
        "execution": "ongoing"
        if action.get("type") == "ongoing" or state.get("status") == 0
        else "ready",
        "frame": int(state.get("current_frames", 0)),
    }


class TDWGoalLedger:
    """Count-based TDW-MAT goal ledger with concrete delivery evidence."""

    def __init__(self, required: Dict[str, int]):
        self.required = {str(name): int(count) for (name, count) in required.items()}
        self.delivered: Dict[int, str] = {}

    def mark_delivered(self, object_id: int, name: str) -> None:
        if name in self.required:
            self.delivered[int(object_id)] = name

    def delivered_counts(self) -> Dict[str, int]:
        counts = Counter(self.delivered.values())
        return {name: counts.get(name, 0) for name in self.required}

    def remaining_counts(self) -> Dict[str, int]:
        delivered = self.delivered_counts()
        return {
            name: max(count - delivered.get(name, 0), 0)
            for (name, count) in self.required.items()
        }

    def view(self) -> dict:
        return {
            "required": self.required,
            "delivered": self.delivered_counts(),
            "remaining": self.remaining_counts(),
        }


class TDWSharedBlackboard:
    """Structured public state shared by the two high-level planners."""

    CLAIM_TTL_FRAMES = 450
    ROOM_CLAIM_TTL_FRAMES = 450
    MAX_DIALOGUE_EVENTS = 24
    MAX_EXECUTION_EVIDENCE = 64
    MAX_REVIEWS = 32
    MAX_PROPOSAL_HISTORY = 32
    MAX_TASK_HISTORY = 64
    MAX_COORDINATION_EVENTS = 64

    def __init__(
        self,
        goal_description: Dict[str, int],
        episode_epoch: int = 0,
    ):
        self.goal_ledger = TDWGoalLedger(goal_description)
        self.episode_epoch = int(episode_epoch)
        self.known_entities: Dict[int, dict] = {}
        self.physical_owners: Dict[int, dict] = {}
        self.container_contents: Dict[int, List[int]] = {}
        self.claims: Dict[int, dict] = {}
        self.room_claims: Dict[int, dict] = {}
        self.room_memory: Dict[str, dict] = {}
        self.current_intents: Dict[int, dict] = {}
        self.reviews: List[dict] = []
        self.execution_evidence: List[dict] = []
        self.proposal_history: List[dict] = []
        self.tasks: Dict[str, dict] = {}
        self.active_tasks: Dict[int, Optional[str]] = {0: None, 1: None}
        self.task_history: List[dict] = []
        self.coordination_events: List[dict] = []
        self._coordination_event_sequence = 0
        self.planning_loop_guards: Dict[int, dict] = {}
        self.task_progress_versions: Counter = Counter()
        self.failure_counts: Counter = Counter()
        self._seen_action_evidence: set = set()
        self.dialogue_events: List[dict] = []
        self._last_observed_messages = {0: None, 1: None}
        self.frame = 0
        self.evaluator_progress = None

    def emit_coordination_event(
        self,
        event: str,
        agent: Optional[int],
        task_id: Optional[str] = None,
        **details: Any,
    ) -> Optional[dict]:
        """Append one minimal, structured coordination lifecycle event.

        Callers must pass coordination identifiers/outcomes, never local
        observations or benchmark-specific geometry.
        """
        self._coordination_event_sequence += 1
        record = {
            "event_id": self._coordination_event_sequence,
            "frame": self.frame,
            "event": str(event),
            "agent": agent,
            "task_id": task_id,
        }
        record.update(
            {
                key: copy.deepcopy(value)
                for (key, value) in details.items()
                if value is not None
            }
        )
        self.coordination_events.append(record)
        self.coordination_events = self.coordination_events[
            -self.MAX_COORDINATION_EVENTS :
        ]
        return record

    def note_task_progress(
        self, task_id: Optional[str], agent: Optional[int], event: str
    ) -> None:
        """Record an adapter-reported task progress boundary.

        Core treats this as an opaque monotonic heartbeat.  It doesn't infer
        progress from distance, confidence, target type, or container state.
        """
        if task_id is None:
            return
        self.task_progress_versions[str(task_id)] += 1
        self.emit_coordination_event(
            "task_progress",
            agent,
            str(task_id),
            outcome=str(event),
            progress_version=int(self.task_progress_versions[str(task_id)]),
        )

    def suspend_active_task(
        self, agent_id: int, reason: str, task_id: Optional[str] = None
    ) -> Optional[str]:
        """Suspend, but never delete, the current task."""
        task_id = task_id or self.active_tasks.get(agent_id)
        task = self.tasks.get(task_id) if task_id else None
        if task is not None and task.get("status") not in (
            "completed",
            "surplus",
            "carried",
        ):
            task.update(
                status="suspended",
                owner=None,
                reason=reason,
                blocked_until=None,
                updated_frame=self.frame,
            )
            self.emit_coordination_event(
                "task_suspended", agent_id, task_id, reason=reason
            )
        if self.active_tasks.get(agent_id) == task_id:
            self.active_tasks[agent_id] = None
        return task_id

    @staticmethod
    def _entity_task_key(object_id: int) -> str:
        return f"entity:{int(object_id)}"

    @staticmethod
    def _room_task_key(room: str) -> str:
        return f"room:{room}"

    @staticmethod
    def _delivery_task_key(agent_id: int) -> str:
        return f"delivery:{int(agent_id)}"

    def _ensure_task(
        self, task_id: str, kind: str, priority: int, **facts: Any
    ) -> dict:
        task = self.tasks.get(task_id)
        if task is None:
            task = {
                "task_id": task_id,
                "kind": kind,
                "status": "pending",
                "priority": int(priority),
                "owner": None,
                "created_frame": self.frame,
                "updated_frame": self.frame,
                "attempts": 0,
                "blocked_until": None,
                "reason": None,
            }
            task["last_owner"] = None
            self.tasks[task_id] = task
        task.update({key: copy.deepcopy(value) for (key, value) in facts.items()})
        task["updated_frame"] = self.frame
        return task

    def _refresh_tasks(self, agents: List[Any]) -> None:
        """Reconcile persistent tasks with physical/evaluator truth."""
        remaining = self.goal_ledger.remaining_counts()
        delivered_ids = set(self.goal_ledger.delivered)
        for object_id, entity in self.known_entities.items():
            entity_type = entity.get("type")
            name = str(entity.get("name") or "")
            if entity_type == 0 and name in self.goal_ledger.required:
                task = self._ensure_task(
                    self._entity_task_key(object_id),
                    "deliver_goal_object",
                    100,
                    object_id=object_id,
                    name=name,
                    room=entity.get("room"),
                    position=entity.get("position"),
                )
                if object_id in delivered_ids:
                    task.update(
                        status="completed",
                        owner=None,
                        reason="evaluator_confirmed_delivery",
                        blocked_until=None,
                    )
                elif object_id in self.physical_owners:
                    task.update(
                        status="carried",
                        owner=self.physical_owners[object_id].get("agent"),
                        reason="physical_ownership",
                    )
                elif remaining.get(name, 0) <= 0:
                    task.update(
                        status="surplus", owner=None, reason="goal_quota_filled"
                    )
                elif not (
                    task.get("status") == "blocked"
                    and self.frame < int(task.get("blocked_until") or 0)
                ):
                    if task.get("status") in (
                        "completed",
                        "surplus",
                        "carried",
                        "blocked",
                    ):
                        task.update(
                            status="pending",
                            owner=None,
                            reason=None,
                            blocked_until=None,
                        )
            elif entity_type == 1:
                task = self._ensure_task(
                    self._entity_task_key(object_id),
                    "container_resource",
                    40,
                    object_id=object_id,
                    name=name,
                    room=entity.get("room"),
                    position=entity.get("position"),
                )
                if object_id in self.physical_owners:
                    task.update(
                        status="in_use",
                        owner=self.physical_owners[object_id].get("agent"),
                        reason="physical_ownership",
                    )
                elif not (
                    task.get("status") == "blocked"
                    and self.frame < int(task.get("blocked_until") or 0)
                ):
                    task.update(
                        status="pending", owner=None, reason=None, blocked_until=None
                    )
        for room, memory in self.room_memory.items():
            task = self._ensure_task(
                self._room_task_key(room), "explore_room", 20, room=room
            )
            previous_status = task.get("status")
            previous_owner = task.get("owner")
            coverage = set(
                (
                    str(value or "none")
                    for value in (memory.get("coverage_by_agent") or {}).values()
                )
            )
            if "all" in coverage:
                task.update(
                    status="completed",
                    owner=None,
                    reason="coverage_complete",
                    blocked_until=None,
                )
                if previous_status != "completed":
                    self.note_task_progress(
                        self._room_task_key(room), previous_owner, "task_completed"
                    )
            elif not (
                task.get("status") == "blocked"
                and self.frame < int(task.get("blocked_until") or 0)
            ):
                if task.get("status") in ("completed", "blocked"):
                    task.update(
                        status="pending", owner=None, reason=None, blocked_until=None
                    )
        for agent_id in range(len(agents)):
            payload = [
                object_id
                for (object_id, owner) in self.physical_owners.items()
                if owner.get("agent") == agent_id
                and self.known_entities.get(object_id, {}).get("type") == 0
                and (object_id not in delivered_ids)
            ]
            delivery_task_id = self._delivery_task_key(agent_id)
            if payload:
                task = self._ensure_task(
                    delivery_task_id,
                    "deliver_payload",
                    130,
                    agent=agent_id,
                    payload=sorted(payload),
                )
                if self.active_tasks.get(agent_id) == delivery_task_id:
                    task.update(status="in_progress", owner=agent_id)
                else:
                    task.update(status="pending", owner=agent_id)
            elif delivery_task_id in self.tasks:
                delivery_task = self.tasks[delivery_task_id]
                if delivery_task.get("status") != "completed":
                    delivery_task.update(
                        status="suspended",
                        owner=None,
                        reason="payload_released_awaiting_evaluator",
                        blocked_until=None,
                        updated_frame=self.frame,
                    )
                if self.active_tasks.get(agent_id) == delivery_task_id:
                    self.active_tasks[agent_id] = None
        for agent_id, task_id in list(self.active_tasks.items()):
            task = self.tasks.get(task_id) if task_id else None
            terminal_statuses = {"completed", "surplus"}
            terminal_statuses.update({"carried", "in_use"})
            if task is None or task.get("status") in terminal_statuses:
                self.active_tasks[agent_id] = None

    def _task_id_from_intent(self, agent_id: int, intent: dict) -> Optional[str]:
        stage = intent.get("stage")
        target_id = intent.get("target_id")
        plan = str(intent.get("plan") or "")
        if stage == "acquire" and target_id is not None:
            return self._entity_task_key(target_id)
        if stage == "deliver":
            return self._delivery_task_key(agent_id)
        if stage == "explore" and plan:
            room = (
                plan[len("go to ") :]
                if plan.startswith("go to ")
                else plan[len("explore current room ") :]
                if plan.startswith("explore current room ")
                else None
            )
            if room:
                return self._room_task_key(room)
        return None

    def sync_tasks(
        self,
        intents: Dict[int, dict],
        planning_boundary_agents: Optional[Iterable[int]] = None,
    ) -> None:
        """Attach current plans to tasks without erasing interrupted work."""
        boundary_agents = set(planning_boundary_agents or [])
        for agent_id, intent in intents.items():
            task_id = self._task_id_from_intent(agent_id, intent)
            if task_id is None:
                if agent_id in boundary_agents and intent.get("stage") == "idle":
                    previous_id = self.active_tasks.get(agent_id)
                    previous = self.tasks.get(previous_id) if previous_id else None
                    if previous is not None and previous.get("status") == "in_progress":
                        previous.update(
                            status="suspended",
                            owner=None,
                            reason="planner_released_task",
                            updated_frame=self.frame,
                        )
                        self.emit_coordination_event(
                            "task_released",
                            agent_id,
                            previous_id,
                            reason="planner_selected_idle",
                        )
                    self.release_task_claims(
                        agent_id, previous_id, "planner_selected_idle"
                    )
                    self.active_tasks[agent_id] = None
                continue
            previous_id = self.active_tasks.get(agent_id)
            if previous_id != task_id:
                previous = self.tasks.get(previous_id) if previous_id else None
                if previous and previous.get("status") == "in_progress":
                    previous.update(
                        status="suspended",
                        owner=None,
                        reason="planner_switched_task",
                        updated_frame=self.frame,
                    )
                    self.emit_coordination_event(
                        "task_suspended",
                        agent_id,
                        previous_id,
                        reason="planner_switched_task",
                    )
                self.release_task_claims(agent_id, previous_id, "planner_switched_task")
                self.task_history.append(
                    {
                        "frame": self.frame,
                        "agent": agent_id,
                        "from": previous_id,
                        "to": task_id,
                    }
                )
            task = self.tasks.get(task_id)
            if task is None:
                kind = (
                    "explore_room"
                    if task_id.startswith("room:")
                    else "deliver_payload"
                    if task_id.startswith("delivery:")
                    else "unknown"
                )
                task = self._ensure_task(task_id, kind, 10)
            non_selectable = {"completed", "surplus", "carried"}
            non_selectable.add("in_use")
            if task.get("status") not in non_selectable:
                task.update(
                    status="in_progress",
                    owner=agent_id,
                    attempts=int(task.get("attempts", 0))
                    + (1 if previous_id != task_id else 0),
                    reason="selected_by_planner",
                    updated_frame=self.frame,
                )
                task["last_owner"] = agent_id
                if previous_id != task_id:
                    self.emit_coordination_event("task_commitment", agent_id, task_id)
            self.active_tasks[agent_id] = task_id
        self.task_history = self.task_history[-self.MAX_TASK_HISTORY :]

    def task_queue(self, agent_id: int, state: dict) -> List[dict]:
        """Return a compact, ordered queue for the LLM decision card."""
        active_id = self.active_tasks.get(agent_id)
        rows = []
        for task_id, task in self.tasks.items():
            hidden_statuses = {"completed", "surplus", "in_use"}
            if task.get("status") in hidden_statuses:
                continue
            owner = task.get("owner")
            if owner not in (None, agent_id):
                continue
            if (
                task_id != active_id
                and owner != agent_id
                and (task.get("last_owner") != agent_id)
            ):
                continue
            object_id = task.get("object_id")
            claim = self.claims.get(object_id) if object_id is not None else None
            if claim and claim.get("agent") != agent_id:
                continue
            row = {
                "task_id": task_id,
                "kind": task.get("kind"),
                "status": task.get("status"),
                "priority": task.get("priority"),
                "object_id": object_id,
                "name": task.get("name"),
                "room": task.get("room"),
                "reason": task.get("reason"),
                "blocked_until": task.get("blocked_until"),
            }
            rows.append(
                (
                    0 if task_id == active_id else 1,
                    0 if task.get("status") == "suspended" else 1,
                    -int(task.get("priority", 0)),
                    task_id,
                    row,
                )
            )
            continue
        return [item[-1] for item in sorted(rows)[:10]]

    def _observe_entity(
        self, entity: dict, agent_id: int, frame: int, evidence_source: str = "direct"
    ) -> None:
        object_id = entity.get("id")
        if object_id is None:
            return
        object_id = int(object_id)
        previous = self.known_entities.get(object_id, {})
        seen_by = set(previous.get("seen_by", []))
        seen_by.add(agent_id)
        position = entity.get("position", previous.get("position"))
        room = entity.get("room", previous.get("room"))
        record = {
            "id": object_id,
            "name": entity.get("name") or previous.get("name"),
            "type": entity.get("type", previous.get("type")),
            "category": _entity_category(entity.get("type", previous.get("type"))),
            "seen_by": sorted(seen_by),
            "last_seen_by": agent_id,
            "last_seen_frame": frame,
            "position": copy.deepcopy(position),
            "room": room,
        }
        self.known_entities[object_id] = record

    def _record_payload(self, item: dict, agent_id: int, arm: str, frame: int) -> None:
        object_id = int(item["id"])
        self._observe_entity(item, agent_id, frame, evidence_source="held")
        self.physical_owners[object_id] = {
            "agent": agent_id,
            "arm": arm,
            "carrier": "hand",
        }
        contained = [
            int(value) for value in item.get("contained", []) if value is not None
        ]
        if item.get("type") == 1:
            self.container_contents[object_id] = contained
            contained_names = item.get("contained_name", [])
            for index, child_id in enumerate(contained):
                name = contained_names[index] if index < len(contained_names) else None
                self.known_entities.setdefault(
                    child_id,
                    {
                        "id": child_id,
                        "name": name,
                        "type": 0,
                        "seen_by": [agent_id],
                        "last_seen_by": agent_id,
                        "last_seen_frame": frame,
                        **{},
                    },
                )
                self.physical_owners[child_id] = {
                    "agent": agent_id,
                    "carrier": "container",
                    "container_id": object_id,
                }

    def observe(
        self,
        states: Dict[str, dict],
        agents: List[Any],
        delivered_objects: Optional[Dict[int, str]] = None,
    ) -> None:
        self.frame = max(
            (int(state.get("current_frames", 0)) for state in states.values())
        )
        previous_owners = self.physical_owners
        previous_delivered_ids = set(self.goal_ledger.delivered)
        self.physical_owners = {}
        self.container_contents = {}
        for agent_id, agent in enumerate(agents):
            state = states[str(agent_id)]
            current_room = getattr(agent, "current_room", None)
            known_rooms = list(getattr(agent, "rooms_name", None) or [])
            if current_room and current_room not in known_rooms:
                known_rooms.append(current_room)
            explored = getattr(agent, "rooms_explored", None) or {}
            for room in known_rooms:
                if room is None or str(room) == "None":
                    continue
                room = str(room)
                entry = self.room_memory.setdefault(
                    room,
                    {
                        "visited_by": [],
                        "coverage_by_agent": {},
                        "last_visit_frame": None,
                    },
                )
                coverage = str(explored.get(room, "none") or "none")
                entry["coverage_by_agent"][str(agent_id)] = coverage
                if room == current_room:
                    visitors = set(entry.get("visited_by", []))
                    visitors.add(agent_id)
                    entry["visited_by"] = sorted(visitors)
                    entry["last_visit_frame"] = self.frame
            for entity in _real_objects(state.get("visible_objects", [])):
                self._observe_entity(
                    entity, agent_id, self.frame, evidence_source="direct"
                )
            for arm_index, item in enumerate(state.get("held_objects", [])):
                if not item or item.get("id") is None:
                    continue
                arm = "left" if arm_index == 0 else "right"
                self._record_payload(item, agent_id, arm, self.frame)
            room_cache = getattr(agent, "object_per_room", None) or {}
            for room, by_type in room_cache.items():
                for entities in by_type.values():
                    for entity in entities:
                        annotated = copy.deepcopy(entity)
                        annotated["room"] = room
                        self._observe_entity(
                            annotated,
                            agent_id,
                            self.frame,
                            evidence_source="local_memory",
                        )
            action_id = int(state.get("action_id", 0))
            evidence_key = (agent_id, action_id)
            if (
                action_id > 0
                and state.get("action_terminal")
                and (evidence_key not in self._seen_action_evidence)
            ):
                evidence = {
                    "evidence_id": f"frame{self.frame}:agent{agent_id}:action{action_id}",
                    "agent": agent_id,
                    "action_id": action_id,
                    "action_type": state.get("action_type"),
                    "status": state.get("action_status"),
                    "valid": bool(state.get("valid", True)),
                    "completed_frame": state.get("action_completed_frame"),
                }
                self.execution_evidence.append(evidence)
                self._seen_action_evidence.add(evidence_key)
                if not evidence["valid"] or evidence["status"] not in (
                    "success",
                    "still_dropping",
                ):
                    self.failure_counts[agent_id] += 1
                else:
                    self.failure_counts[agent_id] = 0
        message_vector = next(
            (
                state.get("messages")
                for state in states.values()
                if state.get("messages") is not None
            ),
            None,
        )
        if message_vector is not None:
            for sender, message in enumerate(message_vector[: len(agents)]):
                normalized = str(message).strip() if message is not None else None
                if normalized and normalized != self._last_observed_messages.get(
                    sender
                ):
                    self.dialogue_events.append(
                        {"frame": self.frame, "sender": sender, "message": normalized}
                    )
                self._last_observed_messages[sender] = normalized
        for object_id, name in (delivered_objects or {}).items():
            self.goal_ledger.mark_delivered(int(object_id), str(name))
        delivered_ids = set(self.goal_ledger.delivered)
        for object_id, owner in self.physical_owners.items():
            previous_owner = previous_owners.get(object_id) or {}
            if owner.get("agent") is not None and (
                previous_owner.get("agent") != owner.get("agent")
                or previous_owner.get("carrier") != owner.get("carrier")
            ):
                task_id = self._entity_task_key(object_id)
                self.note_task_progress(
                    task_id, int(owner["agent"]), "physical_ownership_acquired"
                )
        for object_id in sorted(delivered_ids - previous_delivered_ids):
            previous_owner = previous_owners.get(object_id) or {}
            agent_id = previous_owner.get("agent")
            task_id = self._entity_task_key(object_id)
            self.note_task_progress(task_id, agent_id, "evaluator_confirmed_delivery")
            if agent_id is not None:
                delivery_task_id = self._delivery_task_key(int(agent_id))
                self.note_task_progress(
                    delivery_task_id, int(agent_id), "evaluator_confirmed_delivery"
                )
                delivery_task = self.tasks.get(delivery_task_id)
                if delivery_task is not None:
                    delivery_task.update(
                        status="completed",
                        owner=None,
                        reason="evaluator_confirmed_delivery",
                        blocked_until=None,
                        updated_frame=self.frame,
                    )
        for object_id in list(self.claims):
            claim = self.claims[object_id]
            if (
                object_id in delivered_ids
                or object_id in self.physical_owners
                or self.frame - claim["frame"] > self.CLAIM_TTL_FRAMES
            ):
                del self.claims[object_id]
                self.emit_coordination_event(
                    "claim_released",
                    claim.get("agent"),
                    self._entity_task_key(object_id),
                    claim_scope="object",
                    claim_id=int(object_id),
                    reason="task_completed"
                    if object_id in delivered_ids
                    else "physical_ownership_acquired"
                    if object_id in self.physical_owners
                    else "lease_expired",
                )
        for room_id in list(self.room_claims):
            claim = self.room_claims[room_id]
            if self.frame - claim["frame"] > self.ROOM_CLAIM_TTL_FRAMES:
                del self.room_claims[room_id]
                self.emit_coordination_event(
                    "claim_released",
                    claim.get("agent"),
                    self._room_task_key(str(room_id)),
                    claim_scope="room",
                    claim_id=int(room_id),
                    reason="lease_expired",
                )
        self._refresh_tasks(agents)
        self._compact_history()

    def _compact_history(self) -> None:
        """Bound in-memory history without deleting current physical truth."""
        self.dialogue_events = self.dialogue_events[-self.MAX_DIALOGUE_EVENTS :]
        self.execution_evidence = self.execution_evidence[
            -self.MAX_EXECUTION_EVIDENCE :
        ]
        self.reviews = self.reviews[-self.MAX_REVIEWS :]
        self.proposal_history = self.proposal_history[-self.MAX_PROPOSAL_HISTORY :]
        self.task_history = self.task_history[-self.MAX_TASK_HISTORY :]
        self.coordination_events = self.coordination_events[
            -self.MAX_COORDINATION_EVENTS :
        ]

    def claim(self, object_id: int, agent_id: int, reason: str) -> None:
        entity = self.known_entities.get(int(object_id), {})
        previous = self.claims.get(int(object_id))
        self.claims[int(object_id)] = {
            "agent": agent_id,
            "frame": self.frame,
            "reason": reason,
            "kind": entity.get("category", _entity_category(entity.get("type"))),
        }
        if previous is None or previous.get("agent") != agent_id:
            self.emit_coordination_event(
                "claim_acquired",
                agent_id,
                self._entity_task_key(object_id),
                claim_scope="object",
                claim_id=int(object_id),
                reason=reason,
            )

    def release_agent_claims(
        self, agent_id: int, reason: str = "agent_release"
    ) -> None:
        for object_id in list(self.claims):
            if self.claims[object_id]["agent"] == agent_id:
                del self.claims[object_id]
                self.emit_coordination_event(
                    "claim_released",
                    agent_id,
                    self._entity_task_key(object_id),
                    claim_scope="object",
                    claim_id=int(object_id),
                    reason=reason,
                )

    def release_stale_agent_claims(
        self, agent_id: int, active_target_id: Optional[int]
    ) -> None:
        if active_target_id is None:
            return
        for object_id in list(self.claims):
            if (
                self.claims[object_id]["agent"] == agent_id
                and object_id != active_target_id
            ):
                del self.claims[object_id]
                self.emit_coordination_event(
                    "claim_released",
                    agent_id,
                    self._entity_task_key(object_id),
                    claim_scope="object",
                    claim_id=int(object_id),
                    reason="planner_selected_different_target",
                )

    def claim_room(self, room_id: int, agent_id: int, plan: str) -> None:
        for claimed_room in list(self.room_claims):
            if (
                self.room_claims[claimed_room]["agent"] == agent_id
                and claimed_room != room_id
            ):
                del self.room_claims[claimed_room]
                self.emit_coordination_event(
                    "claim_released",
                    agent_id,
                    self._room_task_key(str(claimed_room)),
                    claim_scope="room",
                    claim_id=int(claimed_room),
                    reason="planner_selected_different_room",
                )
        previous = self.room_claims.get(int(room_id))
        self.room_claims[int(room_id)] = {
            "agent": agent_id,
            "frame": self.frame,
            "plan": plan,
        }
        if previous is None or previous.get("agent") != agent_id:
            self.emit_coordination_event(
                "claim_acquired",
                agent_id,
                self._room_task_key(str(room_id)),
                claim_scope="room",
                claim_id=int(room_id),
                reason="room_exploration_reservation",
            )

    def release_agent_room_claims(
        self, agent_id: int, reason: str = "room_reservation_release"
    ) -> None:
        for room_id in list(self.room_claims):
            if self.room_claims[room_id]["agent"] == agent_id:
                del self.room_claims[room_id]
                self.emit_coordination_event(
                    "claim_released",
                    agent_id,
                    self._room_task_key(str(room_id)),
                    claim_scope="room",
                    claim_id=int(room_id),
                    reason=reason,
                )

    def release_task_claims(
        self, agent_id: int, task_id: Optional[str], reason: str
    ) -> None:
        """Release only the reservation owned by one abandoned V4 task."""
        if task_id is None:
            return
        if task_id.startswith("entity:"):
            try:
                object_id = int(task_id.split(":", 1)[1])
            except (TypeError, ValueError):
                return
            claim = self.claims.get(object_id)
            if claim and claim.get("agent") == agent_id:
                del self.claims[object_id]
                self.emit_coordination_event(
                    "claim_released",
                    agent_id,
                    task_id,
                    claim_scope="object",
                    claim_id=object_id,
                    reason=reason,
                )
            return
        if task_id.startswith("room:"):
            matches = re.findall("\\((\\d+)\\)", task_id)
            if not matches:
                return
            room_id = int(matches[-1])
            claim = self.room_claims.get(room_id)
            if claim and claim.get("agent") == agent_id:
                del self.room_claims[room_id]
                self.emit_coordination_event(
                    "claim_released",
                    agent_id,
                    task_id,
                    claim_scope="room",
                    claim_id=room_id,
                    reason=reason,
                )

    def public_view(self) -> dict:
        def task_summary(task: dict) -> dict:
            return {
                key: copy.deepcopy(task.get(key))
                for key in (
                    "task_id",
                    "kind",
                    "status",
                    "priority",
                    "owner",
                    "last_owner",
                    "object_id",
                    "name",
                    "room",
                    "reason",
                )
                if task.get(key) is not None
            }

        shared_task_ids = {task_id for task_id in self.active_tasks.values() if task_id}
        shared_task_ids.update(
            (self._entity_task_key(object_id) for object_id in self.claims)
        )
        shared_task_ids.update(
            (
                task_id
                for (task_id, task) in self.tasks.items()
                if task.get("owner") is not None
                or task.get("last_owner") is not None
                or task.get("status") == "completed"
            )
        )
        return {
            "frame": self.frame,
            "episode_epoch": self.episode_epoch,
            "goal": self.goal_ledger.view(),
            "claims": copy.deepcopy(self.claims),
            "room_claims": {
                str(room_id): {"agent": claim.get("agent"), "frame": claim.get("frame")}
                for (room_id, claim) in self.room_claims.items()
            },
            "physical_owners": {
                str(object_id): {"agent": owner.get("agent")}
                for (object_id, owner) in self.physical_owners.items()
            },
            "intents": {
                str(agent_id): {
                    "stage": intent.get("stage"),
                    "target_id": intent.get("target_id"),
                    "plan": intent.get("plan"),
                    "execution": intent.get("execution"),
                }
                for (agent_id, intent) in self.current_intents.items()
            },
            "active_tasks": copy.deepcopy(self.active_tasks),
            "tasks": {
                task_id: task_summary(task)
                for (task_id, task) in self.tasks.items()
                if task_id in shared_task_ids
            },
            "coordination_events": copy.deepcopy(self.coordination_events[-12:]),
            "recent_coordination_events": copy.deepcopy(self.coordination_events[-12:]),
            "planning_loop_guards": copy.deepcopy(self.planning_loop_guards),
            "recent_reviews": copy.deepcopy(self.reviews[-6:]),
            "recent_dialogue": copy.deepcopy(self.dialogue_events[-6:]),
        }

    def _payload_view_v4(self, agent_id: int) -> List[dict]:
        """Return only identity facts, not carrier/container internals."""
        return [
            {
                "id": object_id,
                "name": self.known_entities.get(object_id, {}).get("name"),
            }
            for (object_id, owner) in sorted(self.physical_owners.items())
            if owner.get("agent") == agent_id
            and object_id not in self.goal_ledger.delivered
        ]

    def _decision_view_v4(
        self, agent_id: int, state: dict, agents: List[Any], delivery_advisory: dict
    ) -> dict:
        """Compile V4's policy-light, coordination-only decision card.

        The card intentionally omits distances, age/confidence scores,
        deadline estimates, container utility, and coverage heuristics.  Core
        filters only evaluator/ownership/claim facts; all remaining legal
        choices stay visible to the local planner.
        """
        remaining = self.goal_ledger.remaining_counts()
        task_queue = self.task_queue(agent_id, state)

        def task_summary(task_id: Optional[str]) -> Optional[dict]:
            task = self.tasks.get(task_id) if task_id else None
            if task is None:
                return None
            return {
                key: copy.deepcopy(task.get(key))
                for key in (
                    "task_id",
                    "kind",
                    "status",
                    "priority",
                    "owner",
                    "last_owner",
                    "object_id",
                    "name",
                    "room",
                    "reason",
                )
                if task.get(key) is not None
            }

        peer_id = 1 - agent_id
        peer_task_id = self.active_tasks.get(peer_id)
        guard = self.planning_loop_guards.get(agent_id)
        delivery_target = None
        candidates = [
            entity
            for entity in self.known_entities.values()
            if entity.get("type") == 2 and entity.get("position") is not None
        ]
        candidates.sort(key=lambda entity: int(entity.get("id", 0)))
        if candidates:
            target = candidates[0]
            delivery_target = {
                key: copy.deepcopy(target.get(key))
                for key in ("id", "name", "type", "position", "room")
                if target.get(key) is not None
            }
            delivery_target["knowledge_source"] = "verified_shared_task_interface"
            delivery_target["episode_epoch"] = self.episode_epoch
        guidance = [
            "All listed candidates are legal choices; fixed priority and active-task continuity are non-binding coordination hints.",
            "Choose the next task and strategy locally; the Harness does not select an alternative target, room, route, or resource.",
            "Claims and reservations prevent duplicate commitments but do not reveal private observations.",
        ]
        if guard:
            guidance.append(
                "The pending planning-loop guard applies only to the next planning boundary: choose an action other than its exact opaque action_identity once; the suspended task remains available afterwards."
            )
        return {
            "frame_budget": {
                "current": self.frame,
                "remaining": delivery_advisory.get("remaining_frames"),
            },
            "goal": {
                "delivered_total": sum(self.goal_ledger.delivered_counts().values()),
                "required_total": sum(self.goal_ledger.required.values()),
                "remaining": {
                    name: count for (name, count) in remaining.items() if count
                },
            },
            "self": {
                "room": getattr(agents[agent_id], "current_room", None),
                "payload": self._payload_view_v4(agent_id),
            },
            "peer": {
                "payload": self._payload_view_v4(peer_id),
                "commitment": task_summary(peer_task_id),
            },
            "active_task": task_summary(self.active_tasks.get(agent_id)),
            **(
                {"delivery_target": delivery_target}
                if delivery_target is not None
                else {}
            ),
            "task_queue": task_queue,
            "active_claims": {
                str(object_id): {"agent": value.get("agent"), "kind": value.get("kind")}
                for (object_id, value) in self.claims.items()
                if value.get("agent") != agent_id
            },
            "planning_loop_guard": copy.deepcopy(guard),
            "coordination_events": copy.deepcopy(self.coordination_events[-12:]),
            "recent_coordination_events": copy.deepcopy(self.coordination_events[-12:]),
            "recent_relevant_reviews": [
                {
                    "frame": review.get("frame"),
                    "trigger": review.get("trigger"),
                    "reason": str(review.get("reason", ""))[:180],
                }
                for review in self.reviews
                if review.get("proposer") == agent_id
            ][-2:],
            "recent_agent_dialogue": [
                {
                    "frame": event["frame"],
                    "speaker": event["sender"],
                    "message": event["message"][:240],
                }
                for event in self.dialogue_events[-4:]
            ],
            "coordination_guidance": guidance,
        }

    def decision_view(
        self, agent_id: int, state: dict, agents: List[Any], delivery_advisory: dict
    ) -> dict:
        """Compile a small, agent-specific working-memory card.

        The complete evidence ledger remains in ``public_view`` and the JSONL
        trace. The planner receives only current, actionable facts. This is a
        deterministic safety boundary: agent-authored dialogue can influence
        relevance, but cannot rewrite evaluator or physical truth.
        """
        return self._decision_view_v4(agent_id, state, agents, delivery_advisory)

    def prompt_summary(
        self, agent_id: int, state: dict, agents: List[Any], delivery_advisory: dict
    ) -> str:
        return BOARD_PREFIX + json.dumps(
            _jsonable(self.decision_view(agent_id, state, agents, delivery_advisory)),
            ensure_ascii=False,
            separators=(",", ":"),
        )


class TDWPeerConsultCoordinator:
    """Coordinate two CoELA planners without sharing their private images."""

    V4_LOOP_REPEAT_THRESHOLD = 2
    V4_COMMUNICATION_EVENT_TYPES = frozenset(
        {
            "task_commitment",
            "task_suspended",
            "task_released",
            "task_failure",
            "task_progress",
            "task_completed",
            "claim_acquired",
            "claim_released",
        }
    )

    def __init__(
        self,
        agents: List[Any],
        logger: Any,
        output_dir: str,
        review_mode: str = "deterministic",
        max_frames: int = 3000,
        protocol_version: str = PROTOCOL_V4,
    ):
        if len(agents) != 2:
            raise ValueError("peer consult currently requires exactly 2 agents")
        if protocol_version not in SUPPORTED_PROTOCOLS:
            raise ValueError(f"unsupported peer-consult protocol {protocol_version!r}")
        self.agents = agents
        self.logger = logger
        self.output_dir = output_dir
        self.review_mode = review_mode
        self.max_frames = int(max_frames)
        self.protocol_version = protocol_version
        self.run_instance_id = uuid.uuid4().hex
        self.board: Optional[TDWSharedBlackboard] = None
        self.protocol_step = 0
        self.log_path = os.path.join(output_dir, "peer_consult.jsonl")
        self.current_episode_id: Optional[int] = None
        self.episode_log_path: Optional[str] = None
        self.last_actions = {0: None, 1: None}
        self.pending_delivery_attempts: Dict[int, dict] = {}
        self.episode_epoch = 0
        self._revised_this_step: set = set()
        self.v4_loop_state: Dict[int, dict] = {}
        self.v4_pending_boundaries: Dict[int, dict] = {}
        self.v4_seen_execution_evidence: set = set()
        self.v4_guard_blocked_this_step: set = set()
        self.v4_planning_boundaries_this_step: set = set()
        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.log_path, "w", encoding="utf-8") as stream:
            stream.write("")
        for agent in self.agents:
            if hasattr(agent, "fix_lm_satisfied"):
                agent.fix_lm_satisfied = True
            if hasattr(agent, "satisfied"):
                agent.authoritative_satisfied = True
            llm = getattr(agent, "LLM", None)
            if llm is not None:
                llm.peer_consult_protocol = protocol_version
                llm.peer_decision_card = None

    def reset(
        self, goal_description: Dict[str, int], episode_id: Optional[int] = None
    ) -> None:
        self.current_episode_id = int(episode_id) if episode_id is not None else None
        self.episode_log_path = None
        if self.current_episode_id is not None:
            episode_dir = os.path.join(self.output_dir, str(self.current_episode_id))
            os.makedirs(episode_dir, exist_ok=True)
            self.episode_log_path = os.path.join(episode_dir, "peer_consult.jsonl")
            with open(self.episode_log_path, "a", encoding="utf-8"):
                pass
        self.episode_epoch += 1
        self.board = TDWSharedBlackboard(
            goal_description,
            episode_epoch=self.episode_epoch,
        )
        self.protocol_step = 0
        self.last_actions = {0: None, 1: None}
        self.pending_delivery_attempts = {}
        self._revised_this_step = set()
        self.v4_loop_state = {
            agent_id: {
                "last_no_progress_identity": None,
                "last_no_progress_task_id": None,
                "last_no_progress_version": None,
                "consecutive_no_progress": 0,
            }
            for agent_id in range(len(self.agents))
        }
        self.v4_pending_boundaries = {}
        self.v4_seen_execution_evidence = set()
        self.v4_guard_blocked_this_step = set()
        self.v4_planning_boundaries_this_step = set()
        for agent in self.agents:
            agent.object_per_room = {}
            if hasattr(agent, "object_list"):
                agent.object_list = {0: [], 1: [], 2: []}
            if hasattr(agent, "target_pos"):
                agent.target_pos = None
            agent._peer_consult_episode_epoch = self.episode_epoch
            llm = getattr(agent, "LLM", None)
            if llm is not None:
                llm.peer_decision_card = None
        self._log(
            {
                "kind": "reset",
                "protocol": self.protocol_version,
                "goal": goal_description,
                "v4_extensions": {"shared_delivery_target_adapter": True},
                **{"episode_epoch": self.episode_epoch},
            }
        )

    def _log(self, event: dict) -> None:
        event = copy.deepcopy(event)
        event.setdefault("schema_version", "peer_consult_v4.0")
        event.setdefault("protocol", self.protocol_version)
        event.setdefault("run_instance_id", self.run_instance_id)
        event.setdefault("episode_id", self.current_episode_id)
        event.setdefault("episode_epoch", self.episode_epoch)
        event.setdefault("protocol_step", self.protocol_step)
        line = json.dumps(_jsonable(event), ensure_ascii=False) + "\n"
        with open(self.log_path, "a", encoding="utf-8") as stream:
            stream.write(line)
        if self.episode_log_path is not None:
            with open(self.episode_log_path, "a", encoding="utf-8") as stream:
                stream.write(line)

    @staticmethod
    def _held(state: dict) -> List[dict]:
        return _real_objects(state.get("held_objects", []))

    def _sync_authoritative_satisfied(self) -> None:
        """Keep planner delivery memory equal to evaluator evidence.

        A successful physical drop is not necessarily a scored delivery: the
        object can land outside the goal radius. V4 removes the
        optimistic self-report before every planning turn and retains
        only object IDs confirmed by the environment evaluator.
        """
        delivered = sorted(self.board.goal_ledger.delivered)
        for agent_id, agent in enumerate(self.agents):
            if not hasattr(agent, "satisfied"):
                continue
            previous = list(getattr(agent, "satisfied", None) or [])
            if previous != delivered:
                stale = sorted(set(previous) - set(delivered))
                if stale:
                    self.board.reviews.append(
                        {
                            "frame": self.board.frame,
                            "reviewer": 1 - agent_id,
                            "proposer": agent_id,
                            "trigger": "delivery_reconciliation",
                            "verdict": "revise",
                            "reason": "planner-local drop acknowledgement was not confirmed by evaluator evidence",
                            "stale_object_ids": stale,
                        }
                    )
            agent.satisfied = list(delivered)

    def _inject_public_board(
        self, agent_id: int, state: dict, delivery_advisory: dict
    ) -> None:
        agent = self.agents[agent_id]
        history = [
            item
            for item in getattr(agent, "dialogue_history", []) or []
            if not item.startswith(BOARD_PREFIX)
        ]
        history.append(
            self.board.prompt_summary(agent_id, state, self.agents, delivery_advisory)
        )
        agent.dialogue_history = history[-14:]

    @staticmethod
    def _clear_plan(agent: Any) -> None:
        agent.plan = None
        if hasattr(agent, "target_pos"):
            agent.target_pos = None

    def _v4_advisory(self, state: dict) -> dict:
        """Return only raw budget/physical facts needed by shared plumbing."""
        return {
            "remaining_frames": max(
                0, self.max_frames - int(state.get("current_frames", 0))
            ),
            "force_delivery": False,
            "priority_delivery": False,
            "recommendation": "planner_decides",
        }

    def _v4_boundary_descriptor(
        self, agent_id: int, intent: dict, proposed_action: dict
    ) -> Optional[dict]:
        """Return a stable opaque identity for one genuine planning turn."""
        if proposed_action.get("type") == "ongoing":
            return None
        task_id = self.board._task_id_from_intent(agent_id, intent)
        normalized_plan = " ".join(str(intent.get("plan") or "").split())
        identity_source = json.dumps(
            {
                "task_id": task_id,
                "stage": intent.get("stage"),
                "target_id": intent.get("target_id"),
                "plan": normalized_plan,
                "action_type": proposed_action.get("type"),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return {
            "task_id": task_id,
            "action_identity": "action:"
            + hashlib.sha256(identity_source.encode("utf-8")).hexdigest()[:16],
            "stage": intent.get("stage"),
        }

    def _v4_reset_loop_history(self, agent_id: int) -> None:
        self.v4_loop_state[agent_id] = {
            "last_no_progress_identity": None,
            "last_no_progress_task_id": None,
            "last_no_progress_version": None,
            "consecutive_no_progress": 0,
        }

    def _v4_arm_loop_guard(
        self, agent_id: int, descriptor: dict, count: int, outcome: str
    ) -> None:
        task_id = descriptor.get("task_id")
        guard = {
            "task_id": task_id,
            "action_identity": descriptor["action_identity"],
            "reason": "repeated_replan_without_task_progress",
            "consecutive_no_progress": int(count),
            "armed_frame": self.board.frame,
            "applies_to_next_planning_boundary": True,
        }
        self.board.planning_loop_guards[agent_id] = guard
        if task_id is not None:
            self.board.suspend_active_task(
                agent_id, "planning_loop_guard_armed", task_id=task_id
            )
            self.board.release_agent_claims(
                agent_id, reason="planning_loop_guard_armed"
            )
            self.board.release_agent_room_claims(
                agent_id, reason="planning_loop_guard_armed"
            )
        self._clear_plan(self.agents[agent_id])
        self.board.emit_coordination_event(
            "planning_loop_guard_armed",
            agent_id,
            task_id,
            action_identity=descriptor["action_identity"],
            outcome=outcome,
            consecutive_no_progress=int(count),
        )

    def _v4_note_no_progress(
        self,
        agent_id: int,
        descriptor: dict,
        outcome: str,
        progress_version_at_start: int,
    ) -> None:
        task_id = descriptor.get("task_id")
        current_progress = int(self.board.task_progress_versions.get(task_id, 0))
        if current_progress != int(progress_version_at_start):
            self._v4_reset_loop_history(agent_id)
            return
        state = self.v4_loop_state.setdefault(
            agent_id,
            {
                "last_no_progress_identity": None,
                "last_no_progress_task_id": None,
                "last_no_progress_version": None,
                "consecutive_no_progress": 0,
            },
        )
        same = (
            state.get("last_no_progress_identity") == descriptor["action_identity"]
            and state.get("last_no_progress_task_id") == task_id
            and (state.get("last_no_progress_version") == current_progress)
        )
        count = int(state.get("consecutive_no_progress", 0)) + 1 if same else 1
        state.update(
            {
                "last_no_progress_identity": descriptor["action_identity"],
                "last_no_progress_task_id": task_id,
                "last_no_progress_version": current_progress,
                "consecutive_no_progress": count,
            }
        )
        self.board.emit_coordination_event(
            "planning_replan",
            agent_id,
            task_id,
            action_identity=descriptor["action_identity"],
            outcome=outcome,
            task_progress=False,
            consecutive_no_progress=count,
        )
        if (
            count >= self.V4_LOOP_REPEAT_THRESHOLD
            and agent_id not in self.board.planning_loop_guards
        ):
            self._v4_arm_loop_guard(agent_id, descriptor, count, outcome)

    def _v4_consume_execution_evidence(self) -> None:
        """Resolve pending planning boundaries from terminal action facts."""
        for evidence in self.board.execution_evidence:
            evidence_id = evidence.get("evidence_id")
            if evidence_id in self.v4_seen_execution_evidence:
                continue
            self.v4_seen_execution_evidence.add(evidence_id)
            agent_id = int(evidence.get("agent"))
            pending = self.v4_pending_boundaries.pop(agent_id, None)
            if pending is None:
                continue
            progressed = int(
                self.board.task_progress_versions.get(pending.get("task_id"), 0)
            ) != int(pending.get("progress_version_at_start", 0))
            successful = bool(evidence.get("valid", True)) and evidence.get(
                "status"
            ) in ("success", "still_dropping")
            if progressed:
                self._v4_reset_loop_history(agent_id)
                continue
            self._v4_note_no_progress(
                agent_id,
                pending,
                "execution_completed_without_task_progress"
                if successful
                else "execution_replan",
                int(pending.get("progress_version_at_start", 0)),
            )

    def _v4_apply_pending_guards(
        self, actions: Dict[str, dict], intents: Dict[int, dict]
    ) -> None:
        """Consume each armed guard at exactly one planning boundary."""
        self.v4_guard_blocked_this_step = set()
        for agent_id, intent in intents.items():
            if agent_id not in self.v4_planning_boundaries_this_step:
                continue
            descriptor = self._v4_boundary_descriptor(
                agent_id, intent, actions[str(agent_id)]
            )
            if descriptor is None:
                continue
            guard = self.board.planning_loop_guards.get(agent_id)
            if guard is None:
                continue
            del self.board.planning_loop_guards[agent_id]
            if guard.get("task_id") == descriptor.get("task_id") and guard.get(
                "action_identity"
            ) == descriptor.get("action_identity"):
                if self._deny(
                    agent_id,
                    actions,
                    "the exact action was just repeated without task progress; choose any other legal action once",
                    "planning_loop_guard",
                ):
                    if descriptor.get("task_id") is not None:
                        self.board.release_agent_claims(
                            agent_id, reason="planning_loop_guard_applied"
                        )
                        self.board.release_agent_room_claims(
                            agent_id, reason="planning_loop_guard_applied"
                        )
                    self.board.emit_coordination_event(
                        "planning_loop_guard_applied",
                        agent_id,
                        descriptor.get("task_id"),
                        action_identity=descriptor["action_identity"],
                        outcome="replan_required",
                    )
                    self.v4_guard_blocked_this_step.add(agent_id)
            else:
                self.board.emit_coordination_event(
                    "planning_loop_guard_consumed",
                    agent_id,
                    guard.get("task_id"),
                    action_identity=guard.get("action_identity"),
                    outcome="different_action_selected",
                )
            self._v4_reset_loop_history(agent_id)

    def _v4_finalize_planning_boundaries(
        self, actions: Dict[str, dict], intents: Dict[int, dict], review_start: int
    ) -> None:
        """Associate executed/reviewed proposals with their next outcome."""
        reviews = self.board.reviews[review_start:]
        for agent_id, intent in intents.items():
            if agent_id not in self.v4_planning_boundaries_this_step:
                continue
            if agent_id in self.v4_guard_blocked_this_step:
                continue
            descriptor = self._v4_boundary_descriptor(
                agent_id, intent, self._v4_proposed_actions[str(agent_id)]
            )
            if descriptor is None:
                continue
            review = next(
                (
                    item
                    for item in reversed(reviews)
                    if item.get("proposer") == agent_id
                    and item.get("verdict") == "revise"
                ),
                None,
            )
            progress_version = int(
                self.board.task_progress_versions.get(descriptor.get("task_id"), 0)
            )
            if review is not None:
                self._v4_note_no_progress(
                    agent_id, descriptor, "review_replan", progress_version
                )
                continue
            paused = any(
                (
                    item.get("proposer") == agent_id and item.get("verdict") == "pause"
                    for item in reviews
                )
            )
            if paused or actions[str(agent_id)].get("type") == "ongoing":
                continue
            descriptor["progress_version_at_start"] = progress_version
            self.v4_pending_boundaries[agent_id] = descriptor

    def _deny(
        self, agent_id: int, actions: Dict[str, dict], reason: str, trigger: str
    ) -> bool:
        if (
            agent_id in self._revised_this_step
            or actions[str(agent_id)].get("type") == "ongoing"
        ):
            return False
        self._clear_plan(self.agents[agent_id])
        actions[str(agent_id)] = {"type": 1 if agent_id == 0 else 2}
        review = {
            "frame": self.board.frame,
            "reviewer": 1 - agent_id,
            "proposer": agent_id,
            "trigger": trigger,
            "verdict": "revise",
            "reason": reason,
            "replacement": actions[str(agent_id)],
        }
        self.board.reviews.append(review)
        proposal_intent = self.board.current_intents.get(agent_id) or {}
        proposal_task_id = self.board._task_id_from_intent(agent_id, proposal_intent)
        if proposal_task_id is not None:
            self.board.suspend_active_task(
                agent_id, f"review_replan:{trigger}", task_id=proposal_task_id
            )
        if proposal_task_id is not None and trigger in {
            "planning_loop_guard",
            "duplicate_claim",
            "claim_conflict",
            "ownership",
            "verified_fact",
            "physical_legality",
            "room_assignment",
        }:
            self.board.release_agent_claims(agent_id, reason=f"review_replan:{trigger}")
            self.board.release_agent_room_claims(
                agent_id, reason=f"review_replan:{trigger}"
            )
        self._revised_this_step.add(agent_id)
        return True

    @staticmethod
    def _v4_coordination_event_id(message: Any) -> Optional[int]:
        """Parse only the opaque reference emitted by the V4 planner."""
        text = str(message or "").strip().strip("\"'` ")
        if ":" not in text:
            return None
        (prefix, value) = text.split(":", 1)
        if prefix.strip().lower() != "coordination_event":
            return None
        try:
            return int(value.strip())
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _v4_render_coordination_event(event: dict) -> str:
        """Render a fixed-schema public event, never model-authored prose."""
        payload = {
            key: copy.deepcopy(event.get(key))
            for key in (
                "event_id",
                "event",
                "agent",
                "task_id",
                "outcome",
                "progress_version",
            )
            if event.get(key) is not None
        }
        return json.dumps(
            {"coordination_event": payload}, ensure_ascii=False, separators=(",", ":")
        )

    def _reconcile_messages_v4(self, actions: Dict[str, dict]) -> None:
        """Bind communication to one public structured lifecycle event.

        The planner decides whether to communicate and names an event ID.
        Only the fixed-schema public event is sent. Free text is never
        inspected for TDW-specific observations; it is simply not a legal V4
        communication payload.
        """
        for agent_id in range(2):
            action = actions[str(agent_id)]
            if action.get("type") != 6:
                continue
            event_id = self._v4_coordination_event_id(action.get("message"))
            event = next(
                (
                    item
                    for item in reversed(self.board.coordination_events[-12:])
                    if item.get("event_id") == event_id
                    and item.get("event") in self.V4_COMMUNICATION_EVENT_TYPES
                ),
                None,
            )
            if event is not None:
                actions[str(agent_id)] = {
                    "type": 6,
                    "message": self._v4_render_coordination_event(event),
                }
                continue
            replacement = {"type": 8, "delay": 1}
            actions[str(agent_id)] = replacement
            self.board.reviews.append(
                {
                    "frame": self.board.frame,
                    "reviewer": 1 - agent_id,
                    "proposer": agent_id,
                    "trigger": "message_event_reference",
                    "verdict": "revise",
                    "reason": "message did not reference one selectable public coordination event",
                    "replacement": replacement,
                }
            )
            self.board.emit_coordination_event(
                "task_failure",
                agent_id,
                None,
                outcome="invalid_structured_event_reference",
            )
            self._revised_this_step.add(agent_id)

    def _govern_room_reservations_v4(
        self, actions: Dict[str, dict], intents: Dict[int, dict]
    ) -> None:
        """Atomically reserve blind exploration scopes without ranking them."""
        room_interests: Dict[int, List[int]] = {}
        for agent_id, intent in intents.items():
            if agent_id in self._revised_this_step:
                continue
            plan = str(intent.get("plan") or "")
            room_id = intent.get("target_id")
            if room_id is not None and plan.startswith(("go to", "explore")):
                room_interests.setdefault(int(room_id), []).append(agent_id)
        for room_id, reservation in list(self.board.room_claims.items()):
            owner = reservation.get("agent")
            if owner in room_interests.get(int(room_id), []):
                continue
            del self.board.room_claims[room_id]
            self.board.emit_coordination_event(
                "claim_released",
                owner,
                self.board._room_task_key(str(room_id)),
                claim_scope="room",
                claim_id=int(room_id),
                reason="reservation_owner_abandoned_intent",
            )
        for room_id, claimants in room_interests.items():
            existing = self.board.room_claims.get(room_id)
            if existing and existing.get("agent") in claimants:
                winner = int(existing["agent"])
            elif existing:
                winner = int(existing["agent"])
            else:
                ordered = sorted(claimants)
                winner = ordered[(self.protocol_step + room_id) % len(ordered)]
            if winner in claimants:
                self.board.claim_room(
                    room_id, winner, intents[winner].get("plan") or "reserved"
                )
            for agent_id in claimants:
                if agent_id == winner:
                    continue
                self._deny(
                    agent_id,
                    actions,
                    f"exploration scope {room_id} is reserved by agent {winner}; choose another legal action",
                    "room_assignment",
                )

    def _record_v4_delivery_attempts(self, actions: Dict[str, dict]) -> None:
        """Record only ownership truth for a proposed V4 release."""
        delivered_before = set(self.board.goal_ledger.delivered)
        for agent_id in range(2):
            action = actions[str(agent_id)]
            if action.get("type") != 5:
                continue
            drop_arm = action.get("arm")
            released_containers = {
                object_id
                for (object_id, owner) in self.board.physical_owners.items()
                if owner.get("agent") == agent_id
                and owner.get("carrier") == "hand"
                and (owner.get("arm") == drop_arm)
                and (self.board.known_entities.get(object_id, {}).get("type") == 1)
            }
            payload_ids = sorted(
                (
                    object_id
                    for (object_id, owner) in self.board.physical_owners.items()
                    if owner.get("agent") == agent_id
                    and self.board.known_entities.get(object_id, {}).get("type") == 0
                    and (object_id not in delivered_before)
                    and (
                        owner.get("carrier") == "hand"
                        and owner.get("arm") == drop_arm
                        or (
                            owner.get("carrier") == "container"
                            and owner.get("container_id") in released_containers
                        )
                    )
                )
            )
            if not payload_ids:
                continue
            self.pending_delivery_attempts[agent_id] = {
                "frame": self.board.frame,
                "payload_ids": payload_ids,
                "delivered_before": sorted(delivered_before),
            }

    def _reconcile_v4_delivery_attempts(self, states: Dict[str, dict]) -> None:
        """Reconcile release solely from evaluator and ownership facts."""
        delivered_now = set(self.board.goal_ledger.delivered)
        for agent_id, attempt in list(self.pending_delivery_attempts.items()):
            state = states[str(agent_id)]
            if not state.get("action_terminal"):
                continue
            payload_ids = set(attempt.get("payload_ids", []))
            unconfirmed = sorted(payload_ids - delivered_now)
            if not unconfirmed:
                delivery_task_id = self.board._delivery_task_key(agent_id)
                delivery_task = self.board.tasks.get(delivery_task_id)
                if delivery_task is not None:
                    delivery_task.update(
                        status="completed",
                        owner=None,
                        reason="evaluator_confirmed_delivery",
                        blocked_until=None,
                        updated_frame=self.board.frame,
                    )
                self.board.emit_coordination_event(
                    "task_completed",
                    agent_id,
                    delivery_task_id,
                    outcome="evaluator_confirmed_delivery",
                )
                self.pending_delivery_attempts.pop(agent_id, None)
                continue
            still_owned = {
                object_id
                for object_id in unconfirmed
                if (self.board.physical_owners.get(object_id) or {}).get("agent")
                == agent_id
            }
            if still_owned:
                for object_id in sorted(still_owned):
                    self.board.emit_coordination_event(
                        "task_failure",
                        agent_id,
                        self.board._entity_task_key(object_id),
                        outcome="physical_release_failed",
                    )
            released_unconfirmed = [
                object_id for object_id in unconfirmed if object_id not in still_owned
            ]
            for object_id in released_unconfirmed:
                task_id = self.board._entity_task_key(object_id)
                task = self.board.tasks.get(task_id)
                if task is not None and task.get("status") not in (
                    "completed",
                    "surplus",
                    "carried",
                ):
                    task.update(
                        status="suspended",
                        owner=None,
                        reason="release_not_evaluator_confirmed",
                        blocked_until=None,
                        updated_frame=self.board.frame,
                    )
                self.board.emit_coordination_event(
                    "task_failure",
                    agent_id,
                    task_id,
                    outcome="release_not_evaluator_confirmed",
                )
            if released_unconfirmed:
                delivery_task_id = self.board._delivery_task_key(agent_id)
                delivery_task = self.board.tasks.get(delivery_task_id)
                if delivery_task is not None:
                    delivery_task.update(
                        status="suspended",
                        owner=None,
                        reason="release_not_evaluator_confirmed",
                        blocked_until=None,
                        updated_frame=self.board.frame,
                    )
            self.pending_delivery_attempts.pop(agent_id, None)

    def _review_and_govern(
        self,
        states: Dict[str, dict],
        actions: Dict[str, dict],
        intents: Dict[int, dict],
    ) -> None:
        self._v4_apply_pending_guards(actions, intents)
        acquisitions: Dict[int, List[int]] = {}
        for agent_id, intent in intents.items():
            if agent_id in self._revised_this_step:
                continue
            active_target = (
                intent.get("target_id") if intent["stage"] == "acquire" else None
            )
            self.board.release_stale_agent_claims(agent_id, active_target)
            if intent["stage"] == "acquire" and intent["target_id"] is not None:
                acquisitions.setdefault(intent["target_id"], []).append(agent_id)
        for target_id, claimants in acquisitions.items():
            existing = self.board.claims.get(target_id)
            owner = self.board.physical_owners.get(target_id)
            delivered = target_id in self.board.goal_ledger.delivered
            if delivered:
                for agent_id in claimants:
                    self._deny(
                        agent_id,
                        actions,
                        f"object {target_id} is already delivered",
                        "verified_fact",
                    )
                continue
            if owner is not None:
                for agent_id in claimants:
                    relation = (
                        "already carried by this agent"
                        if owner.get("agent") == agent_id
                        else f"carried by agent {owner.get('agent')}"
                    )
                    self._deny(
                        agent_id,
                        actions,
                        f"object {target_id} is {relation}",
                        "ownership",
                    )
                task = self.board.tasks.get(self.board._entity_task_key(target_id))
                if task is not None:
                    task.update(
                        status="carried",
                        owner=owner.get("agent"),
                        reason="physical_ownership",
                        updated_frame=self.board.frame,
                    )
                continue
            if existing and existing["agent"] not in claimants:
                for agent_id in claimants:
                    self._deny(
                        agent_id,
                        actions,
                        f"object {target_id} is claimed by agent {existing['agent']}",
                        "claim_conflict",
                    )
                continue
            if existing and existing["agent"] in claimants:
                winner = existing["agent"]
            else:
                ordered = sorted(claimants)
                winner = ordered[(self.protocol_step + int(target_id)) % len(ordered)]
            self.board.claim(target_id, winner, "accepted_acquire_intent")
            for agent_id in claimants:
                if agent_id != winner:
                    self._deny(
                        agent_id,
                        actions,
                        f"duplicate claim on object {target_id}; agent {winner} owns the claim",
                        "duplicate_claim",
                    )
        for agent_id, intent in intents.items():
            if agent_id in self._revised_this_step:
                continue
            target_id = intent.get("target_id")
            if intent["stage"] != "acquire" or target_id is None:
                continue
            if len(acquisitions.get(target_id, [])) > 1:
                continue
            claim = self.board.claims.get(target_id)
            if claim and claim["agent"] != agent_id:
                self._deny(
                    agent_id,
                    actions,
                    f"object {target_id} is claimed by agent {claim['agent']}",
                    "claim_conflict",
                )
        for agent_id in range(2):
            if agent_id in self._revised_this_step:
                continue
            intent = intents[agent_id]
            if (
                intent["stage"] == "acquire"
                and len(self._held(states[str(agent_id)])) >= 2
            ):
                self._deny(
                    agent_id,
                    actions,
                    "the physical adapter reports no free hand for an acquisition action",
                    "physical_legality",
                )
        self._govern_room_reservations_v4(actions, intents)
        self._reconcile_messages_v4(actions)

    def act(
        self,
        states: Dict[str, dict],
        delivered_objects: Optional[Dict[int, str]] = None,
    ) -> Dict[str, dict]:
        if self.board is None:
            raise RuntimeError("coordinator.reset() must be called first")
        self.protocol_step += 1
        self.board.observe(states, self.agents, delivered_objects)
        self._v4_consume_execution_evidence()
        self._sync_authoritative_satisfied()
        self._revised_this_step = set()
        advisories = {
            agent_id: self._v4_advisory(states[str(agent_id)]) for agent_id in range(2)
        }
        communication_events = {agent_id: [] for agent_id in range(2)}
        self._communication_events_this_step = communication_events
        self._advisories_this_step = advisories
        decision_cards = {}
        for agent_id, agent in enumerate(self.agents):
            decision_cards[agent_id] = self.board.decision_view(
                agent_id, states[str(agent_id)], self.agents, advisories[agent_id]
            )
            llm = getattr(agent, "LLM", None)
            if llm is not None:
                llm.peer_decision_card = decision_cards[agent_id]
            self._inject_public_board(
                agent_id, states[str(agent_id)], advisories[agent_id]
            )
        actions = {}
        v4_history_lengths = {}
        self.v4_planning_boundaries_this_step = set()
        for agent_id, agent in enumerate(self.agents):
            history = getattr(agent, "action_history", None)
            v4_history_lengths[agent_id] = (
                len(history) if isinstance(history, list) else None
            )
        for agent_id, agent in enumerate(self.agents):
            llm = getattr(agent, "LLM", None)
            original_message_gate = getattr(llm, "allow_message_this_turn", None)
            try:
                actions[str(agent_id)] = agent.act(states[str(agent_id)])
            finally:
                if original_message_gate is not None:
                    llm.allow_message_this_turn = original_message_gate
            explicit_boundary = getattr(agent, "_peer_consult_planning_boundary", None)
            history = getattr(agent, "action_history", None)
            history_grew = (
                v4_history_lengths.get(agent_id) is not None
                and isinstance(history, list)
                and (len(history) > v4_history_lengths[agent_id])
            )
            is_boundary = (
                bool(explicit_boundary)
                if explicit_boundary is not None
                else history_grew
            )
            if is_boundary:
                self.v4_planning_boundaries_this_step.add(agent_id)
            if explicit_boundary is not None:
                agent._peer_consult_planning_boundary = False
        intents = {}
        for agent_id, agent in enumerate(self.agents):
            plan = getattr(agent, "plan", None)
            if agent_id in self.v4_planning_boundaries_this_step:
                selected_plan = getattr(agent, "_peer_consult_selected_plan", None)
                if selected_plan is not None:
                    plan = selected_plan
            intents[agent_id] = _intent_from_plan(
                agent_id, plan, actions[str(agent_id)], states[str(agent_id)]
            )
        self.board.current_intents = copy.deepcopy(intents)
        self.board.sync_tasks(intents, self.v4_planning_boundaries_this_step)
        self._v4_proposed_actions = copy.deepcopy(actions)
        review_start = len(self.board.reviews)
        self._review_and_govern(states, actions, intents)
        self._v4_finalize_planning_boundaries(actions, intents, review_start)
        self._record_v4_delivery_attempts(actions)
        validator_reviews = self.board.reviews[review_start:]
        proposals = {
            agent_id: {
                "intent": intents[agent_id],
                "planning_boundary": agent_id in self.v4_planning_boundaries_this_step,
                "task_action": self._v4_boundary_descriptor(
                    agent_id,
                    intents[agent_id],
                    self._v4_proposed_actions[str(agent_id)],
                )
                if agent_id in self.v4_planning_boundaries_this_step
                else None,
                "proposed_action": copy.deepcopy(
                    self._v4_proposed_actions[str(agent_id)]
                ),
                "validator": [
                    {
                        "trigger": item.get("trigger"),
                        "verdict": item.get("verdict"),
                        "reason": item.get("reason"),
                    }
                    for item in validator_reviews
                    if item.get("proposer") == agent_id
                ],
                "final_action": copy.deepcopy(actions[str(agent_id)]),
            }
            for agent_id in range(2)
        }
        self.board.proposal_history.append(
            {"frame": self.board.frame, "proposals": proposals}
        )
        self.board._compact_history()
        self.last_actions = copy.deepcopy(actions)
        self._log(
            {
                "kind": "decision",
                "protocol_step": self.protocol_step,
                "frame": self.board.frame,
                "board": self.board.public_view(),
                "decision_cards": decision_cards,
                "communication_events": communication_events,
                "communication_event_details": None,
                "proposals": proposals,
            }
        )
        return actions

    def observe_outcome(
        self,
        states: Dict[str, dict],
        evaluator_progress: Tuple[int, int, bool],
        info: dict,
        delivered_objects: Optional[Dict[int, str]] = None,
    ) -> None:
        """Record evaluator truth without exposing it in planner prompts."""
        self.board.observe(states, self.agents, delivered_objects)
        self.board.evaluator_progress = tuple(evaluator_progress)
        self._v4_consume_execution_evidence()
        self._reconcile_v4_delivery_attempts(states)
        self._log(
            {
                "kind": "outcome",
                "frame": self.board.frame,
                "evaluator_progress": evaluator_progress,
                "num_frames_for_step": info.get("num_frames_for_step"),
                "evidence": self.board.execution_evidence[-4:],
            }
        )

    def finalize(self, evaluator_progress: Tuple[int, int, bool]) -> None:
        self._log(
            {
                "kind": "final",
                "frame": self.board.frame,
                "goal_ledger": self.board.goal_ledger.view(),
                "evaluator_progress": evaluator_progress,
                "protocol_steps": self.protocol_step,
            }
        )
