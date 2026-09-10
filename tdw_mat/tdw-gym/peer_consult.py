"""Evidence-centric coordination for two TDW-MAT Replicants.

The local perception and skill executors remain inside the published CoELA
``lm_agent`` instances.  This module adds the interaction protocol used by the
new method: a structured memory board, short-lived intents and claims,
container-aware transport evidence, selective peer review, deterministic
conflict checking, and execution evidence keyed by environment action tickets.

The board is centralized for the first TDW-MAT adaptation.  It shares symbolic
facts derived from the two local observations; raw RGB-D observations and the
agents' local occupancy maps are never copied between agents.
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
PROTOCOL_V3 = "PeerConsultV3"
PROTOCOL_V31 = "PeerConsultV3.1"
PROTOCOL_V32 = "PeerConsultV3.2"
PROTOCOL_V33 = "PeerConsultV3.3"
PROTOCOL_V34 = "PeerConsultV3.4"
PROTOCOL_V35 = "PeerConsultV3.5"
PROTOCOL_V4 = "PeerConsultV4"
SPARSE_COMMUNICATION_PROTOCOLS = (
    PROTOCOL_V31, PROTOCOL_V32, PROTOCOL_V33, PROTOCOL_V34, PROTOCOL_V35)
SUPPORTED_PROTOCOLS = (
    PROTOCOL_V3, PROTOCOL_V31, PROTOCOL_V32, PROTOCOL_V33, PROTOCOL_V34,
    PROTOCOL_V35, PROTOCOL_V4)
PROTOCOL_VERSION = PROTOCOL_V3

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
        return {str(key): _jsonable(item) for key, item in value.items()}
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
    matches = re.findall(r"\((\d+)\)", plan)
    return int(matches[-1]) if matches else None


def _intent_from_plan(agent_id: int, plan: Optional[str], action: dict,
                      state: dict) -> dict:
    """Convert a CoELA plan into the short-lived public intent schema."""
    # Keep the semantic stage while a low-level action is in flight.  Calling
    # every ongoing navigation step ``continue`` used to discard the target,
    # so object claims expired precisely while an agent was still pursuing
    # them.
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
    confidence = 0.88
    if not state.get("valid", True):
        confidence = 0.35
    if state.get("action_status") not in (None, "success", "ongoing"):
        confidence = min(confidence, 0.5)
    return {
        "agent": agent_id,
        "stage": stage,
        "target_id": _plan_target_id(plan),
        "plan": plan,
        "action": copy.deepcopy(action),
        "execution": ("ongoing" if action.get("type") == "ongoing" or
                      state.get("status") == 0 else "ready"),
        "confidence": confidence,
        "frame": int(state.get("current_frames", 0)),
    }


class TDWGoalLedger:
    """Count-based TDW-MAT goal ledger with concrete delivery evidence."""

    def __init__(self, required: Dict[str, int]):
        self.required = {str(name): int(count)
                         for name, count in required.items()}
        self.delivered: Dict[int, str] = {}

    def mark_delivered(self, object_id: int, name: str) -> None:
        if name in self.required:
            self.delivered[int(object_id)] = name

    def delivered_counts(self) -> Dict[str, int]:
        counts = Counter(self.delivered.values())
        return {name: counts.get(name, 0) for name in self.required}

    def remaining_counts(self) -> Dict[str, int]:
        delivered = self.delivered_counts()
        return {name: max(count - delivered.get(name, 0), 0)
                for name, count in self.required.items()}

    def complete(self) -> bool:
        return not any(self.remaining_counts().values())

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

    def __init__(self, goal_description: Dict[str, int],
                 semantic_state_v34: bool = False,
                 semantic_state_v35: bool = False,
                 mechanism_v4: bool = False,
                 episode_epoch: int = 0):
        self.goal_ledger = TDWGoalLedger(goal_description)
        # V3.4 is intentionally opt-in so archived V3.x runs retain their
        # exact state semantics. It makes evaluator-delivered object IDs
        # terminal throughout payload and reusable-container bookkeeping.
        self.semantic_state_v34 = bool(semantic_state_v34)
        # V3.5 inherits V3.4's terminal evaluator semantics, then adds
        # episode-scoped evidence and an explicit delivery-confirmation
        # lifecycle.  Keeping a separate flag preserves archived V3.4 runs.
        self.semantic_state_v35 = bool(semantic_state_v35)
        # V4 deliberately carries only generic coordination machinery.  In
        # particular, this flag must not enable V3.x's TDW-specific distance,
        # deadline, container, or navigation policy.
        self.mechanism_v4 = bool(mechanism_v4)
        self.v4_shared_delivery_target_enabled = (
            self.mechanism_v4 and
            os.environ.get("TDW_MAT_V4_SHARED_DELIVERY_TARGET", "0") == "1")
        self.episode_epoch = int(episode_epoch)
        self.known_entities: Dict[int, dict] = {}
        self.physical_owners: Dict[int, dict] = {}
        self.container_contents: Dict[int, List[int]] = {}
        self.container_states: Dict[int, dict] = {}
        self.claims: Dict[int, dict] = {}
        self.room_claims: Dict[int, dict] = {}
        self.room_memory: Dict[str, dict] = {}
        self.target_cooldowns: Dict[str, dict] = {}
        self.current_intents: Dict[int, dict] = {}
        # A payload state is observational evidence, not a promise to follow a
        # long-horizon task contract. It exists only while payload is held.
        self.payload_states: Dict[int, dict] = {}
        self.peer_support_requests: List[dict] = []
        self.reviews: List[dict] = []
        self.execution_evidence: List[dict] = []
        self.proposal_history: List[dict] = []
        # V3.3 keeps unfinished work independently of the one plan string in
        # CoELA.  A plan may be interrupted or rejected, but the underlying
        # task remains pending/suspended until evaluator or physical evidence
        # proves it complete.
        self.tasks: Dict[str, dict] = {}
        self.active_tasks: Dict[int, Optional[str]] = {0: None, 1: None}
        self.task_history: List[dict] = []
        # These are coordination events, not observations.  They contain only
        # task/claim/failure/release lifecycle facts and are safe to expose to
        # both planners without copying RGB-D, maps, paths, or object poses.
        self.coordination_events: List[dict] = []
        self._coordination_event_sequence = 0
        # A guard is a one-boundary exclusion of one opaque action identity.
        # It is intentionally neither a target cooldown nor a replacement
        # policy; the local planner remains responsible for its next choice.
        self.planning_loop_guards: Dict[int, dict] = {}
        self.task_progress_versions: Counter = Counter()
        self.failure_counts: Counter = Counter()
        self.delivery_failure_counts: Counter = Counter()
        self._seen_action_evidence: set = set()
        self.dialogue_events: List[dict] = []
        self._last_observed_messages = {0: None, 1: None}
        self.frame = 0
        self.evaluator_progress = None

    def emit_coordination_event(self, event: str, agent: Optional[int],
                                task_id: Optional[str] = None,
                                **details: Any) -> Optional[dict]:
        """Append one minimal, structured coordination lifecycle event.

        V3.x traces are kept byte-for-byte compatible at the schema level:
        events are collected only for the explicitly selected V4 protocol.
        Callers must pass coordination identifiers/outcomes, never local
        observations or benchmark-specific geometry.
        """
        if not self.mechanism_v4:
            return None
        self._coordination_event_sequence += 1
        record = {
            "event_id": self._coordination_event_sequence,
            "frame": self.frame,
            "event": str(event),
            "agent": agent,
            "task_id": task_id,
        }
        record.update({key: copy.deepcopy(value)
                       for key, value in details.items()
                       if value is not None})
        self.coordination_events.append(record)
        self.coordination_events = self.coordination_events[
            -self.MAX_COORDINATION_EVENTS:]
        return record

    def note_task_progress(self, task_id: Optional[str],
                           agent: Optional[int], event: str) -> None:
        """Record an adapter-reported task progress boundary.

        Core treats this as an opaque monotonic heartbeat.  It doesn't infer
        progress from distance, confidence, target type, or container state.
        """
        if not self.mechanism_v4 or task_id is None:
            return
        self.task_progress_versions[str(task_id)] += 1
        self.emit_coordination_event(
            "task_progress", agent, str(task_id), outcome=str(event),
            progress_version=int(self.task_progress_versions[str(task_id)]))

    def suspend_active_task(
            self, agent_id: int, reason: str,
            task_id: Optional[str] = None) -> Optional[str]:
        """Suspend, but never delete, the current task."""
        task_id = task_id or self.active_tasks.get(agent_id)
        task = self.tasks.get(task_id) if task_id else None
        if task is not None and task.get("status") not in (
                "completed", "surplus", "carried"):
            task.update(status="suspended", owner=None, reason=reason,
                        blocked_until=None, updated_frame=self.frame)
            self.emit_coordination_event(
                "task_suspended", agent_id, task_id, reason=reason)
        if self.active_tasks.get(agent_id) == task_id:
            self.active_tasks[agent_id] = None
        return task_id

    def _container_state(self, object_id: int) -> dict:
        """Return persistent lifecycle state for a transport container."""
        object_id = int(object_id)
        return self.container_states.setdefault(object_id, {
            "container_id": object_id,
            "lifecycle": "available",
            "delivered_contents": [],
            "parked_frame": None,
            "delivery_pending_frame": None,
            "delivery_pending_until": None,
            "delivery_attempts": 0,
            "held_by": None,
            "updated_frame": self.frame,
        })

    def container_parked_at_bed(self, object_id: int) -> bool:
        if not self.semantic_state_v34:
            return False
        state = self.container_states.get(int(object_id)) or {}
        return state.get("lifecycle") == "parked_at_bed"

    def container_delivery_pending(self, object_id: int) -> bool:
        """Return whether a released container is quarantined for scoring.

        TDW releases a container and its children before the evaluator has
        necessarily credited the children.  V3.5 keeps that physical resource
        out of the ordinary acquisition pool during confirmation and retry
        backoff; older protocols retain their previous lifecycle exactly.
        """
        if not self.semantic_state_v35:
            return False
        state = self.container_states.get(int(object_id)) or {}
        if state.get("lifecycle") != "delivery_pending":
            return False
        pending_until = state.get("delivery_pending_until")
        if pending_until is None or self.frame < int(pending_until):
            return True
        state.update({
            "lifecycle": "available",
            "delivery_pending_frame": None,
            "delivery_pending_until": None,
            "held_by": None,
            "updated_frame": self.frame,
        })
        return False

    def mark_container_delivery_pending(self, object_id: int,
                                        until_frame: int,
                                        new_attempt: bool = False) -> None:
        if not self.semantic_state_v35:
            return
        state = self._container_state(object_id)
        if state.get("lifecycle") == "parked_at_bed":
            return
        state.update({
            "lifecycle": "delivery_pending",
            "delivery_pending_frame": (
                state.get("delivery_pending_frame")
                if state.get("delivery_pending_frame") is not None
                else self.frame),
            "delivery_pending_until": max(
                int(until_frame),
                int(state.get("delivery_pending_until") or 0)),
            "delivery_attempts": (int(state.get("delivery_attempts", 0)) +
                                  (1 if new_attempt else 0)),
            "held_by": None,
            "updated_frame": self.frame,
        })

    def confirm_container_delivery(self, object_id: int,
                                   delivered_contents: Iterable[int]) -> None:
        if not self.semantic_state_v35:
            return
        state = self._container_state(object_id)
        credited = set(state.get("delivered_contents") or [])
        credited.update(int(value) for value in delivered_contents)
        state.update({
            "lifecycle": "parked_at_bed",
            "delivered_contents": sorted(credited),
            "parked_frame": (state.get("parked_frame")
                             if state.get("parked_frame") is not None
                             else self.frame),
            "delivery_pending_frame": None,
            "delivery_pending_until": None,
            "held_by": None,
            "updated_frame": self.frame,
        })

    def _reconcile_container_lifecycle(
            self, delivered_ids: set,
            previous_owners: Dict[int, dict],
            previous_contents: Dict[int, List[int]]) -> None:
        """Retire containers whose contents received evaluator credit.

        A TDW drop leaves an object physically inside its container. Without
        a persistent lifecycle, re-grasping that container resurrects the
        credited child as a fresh payload. Evaluator credit for a contained
        child is sufficient evidence that the containing resource was parked
        at the goal location.
        """
        if not self.semantic_state_v34:
            return

        delivered_by_container: Dict[int, set] = {}
        owner_sources = (previous_owners, self.physical_owners)
        content_sources = (previous_contents, self.container_contents)
        for child_id in delivered_ids:
            for owners in owner_sources:
                owner = owners.get(int(child_id)) or {}
                if (owner.get("carrier") == "container" and
                        owner.get("container_id") is not None):
                    container_id = int(owner["container_id"])
                    delivered_by_container.setdefault(
                        container_id, set()).add(int(child_id))
            for contents in content_sources:
                for container_id, child_ids in contents.items():
                    if int(child_id) in child_ids:
                        delivered_by_container.setdefault(
                            int(container_id), set()).add(int(child_id))

        for container_id, entity in self.known_entities.items():
            if entity.get("type") != 1:
                continue
            lifecycle = self._container_state(container_id)
            owner = self.physical_owners.get(container_id)
            lifecycle["held_by"] = (
                owner.get("agent") if owner is not None else None)
            lifecycle["updated_frame"] = self.frame

        for container_id, child_ids in delivered_by_container.items():
            lifecycle = self._container_state(container_id)
            credited = set(lifecycle.get("delivered_contents") or [])
            credited.update(int(value) for value in child_ids)
            lifecycle.update({
                "lifecycle": "parked_at_bed",
                "delivered_contents": sorted(credited),
                "parked_frame": (lifecycle.get("parked_frame")
                                 if lifecycle.get("parked_frame") is not None
                else self.frame),
                "delivery_pending_frame": None,
                "delivery_pending_until": None,
                "updated_frame": self.frame,
            })

    @staticmethod
    def _entity_task_key(object_id: int) -> str:
        return f"entity:{int(object_id)}"

    @staticmethod
    def _room_task_key(room: str) -> str:
        return f"room:{room}"

    @staticmethod
    def _delivery_task_key(agent_id: int) -> str:
        return f"delivery:{int(agent_id)}"

    def _ensure_task(self, task_id: str, kind: str, priority: int,
                     **facts: Any) -> dict:
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
            if self.mechanism_v4:
                task["last_owner"] = None
            self.tasks[task_id] = task
        task.update({key: copy.deepcopy(value)
                     for key, value in facts.items()})
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
                    self._entity_task_key(object_id), "deliver_goal_object",
                    100, object_id=object_id, name=name,
                    room=entity.get("room"),
                    position=entity.get("position"))
                if object_id in delivered_ids:
                    task.update(status="completed", owner=None,
                                reason="evaluator_confirmed_delivery",
                                blocked_until=None)
                elif object_id in self.physical_owners:
                    task.update(
                        status="carried",
                        owner=self.physical_owners[object_id].get("agent"),
                        reason="physical_ownership")
                elif remaining.get(name, 0) <= 0:
                    task.update(status="surplus", owner=None,
                                reason="goal_quota_filled")
                elif not (task.get("status") == "blocked" and
                          self.frame < int(task.get("blocked_until") or 0)):
                    if task.get("status") in (
                            "completed", "surplus", "carried", "blocked"):
                        task.update(status="pending", owner=None,
                                    reason=None, blocked_until=None)
            elif entity_type == 1:
                task = self._ensure_task(
                    self._entity_task_key(object_id), "container_resource",
                    40, object_id=object_id, name=name,
                    room=entity.get("room"),
                    position=entity.get("position"))
                if object_id in self.physical_owners:
                    task.update(
                        status="in_use",
                        owner=self.physical_owners[object_id].get("agent"),
                        reason="physical_ownership")
                elif self.container_parked_at_bed(object_id):
                    task.update(
                        status="parked_at_bed", owner=None,
                        reason="container_contents_evaluator_delivered",
                        blocked_until=None)
                elif self.container_delivery_pending(object_id):
                    lifecycle = self.container_states.get(object_id) or {}
                    task.update(
                        status="delivery_pending", owner=None,
                        reason="awaiting_evaluator_delivery_confirmation",
                        blocked_until=lifecycle.get(
                            "delivery_pending_until"))
                elif not (task.get("status") == "blocked" and
                          self.frame < int(task.get("blocked_until") or 0)):
                    task.update(status="pending", owner=None,
                                reason=None, blocked_until=None)

        for room, memory in self.room_memory.items():
            task = self._ensure_task(
                self._room_task_key(room), "explore_room", 20, room=room)
            previous_status = task.get("status")
            previous_owner = task.get("owner")
            coverage = set(str(value or "none") for value in
                           (memory.get("coverage_by_agent") or {}).values())
            if "all" in coverage:
                task.update(status="completed", owner=None,
                            reason="coverage_complete", blocked_until=None)
                if previous_status != "completed":
                    self.note_task_progress(
                        self._room_task_key(room), previous_owner,
                        "task_completed")
            elif not (task.get("status") == "blocked" and
                      self.frame < int(task.get("blocked_until") or 0)):
                if task.get("status") in ("completed", "blocked"):
                    task.update(status="pending", owner=None,
                                reason=None, blocked_until=None)

        for agent_id in range(len(agents)):
            payload = [object_id for object_id, owner in
                       self.physical_owners.items()
                       if owner.get("agent") == agent_id and
                       self.known_entities.get(object_id, {}).get("type") == 0
                       and object_id not in delivered_ids]
            delivery_task_id = self._delivery_task_key(agent_id)
            if payload:
                task = self._ensure_task(
                    delivery_task_id, "deliver_payload", 130,
                    agent=agent_id, payload=sorted(payload))
                if self.active_tasks.get(agent_id) == delivery_task_id:
                    task.update(status="in_progress", owner=agent_id)
                else:
                    task.update(status="pending", owner=agent_id)
            elif delivery_task_id in self.tasks:
                if self.mechanism_v4:
                    delivery_task = self.tasks[delivery_task_id]
                    if delivery_task.get("status") != "completed":
                        delivery_task.update(
                            status="suspended", owner=None,
                            reason="payload_released_awaiting_evaluator",
                            blocked_until=None, updated_frame=self.frame)
                    if self.active_tasks.get(agent_id) == delivery_task_id:
                        self.active_tasks[agent_id] = None
                else:
                    self.tasks[delivery_task_id].update(
                        status="completed", owner=None,
                        reason="payload_released", blocked_until=None,
                        updated_frame=self.frame)

        for agent_id, task_id in list(self.active_tasks.items()):
            task = self.tasks.get(task_id) if task_id else None
            terminal_statuses = {"completed", "surplus"}
            if self.mechanism_v4:
                terminal_statuses.update({"carried", "in_use"})
            if self.semantic_state_v34:
                terminal_statuses.update({"carried", "parked_at_bed"})
            if self.semantic_state_v35:
                terminal_statuses.add("delivery_pending")
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
            room = plan[len("go to "):] if plan.startswith("go to ") else (
                plan[len("explore current room "):]
                if plan.startswith("explore current room ") else None)
            if room:
                return self._room_task_key(room)
        return None

    def sync_tasks(
            self, intents: Dict[int, dict],
            planning_boundary_agents: Optional[Iterable[int]] = None) -> None:
        """Attach current plans to tasks without erasing interrupted work."""
        boundary_agents = set(planning_boundary_agents or [])
        for agent_id, intent in intents.items():
            task_id = self._task_id_from_intent(agent_id, intent)
            # Communication is not a task switch. A V4 planner-selected idle
            # at a genuine boundary is an explicit release; executor-internal
            # plan=None steps are excluded by ``boundary_agents``.
            if task_id is None:
                if (self.mechanism_v4 and agent_id in boundary_agents and
                        intent.get("stage") == "idle"):
                    previous_id = self.active_tasks.get(agent_id)
                    previous = (self.tasks.get(previous_id)
                                if previous_id else None)
                    if previous is not None and previous.get("status") == (
                            "in_progress"):
                        previous.update(
                            status="suspended", owner=None,
                            reason="planner_released_task",
                            updated_frame=self.frame)
                        self.emit_coordination_event(
                            "task_released", agent_id, previous_id,
                            reason="planner_selected_idle")
                    self.release_task_claims(
                        agent_id, previous_id,
                        "planner_selected_idle")
                    self.active_tasks[agent_id] = None
                continue
            previous_id = self.active_tasks.get(agent_id)
            if previous_id != task_id:
                previous = self.tasks.get(previous_id) if previous_id else None
                if previous and previous.get("status") == "in_progress":
                    previous.update(status="suspended", owner=None,
                                    reason="planner_switched_task",
                                    updated_frame=self.frame)
                    self.emit_coordination_event(
                        "task_suspended", agent_id, previous_id,
                        reason="planner_switched_task")
                self.release_task_claims(
                    agent_id, previous_id, "planner_switched_task")
                self.task_history.append({
                    "frame": self.frame,
                    "agent": agent_id,
                    "from": previous_id,
                    "to": task_id,
                })
            task = self.tasks.get(task_id)
            if task is None:
                kind = ("explore_room" if task_id.startswith("room:")
                        else "deliver_payload" if task_id.startswith(
                            "delivery:") else "unknown")
                task = self._ensure_task(task_id, kind, 10)
            non_selectable = {"completed", "surplus", "carried"}
            if self.mechanism_v4:
                non_selectable.add("in_use")
            if self.semantic_state_v35:
                non_selectable.add("delivery_pending")
            if task.get("status") not in non_selectable:
                task.update(status="in_progress", owner=agent_id,
                            attempts=int(task.get("attempts", 0)) +
                            (1 if previous_id != task_id else 0),
                            reason="selected_by_planner",
                            updated_frame=self.frame)
                if self.mechanism_v4:
                    task["last_owner"] = agent_id
                if previous_id != task_id:
                    self.emit_coordination_event(
                        "task_commitment", agent_id, task_id)
            self.active_tasks[agent_id] = task_id
        self.task_history = self.task_history[-self.MAX_TASK_HISTORY:]

    def block_active_task(self, agent_id: int, reason: str,
                          duration_frames: int = 120) -> None:
        task_id = self.active_tasks.get(agent_id)
        task = self.tasks.get(task_id) if task_id else None
        if task is not None and task.get("status") not in (
                "completed", "surplus", "carried"):
            task.update(status="blocked", owner=None, reason=reason,
                        blocked_until=self.frame + int(duration_frames),
                        updated_frame=self.frame)
        self.active_tasks[agent_id] = None

    def task_queue(self, agent_id: int, state: dict) -> List[dict]:
        """Return a compact, ordered queue for the LLM decision card."""
        active_id = self.active_tasks.get(agent_id)
        rows = []
        for task_id, task in self.tasks.items():
            hidden_statuses = {"completed", "surplus", "in_use"}
            if self.semantic_state_v34:
                hidden_statuses.update({"carried", "parked_at_bed"})
            if self.semantic_state_v35:
                hidden_statuses.add("delivery_pending")
            if task.get("status") in hidden_statuses:
                continue
            owner = task.get("owner")
            if owner not in (None, agent_id):
                continue
            if (self.mechanism_v4 and task_id != active_id and
                    owner != agent_id and
                    task.get("last_owner") != agent_id):
                # An unowned pending task may originate solely from the
                # peer's private discovery.  V4 shares commitments and
                # claims, not an automatic union of private observations.
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
            if self.mechanism_v4:
                # V4 queue order is coordination-only.  It carries stable
                # continuity and fixed priority, but no benchmark geometry,
                # confidence, utility, or feasibility estimate.
                rows.append((
                    0 if task_id == active_id else 1,
                    0 if task.get("status") == "suspended" else 1,
                    -int(task.get("priority", 0)),
                    task_id,
                    row,
                ))
                continue
            distance = self._distance(task.get("position"), state)
            row["distance_m"] = (
                round(distance, 1) if distance is not None else None)
            rows.append((
                0 if task_id == active_id else 1,
                0 if task.get("status") == "suspended" else 1,
                0 if task.get("status") != "blocked" else 1,
                -int(task.get("priority", 0)),
                distance if distance is not None else float("inf"),
                task_id,
                row,
            ))
        return [item[-1] for item in sorted(rows)[:10]]

    def _observe_entity(self, entity: dict, agent_id: int, frame: int,
                        evidence_source: str = "direct") -> None:
        object_id = entity.get("id")
        if object_id is None:
            return
        object_id = int(object_id)
        previous = self.known_entities.get(object_id, {})
        if (self.semantic_state_v35 and previous and
                int(previous.get("episode_epoch", -1)) !=
                self.episode_epoch):
            previous = {}
        seen_by = set(previous.get("seen_by", []))
        seen_by.add(agent_id)
        position = entity.get("position", previous.get("position"))
        room = entity.get("room", previous.get("room"))
        record = {
            "id": object_id,
            "name": entity.get("name") or previous.get("name"),
            "type": entity.get("type", previous.get("type")),
            "category": _entity_category(
                entity.get("type", previous.get("type"))),
            "seen_by": sorted(seen_by),
            "last_seen_by": agent_id,
            "last_seen_frame": frame,
            "position": copy.deepcopy(position),
            "room": room,
        }
        if self.semantic_state_v35:
            is_direct = evidence_source in ("direct", "held")
            record.update({
                "episode_epoch": self.episode_epoch,
                "evidence_source": evidence_source,
                "last_memory_frame": (frame if not is_direct else
                                      previous.get("last_memory_frame")),
                "last_direct_seen_frame": (
                    frame if is_direct else
                    previous.get("last_direct_seen_frame")),
                "last_direct_seen_by": (
                    agent_id if is_direct else
                    previous.get("last_direct_seen_by")),
            })
            # A semantic-memory publication must not masquerade as a new
            # camera observation.  This distinction lets delivery recovery
            # require genuinely fresh bed evidence after a failed hypothesis.
            if not is_direct:
                record["last_seen_frame"] = previous.get(
                    "last_seen_frame", frame)
                record["last_seen_by"] = previous.get(
                    "last_seen_by", agent_id)
        self.known_entities[object_id] = record

    def _record_payload(self, item: dict, agent_id: int, arm: str,
                        frame: int) -> None:
        object_id = int(item["id"])
        self._observe_entity(item, agent_id, frame, evidence_source="held")
        self.physical_owners[object_id] = {
            "agent": agent_id,
            "arm": arm,
            "carrier": "hand",
        }
        contained = [int(value) for value in item.get("contained", [])
                     if value is not None]
        if item.get("type") == 1:
            self.container_contents[object_id] = contained
            contained_names = item.get("contained_name", [])
            for index, child_id in enumerate(contained):
                name = (contained_names[index]
                        if index < len(contained_names) else None)
                self.known_entities.setdefault(child_id, {
                    "id": child_id,
                    "name": name,
                    "type": 0,
                    "seen_by": [agent_id],
                    "last_seen_by": agent_id,
                    "last_seen_frame": frame,
                    **({
                        "episode_epoch": self.episode_epoch,
                        "evidence_source": "held",
                        "last_direct_seen_frame": frame,
                        "last_direct_seen_by": agent_id,
                    } if self.semantic_state_v35 else {}),
                })
                self.physical_owners[child_id] = {
                    "agent": agent_id,
                    "carrier": "container",
                    "container_id": object_id,
                }

    def observe(self, states: Dict[str, dict], agents: List[Any],
                delivered_objects: Optional[Dict[int, str]] = None) -> None:
        self.frame = max(int(state.get("current_frames", 0))
                         for state in states.values())
        previous_owners = self.physical_owners
        previous_contents = self.container_contents
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
                entry = self.room_memory.setdefault(room, {
                    "visited_by": [],
                    "coverage_by_agent": {},
                    "last_visit_frame": None,
                })
                coverage = str(explored.get(room, "none") or "none")
                entry["coverage_by_agent"][str(agent_id)] = coverage
                if room == current_room:
                    visitors = set(entry.get("visited_by", []))
                    visitors.add(agent_id)
                    entry["visited_by"] = sorted(visitors)
                    entry["last_visit_frame"] = self.frame
            for entity in _real_objects(state.get("visible_objects", [])):
                self._observe_entity(
                    entity, agent_id, self.frame, evidence_source="direct")
            for arm_index, item in enumerate(state.get("held_objects", [])):
                if not item or item.get("id") is None:
                    continue
                arm = "left" if arm_index == 0 else "right"
                self._record_payload(item, agent_id, arm, self.frame)

            # Publish the compact room/pose facts already present in CoELA's
            # private semantic memory. RGB-D and occupancy grids remain local.
            room_cache = (getattr(agent, "object_per_room", None) or {})
            if (self.semantic_state_v35 and
                    int(getattr(agent, "_peer_consult_episode_epoch", -1)) !=
                    self.episode_epoch):
                room_cache = {}
            for room, by_type in room_cache.items():
                for entities in by_type.values():
                    for entity in entities:
                        annotated = copy.deepcopy(entity)
                        annotated["room"] = room
                        self._observe_entity(
                            annotated, agent_id, self.frame,
                            evidence_source="local_memory")

            action_id = int(state.get("action_id", 0))
            evidence_key = (agent_id, action_id)
            if (action_id > 0 and state.get("action_terminal") and
                    evidence_key not in self._seen_action_evidence):
                evidence = {
                    "evidence_id": f"frame{self.frame}:agent{agent_id}:"
                                   f"action{action_id}",
                    "agent": agent_id,
                    "action_id": action_id,
                    "action_type": state.get("action_type"),
                    "status": state.get("action_status"),
                    "valid": bool(state.get("valid", True)),
                    "completed_frame": state.get("action_completed_frame"),
                }
                self.execution_evidence.append(evidence)
                self._seen_action_evidence.add(evidence_key)
                if (not evidence["valid"] or evidence["status"] not in
                        ("success", "still_dropping")):
                    self.failure_counts[agent_id] += 1
                else:
                    self.failure_counts[agent_id] = 0

        # Natural-language messages are authored by the embodied agents. Keep
        # a grounded event archive so an unresolved request doesn't disappear
        # merely because CoELA only places a short dialogue tail in its prompt.
        # Both observations contain the same message vector, so read one copy.
        message_vector = next((state.get("messages") for state in
                               states.values() if state.get("messages")
                               is not None), None)
        if message_vector is not None:
            for sender, message in enumerate(message_vector[:len(agents)]):
                normalized = (str(message).strip()
                              if message is not None else None)
                if (normalized and normalized !=
                        self._last_observed_messages.get(sender)):
                    self.dialogue_events.append({
                        "frame": self.frame,
                        "sender": sender,
                        "message": normalized,
                    })
                self._last_observed_messages[sender] = normalized

        # Agent-local ``satisfied`` lists are planner bookkeeping, not
        # delivery evidence: legacy CoELA can add an object merely because a
        # teammate is holding it.  Accept only the environment evaluator's
        # monotonic object-id mapping.
        for object_id, name in (delivered_objects or {}).items():
            self.goal_ledger.mark_delivered(int(object_id), str(name))

        delivered_ids = set(self.goal_ledger.delivered)
        if self.mechanism_v4:
            # These are the only progress facts consumed by the generic loop
            # monitor.  They come from physical ownership/evaluator truth and
            # contain no distance, confidence, route, or container policy.
            for object_id, owner in self.physical_owners.items():
                previous_owner = previous_owners.get(object_id) or {}
                if (owner.get("agent") is not None and
                        (previous_owner.get("agent") != owner.get("agent") or
                         previous_owner.get("carrier") !=
                         owner.get("carrier"))):
                    task_id = self._entity_task_key(object_id)
                    self.note_task_progress(
                        task_id, int(owner["agent"]),
                        "physical_ownership_acquired")
            for object_id in sorted(delivered_ids - previous_delivered_ids):
                previous_owner = previous_owners.get(object_id) or {}
                agent_id = previous_owner.get("agent")
                task_id = self._entity_task_key(object_id)
                self.note_task_progress(
                    task_id, agent_id, "evaluator_confirmed_delivery")
                if agent_id is not None:
                    delivery_task_id = self._delivery_task_key(int(agent_id))
                    self.note_task_progress(
                        delivery_task_id, int(agent_id),
                        "evaluator_confirmed_delivery")
                    delivery_task = self.tasks.get(delivery_task_id)
                    if delivery_task is not None:
                        delivery_task.update(
                            status="completed", owner=None,
                            reason="evaluator_confirmed_delivery",
                            blocked_until=None, updated_frame=self.frame)
        self._reconcile_container_lifecycle(
            delivered_ids, previous_owners, previous_contents)
        if self.semantic_state_v34:
            # Evaluator-confirmed IDs are terminal task evidence. TDW can
            # still report a credited child inside a re-grasped container;
            # that physical artifact must not resurrect planner payload.
            for object_id in delivered_ids:
                self.physical_owners.pop(object_id, None)
            self.container_contents = {
                container_id: [
                    child_id for child_id in child_ids
                    if child_id not in delivered_ids
                ]
                for container_id, child_ids in self.container_contents.items()
            }
        for object_id in list(self.claims):
            claim = self.claims[object_id]
            if (object_id in delivered_ids or
                    (self.mechanism_v4 and
                     object_id in self.physical_owners) or
                    self.frame - claim["frame"] > self.CLAIM_TTL_FRAMES):
                del self.claims[object_id]
                self.emit_coordination_event(
                    "claim_released", claim.get("agent"),
                    self._entity_task_key(object_id),
                    claim_scope="object", claim_id=int(object_id),
                    reason=(
                        "task_completed" if object_id in delivered_ids else
                        "physical_ownership_acquired"
                        if object_id in self.physical_owners else
                        "lease_expired"))
        for room_id in list(self.room_claims):
            claim = self.room_claims[room_id]
            if self.frame - claim["frame"] > self.ROOM_CLAIM_TTL_FRAMES:
                del self.room_claims[room_id]
                self.emit_coordination_event(
                    "claim_released", claim.get("agent"),
                    self._room_task_key(str(room_id)),
                    claim_scope="room", claim_id=int(room_id),
                    reason="lease_expired")
        for key in list(self.target_cooldowns):
            if self.frame >= self.target_cooldowns[key]["until_frame"]:
                del self.target_cooldowns[key]

        for agent_id in range(len(agents)):
            payload = [object_id for object_id, owner in
                       self.physical_owners.items()
                       if owner.get("agent") == agent_id]
            target_payload = [object_id for object_id in payload
                              if self.known_entities.get(object_id, {}).get(
                                  "type") == 0]
            loaded_containers = [object_id for object_id in payload
                                 if self.container_contents.get(object_id)]
            if target_payload or loaded_containers:
                previous = self.payload_states.get(agent_id)
                payload_ids = sorted(set(target_payload))
                previous_ids = set((previous or {}).get("payload", []))
                preserve_start = bool(previous_ids.intersection(payload_ids))
                self.payload_states[agent_id] = {
                    "kind": "active_payload",
                    "payload": payload_ids,
                    "loaded_containers": loaded_containers,
                    "started_frame": (previous.get("started_frame")
                                      if previous and preserve_start
                                      else self.frame),
                    "updated_frame": self.frame,
                }
            else:
                self.payload_states.pop(agent_id, None)

        self._refresh_tasks(agents)
        self._compact_history()

    def _compact_history(self) -> None:
        """Bound in-memory history without deleting current physical truth."""
        self.dialogue_events = self.dialogue_events[-self.MAX_DIALOGUE_EVENTS:]
        self.execution_evidence = self.execution_evidence[
            -self.MAX_EXECUTION_EVIDENCE:]
        self.reviews = self.reviews[-self.MAX_REVIEWS:]
        self.proposal_history = self.proposal_history[
            -self.MAX_PROPOSAL_HISTORY:]
        self.task_history = self.task_history[-self.MAX_TASK_HISTORY:]
        self.coordination_events = self.coordination_events[
            -self.MAX_COORDINATION_EVENTS:]

    def claim(self, object_id: int, agent_id: int, reason: str) -> None:
        entity = self.known_entities.get(int(object_id), {})
        previous = self.claims.get(int(object_id))
        self.claims[int(object_id)] = {
            "agent": agent_id,
            "frame": self.frame,
            "reason": reason,
            "kind": entity.get("category", _entity_category(
                entity.get("type"))),
        }
        if previous is None or previous.get("agent") != agent_id:
            self.emit_coordination_event(
                "claim_acquired", agent_id,
                self._entity_task_key(object_id),
                claim_scope="object", claim_id=int(object_id),
                reason=reason)

    def release_agent_claims(self, agent_id: int,
                             reason: str = "agent_release") -> None:
        for object_id in list(self.claims):
            if self.claims[object_id]["agent"] == agent_id:
                del self.claims[object_id]
                self.emit_coordination_event(
                    "claim_released", agent_id,
                    self._entity_task_key(object_id),
                    claim_scope="object", claim_id=int(object_id),
                    reason=reason)

    def release_stale_agent_claims(self, agent_id: int,
                                   active_target_id: Optional[int]) -> None:
        # A communication or a one-step recovery turn must not silently erase
        # an otherwise valid intent. Expiry remains bounded by CLAIM_TTL_FRAMES;
        # selecting a different concrete target releases the older intent.
        if active_target_id is None:
            return
        for object_id in list(self.claims):
            if (self.claims[object_id]["agent"] == agent_id and
                    object_id != active_target_id):
                del self.claims[object_id]
                self.emit_coordination_event(
                    "claim_released", agent_id,
                    self._entity_task_key(object_id),
                    claim_scope="object", claim_id=int(object_id),
                    reason="planner_selected_different_target")

    def claim_room(self, room_id: int, agent_id: int, plan: str) -> None:
        for claimed_room in list(self.room_claims):
            if (self.room_claims[claimed_room]["agent"] == agent_id and
                    claimed_room != room_id):
                del self.room_claims[claimed_room]
                self.emit_coordination_event(
                    "claim_released", agent_id,
                    self._room_task_key(str(claimed_room)),
                    claim_scope="room", claim_id=int(claimed_room),
                    reason="planner_selected_different_room")
        previous = self.room_claims.get(int(room_id))
        self.room_claims[int(room_id)] = {
            "agent": agent_id,
            "frame": self.frame,
            "plan": plan,
        }
        if previous is None or previous.get("agent") != agent_id:
            self.emit_coordination_event(
                "claim_acquired", agent_id,
                self._room_task_key(str(room_id)),
                claim_scope="room", claim_id=int(room_id),
                reason="room_exploration_reservation")

    def release_agent_room_claims(
            self, agent_id: int,
            reason: str = "room_reservation_release") -> None:
        for room_id in list(self.room_claims):
            if self.room_claims[room_id]["agent"] == agent_id:
                del self.room_claims[room_id]
                self.emit_coordination_event(
                    "claim_released", agent_id,
                    self._room_task_key(str(room_id)),
                    claim_scope="room", claim_id=int(room_id),
                    reason=reason)

    def release_task_claims(self, agent_id: int, task_id: Optional[str],
                            reason: str) -> None:
        """Release only the reservation owned by one abandoned V4 task."""
        if not self.mechanism_v4 or task_id is None:
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
                    "claim_released", agent_id, task_id,
                    claim_scope="object", claim_id=object_id,
                    reason=reason)
            return
        if task_id.startswith("room:"):
            matches = re.findall(r"\((\d+)\)", task_id)
            if not matches:
                return
            room_id = int(matches[-1])
            claim = self.room_claims.get(room_id)
            if claim and claim.get("agent") == agent_id:
                del self.room_claims[room_id]
                self.emit_coordination_event(
                    "claim_released", agent_id, task_id,
                    claim_scope="room", claim_id=room_id,
                    reason=reason)

    @staticmethod
    def _cooldown_key(agent_id: int, target_id: int) -> str:
        return f"{int(agent_id)}:{int(target_id)}"

    def cooldown_target(self, agent_id: int, target_id: int,
                        duration_frames: int, reason: str) -> None:
        self.target_cooldowns[self._cooldown_key(agent_id, target_id)] = {
            "agent": int(agent_id),
            "target_id": int(target_id),
            "started_frame": self.frame,
            "until_frame": self.frame + int(duration_frames),
            "reason": reason,
        }

    def target_on_cooldown(self, agent_id: int, target_id: int) -> bool:
        value = self.target_cooldowns.get(
            self._cooldown_key(agent_id, target_id))
        return bool(value and self.frame < value["until_frame"])

    def public_view(self) -> dict:
        if self.mechanism_v4:
            def task_summary(task: dict) -> dict:
                return {
                    key: copy.deepcopy(task.get(key))
                    for key in (
                        "task_id", "kind", "status", "priority", "owner",
                        "last_owner", "object_id", "name", "room", "reason")
                    if task.get(key) is not None
                }

            shared_task_ids = {
                task_id for task_id in self.active_tasks.values() if task_id
            }
            shared_task_ids.update(
                self._entity_task_key(object_id)
                for object_id in self.claims)
            shared_task_ids.update(
                task_id for task_id, task in self.tasks.items()
                if task.get("owner") is not None or
                task.get("last_owner") is not None or
                task.get("status") == "completed")
            return {
                "frame": self.frame,
                "episode_epoch": self.episode_epoch,
                "goal": self.goal_ledger.view(),
                "claims": copy.deepcopy(self.claims),
                "room_claims": {
                    str(room_id): {
                        "agent": claim.get("agent"),
                        "frame": claim.get("frame"),
                    }
                    for room_id, claim in self.room_claims.items()
                },
                "physical_owners": {
                    str(object_id): {"agent": owner.get("agent")}
                    for object_id, owner in self.physical_owners.items()
                },
                "intents": {
                    str(agent_id): {
                        "stage": intent.get("stage"),
                        "target_id": intent.get("target_id"),
                        "plan": intent.get("plan"),
                        "execution": intent.get("execution"),
                    }
                    for agent_id, intent in self.current_intents.items()
                },
                "active_tasks": copy.deepcopy(self.active_tasks),
                "tasks": {
                    task_id: task_summary(task)
                    for task_id, task in self.tasks.items()
                    if task_id in shared_task_ids
                },
                "coordination_events": copy.deepcopy(
                    self.coordination_events[-12:]),
                "recent_coordination_events": copy.deepcopy(
                    self.coordination_events[-12:]),
                "planning_loop_guards": copy.deepcopy(
                    self.planning_loop_guards),
                "recent_reviews": copy.deepcopy(self.reviews[-6:]),
                "recent_dialogue": copy.deepcopy(self.dialogue_events[-6:]),
            }

        entities = {
            object_id: {
                "name": value.get("name"),
                "type": value.get("type"),
                "category": value.get("category", _entity_category(
                    value.get("type"))),
                "last_seen_by": value.get("last_seen_by"),
                "last_seen_frame": value.get("last_seen_frame"),
                "room": value.get("room"),
                "position": value.get("position"),
                **({
                    "episode_epoch": value.get("episode_epoch"),
                    "evidence_source": value.get("evidence_source"),
                    "last_direct_seen_frame": value.get(
                        "last_direct_seen_frame"),
                } if self.semantic_state_v35 else {}),
            }
            for object_id, value in self.known_entities.items()
            if value.get("type") in (0, 1, 2)
        }
        view = {
            "frame": self.frame,
            **({"episode_epoch": self.episode_epoch}
               if self.semantic_state_v35 else {}),
            "goal": self.goal_ledger.view(),
            "coordination_policy": {
                "planning": "prefer known useful targets over blind search",
                "delivery": ("compare safe delivery with the cost and risk "
                             "of one more pickup; free capacity alone is not "
                             "a reason to keep collecting"),
                "division": ("avoid duplicate objects; sharing a room is "
                             "allowed when distinct useful targets exist"),
            },
            "claims": self.claims,
            "room_claims": self.room_claims,
            "recovery_cooldowns": self.target_cooldowns,
            "physical_owners": self.physical_owners,
            "container_contents": self.container_contents,
            "container_states": self.container_states,
            "payload_states": self.payload_states,
            "room_memory": self.room_memory,
            "intents": self.current_intents,
            "active_tasks": self.active_tasks,
            "tasks": self.tasks,
            "recent_task_transitions": self.task_history[-8:],
            "known_entities": entities,
            "recent_failures": dict(self.failure_counts),
            "unconfirmed_delivery_counts": dict(
                self.delivery_failure_counts),
            "recent_reviews": self.reviews[-6:],
            "recent_dialogue": self.dialogue_events[-6:],
        }
        return view

    @staticmethod
    def _distance(position: Any, state: dict) -> Optional[float]:
        if position is None:
            return None
        try:
            source = np.asarray(state["agent"][:3], dtype=float)[[0, 2]]
            target = np.asarray(position, dtype=float)[[0, 2]]
            return float(np.linalg.norm(target - source))
        except (KeyError, TypeError, ValueError, IndexError):
            return None

    def _payload_view(self, agent_id: int) -> List[dict]:
        payload = []
        delivered_ids = set(self.goal_ledger.delivered)
        for object_id, owner in self.physical_owners.items():
            if (owner.get("agent") != agent_id or
                    (self.semantic_state_v34 and
                     object_id in delivered_ids)):
                continue
            entity = self.known_entities.get(object_id, {})
            payload.append({
                "id": object_id,
                "name": entity.get("name"),
                "carrier": owner.get("carrier"),
                "container_id": owner.get("container_id"),
            })
        return payload

    def _payload_view_v4(self, agent_id: int) -> List[dict]:
        """Return only identity facts, not carrier/container internals."""
        return [
            {
                "id": object_id,
                "name": self.known_entities.get(object_id, {}).get("name"),
            }
            for object_id, owner in sorted(self.physical_owners.items())
            if owner.get("agent") == agent_id and
            object_id not in self.goal_ledger.delivered
        ]

    def _decision_view_v4(self, agent_id: int, state: dict,
                          agents: List[Any],
                          delivery_advisory: dict) -> dict:
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
                    "task_id", "kind", "status", "priority", "owner",
                    "last_owner", "object_id", "name", "room", "reason")
                if task.get(key) is not None
            }

        peer_id = 1 - agent_id
        peer_task_id = self.active_tasks.get(peer_id)
        guard = self.planning_loop_guards.get(agent_id)
        delivery_target = None
        if self.v4_shared_delivery_target_enabled:
            candidates = [
                entity for entity in self.known_entities.values()
                if entity.get("type") == 2 and
                entity.get("position") is not None
            ]
            candidates.sort(key=lambda entity: int(entity.get("id", 0)))
            if candidates:
                target = candidates[0]
                delivery_target = {
                    key: copy.deepcopy(target.get(key))
                    for key in ("id", "name", "type", "position", "room")
                    if target.get(key) is not None
                }
                delivery_target["knowledge_source"] = (
                    "verified_shared_task_interface")
        guidance = [
            "All listed candidates are legal choices; fixed priority and "
            "active-task continuity are non-binding coordination hints.",
            "Choose the next task and strategy locally; the Harness does "
            "not select an alternative target, room, route, or resource.",
            "Claims and reservations prevent duplicate commitments but do "
            "not reveal private observations.",
        ]
        if guard:
            guidance.append(
                "The pending planning-loop guard applies only to the next "
                "planning boundary: choose an action other than its exact "
                "opaque action_identity once; the suspended task remains "
                "available afterwards.")
        return {
            "frame_budget": {
                "current": self.frame,
                "remaining": delivery_advisory.get("remaining_frames"),
            },
            "goal": {
                "delivered_total": sum(
                    self.goal_ledger.delivered_counts().values()),
                "required_total": sum(self.goal_ledger.required.values()),
                "remaining": {name: count for name, count in
                              remaining.items() if count},
            },
            "self": {
                "room": getattr(agents[agent_id], "current_room", None),
                "payload": self._payload_view_v4(agent_id),
            },
            "peer": {
                "payload": self._payload_view_v4(peer_id),
                "commitment": task_summary(peer_task_id),
            },
            "active_task": task_summary(
                self.active_tasks.get(agent_id)),
            **({"delivery_target": delivery_target}
               if delivery_target is not None else {}),
            "task_queue": task_queue,
            "active_claims": {
                str(object_id): {
                    "agent": value.get("agent"),
                    "kind": value.get("kind"),
                }
                for object_id, value in self.claims.items()
                if value.get("agent") != agent_id
            },
            "planning_loop_guard": copy.deepcopy(guard),
            "coordination_events": copy.deepcopy(
                self.coordination_events[-12:]),
            "recent_coordination_events": copy.deepcopy(
                self.coordination_events[-12:]),
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

    def decision_view(self, agent_id: int, state: dict,
                      agents: List[Any], delivery_advisory: dict) -> dict:
        """Compile a small, agent-specific working-memory card.

        The complete evidence ledger remains in ``public_view`` and the JSONL
        trace. The planner receives only current, actionable facts. This is a
        deterministic safety boundary: agent-authored dialogue can influence
        relevance, but cannot rewrite evaluator or physical truth.
        """
        if self.mechanism_v4:
            return self._decision_view_v4(
                agent_id, state, agents, delivery_advisory)

        remaining = self.goal_ledger.remaining_counts()
        delivered_ids = set(self.goal_ledger.delivered)
        current_room = getattr(agents[agent_id], "current_room", None)
        candidates = []
        for object_id, entity in self.known_entities.items():
            name = str(entity.get("name") or "")
            if (entity.get("type") != 0 or remaining.get(name, 0) <= 0 or
                    object_id in delivered_ids or
                    object_id in self.physical_owners or
                    self.target_on_cooldown(agent_id, object_id)):
                continue
            claim = self.claims.get(object_id)
            if claim and claim.get("agent") != agent_id:
                continue
            distance = self._distance(entity.get("position"), state)
            room = entity.get("room")
            candidates.append((
                0 if room == current_room else 1,
                distance if distance is not None else float("inf"),
                self.frame - int(entity.get("last_seen_frame") or 0),
                object_id,
                {
                    "id": object_id,
                    "name": name,
                    "room": room,
                    "distance_m": (round(distance, 1)
                                   if distance is not None else None),
                    "seen_by": entity.get("seen_by", []),
                    "age_frames": max(
                        0, self.frame - int(
                            entity.get("last_seen_frame") or 0)),
                },
            ))
        actionable = [item[-1] for item in sorted(candidates)[:5]]

        held = _real_objects(state.get("held_objects", []))
        holds_container = any(item.get("type") == 1 for item in held)
        container_candidates = []
        if len(held) < 2 and not holds_container:
            for object_id, entity in self.known_entities.items():
                if (entity.get("type") != 1 or
                        object_id in self.physical_owners or
                        self.container_parked_at_bed(object_id) or
                        self.container_delivery_pending(object_id) or
                        self.target_on_cooldown(agent_id, object_id)):
                    continue
                claim = self.claims.get(object_id)
                if claim and claim.get("agent") != agent_id:
                    continue
                room = entity.get("room")
                distance = self._distance(entity.get("position"), state)
                same_room_targets = sum(
                    1 for candidate in self.known_entities.values()
                    if (candidate.get("type") == 0 and
                        candidate.get("room") == room and
                        remaining.get(str(candidate.get("name") or ""), 0)
                        > 0 and
                        candidate.get("id") not in self.physical_owners and
                        candidate.get("id") not in delivered_ids))
                known_contents = self.container_contents.get(object_id, [])
                container_candidates.append((
                    0 if room == current_room else 1,
                    -same_room_targets,
                    distance if distance is not None else float("inf"),
                    object_id,
                    {
                        "id": object_id,
                        "name": entity.get("name"),
                        "room": room,
                        "distance_m": (round(distance, 1)
                                       if distance is not None else None),
                        "free_slots": max(0, 3 - len(known_contents)),
                        "same_room_useful_targets": same_room_targets,
                        "age_frames": max(
                            0, self.frame - int(
                                entity.get("last_seen_frame") or 0)),
                    },
                ))
        actionable_containers = [
            item[-1] for item in sorted(container_candidates)[:3]
        ]

        room_facts: Dict[str, dict] = {
            room: {
                "known_useful_targets": 0,
                "known_containers": 0,
                "last_seen_frame": 0,
            }
            for room in self.room_memory
        }
        for entity in self.known_entities.values():
            room = entity.get("room")
            name = str(entity.get("name") or "")
            if room is None:
                continue
            entry = room_facts.setdefault(str(room), {
                "known_useful_targets": 0,
                "known_containers": 0,
                "last_seen_frame": 0,
            })
            if (entity.get("type") == 0 and remaining.get(name, 0) > 0 and
                    entity.get("id") not in self.physical_owners and
                    entity.get("id") not in delivered_ids):
                entry["known_useful_targets"] += 1
            if (entity.get("type") == 1 and
                    entity.get("id") not in self.physical_owners and
                    not self.container_parked_at_bed(entity.get("id"))):
                entry["known_containers"] += 1
            entry["last_seen_frame"] = max(
                entry["last_seen_frame"],
                int(entity.get("last_seen_frame") or 0))

        current_agents: Dict[str, List[int]] = {}
        for candidate_agent_id, candidate_agent in enumerate(agents):
            room = getattr(candidate_agent, "current_room", None)
            if room is not None:
                current_agents.setdefault(str(room), []).append(
                    candidate_agent_id)
        room_rows = []
        coverage_rank = {"unseen": 0, "partial": 1, "all": 2}
        for room, value in room_facts.items():
            memory = self.room_memory.get(room, {})
            coverage_values = set(
                str(item or "none") for item in
                (memory.get("coverage_by_agent") or {}).values())
            if "all" in coverage_values:
                coverage = "all"
            elif memory.get("visited_by") or coverage_values - {"none"}:
                coverage = "partial"
            else:
                coverage = "unseen"
            last_visit = memory.get("last_visit_frame")
            room_rows.append({
                "room": room,
                "coverage": coverage,
                "visited_by": memory.get("visited_by", []),
                "current_agents": current_agents.get(room, []),
                "known_useful_targets": value["known_useful_targets"],
                "known_containers": value["known_containers"],
                "last_visit_age_frames": (
                    self.frame - int(last_visit)
                    if last_visit is not None else None),
                "last_evidence_age_frames": max(
                    0, self.frame - value["last_seen_frame"]),
            })
        ranked_rooms = sorted(
            room_rows,
            key=lambda item: (
                item["room"] != current_room,
                -item["known_useful_targets"],
                coverage_rank[item["coverage"]],
                -(item["last_visit_age_frames"] or 0),
                item["room"]),
        )[:6]

        peer_id = 1 - agent_id
        peer_intent = self.current_intents.get(peer_id) or {}
        relevant_reviews = [
            {
                "frame": review.get("frame"),
                "trigger": review.get("trigger"),
                "reason": str(review.get("reason", ""))[:180],
            }
            for review in self.reviews
            if review.get("proposer") == agent_id
        ][-2:]
        recent_dialogue = [
            {
                "frame": event["frame"],
                "speaker": event["sender"],
                "message": event["message"][:240],
            }
            for event in self.dialogue_events[-4:]
        ]
        payload_state = self.payload_states.get(agent_id)
        held_containers = []
        for item in held:
            if item.get("type") != 1:
                continue
            used = len([value for value in item.get("contained", [])
                        if value is not None])
            held_containers.append({
                "id": item.get("id"),
                "name": item.get("name"),
                "used_slots": used,
                "free_slots": max(0, 3 - used),
            })
        guidance = [
            "Free capacity alone is not a reason to keep collecting.",
            "Containers are transport resources, never goal-quota items; "
            "use them when nearby targets make the route worthwhile.",
            "Prefer a known useful target over blind room search.",
            "Avoid only true object conflicts; distinct targets in one "
            "room may be pursued concurrently.",
            "Use room coverage and evidence age to avoid repeating a "
            "fully explored room without a concrete reason.",
            "Treat delivery_advisory as risk evidence, not a long-term "
            "task contract.",
        ]
        if "communication_events" in delivery_advisory:
            guidance.append(
                "Natural language is event-triggered. Do not restate Memory "
                "Board facts unless communication_events is non-empty.")
        task_queue = self.task_queue(agent_id, state)
        if task_queue:
            guidance.extend([
                "Continue the active task unless it completed, became "
                "physically impossible, conflicted, or was blocked by a "
                "review.",
                "Resume suspended goal-object tasks before starting blind "
                "exploration; explore only when no unblocked known useful "
                "goal-object task is available.",
                "A message is optional even when a communication event is "
                "open. Send only if it changes the peer's next decision, "
                "and never repeat a previously reported fact.",
            ])
        return {
            "frame_budget": {
                "current": self.frame,
                "remaining": delivery_advisory.get("remaining_frames"),
            },
            "goal": {
                "delivered_total": sum(
                    self.goal_ledger.delivered_counts().values()),
                "required_total": sum(self.goal_ledger.required.values()),
                "remaining": {name: count for name, count in
                              remaining.items() if count},
            },
            "self": {
                "room": current_room,
                "payload": self._payload_view(agent_id),
                "payload_age_frames": (
                    self.frame - payload_state.get(
                        "started_frame", self.frame)
                    if payload_state else 0),
                "held_containers": held_containers,
                "consecutive_action_failures": int(
                    self.failure_counts.get(agent_id, 0)),
            },
            "peer": {
                "room": getattr(agents[peer_id], "current_room", None),
                "payload": self._payload_view(peer_id),
                "intent": {
                    "stage": peer_intent.get("stage"),
                    "target_id": peer_intent.get("target_id"),
                    "plan": peer_intent.get("plan"),
                },
            },
            "delivery_advisory": delivery_advisory,
            "active_task": self.tasks.get(self.active_tasks.get(agent_id)),
            "task_queue": task_queue,
            "actionable_targets": actionable,
            "actionable_containers": actionable_containers,
            "room_summary": ranked_rooms,
            "active_claims": {
                str(object_id): value for object_id, value in
                self.claims.items() if value.get("agent") != agent_id
            },
            "recent_relevant_reviews": relevant_reviews,
            "recent_agent_dialogue": recent_dialogue,
            "coordination_guidance": guidance,
        }

    def prompt_summary(self, agent_id: int, state: dict,
                       agents: List[Any], delivery_advisory: dict) -> str:
        return BOARD_PREFIX + json.dumps(
            _jsonable(self.decision_view(
                agent_id, state, agents, delivery_advisory)),
            ensure_ascii=False,
            separators=(",", ":"))


class TDWPeerConsultCoordinator:
    """Coordinate two CoELA planners without sharing their private images."""

    NAV_NO_PROGRESS_FRAMES = 120
    NAV_PROGRESS_DISTANCE = 0.75
    FAILURE_RECOVERY_THRESHOLD = 3
    TARGET_COOLDOWN_FRAMES = 360
    DELIVERY_SAFETY_MARGIN_FRAMES = 180
    DELIVERY_RECOMMEND_AGE_FRAMES = 420
    V31_DELIVERY_PRIORITY_BUFFER_FRAMES = 900
    V31_DELIVERY_PRIORITY_PAYLOAD_AGE_FRAMES = 720
    ROOM_CONFLICT_CONFIRM_FRAMES = 120
    ROOM_REASSIGN_COOLDOWN_FRAMES = 360
    V32_DIRECT_DROP_DISTANCE = 2.5
    V32_ACQUIRE_FIXED_FRAMES = 100
    V32_ACQUIRE_TRAVEL_FRAMES_PER_METER = 28
    V32_ACQUIRE_SAFETY_FRAMES = 60
    V33_TASK_BLOCK_FRAMES = 120
    V35_DELIVERY_CONFIRMATION_GRACE_FRAMES = 60
    V35_DELIVERY_RETRY_BASE_FRAMES = 180
    V35_DELIVERY_RETRY_INCREMENT_FRAMES = 90
    V35_DELIVERY_RETRY_MAX_FRAMES = 540
    COMMUNICATION_COOLDOWN_FRAMES = 240
    COMMUNICATION_DUPLICATE_FRAMES = 720
    V4_LOOP_REPEAT_THRESHOLD = 2
    V4_COMMUNICATION_EVENT_TYPES = frozenset({
        "task_commitment", "task_suspended", "task_released",
        "task_failure", "task_progress", "task_completed",
        "claim_acquired", "claim_released",
    })

    def __init__(self, agents: List[Any], logger: Any, output_dir: str,
                 review_mode: str = "deterministic",
                 max_frames: int = 3000,
                 protocol_version: str = PROTOCOL_V3):
        if len(agents) != 2:
            raise ValueError("peer consult currently requires exactly 2 agents")
        if protocol_version not in SUPPORTED_PROTOCOLS:
            raise ValueError(
                f"unsupported peer-consult protocol {protocol_version!r}")
        self.agents = agents
        self.logger = logger
        self.output_dir = output_dir
        self.review_mode = review_mode
        self.max_frames = int(max_frames)
        self.protocol_version = protocol_version
        self.run_instance_id = uuid.uuid4().hex
        self.mechanism_v4 = protocol_version == PROTOCOL_V4
        # Opt-in TDW adapter capability; it does not constrain V4 planning.
        self.v4_shared_delivery_target_enabled = (
            self.mechanism_v4 and
            os.environ.get("TDW_MAT_V4_SHARED_DELIVERY_TARGET", "0") == "1")
        self.event_triggered_communication = (
            protocol_version in SPARSE_COMMUNICATION_PROTOCOLS)
        self.authoritative_delivery_bookkeeping = (
            protocol_version in (
                PROTOCOL_V32, PROTOCOL_V33, PROTOCOL_V34, PROTOCOL_V35,
                PROTOCOL_V4))
        self.persistent_task_protocol = (
            protocol_version in (
                PROTOCOL_V33, PROTOCOL_V34, PROTOCOL_V35, PROTOCOL_V4))
        self.semantic_state_v34 = protocol_version in (
            PROTOCOL_V34, PROTOCOL_V35)
        self.semantic_state_v35 = protocol_version == PROTOCOL_V35
        self.board: Optional[TDWSharedBlackboard] = None
        self.protocol_step = 0
        self.log_path = os.path.join(output_dir, "peer_consult.jsonl")
        self.current_episode_id: Optional[int] = None
        self.episode_log_path: Optional[str] = None
        self.last_actions = {0: None, 1: None}
        self.nav_trackers: Dict[int, dict] = {}
        self.room_conflicts: Dict[int, dict] = {}
        self.communication_snapshots: Dict[int, dict] = {}
        self.communication_seen_entities = {
            0: {"targets": set(), "containers": set()},
            1: {"targets": set(), "containers": set()},
        }
        self.communication_event_details: Dict[int, dict] = {}
        self.communication_signature_frames: Dict[int, Dict[tuple, int]] = {
            0: {}, 1: {}}
        self.last_communication_frame = {0: -10**9, 1: -10**9}
        self.last_communication_message = {0: None, 1: None}
        self.pending_delivery_attempts: Dict[int, dict] = {}
        self.invalid_bed_hypotheses: Dict[int, dict] = {}
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
        # Correct legacy CoELA bookkeeping before either planner is reset.
        # Otherwise seeing a teammate carry an object permanently labels it
        # as delivered in the observer's private prompt state.
        for agent in self.agents:
            if hasattr(agent, "fix_lm_satisfied"):
                agent.fix_lm_satisfied = True
            if hasattr(agent, "satisfied"):
                agent.authoritative_satisfied = (
                    self.authoritative_delivery_bookkeeping)
            # These high-level switches are consumed by lm_agent and LLM;
            # the local navigation implementation remains unchanged.
            # V4's persistence lives in the Harness board.  These legacy
            # switches alter CoELA's candidate policy/explorer, so V4 leaves
            # both off and retains the original navigation/exploration path.
            agent.stable_task_protocol = (
                self.persistent_task_protocol and not self.mechanism_v4)
            agent.coverage_exploration = (
                self.persistent_task_protocol and not self.mechanism_v4)
            llm = getattr(agent, "LLM", None)
            if llm is not None:
                llm.peer_consult_protocol = protocol_version
                llm.peer_decision_card = None

    def reset(self, goal_description: Dict[str, int],
              episode_id: Optional[int] = None) -> None:
        self.current_episode_id = (
            int(episode_id) if episode_id is not None else None)
        self.episode_log_path = None
        if self.mechanism_v4 and self.current_episode_id is not None:
            episode_dir = os.path.join(
                self.output_dir, str(self.current_episode_id))
            os.makedirs(episode_dir, exist_ok=True)
            self.episode_log_path = os.path.join(
                episode_dir, "peer_consult.jsonl")
            # Preserve an incomplete prior attempt. Reset records and the
            # run-instance ID make appended attempts unambiguous.
            with open(self.episode_log_path, "a", encoding="utf-8"):
                pass
        if self.semantic_state_v35 or self.mechanism_v4:
            self.episode_epoch += 1
        self.board = TDWSharedBlackboard(
            goal_description, semantic_state_v34=self.semantic_state_v34,
            semantic_state_v35=self.semantic_state_v35,
            mechanism_v4=self.mechanism_v4,
            episode_epoch=self.episode_epoch)
        self.protocol_step = 0
        self.last_actions = {0: None, 1: None}
        self.nav_trackers = {}
        self.room_conflicts = {}
        self.communication_snapshots = {}
        self.communication_seen_entities = {
            0: {"targets": set(), "containers": set()},
            1: {"targets": set(), "containers": set()},
        }
        self.communication_event_details = {}
        self.communication_signature_frames = {0: {}, 1: {}}
        self.last_communication_frame = {0: -10**9, 1: -10**9}
        self.last_communication_message = {0: None, 1: None}
        self.pending_delivery_attempts = {}
        self.invalid_bed_hypotheses = {}
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
        if self.semantic_state_v35 or self.mechanism_v4:
            # ``lm_agent.reset()`` historically omitted these caches.  The
            # board is new, but observe() runs before the next agent.act(), so
            # stale room entities could otherwise repopulate it at frame 0.
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
        self._log({
            "kind": "reset",
            "protocol": self.protocol_version,
            "goal": goal_description,
            **({
                "v4_extensions": {
                    "shared_delivery_target_adapter": True,
                }
            } if self.v4_shared_delivery_target_enabled else {}),
            **({"episode_epoch": self.episode_epoch}
               if (self.semantic_state_v35 or self.mechanism_v4) else {}),
        })

    def _log(self, event: dict) -> None:
        if self.mechanism_v4:
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
        if self.mechanism_v4 and self.episode_log_path is not None:
            with open(self.episode_log_path, "a", encoding="utf-8") as stream:
                stream.write(line)

    @staticmethod
    def _bed_known(agent: Any) -> bool:
        object_list = getattr(agent, "object_list", None) or {}
        return bool(object_list.get(2, []))

    @staticmethod
    def _same_bed_hypothesis(bed: dict, hypothesis: dict) -> bool:
        bed_id = bed.get("id")
        invalid_id = hypothesis.get("bed_object_id")
        if bed_id is not None and invalid_id is not None:
            return int(bed_id) == int(invalid_id)
        try:
            bed_position = np.asarray(bed.get("position"), dtype=float)
            invalid_position = np.asarray(
                hypothesis.get("bed_position"), dtype=float)
            return bool(np.linalg.norm(
                bed_position[[0, 2]] - invalid_position[[0, 2]]) < 1.0)
        except (TypeError, ValueError, IndexError):
            return False

    def _bed_hypothesis_is_fresh(self, agent_id: int, bed: dict,
                                 hypothesis: dict) -> bool:
        if not self._same_bed_hypothesis(bed, hypothesis):
            return True
        entity = self.board.known_entities.get(int(bed.get("id", -1)), {})
        return int(entity.get("last_direct_seen_frame") or -1) > int(
            hypothesis.get("invalidated_frame", self.board.frame))

    def _retire_invalid_bed_from_private_memory(
            self, agent_id: int, hypothesis: dict) -> None:
        """Remove one failed bed hypothesis from CoELA's semantic cache.

        This doesn't alter perception or navigation.  It prevents a stale
        episode/local-memory record from blocking V3.4's shared-bed injection;
        a subsequent direct observation can populate the normal cache again.
        """
        if not self.semantic_state_v35:
            return
        agent = self.agents[agent_id]
        bed_id = hypothesis.get("bed_object_id")

        def keep(entity: dict) -> bool:
            return not self._same_bed_hypothesis(entity, hypothesis)

        object_list = getattr(agent, "object_list", None)
        if isinstance(object_list, dict):
            object_list[2] = [entity for entity in object_list.get(2, [])
                              if keep(entity)]
        for by_type in (getattr(agent, "object_per_room", None) or {}).values():
            if isinstance(by_type, dict):
                by_type[2] = [entity for entity in by_type.get(2, [])
                              if keep(entity)]
        if bed_id is not None:
            object_info = getattr(agent, "object_info", None)
            if isinstance(object_info, dict):
                object_info.pop(int(bed_id), None)
            id_map = getattr(agent, "id_map", None)
            object_map = getattr(agent, "object_map", None)
            if isinstance(id_map, np.ndarray):
                mask = id_map == int(bed_id)
                if isinstance(object_map, np.ndarray):
                    object_map[mask] = 0
                id_map[mask] = 0
        self._clear_plan(agent)

    def _invalidate_bed_hypothesis(self, agent_id: int,
                                   attempt: dict) -> None:
        if not self.semantic_state_v35:
            return
        failure_count = max(
            [int(self.board.delivery_failure_counts.get(object_id, 0))
             for object_id in attempt.get("payload_ids", [])] or [1])
        hypothesis = {
            "bed_object_id": attempt.get("bed_object_id"),
            "bed_position": copy.deepcopy(attempt.get("bed_position")),
            "bed_knowledge_source": attempt.get("bed_knowledge_source"),
            "invalidated_frame": self.board.frame,
            "failure_count": failure_count,
            "reason": "delivery_not_credited_relocalize_or_reapproach",
        }
        self.invalid_bed_hypotheses[agent_id] = hypothesis
        self._retire_invalid_bed_from_private_memory(agent_id, hypothesis)

    def _bed_evidence(self, agent_id: int) -> Tuple[List[dict], str]:
        local = list((getattr(self.agents[agent_id], "object_list", None)
                      or {}).get(2, []))
        hypothesis = (self.invalid_bed_hypotheses.get(agent_id)
                      if self.semantic_state_v35 else None)
        if hypothesis:
            local = [bed for bed in local if self._bed_hypothesis_is_fresh(
                agent_id, bed, hypothesis)]
        if local:
            if hypothesis:
                self.invalid_bed_hypotheses.pop(agent_id, None)
            return local, "agent_local"
        if self.semantic_state_v34:
            shared = [copy.deepcopy(entity) for entity in
                      self.board.known_entities.values()
                      if entity.get("type") == 2 and
                      (not self.semantic_state_v35 or
                       int(entity.get("episode_epoch", -1)) ==
                       self.episode_epoch)]
            if hypothesis:
                shared = [bed for bed in shared
                          if self._bed_hypothesis_is_fresh(
                              agent_id, bed, hypothesis)]
            if shared:
                if hypothesis:
                    self.invalid_bed_hypotheses.pop(agent_id, None)
                return shared, "shared_memory_board"
        return [], ("relocalization_required" if hypothesis else "unknown")

    @staticmethod
    def _held(state: dict) -> List[dict]:
        return _real_objects(state.get("held_objects", []))

    @classmethod
    def _can_collect_more(cls, state: dict) -> bool:
        """Return whether the physical payload still has useful capacity."""
        held = cls._held(state)
        if len(held) < 2:
            return True
        targets = [item for item in held if item.get("type") == 0]
        containers = [item for item in held if item.get("type") == 1]
        return bool(targets and any(
            len([value for value in container.get("contained", [])
                 if value is not None]) < 3
            for container in containers))

    def _useful_payload_count(self, state: dict) -> int:
        remaining = Counter(self.board.goal_ledger.remaining_counts())
        payload_names = []
        delivered_ids = set(self.board.goal_ledger.delivered)
        for item in self._held(state):
            if (item.get("type") == 0 and item.get("name") is not None and
                    (not self.semantic_state_v34 or
                     item.get("id") not in delivered_ids)):
                payload_names.append(str(item["name"]))
            if item.get("type") == 1:
                contained_ids = list(item.get("contained", []))
                for index, name in enumerate(
                        item.get("contained_name", [])):
                    if name is None:
                        continue
                    child_id = (contained_ids[index]
                                if index < len(contained_ids) else None)
                    if (self.semantic_state_v34 and
                            child_id in delivered_ids):
                        continue
                    payload_names.append(str(name))
        payload = Counter(payload_names)
        return sum(min(count, remaining.get(name, 0))
                   for name, count in payload.items())

    def _same_room_actionable_target(self, agent_id: int,
                                     state: dict) -> bool:
        remaining = self.board.goal_ledger.remaining_counts()
        current_room = getattr(self.agents[agent_id], "current_room", None)
        for object_id, entity in self.board.known_entities.items():
            name = str(entity.get("name") or "")
            if (entity.get("type") != 0 or entity.get("room") != current_room
                    or remaining.get(name, 0) <= 0
                    or object_id in self.board.goal_ledger.delivered
                    or object_id in self.board.physical_owners
                    or self.board.target_on_cooldown(agent_id, object_id)):
                continue
            claim = self.board.claims.get(object_id)
            if claim is None or claim.get("agent") == agent_id:
                distance = self.board._distance(
                    entity.get("position"), state)
                if distance is None or distance <= 3.5:
                    return True
        return False

    def _delivery_advisory(self, agent_id: int, state: dict) -> dict:
        useful_payload = self._useful_payload_count(state)
        remaining_frames = max(
            0, self.max_frames - int(state.get("current_frames", 0)))
        payload_state = self.board.payload_states.get(agent_id)
        payload_age = (self.board.frame - payload_state.get(
            "started_frame", self.board.frame) if payload_state else 0)
        held = self._held(state)
        delivered_ids = set(self.board.goal_ledger.delivered)
        direct_targets = sum(
            item.get("type") == 0 and
            (not self.semantic_state_v34 or
             item.get("id") not in delivered_ids)
            for item in held)
        containers = [item for item in held if item.get("type") == 1]
        def useful_contained(item: dict) -> List[int]:
            return [
                value for value in item.get("contained", [])
                if value is not None and
                (not self.semantic_state_v34 or value not in delivered_ids)
            ]
        container_used_slots = sum(
            len([value for value in item.get("contained", [])
                 if value is not None]) for item in containers)
        container_free_slots = sum(
            max(0, 3 - len([value for value in item.get("contained", [])
                            if value is not None]))
            for item in containers)
        container_useful_slots = sum(
            len(useful_contained(item)) for item in containers)
        bed_distance = None
        bed_objects, bed_source = self._bed_evidence(agent_id)
        if bed_objects:
            bed_distance = self.board._distance(
                bed_objects[0].get("position"), state)
        consecutive_failures = int(
            self.board.failure_counts.get(agent_id, 0))
        estimated_delivery_frames = int(
            150 + 32 * (bed_distance if bed_distance is not None else 8)
            + 45 * consecutive_failures)
        force_delivery = bool(
            useful_payload and bed_objects and
            remaining_frames <= estimated_delivery_frames +
            self.DELIVERY_SAFETY_MARGIN_FRAMES)
        nearby_target = self._same_room_actionable_target(agent_id, state)
        priority_delivery = force_delivery
        if self.protocol_version in SPARSE_COMMUNICATION_PROTOCOLS:
            priority_delivery = bool(
                useful_payload and bed_objects and (
                    remaining_frames <= estimated_delivery_frames +
                    self.V31_DELIVERY_PRIORITY_BUFFER_FRAMES or
                    (payload_age >=
                     self.V31_DELIVERY_PRIORITY_PAYLOAD_AGE_FRAMES and
                     not nearby_target)))
        recommend_delivery = bool(
            useful_payload and bed_objects and (
                priority_delivery or
                (payload_age >= self.DELIVERY_RECOMMEND_AGE_FRAMES and
                 not nearby_target) or
                remaining_frames <= estimated_delivery_frames + 600))
        if not useful_payload:
            recommendation = "collect_or_explore"
            reason = "no useful target payload is currently held"
        elif not bed_objects:
            recommendation = "locate_bed"
            reason = "useful payload is held but the bed is not known"
        elif priority_delivery:
            recommendation = "delivery_urgent"
            reason = ("current payload should be banked before another "
                      "pickup consumes the safe delivery budget")
        elif recommend_delivery:
            recommendation = "delivery_preferred"
            reason = ("useful payload has aged or no nearby useful target "
                      "justifies another pickup")
        else:
            recommendation = "delivery_optional"
            reason = ("a nearby useful target may still justify collection; "
                      "free capacity alone does not")
        pending_confirmation = sorted({
            int(object_id)
            for attempt in self.pending_delivery_attempts.values()
            if attempt.get("phase") == "confirmation_grace"
            for object_id in attempt.get("payload_ids", [])
        }) if self.semantic_state_v35 else []
        invalid_bed = (self.invalid_bed_hypotheses.get(agent_id)
                       if self.semantic_state_v35 else None)
        return {
            "useful_payload_count": useful_payload,
            "direct_target_count": direct_targets,
            "held_container_count": len(containers),
            "container_used_slots": container_used_slots,
            "container_useful_slots": container_useful_slots,
            "container_free_slots": container_free_slots,
            "remaining_frames": remaining_frames,
            "payload_age_frames": payload_age,
            "bed_distance_m": (round(bed_distance, 1)
                               if bed_distance is not None else None),
            "bed_known": bool(bed_objects),
            "bed_object_id": (bed_objects[0].get("id")
                              if bed_objects else None),
            "bed_position": (copy.deepcopy(
                bed_objects[0].get("position")) if bed_objects else None),
            "bed_knowledge_source": bed_source,
            "estimated_delivery_frames": estimated_delivery_frames,
            "nearby_useful_target": nearby_target,
            "recommendation": recommendation,
            "reason": reason,
            "force_delivery": force_delivery,
            "priority_delivery": priority_delivery,
            **({
                "delivery_confirmation_pending_object_ids":
                    pending_confirmation,
                "delivery_relocalization_required": bool(invalid_bed),
                "invalidated_bed_id": (
                    invalid_bed.get("bed_object_id")
                    if invalid_bed else None),
            } if self.semantic_state_v35 else {}),
        }

    def _sync_authoritative_satisfied(self) -> None:
        """Keep planner delivery memory equal to evaluator evidence.

        A successful physical drop is not necessarily a scored delivery: the
        object can land outside the goal radius.  V3.2 therefore removes the
        legacy optimistic self-report before every planning turn and retains
        only object IDs confirmed by the environment evaluator.
        """
        if not self.authoritative_delivery_bookkeeping:
            return
        delivered = sorted(self.board.goal_ledger.delivered)
        for agent_id, agent in enumerate(self.agents):
            if not hasattr(agent, "satisfied"):
                continue
            previous = list(getattr(agent, "satisfied", None) or [])
            if previous != delivered:
                stale = sorted(set(previous) - set(delivered))
                if stale:
                    self.board.reviews.append({
                        "frame": self.board.frame,
                        "reviewer": 1 - agent_id,
                        "proposer": agent_id,
                        "trigger": "delivery_reconciliation",
                        "verdict": "revise",
                        "reason": ("planner-local drop acknowledgement was "
                                   "not confirmed by evaluator evidence"),
                        "stale_object_ids": stale,
                    })
            agent.satisfied = list(delivered)

    def _local_bed_position(self, agent_id: int) -> Optional[np.ndarray]:
        beds, _ = self._bed_evidence(agent_id)
        if not beds or beds[0].get("position") is None:
            return None
        return np.asarray(beds[0]["position"], dtype=float)

    def _estimated_acquire_delivery_frames(
            self, agent_id: int, state: dict,
            target_id: int) -> Optional[int]:
        entity = self.board.known_entities.get(target_id, {})
        target_position = entity.get("position")
        bed_position = self._local_bed_position(agent_id)
        if target_position is None or bed_position is None:
            return None
        agent_position = np.asarray(state["agent"][:3], dtype=float)
        target_position = np.asarray(target_position, dtype=float)
        route_distance = (
            np.linalg.norm(agent_position[[0, 2]] -
                           target_position[[0, 2]]) +
            np.linalg.norm(target_position[[0, 2]] -
                           bed_position[[0, 2]]))
        failures = int(self.board.failure_counts.get(agent_id, 0))
        return int(
            self.V32_ACQUIRE_FIXED_FRAMES +
            self.V32_ACQUIRE_TRAVEL_FRAMES_PER_METER * route_distance +
            45 * failures)

    def _communication_events(self, agent_id: int, state: dict,
                              delivery_advisory: dict) -> List[str]:
        """Return sparse, physically grounded reasons to open a message slot.

        The Memory Board already shares ordinary state. Natural language is
        therefore reserved for changes that can alter the peer's next choice,
        instead of being offered on every LLM replanning turn.
        """
        if not self.event_triggered_communication:
            return ["legacy_open_channel"]

        remaining = self.board.goal_ledger.remaining_counts()
        delivered_ids = frozenset(self.board.goal_ledger.delivered)
        visible_targets = frozenset(
            int(item["id"]) for item in _real_objects(
                state.get("visible_objects", []))
            if (item.get("type") == 0 and
                remaining.get(str(item.get("name") or ""), 0) > 0 and
                int(item["id"]) not in delivered_ids))
        visible_containers = frozenset(
            int(item["id"]) for item in _real_objects(
                state.get("visible_objects", []))
            if (item.get("type") == 1 and
                not self.board.container_parked_at_bed(int(item["id"]))))
        payload = tuple(sorted(
            (int(object_id), str(owner.get("carrier")),
             owner.get("container_id"))
            for object_id, owner in self.board.physical_owners.items()
            if (owner.get("agent") == agent_id and
                (not self.semantic_state_v34 or
                 object_id not in delivered_ids))))
        failure_alert = (
            int(self.board.failure_counts.get(agent_id, 0)) >=
            self.FAILURE_RECOVERY_THRESHOLD)
        delivery_priority = bool(
            delivery_advisory.get("priority_delivery"))
        current = {
            "visible_targets": visible_targets,
            "visible_containers": visible_containers,
            "payload": payload,
            "delivered_ids": delivered_ids,
            "failure_alert": failure_alert,
            "delivery_priority": delivery_priority,
        }
        previous = self.communication_snapshots.get(agent_id)
        self.communication_snapshots[agent_id] = current
        if self.semantic_state_v34:
            seen = self.communication_seen_entities[agent_id]
            new_target_ids = sorted(visible_targets - seen["targets"])
            new_container_ids = sorted(
                visible_containers - seen["containers"])
            seen["targets"].update(visible_targets)
            seen["containers"].update(visible_containers)

            previous_payload = tuple(previous["payload"]) if previous else ()
            previous_delivered = (previous["delivered_ids"]
                                  if previous else frozenset())
            delivered_delta = sorted(delivered_ids - previous_delivered)
            events = []
            if new_target_ids:
                events.append("new_target_evidence")
            if new_container_ids:
                events.append("new_container_evidence")
            if payload != previous_payload and (payload or previous_payload):
                events.append("payload_change")
            previously_carried = {item[0] for item in previous_payload}
            delivery_progress_ids = sorted(
                set(delivered_delta).intersection(previously_carried))
            if delivery_progress_ids:
                events.append("delivery_progress")
            if failure_alert and not (
                    previous and previous["failure_alert"]):
                events.append("recovery_request")
            if delivery_priority and not (
                    previous and previous["delivery_priority"]):
                events.append("delivery_priority")

            details = {
                "events": list(events),
                "new_target_ids": new_target_ids,
                "new_container_ids": new_container_ids,
                "payload": [list(value) for value in payload],
                "payload_added_ids": sorted(
                    {value[0] for value in payload} -
                    {value[0] for value in previous_payload}),
                "payload_removed_ids": sorted(
                    {value[0] for value in previous_payload} -
                    {value[0] for value in payload}),
                "delivery_progress_ids": delivery_progress_ids,
                "failure_alert": failure_alert,
                "delivery_priority": delivery_priority,
            }
            details["semantic_signature"] = (
                tuple(events), tuple(new_target_ids),
                tuple(new_container_ids),
                tuple(payload) if "payload_change" in events else (),
                tuple(delivery_progress_ids),
                failure_alert if "recovery_request" in events else False,
                delivery_priority if "delivery_priority" in events else False,
            )
            self.communication_event_details[agent_id] = details
            if not events:
                return []
            since_message = (self.board.frame -
                             self.last_communication_frame[agent_id])
            critical = {"delivery_progress", "delivery_priority"}
            if (since_message < self.COMMUNICATION_COOLDOWN_FRAMES and
                    not critical.intersection(events)):
                return []
            return events

        if previous is None:
            return (["delivery_priority"] if delivery_priority else [])

        events = []
        if visible_targets - previous["visible_targets"]:
            events.append("new_target_evidence")
        if visible_containers - previous["visible_containers"]:
            events.append("new_container_evidence")
        if payload != previous["payload"]:
            events.append("payload_change")
        previously_carried = {item[0] for item in previous["payload"]}
        if ((delivered_ids - previous["delivered_ids"]) &
                previously_carried):
            events.append("delivery_progress")
        if failure_alert and not previous["failure_alert"]:
            events.append("recovery_request")
        if delivery_priority and not previous["delivery_priority"]:
            events.append("delivery_priority")

        if not events:
            return []
        since_message = (self.board.frame -
                         self.last_communication_frame[agent_id])
        critical = {"delivery_progress", "delivery_priority"}
        if (since_message < self.COMMUNICATION_COOLDOWN_FRAMES and
                not critical.intersection(events)):
            return []
        return events

    def _prepare_transport_decision(self, agent_id: int, state: dict,
                                    delivery_advisory: dict) -> None:
        """Clear physically stale plans while preserving planner autonomy.

        The only forced choice is deadline safety. A full hand or a partially
        loaded container is evidence for the next LLM decision, not a hard
        coded "fill then deliver" contract.
        """
        if state.get("status") == 0:
            return
        agent = self.agents[agent_id]
        held = self._held(state)
        targets = [item for item in held if item.get("type") == 0]
        containers = [item for item in held if item.get("type") == 1]
        loaded = [item for item in containers
                  if any(value is not None
                         for value in item.get("contained", []))]

        plan = getattr(agent, "plan", None) or ""
        physically_full = not self._can_collect_more(state)
        stale_acquisition = bool(
            plan.startswith("go grasp") and (
                physically_full or
                (targets and containers) or
                (containers and plan.startswith("go grasp container"))))
        if stale_acquisition:
            self._clear_plan(agent)

        # A partial payload is protected only at the deadline. At all other
        # times the planner sees collection, loading, and delivery evidence.
        if ((targets or loaded) and
                delivery_advisory.get("priority_delivery") and
                self._bed_known(agent)):
            forced_plan = "transport objects I'm holding to the bed"
            if agent.plan != forced_plan:
                agent.plan = forced_plan
                # Without this reset, goput() can reuse the just-grasped
                # object's position and immediately drop it on the floor.
                agent.target_pos = None

    def _inject_public_board(self, agent_id: int, state: dict,
                             delivery_advisory: dict) -> None:
        agent = self.agents[agent_id]
        history = [item for item in (getattr(agent, "dialogue_history", [])
                                     or [])
                   if not item.startswith(BOARD_PREFIX)]
        history.append(self.board.prompt_summary(
            agent_id, state, self.agents, delivery_advisory))
        # The planner selects a peer-consult-specific history window. Keep a
        # slightly larger raw archive here so recent agent-authored messages
        # remain available for that selection.
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
                0, self.max_frames - int(state.get("current_frames", 0))),
            "force_delivery": False,
            "priority_delivery": False,
            "recommendation": "planner_decides",
        }

    def _v4_boundary_descriptor(self, agent_id: int, intent: dict,
                                proposed_action: dict) -> Optional[dict]:
        """Return a stable opaque identity for one genuine planning turn."""
        if proposed_action.get("type") == "ongoing":
            return None
        task_id = self.board._task_id_from_intent(agent_id, intent)
        normalized_plan = " ".join(
            str(intent.get("plan") or "").split())
        identity_source = json.dumps({
            "task_id": task_id,
            "stage": intent.get("stage"),
            "target_id": intent.get("target_id"),
            "plan": normalized_plan,
            "action_type": proposed_action.get("type"),
        }, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return {
            "task_id": task_id,
            "action_identity": "action:" + hashlib.sha256(
                identity_source.encode("utf-8")).hexdigest()[:16],
            "stage": intent.get("stage"),
        }

    def _v4_reset_loop_history(self, agent_id: int) -> None:
        self.v4_loop_state[agent_id] = {
            "last_no_progress_identity": None,
            "last_no_progress_task_id": None,
            "last_no_progress_version": None,
            "consecutive_no_progress": 0,
        }

    def _v4_arm_loop_guard(self, agent_id: int, descriptor: dict,
                           count: int, outcome: str) -> None:
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
                agent_id, "planning_loop_guard_armed", task_id=task_id)
            self.board.release_agent_claims(
                agent_id, reason="planning_loop_guard_armed")
            self.board.release_agent_room_claims(
                agent_id, reason="planning_loop_guard_armed")
        self._clear_plan(self.agents[agent_id])
        self.board.emit_coordination_event(
            "planning_loop_guard_armed", agent_id, task_id,
            action_identity=descriptor["action_identity"],
            outcome=outcome, consecutive_no_progress=int(count))

    def _v4_note_no_progress(self, agent_id: int, descriptor: dict,
                             outcome: str,
                             progress_version_at_start: int) -> None:
        task_id = descriptor.get("task_id")
        current_progress = int(
            self.board.task_progress_versions.get(task_id, 0))
        if current_progress != int(progress_version_at_start):
            self._v4_reset_loop_history(agent_id)
            return
        state = self.v4_loop_state.setdefault(agent_id, {
            "last_no_progress_identity": None,
            "last_no_progress_task_id": None,
            "last_no_progress_version": None,
            "consecutive_no_progress": 0,
        })
        same = (
            state.get("last_no_progress_identity") ==
            descriptor["action_identity"] and
            state.get("last_no_progress_task_id") == task_id and
            state.get("last_no_progress_version") == current_progress)
        count = int(state.get("consecutive_no_progress", 0)) + 1 \
            if same else 1
        state.update({
            "last_no_progress_identity": descriptor["action_identity"],
            "last_no_progress_task_id": task_id,
            "last_no_progress_version": current_progress,
            "consecutive_no_progress": count,
        })
        self.board.emit_coordination_event(
            "planning_replan", agent_id, task_id,
            action_identity=descriptor["action_identity"],
            outcome=outcome, task_progress=False,
            consecutive_no_progress=count)
        if (count >= self.V4_LOOP_REPEAT_THRESHOLD and
                agent_id not in self.board.planning_loop_guards):
            self._v4_arm_loop_guard(
                agent_id, descriptor, count, outcome)

    def _v4_consume_execution_evidence(self) -> None:
        """Resolve pending planning boundaries from terminal action facts."""
        if not self.mechanism_v4:
            return
        for evidence in self.board.execution_evidence:
            evidence_id = evidence.get("evidence_id")
            if evidence_id in self.v4_seen_execution_evidence:
                continue
            self.v4_seen_execution_evidence.add(evidence_id)
            agent_id = int(evidence.get("agent"))
            pending = self.v4_pending_boundaries.pop(agent_id, None)
            if pending is None:
                continue
            progressed = int(self.board.task_progress_versions.get(
                pending.get("task_id"), 0)) != int(
                    pending.get("progress_version_at_start", 0))
            successful = (
                bool(evidence.get("valid", True)) and
                evidence.get("status") in ("success", "still_dropping"))
            if progressed:
                self._v4_reset_loop_history(agent_id)
                continue
            self._v4_note_no_progress(
                agent_id, pending,
                ("execution_completed_without_task_progress"
                 if successful else "execution_replan"),
                int(pending.get("progress_version_at_start", 0)))

    def _v4_apply_pending_guards(
            self, actions: Dict[str, dict], intents: Dict[int, dict]) -> None:
        """Consume each armed guard at exactly one planning boundary."""
        self.v4_guard_blocked_this_step = set()
        for agent_id, intent in intents.items():
            if agent_id not in self.v4_planning_boundaries_this_step:
                continue
            descriptor = self._v4_boundary_descriptor(
                agent_id, intent, actions[str(agent_id)])
            if descriptor is None:
                continue
            guard = self.board.planning_loop_guards.get(agent_id)
            if guard is None:
                continue
            del self.board.planning_loop_guards[agent_id]
            if (guard.get("task_id") == descriptor.get("task_id") and
                    guard.get("action_identity") ==
                    descriptor.get("action_identity")):
                if self._deny(
                        agent_id, actions,
                        "the exact action was just repeated without task "
                        "progress; choose any other legal action once",
                        "planning_loop_guard"):
                    if descriptor.get("task_id") is not None:
                        self.board.release_agent_claims(
                            agent_id, reason="planning_loop_guard_applied")
                        self.board.release_agent_room_claims(
                            agent_id, reason="planning_loop_guard_applied")
                    self.board.emit_coordination_event(
                        "planning_loop_guard_applied", agent_id,
                        descriptor.get("task_id"),
                        action_identity=descriptor["action_identity"],
                        outcome="replan_required")
                    self.v4_guard_blocked_this_step.add(agent_id)
            else:
                self.board.emit_coordination_event(
                    "planning_loop_guard_consumed", agent_id,
                    guard.get("task_id"),
                    action_identity=guard.get("action_identity"),
                    outcome="different_action_selected")
            self._v4_reset_loop_history(agent_id)

    def _v4_finalize_planning_boundaries(
            self, actions: Dict[str, dict], intents: Dict[int, dict],
            review_start: int) -> None:
        """Associate executed/reviewed proposals with their next outcome."""
        reviews = self.board.reviews[review_start:]
        for agent_id, intent in intents.items():
            if agent_id not in self.v4_planning_boundaries_this_step:
                continue
            if agent_id in self.v4_guard_blocked_this_step:
                continue
            descriptor = self._v4_boundary_descriptor(
                agent_id, intent,
                self._v4_proposed_actions[str(agent_id)])
            if descriptor is None:
                continue
            review = next((
                item for item in reversed(reviews)
                if item.get("proposer") == agent_id and
                item.get("verdict") == "revise"), None)
            progress_version = int(
                self.board.task_progress_versions.get(
                    descriptor.get("task_id"), 0))
            if review is not None:
                self._v4_note_no_progress(
                    agent_id, descriptor, "review_replan",
                    progress_version)
                continue
            # A physical pause preserves the same task/action; it is neither
            # a new attempt nor a failure boundary.
            paused = any(
                item.get("proposer") == agent_id and
                item.get("verdict") == "pause" for item in reviews)
            if paused or actions[str(agent_id)].get("type") == "ongoing":
                continue
            descriptor["progress_version_at_start"] = progress_version
            self.v4_pending_boundaries[agent_id] = descriptor

    def _deny(self, agent_id: int, actions: Dict[str, dict], reason: str,
              trigger: str) -> bool:
        if (agent_id in self._revised_this_step or
                actions[str(agent_id)].get("type") == "ongoing"):
            return False
        self._clear_plan(self.agents[agent_id])
        actions[str(agent_id)] = {
            "type": 1 if agent_id == 0 else 2,
        }
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
        if self.mechanism_v4:
            # V4 has no time-based task block.  A reviewed task persists as
            # suspended and can be selected again after the planner makes a
            # different choice.
            proposal_intent = self.board.current_intents.get(agent_id) or {}
            proposal_task_id = self.board._task_id_from_intent(
                agent_id, proposal_intent)
            if proposal_task_id is not None:
                self.board.suspend_active_task(
                    agent_id, f"review_replan:{trigger}",
                    task_id=proposal_task_id)
            if proposal_task_id is not None and trigger in {
                    "planning_loop_guard", "duplicate_claim",
                    "claim_conflict", "ownership", "verified_fact",
                    "physical_legality", "room_assignment"}:
                self.board.release_agent_claims(
                    agent_id, reason=f"review_replan:{trigger}")
                self.board.release_agent_room_claims(
                    agent_id, reason=f"review_replan:{trigger}")
        elif (self.persistent_task_protocol and trigger in {
                "recovery_cooldown", "duplicate_claim", "claim_conflict",
                "ownership", "verified_fact", "payload_feasibility",
                "goal_quota", "late_acquisition", "room_assignment",
                "navigation_recovery"}):
            self.board.block_active_task(
                agent_id, reason, self.V33_TASK_BLOCK_FRAMES)
        self._revised_this_step.add(agent_id)
        return True

    def _pause_action(self, agent_id: int, actions: Dict[str, dict],
                      reason: str, trigger: str) -> bool:
        """Serialize one physical step without deleting the active task."""
        if (agent_id in self._revised_this_step or
                actions[str(agent_id)].get("type") == "ongoing"):
            return False
        actions[str(agent_id)] = {
            "type": 1 if agent_id == 0 else 2,
        }
        self.board.reviews.append({
            "frame": self.board.frame,
            "reviewer": 1 - agent_id,
            "proposer": agent_id,
            "trigger": trigger,
            "verdict": "pause",
            "reason": reason,
            "replacement": actions[str(agent_id)],
            "preserved_task": self.board.active_tasks.get(agent_id),
        })
        self._revised_this_step.add(agent_id)
        return True

    def _assign_alternative_target(self, agent_id: int, state: dict,
                                   excluded_target: int,
                                   excluded_names: Optional[set] = None
                                   ) -> Optional[int]:
        """Cache a reachable-looking, still-needed target for the next step."""
        if self.persistent_task_protocol:
            # V3.3 exposes feasible pending work in the task queue and lets
            # the planner choose. Governance must not create a new objective.
            return None
        agent = self.agents[agent_id]
        remaining = self.board.goal_ledger.remaining_counts()
        current_room = getattr(agent, "current_room", None)
        candidates: Dict[int, dict] = {}

        for room, by_type in (getattr(
                agent, "object_per_room", None) or {}).items():
            for entity in by_type.get(0, []):
                annotated = copy.deepcopy(entity)
                annotated["room"] = room
                candidates[int(entity["id"])] = annotated
        for entity in (getattr(agent, "object_list", None) or {}).get(0, []):
            candidates.setdefault(int(entity["id"]), entity)

        agent_position = np.asarray(state["agent"][:3], dtype=float)[[0, 2]]
        ranked = []
        for target_id, entity in candidates.items():
            if target_id == excluded_target:
                continue
            name = str(entity.get("name", ""))
            if (remaining.get(name, 0) <= 0 or
                    name in (excluded_names or set())):
                continue
            if (target_id in self.board.goal_ledger.delivered or
                    target_id in self.board.physical_owners or
                    self.board.target_on_cooldown(agent_id, target_id)):
                continue
            claim = self.board.claims.get(target_id)
            if claim and claim["agent"] != agent_id:
                continue
            position = entity.get("position")
            distance = float("inf")
            if position is not None:
                candidate_position = np.asarray(position, dtype=float)[[0, 2]]
                distance = float(np.linalg.norm(
                    candidate_position - agent_position))
            room_penalty = 0 if entity.get("room") == current_room else 1
            ranked.append((room_penalty, distance, target_id, name))

        if not ranked:
            return None
        _, _, target_id, name = min(ranked)
        agent.plan = f"go grasp target object <{name}> ({target_id})"
        agent.target_pos = None
        return target_id

    def _assign_alternative_room(self, agent_id: int,
                                 excluded_room_id: int) -> Optional[str]:
        """Select a short-horizon alternative room, not a lasting contract."""
        if self.persistent_task_protocol:
            return None
        agent = self.agents[agent_id]
        remaining = self.board.goal_ledger.remaining_counts()
        peer_room = getattr(self.agents[1 - agent_id], "current_room", None)
        ranked = []
        for room in (getattr(agent, "rooms_name", None) or
                     list((getattr(agent, "object_per_room", None) or {}))):
            room_match = re.findall(r"\((\d+)\)", str(room))
            room_id = int(room_match[-1]) if room_match else None
            if room_id == excluded_room_id:
                continue
            useful = 0
            for entity in self.board.known_entities.values():
                name = str(entity.get("name") or "")
                if (entity.get("room") == room and entity.get("type") == 0
                        and remaining.get(name, 0) > 0
                        and entity.get("id") not in
                        self.board.physical_owners):
                    useful += 1
            explored = (getattr(agent, "rooms_explored", None) or {}).get(
                room, "none")
            ranked.append((
                1 if room == peer_room else 0,
                0 if useful else 1,
                0 if explored != "all" else 1,
                -useful,
                str(room),
            ))
        if not ranked:
            return None
        room = min(ranked)[-1]
        agent.plan = (f"explore current room {room}"
                      if room == getattr(agent, "current_room", None)
                      else f"go to {room}")
        agent.target_pos = None
        return room

    def _grounded_message(self, agent_id: int, state: dict) -> str:
        room = getattr(self.agents[agent_id], "current_room", None)
        held_descriptions = []
        for item in self._held(state):
            description = f"<{item.get('name')}> ({item.get('id')})"
            contained = [name for name in item.get("contained_name", [])
                         if name is not None]
            if contained:
                description += " containing " + ", ".join(contained)
            held_descriptions.append(description)
        payload = (", ".join(held_descriptions)
                   if held_descriptions else "nothing")
        remaining = self.board.goal_ledger.remaining_counts()
        remaining_text = ", ".join(
            f"{count} {name}" for name, count in remaining.items() if count)
        return (f"Grounded update: I am in {room}. I am holding {payload}. "
                f"Remaining goal: {remaining_text or 'complete'}. "
                "Please work on a distinct reachable target or room.")

    def _message_needs_reconciliation(self, agent_id: int,
                                      message: str) -> bool:
        normalized = message.lower().replace("’", "'")
        owners = self.board.physical_owners
        known = self.board.known_entities
        container_names = {
            str(entity.get("name", "")).lower()
            for entity in known.values() if entity.get("type") == 1
        }
        mentioned_containers = [name for name in container_names
                                if name and name in normalized]
        if (len(set(mentioned_containers)) >= 2 and
                re.search(r"\b(?:inside|into)\b", normalized)):
            return True

        ownership_patterns = (
            (agent_id, r"(?:i am|i'm)\s+(?:currently\s+)?holding\s+"
                       r"(.*?)(?:[.;]|$)"),
            (1 - agent_id, r"(?:you are|you're)\s+(?:currently\s+)?holding\s+"
                           r"(.*?)(?:[.;]|$)"),
        )
        for expected_owner, pattern in ownership_patterns:
            for match in re.finditer(pattern, normalized):
                clause = match.group(1)
                for object_id in map(int, re.findall(r"\((\d+)\)", clause)):
                    owner = owners.get(object_id)
                    if owner is None or owner.get("agent") != expected_owner:
                        return True
                for object_id, entity in known.items():
                    name = str(entity.get("name", "")).lower()
                    if name and name in clause:
                        owner = owners.get(object_id)
                        if owner is None or owner.get("agent") != expected_owner:
                            return True
        return False

    def _reconcile_messages(self, states: Dict[str, dict],
                            actions: Dict[str, dict]) -> None:
        for agent_id in range(2):
            action = actions[str(agent_id)]
            if action.get("type") != 6:
                continue
            message = str(action.get("message", ""))
            if not self._message_needs_reconciliation(agent_id, message):
                continue
            replacement = {
                "type": 6,
                "message": self._grounded_message(
                    agent_id, states[str(agent_id)]),
            }
            actions[str(agent_id)] = replacement
            self.board.reviews.append({
                "frame": self.board.frame,
                "reviewer": 1 - agent_id,
                "proposer": agent_id,
                "trigger": "message_reconciliation",
                "verdict": "revise",
                "reason": "message contradicted physical ownership or "
                          "requested unsupported container nesting",
                "replacement": replacement,
            })
            self._revised_this_step.add(agent_id)

    @staticmethod
    def _v4_coordination_event_id(message: Any) -> Optional[int]:
        """Parse only the opaque reference emitted by the V4 planner."""
        text = str(message or "").strip().strip('"\'` ')
        if ":" not in text:
            return None
        prefix, value = text.split(":", 1)
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
                "event_id", "event", "agent", "task_id", "outcome",
                "progress_version")
            if event.get(key) is not None
        }
        return json.dumps(
            {"coordination_event": payload}, ensure_ascii=False,
            separators=(",", ":"))

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
            event_id = self._v4_coordination_event_id(
                action.get("message"))
            event = next((
                item for item in reversed(
                    self.board.coordination_events[-12:])
                if item.get("event_id") == event_id and
                item.get("event") in self.V4_COMMUNICATION_EVENT_TYPES
            ), None)
            if event is not None:
                actions[str(agent_id)] = {
                    "type": 6,
                    "message": self._v4_render_coordination_event(event),
                }
                continue
            replacement = {"type": 8, "delay": 1}
            actions[str(agent_id)] = replacement
            self.board.reviews.append({
                "frame": self.board.frame,
                "reviewer": 1 - agent_id,
                "proposer": agent_id,
                "trigger": "message_event_reference",
                "verdict": "revise",
                "reason": "message did not reference one selectable public "
                          "coordination event",
                "replacement": replacement,
            })
            self.board.emit_coordination_event(
                "task_failure", agent_id, None,
                outcome="invalid_structured_event_reference")
            self._revised_this_step.add(agent_id)

    def _govern_communications(self, actions: Dict[str, dict],
                               communication_events: Dict[int, List[str]],
                               advisories: Dict[int, dict]) -> None:
        """Suppress routine, duplicate, or delivery-blocking language turns."""
        if not self.event_triggered_communication:
            return
        for agent_id in range(2):
            action = actions[str(agent_id)]
            if action.get("type") != 6:
                continue
            message = " ".join(str(action.get("message", "")).split())
            events = communication_events.get(agent_id, [])
            event_details = self.communication_event_details.get(
                agent_id, {})
            semantic_signature = event_details.get("semantic_signature")
            if self.semantic_state_v34 and semantic_signature:
                previous_signature_frame = (
                    self.communication_signature_frames[agent_id].get(
                        semantic_signature))
                duplicate = bool(
                    previous_signature_frame is not None and
                    self.board.frame - previous_signature_frame <
                    self.COMMUNICATION_DUPLICATE_FRAMES)
            else:
                duplicate = bool(
                    message and message ==
                    self.last_communication_message.get(agent_id) and
                    self.board.frame -
                    self.last_communication_frame[agent_id] <
                    self.COMMUNICATION_DUPLICATE_FRAMES)
            critical_event = bool(
                {"delivery_progress", "delivery_priority"}.intersection(
                    events))
            delivery_blocks_message = bool(
                advisories[agent_id].get("priority_delivery") and
                not (self.semantic_state_v34 and critical_event))
            if not events or duplicate or delivery_blocks_message:
                if delivery_blocks_message:
                    reason = ("delivery priority is active; do not spend the "
                              "next environment action on dialogue")
                elif duplicate:
                    reason = "message repeats a recent grounded update"
                else:
                    reason = ("the shared Memory Board has no new event that "
                              "requires natural-language coordination")
                replacement = {"type": 8, "delay": 1}
                actions[str(agent_id)] = replacement
                self.board.reviews.append({
                    "frame": self.board.frame,
                    "reviewer": 1 - agent_id,
                    "proposer": agent_id,
                    "trigger": "communication_gate",
                    "verdict": "revise",
                    "reason": reason,
                    "communication_events": events,
                    "communication_event_details": event_details,
                    "replacement": replacement,
                })
                self._revised_this_step.add(agent_id)
                continue
            self.last_communication_frame[agent_id] = self.board.frame
            self.last_communication_message[agent_id] = message
            if self.semantic_state_v34 and semantic_signature:
                signature_frames = self.communication_signature_frames[
                    agent_id]
                signature_frames[semantic_signature] = self.board.frame
                cutoff = (self.board.frame -
                          self.COMMUNICATION_DUPLICATE_FRAMES)
                self.communication_signature_frames[agent_id] = {
                    signature: frame for signature, frame in
                    signature_frames.items() if frame >= cutoff
                }

    def _govern_room_assignments(self, actions: Dict[str, dict],
                                 intents: Dict[int, dict]) -> None:
        room_interests: Dict[int, List[int]] = {}
        for agent_id, intent in intents.items():
            plan = intent.get("plan") or ""
            room_id = intent.get("target_id")
            if (room_id is not None and
                    plan.startswith(("go to", "explore"))):
                room_interests.setdefault(room_id, []).append(agent_id)

        for room_id, claimants in room_interests.items():
            useful_targets = {
                object_id for object_id, entity in
                self.board.known_entities.items()
                if (entity.get("type") == 0 and
                    entity.get("id") not in self.board.physical_owners and
                    self.board.goal_ledger.remaining_counts().get(
                        str(entity.get("name") or ""), 0) > 0 and
                    re.findall(r"\((\d+)\)", str(entity.get("room"))) and
                    int(re.findall(
                        r"\((\d+)\)", str(entity.get("room")))[-1]) ==
                    room_id)
            }
            # Co-location is complementary rather than conflicting when the
            # room contains enough distinct useful targets for both agents.
            if len(useful_targets) >= len(claimants):
                self.board.room_claims.pop(room_id, None)
                self.room_conflicts.pop(room_id, None)
                continue
            existing = self.board.room_claims.get(room_id)
            winner = existing["agent"] if existing else min(claimants)
            winner_plan = intents.get(winner, {}).get("plan") or "assigned"
            self.board.claim_room(room_id, winner, winner_plan)
            if self.protocol_version in SPARSE_COMMUNICATION_PROTOCOLS:
                conflict = self.room_conflicts.get(room_id)
                claimant_signature = tuple(sorted(claimants))
                if (conflict is None or
                        conflict.get("claimants") != claimant_signature):
                    self.room_conflicts[room_id] = {
                        "claimants": claimant_signature,
                        "winner": winner,
                        "first_frame": self.board.frame,
                        "last_review_frame": -10**9,
                    }
                    continue
                if (self.board.frame - conflict["first_frame"] <
                        self.ROOM_CONFLICT_CONFIRM_FRAMES or
                        self.board.frame - conflict["last_review_frame"] <
                        self.ROOM_REASSIGN_COOLDOWN_FRAMES):
                    continue
                conflict["last_review_frame"] = self.board.frame
            for agent_id in claimants:
                if agent_id == winner:
                    continue
                if self._deny(
                        agent_id, actions,
                        f"room {room_id} has insufficient distinct useful "
                        f"targets for both agents; agent {winner} keeps it",
                        "room_assignment"):
                    self._assign_alternative_room(agent_id, room_id)

    def _govern_room_reservations_v4(
            self, actions: Dict[str, dict], intents: Dict[int, dict]) -> None:
        """Atomically reserve blind exploration scopes without ranking them."""
        room_interests: Dict[int, List[int]] = {}
        for agent_id, intent in intents.items():
            if agent_id in self._revised_this_step:
                continue
            plan = str(intent.get("plan") or "")
            room_id = intent.get("target_id")
            if (room_id is not None and
                    plan.startswith(("go to", "explore"))):
                room_interests.setdefault(int(room_id), []).append(agent_id)

        for room_id, reservation in list(self.board.room_claims.items()):
            owner = reservation.get("agent")
            if owner in room_interests.get(int(room_id), []):
                continue
            del self.board.room_claims[room_id]
            self.board.emit_coordination_event(
                "claim_released", owner,
                self.board._room_task_key(str(room_id)),
                claim_scope="room", claim_id=int(room_id),
                reason="reservation_owner_abandoned_intent")

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
                    room_id, winner,
                    intents[winner].get("plan") or "reserved")
            for agent_id in claimants:
                if agent_id == winner:
                    continue
                self._deny(
                    agent_id, actions,
                    f"exploration scope {room_id} is reserved by agent "
                    f"{winner}; choose another legal action",
                    "room_assignment")

    def _govern_goal_type_quotas(self, states: Dict[str, dict],
                                 actions: Dict[str, dict],
                                 intents: Dict[int, dict]) -> None:
        """Avoid pursuing surplus instances after the remaining quota is reserved."""
        remaining = self.board.goal_ledger.remaining_counts()
        held_by_name = Counter()
        for object_id in self.board.physical_owners:
            entity = self.board.known_entities.get(object_id, {})
            if entity.get("type") == 0 and entity.get("name") is not None:
                held_by_name[str(entity["name"])] += 1

        proposals: Dict[str, List[Tuple[int, int, float]]] = {}
        for agent_id, intent in intents.items():
            if intent.get("stage") != "acquire":
                continue
            target_id = intent.get("target_id")
            entity = self.board.known_entities.get(target_id, {})
            # Containers are transport resources. They may be claimed to
            # avoid duplicate pursuit, but they never consume goal quota.
            if entity.get("type") != 0:
                continue
            name = str(entity.get("name") or "")
            if not name:
                continue
            distance = self.board._distance(
                entity.get("position"), states[str(agent_id)])
            proposals.setdefault(name, []).append((
                agent_id, target_id,
                distance if distance is not None else float("inf")))

        for name, candidates in proposals.items():
            available_slots = max(
                0, int(remaining.get(name, 0)) - held_by_name.get(name, 0))
            ranked = sorted(candidates, key=lambda item: (
                item[2],
                0 if self.board.claims.get(item[1], {}).get("agent") ==
                item[0] else 1,
                item[0]))
            for agent_id, target_id, _ in ranked[available_slots:]:
                if self._deny(
                        agent_id, actions,
                        f"remaining {name} quota is already covered by "
                        "physical payload or a closer current proposal",
                        "goal_quota"):
                    self.board.claims.pop(target_id, None)
                    self._assign_alternative_target(
                        agent_id, states[str(agent_id)], target_id,
                        excluded_names={name})

    def _assign_feasible_completion_target(
            self, agent_id: int, state: dict,
            excluded_target: int) -> Optional[int]:
        """Choose a locally known target that can still be banked in time."""
        if self.persistent_task_protocol:
            return None
        agent = self.agents[agent_id]
        remaining = self.board.goal_ledger.remaining_counts()
        remaining_frames = max(
            0, self.max_frames - int(state.get("current_frames", 0)))
        local_ids = set()
        for by_type in (getattr(agent, "object_per_room", None) or {}).values():
            local_ids.update(int(item["id"])
                             for item in by_type.get(0, []))
        local_ids.update(int(item["id"]) for item in
                         (getattr(agent, "object_list", None) or {}).get(0, []))

        ranked = []
        for target_id in local_ids:
            if target_id == excluded_target:
                continue
            entity = self.board.known_entities.get(target_id, {})
            name = str(entity.get("name") or "")
            if (remaining.get(name, 0) <= 0 or
                    target_id in self.board.goal_ledger.delivered or
                    target_id in self.board.physical_owners or
                    self.board.target_on_cooldown(agent_id, target_id)):
                continue
            claim = self.board.claims.get(target_id)
            if claim and claim.get("agent") != agent_id:
                continue
            estimate = self._estimated_acquire_delivery_frames(
                agent_id, state, target_id)
            if (estimate is None or
                    estimate + self.V32_ACQUIRE_SAFETY_FRAMES >=
                    remaining_frames):
                continue
            ranked.append((estimate, target_id, name))
        if not ranked:
            return None
        _, target_id, name = min(ranked)
        agent.plan = f"go grasp target object <{name}> ({target_id})"
        agent.target_pos = None
        return target_id

    def _govern_late_acquisitions(self, states: Dict[str, dict],
                                  actions: Dict[str, dict],
                                  intents: Dict[int, dict]) -> None:
        """Reject pickups whose estimated route cannot finish before timeout."""
        if self.protocol_version not in (
                PROTOCOL_V32, PROTOCOL_V33, PROTOCOL_V34, PROTOCOL_V35):
            return
        for agent_id, intent in intents.items():
            if intent.get("stage") != "acquire":
                continue
            target_id = intent.get("target_id")
            if target_id is None or self._useful_payload_count(
                    states[str(agent_id)]):
                continue
            entity = self.board.known_entities.get(target_id, {})
            if (self.protocol_version in (
                    PROTOCOL_V33, PROTOCOL_V34, PROTOCOL_V35) and
                    entity.get("type") != 0):
                # A container is a reusable resource, not a goal object.
                # V3.2 accidentally applied the target deadline rule to
                # baskets/trays and removed useful carrying capacity.
                continue
            estimate = self._estimated_acquire_delivery_frames(
                agent_id, states[str(agent_id)], target_id)
            if estimate is None:
                continue
            remaining_frames = max(
                0, self.max_frames - int(
                    states[str(agent_id)].get("current_frames", 0)))
            if (remaining_frames >
                    estimate + self.V32_ACQUIRE_SAFETY_FRAMES):
                continue
            if self._deny(
                    agent_id, actions,
                    f"target {target_id} needs about {estimate} frames to "
                    f"acquire and bank, but only {remaining_frames} remain",
                    "late_acquisition"):
                self.board.cooldown_target(
                    agent_id, target_id, self.TARGET_COOLDOWN_FRAMES,
                    "insufficient acquire-and-deliver budget")
                self._assign_feasible_completion_target(
                    agent_id, states[str(agent_id)], target_id)

    def _govern_delivery_commit(self, states: Dict[str, dict],
                                actions: Dict[str, dict],
                                advisories: Dict[int, dict]) -> None:
        """Commit a priority payload once it is safely inside drop range."""
        if self.protocol_version != PROTOCOL_V32:
            return
        remaining = self.board.goal_ledger.remaining_counts()
        for agent_id in range(2):
            action = actions[str(agent_id)]
            if (action.get("type") in (5, "ongoing") or
                    not advisories[agent_id].get("priority_delivery")):
                continue
            bed_position = self._local_bed_position(agent_id)
            if bed_position is None:
                continue
            agent_position = np.asarray(
                states[str(agent_id)]["agent"][:3], dtype=float)
            distance = float(np.linalg.norm(
                agent_position[[0, 2]] - bed_position[[0, 2]]))
            if distance > self.V32_DIRECT_DROP_DISTANCE:
                continue
            current_room = str(
                getattr(self.agents[agent_id], "current_room", ""))
            if "Bedroom" not in current_room:
                continue
            arm = None
            for arm_name, item in zip(
                    ("left", "right"),
                    states[str(agent_id)].get("held_objects", [])):
                if not item or item.get("id") is None:
                    continue
                names = []
                if item.get("type") == 0:
                    names.append(str(item.get("name") or ""))
                elif item.get("type") == 1:
                    names.extend(str(name) for name in
                                 item.get("contained_name", [])
                                 if name is not None)
                if any(remaining.get(name, 0) > 0 for name in names):
                    arm = arm_name
                    break
            if arm is None:
                continue
            actions[str(agent_id)] = {"type": 5, "arm": arm}
            self.board.reviews.append({
                "frame": self.board.frame,
                "reviewer": 1 - agent_id,
                "proposer": agent_id,
                "trigger": "delivery_commit",
                "verdict": "revise",
                "reason": ("priority payload is inside the public goal "
                           "drop radius; commit delivery now"),
                "replacement": actions[str(agent_id)],
            })
            self._revised_this_step.add(agent_id)

    def _record_delivery_attempts(self, actions: Dict[str, dict]) -> None:
        """Remember payload evidence before TDW executes a drop action."""
        if not self.authoritative_delivery_bookkeeping:
            return
        delivered_before = set(self.board.goal_ledger.delivered)
        for agent_id in range(2):
            action = actions[str(agent_id)]
            if action.get("type") != 5:
                continue
            if self.semantic_state_v35 and agent_id in self.pending_delivery_attempts:
                # A second release can't replace unresolved evidence from the
                # first one. Governance normally prevents this; retain the
                # first ticket defensively if an integration bypasses it.
                continue
            drop_arm = action.get("arm")
            container_ids = sorted(
                object_id for object_id, owner in
                self.board.physical_owners.items()
                if (owner.get("agent") == agent_id and
                    self.board.known_entities.get(object_id, {}).get(
                        "type") == 1 and
                    (not self.semantic_state_v35 or
                     owner.get("arm") == drop_arm)))
            container_payloads: Dict[int, List[int]] = {
                container_id: [] for container_id in container_ids}
            payload_ids = []
            for object_id, owner in self.board.physical_owners.items():
                if (owner.get("agent") != agent_id or
                        self.board.known_entities.get(object_id, {}).get(
                            "type") != 0 or
                        object_id in delivered_before):
                    continue
                if self.semantic_state_v35:
                    if owner.get("carrier") == "hand":
                        released = owner.get("arm") == drop_arm
                    else:
                        container_id = owner.get("container_id")
                        released = container_id in container_ids
                        if released:
                            container_payloads[int(container_id)].append(
                                int(object_id))
                    if not released:
                        continue
                payload_ids.append(int(object_id))
            payload_ids = sorted(payload_ids)
            if not payload_ids:
                continue
            advisory = self._advisories_this_step.get(agent_id, {})
            attempt = {
                "frame": self.board.frame,
                "payload_ids": payload_ids,
                "delivered_before": sorted(delivered_before),
                "action": copy.deepcopy(action),
            }
            if self.semantic_state_v35:
                attempt.update({
                    "phase": "executing",
                    "container_ids": container_ids,
                    "container_payloads": container_payloads,
                    "bed_object_id": advisory.get("bed_object_id"),
                    "bed_position": copy.deepcopy(
                        advisory.get("bed_position")),
                    "bed_knowledge_source": advisory.get(
                        "bed_knowledge_source"),
                })
                pending_until = (
                    self.board.frame + 30 +
                    self.V35_DELIVERY_CONFIRMATION_GRACE_FRAMES)
                for container_id in container_ids:
                    self.board.mark_container_delivery_pending(
                        container_id, pending_until, new_attempt=True)
            self.pending_delivery_attempts[agent_id] = attempt

    def _record_v4_delivery_attempts(self,
                                     actions: Dict[str, dict]) -> None:
        """Record only ownership truth for a proposed V4 release."""
        delivered_before = set(self.board.goal_ledger.delivered)
        for agent_id in range(2):
            action = actions[str(agent_id)]
            if action.get("type") != 5:
                continue
            drop_arm = action.get("arm")
            released_containers = {
                object_id for object_id, owner in
                self.board.physical_owners.items()
                if owner.get("agent") == agent_id and
                owner.get("carrier") == "hand" and
                owner.get("arm") == drop_arm and
                self.board.known_entities.get(object_id, {}).get("type") == 1
            }
            payload_ids = sorted(
                object_id for object_id, owner in
                self.board.physical_owners.items()
                if owner.get("agent") == agent_id and
                self.board.known_entities.get(object_id, {}).get("type") == 0
                and object_id not in delivered_before and (
                    (owner.get("carrier") == "hand" and
                     owner.get("arm") == drop_arm) or
                    (owner.get("carrier") == "container" and
                     owner.get("container_id") in released_containers)))
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
                        status="completed", owner=None,
                        reason="evaluator_confirmed_delivery",
                        blocked_until=None, updated_frame=self.board.frame)
                self.board.emit_coordination_event(
                    "task_completed", agent_id,
                    delivery_task_id,
                    outcome="evaluator_confirmed_delivery")
                self.pending_delivery_attempts.pop(agent_id, None)
                continue
            still_owned = {
                object_id for object_id in unconfirmed
                if (self.board.physical_owners.get(object_id) or {}).get(
                    "agent") == agent_id
            }
            if still_owned:
                for object_id in sorted(still_owned):
                    self.board.emit_coordination_event(
                        "task_failure", agent_id,
                        self.board._entity_task_key(object_id),
                        outcome="physical_release_failed")
            released_unconfirmed = [
                object_id for object_id in unconfirmed
                if object_id not in still_owned
            ]
            for object_id in released_unconfirmed:
                task_id = self.board._entity_task_key(object_id)
                task = self.board.tasks.get(task_id)
                if task is not None and task.get("status") not in (
                        "completed", "surplus", "carried"):
                    task.update(
                        status="suspended", owner=None,
                        reason="release_not_evaluator_confirmed",
                        blocked_until=None, updated_frame=self.board.frame)
                self.board.emit_coordination_event(
                    "task_failure", agent_id, task_id,
                    outcome="release_not_evaluator_confirmed")
            if released_unconfirmed:
                delivery_task_id = self.board._delivery_task_key(agent_id)
                delivery_task = self.board.tasks.get(delivery_task_id)
                if delivery_task is not None:
                    delivery_task.update(
                        status="suspended", owner=None,
                        reason="release_not_evaluator_confirmed",
                        blocked_until=None, updated_frame=self.board.frame)
            self.pending_delivery_attempts.pop(agent_id, None)

    def _reconcile_delivery_attempts(self, states: Dict[str, dict]) -> None:
        """Use evaluator progress, not a successful arm drop, as completion."""
        delivered = set(self.board.goal_ledger.delivered)
        for agent_id, attempt in list(self.pending_delivery_attempts.items()):
            state = states[str(agent_id)]
            if state.get("status") == 0:
                continue
            action_type = str(state.get("action_type"))
            if action_type not in ("5", "drop", "Drop"):
                # The next terminal state can already describe a later action
                # in synthetic/unit integrations. Reconcile once time moved.
                if self.board.frame <= attempt["frame"]:
                    continue
            payload_ids = set(attempt["payload_ids"])
            confirmed = sorted(payload_ids.intersection(delivered))
            unconfirmed = sorted(payload_ids - delivered)
            if self.semantic_state_v35:
                if not unconfirmed:
                    for container_id, child_ids in (
                            attempt.get("container_payloads") or {}).items():
                        credited = set(child_ids).intersection(delivered)
                        if credited:
                            self.board.confirm_container_delivery(
                                int(container_id), credited)
                    self.board.reviews.append({
                        "frame": self.board.frame,
                        "reviewer": "evaluator",
                        "proposer": agent_id,
                        "trigger": "delivery_confirmation",
                        "verdict": "accept",
                        "attempt_frame": attempt["frame"],
                        "confirmed_object_ids": confirmed,
                        "unconfirmed_object_ids": [],
                        "reason": "all released goal objects were credited",
                    })
                    self.pending_delivery_attempts.pop(agent_id, None)
                    continue

                if attempt.get("phase") != "confirmation_grace":
                    grace_until = (
                        self.board.frame +
                        self.V35_DELIVERY_CONFIRMATION_GRACE_FRAMES)
                    attempt.update({
                        "phase": "confirmation_grace",
                        "terminal_frame": self.board.frame,
                        "grace_until": grace_until,
                    })
                    for object_id in unconfirmed:
                        for cooldown_agent in range(2):
                            self.board.cooldown_target(
                                cooldown_agent, object_id,
                                self.V35_DELIVERY_CONFIRMATION_GRACE_FRAMES,
                                "awaiting evaluator delivery confirmation")
                        task = self.board.tasks.get(
                            self.board._entity_task_key(object_id))
                        if task is not None:
                            task.update(
                                status="blocked", owner=None,
                                reason="delivery_confirmation_grace",
                                blocked_until=grace_until,
                                updated_frame=self.board.frame)
                    for container_id in attempt.get("container_ids", []):
                        for cooldown_agent in range(2):
                            self.board.cooldown_target(
                                cooldown_agent, container_id,
                                self.V35_DELIVERY_CONFIRMATION_GRACE_FRAMES,
                                "container awaiting delivery confirmation")
                        self.board.mark_container_delivery_pending(
                            container_id, grace_until)
                    self.board.reviews.append({
                        "frame": self.board.frame,
                        "reviewer": "evaluator",
                        "proposer": agent_id,
                        "trigger": "delivery_confirmation_pending",
                        "verdict": "defer",
                        "attempt_frame": attempt["frame"],
                        "unconfirmed_object_ids": unconfirmed,
                        "grace_until": grace_until,
                        "reason": ("physical release is terminal but scoring "
                                   "gets a settling/confirmation grace period"),
                    })
                    continue
                if self.board.frame < int(attempt["grace_until"]):
                    continue

                retry_frames = self.V35_DELIVERY_RETRY_BASE_FRAMES
                for object_id in unconfirmed:
                    self.board.delivery_failure_counts[object_id] += 1
                    failures = int(
                        self.board.delivery_failure_counts[object_id])
                    retry_frames = max(retry_frames, min(
                        self.V35_DELIVERY_RETRY_BASE_FRAMES +
                        (failures - 1) *
                        self.V35_DELIVERY_RETRY_INCREMENT_FRAMES,
                        self.V35_DELIVERY_RETRY_MAX_FRAMES))
                    for cooldown_agent in range(2):
                        self.board.cooldown_target(
                            cooldown_agent, object_id, retry_frames,
                            "delivery unconfirmed; relocalize bed and "
                            "change approach before retry")
                    task = self.board.tasks.get(
                        self.board._entity_task_key(object_id))
                    if task is not None:
                        task.update(
                            status="blocked", owner=None,
                            priority=max(int(task.get("priority", 100)), 120),
                            reason="delivery_unconfirmed_relocalize_bed",
                            blocked_until=self.board.frame + retry_frames,
                            updated_frame=self.board.frame)
                for container_id in attempt.get("container_ids", []):
                    for cooldown_agent in range(2):
                        self.board.cooldown_target(
                            cooldown_agent, container_id, retry_frames,
                            "container delivery unconfirmed")
                    self.board.mark_container_delivery_pending(
                        container_id, self.board.frame + retry_frames)
                self._invalidate_bed_hypothesis(agent_id, attempt)
                self.board.reviews.append({
                    "frame": self.board.frame,
                    "reviewer": "evaluator",
                    "proposer": agent_id,
                    "trigger": "delivery_confirmation",
                    "verdict": "revise",
                    "attempt_frame": attempt["frame"],
                    "confirmed_object_ids": confirmed,
                    "unconfirmed_object_ids": unconfirmed,
                    "retry_after_frame": self.board.frame + retry_frames,
                    "invalidated_bed_id": attempt.get("bed_object_id"),
                    "reason": ("confirmation grace expired without credit; "
                               "cool down payload/container and relocalize "
                               "the bed or use a different approach"),
                })
                self.pending_delivery_attempts.pop(agent_id, None)
                continue

            verdict = "accept" if not unconfirmed else "revise"
            review = {
                "frame": self.board.frame,
                "reviewer": "evaluator",
                "proposer": agent_id,
                "trigger": "delivery_confirmation",
                "verdict": verdict,
                "attempt_frame": attempt["frame"],
                "confirmed_object_ids": confirmed,
                "unconfirmed_object_ids": unconfirmed,
                "reason": ("all released goal objects were credited"
                           if not unconfirmed else
                           "physical release did not produce evaluator "
                           "credit; keep those object tasks pending"),
            }
            self.board.reviews.append(review)
            for object_id in unconfirmed:
                self.board.delivery_failure_counts[object_id] += 1
                task = self.board.tasks.get(
                    self.board._entity_task_key(object_id))
                if task is not None:
                    task.update(
                        status="pending",
                        owner=None,
                        priority=max(int(task.get("priority", 100)), 120),
                        reason="delivery_unconfirmed_recover_and_reapproach",
                        blocked_until=None,
                        updated_frame=self.board.frame)
            self.pending_delivery_attempts.pop(agent_id, None)

    def _govern_navigation(self, states: Dict[str, dict],
                           actions: Dict[str, dict],
                           intents: Dict[int, dict]) -> None:
        for agent_id, intent in intents.items():
            plan = intent.get("plan") or ""
            if not plan.startswith(("go to", "go grasp", "transport")):
                self.nav_trackers.pop(agent_id, None)
                continue
            position = np.asarray(states[str(agent_id)]["agent"][:3],
                                  dtype=float)[[0, 2]]
            frame = self.board.frame
            tracker = self.nav_trackers.get(agent_id)
            if tracker is None or tracker["plan"] != plan:
                self.nav_trackers[agent_id] = {
                    "plan": plan,
                    "anchor": position,
                    "frame": frame,
                }
                continue
            if (np.linalg.norm(position - tracker["anchor"]) >=
                    self.NAV_PROGRESS_DISTANCE):
                tracker["anchor"] = position
                tracker["frame"] = frame
                continue
            stalled = frame - tracker["frame"] >= self.NAV_NO_PROGRESS_FRAMES
            repeated_failure = (self.board.failure_counts.get(agent_id, 0) >=
                                self.FAILURE_RECOVERY_THRESHOLD)
            if not (stalled or repeated_failure):
                continue
            if self._deny(agent_id, actions,
                          "navigation plan made no spatial progress; clear "
                          "the cached target and replan",
                          "navigation_recovery"):
                target_id = _plan_target_id(plan)
                if plan.startswith("go grasp") and target_id is not None:
                    self.board.cooldown_target(
                        agent_id, target_id, self.TARGET_COOLDOWN_FRAMES,
                        "navigation made no spatial progress")
                    self._assign_alternative_target(
                        agent_id, states[str(agent_id)], target_id)
                self.board.release_agent_room_claims(agent_id)
                self.nav_trackers.pop(agent_id, None)

    def _review_and_govern(self, states: Dict[str, dict],
                           actions: Dict[str, dict],
                           intents: Dict[int, dict]) -> None:
        if self.mechanism_v4:
            # Apply the one-boundary loop exclusion before acquiring claims.
            # A denied repeat must not transiently reserve its old resource.
            self._v4_apply_pending_guards(actions, intents)
        acquisitions: Dict[int, List[int]] = {}
        pending_delivery_ids = set()
        if self.semantic_state_v35:
            for attempt in self.pending_delivery_attempts.values():
                pending_delivery_ids.update(attempt.get("payload_ids", []))
                pending_delivery_ids.update(attempt.get("container_ids", []))
        for agent_id, intent in intents.items():
            if agent_id in self._revised_this_step:
                continue
            active_target = (intent.get("target_id")
                             if intent["stage"] == "acquire" else None)
            self.board.release_stale_agent_claims(agent_id, active_target)
            if active_target in pending_delivery_ids:
                self._deny(
                    agent_id, actions,
                    f"object {active_target} is awaiting evaluator delivery "
                    "confirmation and cannot be re-grasped yet",
                    "delivery_confirmation_pending")
                continue
            if (active_target is not None and
                    self.board.target_on_cooldown(agent_id, active_target)):
                cooldown = self.board.target_cooldowns[
                    self.board._cooldown_key(agent_id, active_target)]
                if self._deny(
                        agent_id, actions,
                        f"target {active_target} is cooling down until frame "
                        f"{cooldown['until_frame']}; select another target",
                        "recovery_cooldown"):
                    self._assign_alternative_target(
                        agent_id, states[str(agent_id)], active_target)
                continue
            if intent["stage"] == "acquire" and intent["target_id"] is not None:
                acquisitions.setdefault(intent["target_id"], []).append(agent_id)

        for target_id, claimants in acquisitions.items():
            existing = self.board.claims.get(target_id)
            owner = self.board.physical_owners.get(target_id)
            delivered = target_id in self.board.goal_ledger.delivered
            if delivered:
                for agent_id in claimants:
                    self._deny(agent_id, actions,
                               f"object {target_id} is already delivered",
                               "verified_fact")
                continue
            if (self.semantic_state_v34 and
                    self.board.container_parked_at_bed(target_id)):
                for agent_id in claimants:
                    self._deny(
                        agent_id, actions,
                        f"container {target_id} is parked at the bed after "
                        "an evaluator-confirmed delivery and is retired",
                        "verified_fact")
                continue
            if owner is not None:
                for agent_id in claimants:
                    if (self.mechanism_v4 or self.semantic_state_v34 or
                            owner.get("agent") != agent_id):
                        relation = ("already carried by this agent"
                                    if owner.get("agent") == agent_id else
                                    f"carried by agent {owner.get('agent')}")
                        self._deny(agent_id, actions,
                                   f"object {target_id} is {relation}",
                                   "ownership")
                if self.semantic_state_v34 or self.mechanism_v4:
                    task = self.board.tasks.get(
                        self.board._entity_task_key(target_id))
                    if task is not None:
                        task.update(
                            status="carried",
                            owner=owner.get("agent"),
                            reason="physical_ownership",
                            updated_frame=self.board.frame)
                continue

            if existing and existing["agent"] not in claimants:
                for agent_id in claimants:
                    self._deny(agent_id, actions,
                               f"object {target_id} is claimed by agent "
                               f"{existing['agent']}", "claim_conflict")
                continue

            if existing and existing["agent"] in claimants:
                winner = existing["agent"]
            elif self.mechanism_v4:
                ordered = sorted(claimants)
                winner = ordered[
                    (self.protocol_step + int(target_id)) % len(ordered)]
            else:
                winner = min(claimants)
            self.board.claim(target_id, winner, "accepted_acquire_intent")
            for agent_id in claimants:
                if agent_id != winner:
                    self._deny(agent_id, actions,
                               f"duplicate claim on object {target_id}; "
                               f"agent {winner} owns the claim",
                               "duplicate_claim")

        for agent_id, intent in intents.items():
            if agent_id in self._revised_this_step:
                continue
            target_id = intent.get("target_id")
            if intent["stage"] != "acquire" or target_id is None:
                continue
            # Multi-claim conflicts were fully resolved in the pass above;
            # don't overwrite that more informative peer-review evidence.
            if len(acquisitions.get(target_id, [])) > 1:
                continue
            claim = self.board.claims.get(target_id)
            if claim and claim["agent"] != agent_id:
                self._deny(agent_id, actions,
                           f"object {target_id} is claimed by agent "
                           f"{claim['agent']}", "claim_conflict")

        if self.mechanism_v4:
            # Factual-only legality: hands are a physical capability exposed
            # by the adapter.  No distance, deadline, target utility, or
            # container-fill inference is made here.
            for agent_id in range(2):
                if agent_id in self._revised_this_step:
                    continue
                intent = intents[agent_id]
                if (intent["stage"] == "acquire" and
                        len(self._held(states[str(agent_id)])) >= 2):
                    self._deny(
                        agent_id, actions,
                        "the physical adapter reports no free hand for an "
                        "acquisition action",
                        "physical_legality")
            # Reservation/claim coordination and natural-language messages
            # remain available.  The legacy navigation, budget, quota,
            # payload, delivery, and container policies are deliberately not
            # executed for V4.
            self._govern_room_reservations_v4(actions, intents)
            self._reconcile_messages_v4(actions)
        else:
            # Reject only physically impossible acquisitions or deadline-unsafe
            # ones. This is a one-step feasibility check, not a task contract.
            for agent_id in range(2):
                intent = intents[agent_id]
                advisory = self._delivery_advisory(
                    agent_id, states[str(agent_id)])
                if (intent["stage"] == "acquire" and (
                        not self._can_collect_more(states[str(agent_id)]) or
                        advisory.get("priority_delivery"))):
                    self._deny(agent_id, actions,
                               "safe delivery dominates another acquisition "
                               "under current payload and frame risk",
                               "payload_feasibility")

            self._govern_goal_type_quotas(states, actions, intents)
            self._govern_late_acquisitions(states, actions, intents)
            self._govern_room_assignments(actions, intents)
            self._reconcile_messages(states, actions)
            self._govern_communications(
                actions, self._communication_events_this_step,
                self._advisories_this_step)
            self._govern_navigation(states, actions, intents)
            self._govern_delivery_commit(
                states, actions, self._advisories_this_step)

        # TDW advances both bodies concurrently.  Serialize two close, forward
        # moves to avoid a head-on collision at narrow passages.
        if (not self.mechanism_v4 and
                actions["0"].get("type") == 0 and
                actions["1"].get("type") == 0):
            p0 = np.asarray(states["0"]["agent"][:3], dtype=float)
            p1 = np.asarray(states["1"]["agent"][:3], dtype=float)
            if np.linalg.norm(p0[[0, 2]] - p1[[0, 2]]) < 1.25:
                if self.persistent_task_protocol:
                    self._pause_action(
                        1, actions,
                        "nearby concurrent forward moves are serialized",
                        "spatial_conflict")
                else:
                    self._deny(
                        1, actions,
                        "nearby concurrent forward moves are serialized",
                        "spatial_conflict")

    def act(self, states: Dict[str, dict],
            delivered_objects: Optional[Dict[int, str]] = None
            ) -> Dict[str, dict]:
        if self.board is None:
            raise RuntimeError("coordinator.reset() must be called first")
        self.protocol_step += 1
        self.board.observe(states, self.agents, delivered_objects)
        self._v4_consume_execution_evidence()
        self._sync_authoritative_satisfied()
        self._revised_this_step = set()

        advisories = {
            agent_id: (self._v4_advisory(states[str(agent_id)])
                       if self.mechanism_v4 else
                       self._delivery_advisory(
                           agent_id, states[str(agent_id)]))
            for agent_id in range(2)
        }
        communication_events = (
            {agent_id: [] for agent_id in range(2)}
            if self.mechanism_v4 else
            {
                agent_id: self._communication_events(
                    agent_id, states[str(agent_id)], advisories[agent_id])
                for agent_id in range(2)
            })
        if self.event_triggered_communication:
            for agent_id in range(2):
                advisories[agent_id]["communication_events"] = (
                    communication_events[agent_id])
                if self.semantic_state_v34:
                    advisories[agent_id]["communication_event_details"] = (
                        copy.deepcopy(self.communication_event_details.get(
                            agent_id, {})))
        self._communication_events_this_step = communication_events
        self._advisories_this_step = advisories
        decision_cards = {}
        for agent_id, agent in enumerate(self.agents):
            decision_cards[agent_id] = self.board.decision_view(
                agent_id, states[str(agent_id)], self.agents,
                advisories[agent_id])
            llm = getattr(agent, "LLM", None)
            if llm is not None:
                llm.peer_decision_card = (
                    decision_cards[agent_id]
                    if self.persistent_task_protocol else None)
            self._inject_public_board(
                agent_id, states[str(agent_id)], advisories[agent_id])
            if not self.mechanism_v4:
                self._prepare_transport_decision(
                    agent_id, states[str(agent_id)], advisories[agent_id])

        actions = {}
        v4_history_lengths = {}
        if self.mechanism_v4:
            self.v4_planning_boundaries_this_step = set()
            for agent_id, agent in enumerate(self.agents):
                history = getattr(agent, "action_history", None)
                v4_history_lengths[agent_id] = (
                    len(history) if isinstance(history, list) else None)
        for agent_id, agent in enumerate(self.agents):
            llm = getattr(agent, "LLM", None)
            original_message_gate = getattr(
                llm, "allow_message_this_turn", None)
            if (self.event_triggered_communication and llm is not None and
                    original_message_gate is not None):
                llm.allow_message_this_turn = bool(
                    communication_events[agent_id])
            try:
                actions[str(agent_id)] = agent.act(states[str(agent_id)])
            finally:
                if original_message_gate is not None:
                    llm.allow_message_this_turn = original_message_gate
            if self.mechanism_v4:
                explicit_boundary = getattr(
                    agent, "_peer_consult_planning_boundary", None)
                history = getattr(agent, "action_history", None)
                history_grew = (
                    v4_history_lengths.get(agent_id) is not None and
                    isinstance(history, list) and
                    len(history) > v4_history_lengths[agent_id])
                is_boundary = (bool(explicit_boundary)
                               if explicit_boundary is not None else
                               history_grew)
                if is_boundary:
                    self.v4_planning_boundaries_this_step.add(agent_id)
                if explicit_boundary is not None:
                    # A test/adapter marker is edge-triggered, just like an
                    # action_history append from a real LLM_plan() call.
                    agent._peer_consult_planning_boundary = False
        intents = {}
        for agent_id, agent in enumerate(self.agents):
            plan = getattr(agent, "plan", None)
            if (self.mechanism_v4 and
                    agent_id in self.v4_planning_boundaries_this_step):
                # Some original CoELA executors consume an in-range plan
                # (grasp/message/drop) before returning the first low-level
                # action. lm_agent exposes the just-selected high-level plan
                # as an edge marker so the planning boundary retains its
                # task/action identity without changing executor behavior.
                selected_plan = getattr(
                    agent, "_peer_consult_selected_plan", None)
                if selected_plan is not None:
                    plan = selected_plan
            intents[agent_id] = _intent_from_plan(
                agent_id, plan, actions[str(agent_id)],
                states[str(agent_id)])
        if self.mechanism_v4:
            # V4 identities/events never consume or expose a model-generated
            # confidence score.
            for intent in intents.values():
                intent.pop("confidence", None)
        self.board.current_intents = copy.deepcopy(intents)
        if self.persistent_task_protocol:
            self.board.sync_tasks(
                intents,
                (self.v4_planning_boundaries_this_step
                 if self.mechanism_v4 else None))
        self._v4_proposed_actions = copy.deepcopy(actions)
        review_start = len(self.board.reviews)
        self._review_and_govern(states, actions, intents)
        if self.mechanism_v4:
            self._v4_finalize_planning_boundaries(
                actions, intents, review_start)
            self._record_v4_delivery_attempts(actions)
        else:
            self._record_delivery_attempts(actions)

        if self.mechanism_v4:
            validator_reviews = self.board.reviews[review_start:]
            proposals = {
                agent_id: {
                    "intent": intents[agent_id],
                    "planning_boundary": (
                        agent_id in self.v4_planning_boundaries_this_step),
                    "task_action": (
                        self._v4_boundary_descriptor(
                            agent_id, intents[agent_id],
                            self._v4_proposed_actions[str(agent_id)])
                        if agent_id in
                        self.v4_planning_boundaries_this_step else None),
                    "proposed_action": copy.deepcopy(
                        self._v4_proposed_actions[str(agent_id)]),
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
        else:
            proposals = {
                agent_id: {
                    "intent": intents[agent_id],
                    "final_action": copy.deepcopy(actions[str(agent_id)]),
                }
                for agent_id in range(2)
            }
        self.board.proposal_history.append({
            "frame": self.board.frame,
            "proposals": proposals,
        })
        self.board._compact_history()
        self.last_actions = copy.deepcopy(actions)
        self._log({
            "kind": "decision",
            "protocol_step": self.protocol_step,
            "frame": self.board.frame,
            "board": self.board.public_view(),
            "decision_cards": decision_cards,
            "communication_events": communication_events,
            "communication_event_details": (
                self.communication_event_details
                if self.semantic_state_v34 else None),
            "proposals": proposals,
        })
        return actions

    def observe_outcome(self, states: Dict[str, dict],
                        evaluator_progress: Tuple[int, int, bool],
                        info: dict,
                        delivered_objects: Optional[Dict[int, str]] = None
                        ) -> None:
        """Record evaluator truth without exposing it in planner prompts."""
        self.board.observe(states, self.agents, delivered_objects)
        self.board.evaluator_progress = tuple(evaluator_progress)
        self._v4_consume_execution_evidence()
        if self.mechanism_v4:
            self._reconcile_v4_delivery_attempts(states)
        else:
            self._reconcile_delivery_attempts(states)
        self._log({
            "kind": "outcome",
            "frame": self.board.frame,
            "evaluator_progress": evaluator_progress,
            "num_frames_for_step": info.get("num_frames_for_step"),
            "evidence": self.board.execution_evidence[-4:],
        })

    def finalize(self, evaluator_progress: Tuple[int, int, bool]) -> None:
        self._log({
            "kind": "final",
            "frame": self.board.frame,
            "goal_ledger": self.board.goal_ledger.view(),
            "evaluator_progress": evaluator_progress,
            "protocol_steps": self.protocol_step,
        })
