import json
import os
import numpy as np
import cv2
import pyastar2d as pyastar
import random
import time
import math
import copy
from PIL import Image
from agent_memory import AgentMemory

from LLM.LLM import (LLM, action_allowed_for_role, normalize_agent_role,
                     plan_allowed_for_role)

CELL_SIZE = 0.125
ANGLE = 15

_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}


def _env_flag_enabled(name):
    """Return True only when an environment flag is explicitly enabled."""
    return os.getenv(name, "").strip().lower() in _TRUE_ENV_VALUES


def _role_from_embodiment(embodiment):
    return normalize_agent_role(embodiment)


def _resolve_team_roles(args, agent_id, explicit_role=None):
    """Resolve Human/Scout roles while preserving legacy Human/Human runs."""
    embodiments = getattr(args, "embodiments", None)
    if isinstance(embodiments, str):
        embodiments = [embodiments]
    if not embodiments:
        self_role = normalize_agent_role(explicit_role)
        return self_role, "human"

    roles = [_role_from_embodiment(item) for item in embodiments]
    self_role = (normalize_agent_role(explicit_role) if explicit_role is not None
                 else roles[agent_id] if agent_id < len(roles) else "human")
    opponent_id = 1 - agent_id
    opponent_role = (roles[opponent_id] if opponent_id < len(roles)
                     else "human")
    return self_role, opponent_role

class lm_agent:
    def  __init__(self, agent_id, logger, max_frames, args,
                  output_dir='results', agent_role=None):
        self.with_oppo = None
        self.oppo_pos = None
        self.with_character = None
        self.color2id = None
        self.satisfied = None
        # Disabled by default to preserve the published repository behavior.
        # In corrected mode, `satisfied` means locally confirmed delivery only;
        # teammate-held and invalid objects use separate temporary/ignore state.
        self.fix_lm_satisfied = _env_flag_enabled("TDW_MAT_FIX_LM_SATISFIED")
        # PeerConsult V3.2 can promote evaluator-confirmed delivery to the
        # sole source of truth.  The coordinator enables this per protocol so
        # legacy and earlier ablation behavior remains unchanged.
        self.authoritative_satisfied = False
        self.stable_task_protocol = False
        self.coverage_exploration = False
        self.object_list = None
        self.container_held = None
        self.gt_mask = None
        self.object_info = {} # {id: {id: xx, type: 0/1/2, name: sss, position: x,y,z}}
        self.object_per_room = {} # {room_name: {0/1/2: [{id: xx, type: 0/1/2, name: sss, position: x,y,z}]}}
        self.id_map = None
        self.object_map = None
        self.agent_id = agent_id
        self.agent_type = 'lm_agent'
        self.agent_role, self.opponent_role = _resolve_team_roles(
            args, agent_id, explicit_role=agent_role)
        self.agent_names = ["Alice", "Bob"]
        self.opponent_agent_id = 1 - agent_id
        self.env_api = None
        self.max_frames = max_frames
        self.output_dir = output_dir
        self.map_size = (240, 120)
        self.save_img = True
        self._scene_bounds = {
            "x_min": -15,
            "x_max": 15,
            "z_min": -7.5,
            "z_max": 7.5
        }
        self.max_nav_steps = 80
        self.max_move_steps = 150
        self.logger = logger
        random.seed(1024)
        self.debug = True

        self.new_object_list = None
        # V3.4 keeps first-observation events until an LLM decision boundary.
        # This is separate from ``new_object_list``, which is intentionally a
        # one-step legacy compatibility view.
        self.pending_perception_events = []
        self.visible_objects = None
        self.num_frames = None
        self.steps = None
        self.obs = None
        self.local_step = 0
        self.episode_provenance = None

        self.last_action = None
        self.pre_action = None

        self.goal_objects = None
        self.dropping_object = None

        self.source = args.source
        self.lm_id = args.lm_id
        self.prompt_template_path = args.prompt_template_path
        self.communication = args.communication
        self.cot = args.cot
        self.args = args
        self.LLM = LLM(
            self.source,
            self.lm_id,
            self.prompt_template_path,
            self.communication,
            self.cot,
            self.args,
            self.agent_id,
            agent_role=self.agent_role,
            opponent_role=self.opponent_role,
        )
        self.action_history = []
        self.dialogue_history = []
        self.plan = None
        self.target_pos = None

        self.rooms_name = None
        self.rooms_explored = {}
        self.position = None
        self.forward = None
        self.current_room = None
        self.holding_objects_id = None
        self.oppo_holding_objects_id = None
        self.oppo_last_room = None
        self.rotated = None
        self.navigation_threshold = 5
        self.detection_threshold = 5


    def pos2map(self, x, z):
        i = int(round((x - self._scene_bounds["x_min"]) / CELL_SIZE))
        j = int(round((z - self._scene_bounds["z_min"]) / CELL_SIZE))
        return i, j

    def map2pos(self, i, j):
        x = i * CELL_SIZE + self._scene_bounds["x_min"]
        z = j * CELL_SIZE + self._scene_bounds["z_min"]
        return x, z

    def get_pc(self, color):
        depth = self.obs['depth'].copy()
        mask = np.any(self.obs['seg_mask'] != np.asarray(color), axis=-1)
        depth[mask] = 1e9
        #camera info
        FOV = self.obs['FOV']
        W, H = depth.shape
        cx = W / 2.
        cy = H / 2.
        fx = cx / np.tan(math.radians(FOV / 2.))
        fy = cy / np.tan(math.radians(FOV / 2.))

        #Ego
        x_index = np.linspace(0, W - 1, W)
        y_index = np.linspace(0, H - 1, H)
        xx, yy = np.meshgrid(x_index, y_index)

        xx = (xx - cx) / fx * depth
        yy = (yy - cy) / fy * depth

        index = np.where((depth > 0) & (depth < 10))
        xx = xx[index].copy().reshape(-1)
        yy = yy[index].copy().reshape(-1)
        depth = depth[index].copy().reshape(-1)

        pc = np.stack((xx, yy, depth, np.ones_like(xx)))

        pc = pc.reshape(4, -1)

        E = self.obs['camera_matrix']
        inv_E = np.linalg.inv(np.array(E).reshape((4, 4)))
        rot = np.array([[1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, -1, 0],
                        [0, 0, 0, 1]])
        inv_E = np.dot(inv_E, rot)
        rpc = np.dot(inv_E, pc)
        return rpc[:3]

    def cal_object_position(self, o_dict):
        pc = self.get_pc(o_dict['seg_color'])
        if pc.shape[1] < 5:
            return None
        position = pc.mean(1)
        return position[:3]

    def filtered(self, all_visible_objects):
        visible_obj = []
        for o in all_visible_objects:
            if o['type'] is not None and o['type'] < 4:
                visible_obj.append(o)
        return visible_obj

    def _uses_v34_protocol(self):
        return getattr(self.LLM, "peer_consult_protocol", None) in (
            "PeerConsultV3.4", "PeerConsultV3.5")

    def _uses_v35_protocol(self):
        return (getattr(self.LLM, "peer_consult_protocol", None) ==
                "PeerConsultV3.5")

    def _uses_v4_protocol(self):
        """Return whether the policy-light V4 Harness is active.

        V4 deliberately isn't folded into either of the legacy version
        helpers above.  Those helpers select V3.4/V3.5 perception, navigation,
        container and recovery semantics that aren't part of the V4 Core.
        """
        return (getattr(self.LLM, "peer_consult_protocol", None) ==
                "PeerConsultV4")

    def _uses_shared_delivery_target_adapter(self):
        """Return whether a verified shared task target may feed executor IO.

        V3.4/V3.5 retain their published behavior.  V4 exposes the same
        factual bridge only under an explicit adapter flag so its original
        executor-isolation ablation remains reproducible.
        """
        return (self._uses_v34_protocol() or
                (self._uses_v4_protocol() and
                 os.getenv("TDW_MAT_V4_SHARED_DELIVERY_TARGET", "0") == "1"))

    def _uses_episode_isolation_protocol(self):
        """Protocols whose planner views must be cleared between episodes.

        Episode isolation is a protocol-agnostic correctness invariant, not a
        V3.5 policy.  Keeping it behind a separate capability lets V4 reuse the
        reset safety without inheriting any V3.4/V3.5 executor behavior.
        """
        return self._uses_v35_protocol() or self._uses_v4_protocol()

    @staticmethod
    def _episode_provenance_from_output_dir(output_dir):
        """Return a stable, local-only episode identity for cached evidence."""
        if output_dir is None:
            return None
        return os.path.normcase(os.path.abspath(os.fspath(output_dir)))

    @staticmethod
    def _is_shared_bed_record(record):
        return (record.get("knowledge_source") ==
                "shared_memory_board")

    def _prune_v35_shared_beds(self):
        """Drop stale shared beds and let genuine local perception win."""
        beds = list((self.object_list or {}).get(2, []))
        local_beds = [record for record in beds
                      if not self._is_shared_bed_record(record)]
        if local_beds:
            retained = local_beds
        else:
            retained = [
                record for record in beds
                if (record.get("episode_provenance") ==
                    self.episode_provenance)
            ]
        self.object_list[2] = retained

        # ``object_per_room`` is a second view over the same entities. Keep it
        # consistent so a stale shared bed cannot survive in the prompt after
        # the flat list has selected a locally observed bed.
        retained_ids = {id(record) for record in retained}
        for room_objects in (self.object_per_room or {}).values():
            room_beds = list(room_objects.get(2, []))
            room_objects[2] = [
                record for record in room_beds
                if (not self._is_shared_bed_record(record) or
                    id(record) in retained_ids)
            ]

    def _queue_perception_event(self, object_id):
        """Persist a first observation until it is presented to V3.4's LLM."""
        if not self._uses_v34_protocol():
            return
        if any(event.get("object_id") == object_id
               for event in self.pending_perception_events):
            return
        entity = self.object_info.get(object_id) or {}
        object_type = entity.get("type")
        event_kind = {
            0: "new_target_evidence",
            1: "new_container_evidence",
            2: "new_bed_evidence",
        }.get(object_type)
        if event_kind is None:
            return
        position = entity.get("position")
        serialized_position = None
        if position is not None:
            try:
                serialized_position = [float(value) for value in position[:3]]
            except (TypeError, ValueError, IndexError):
                serialized_position = None
        room = None
        if position is not None:
            try:
                room = self.env_api['belongs_to_which_room'](position)
            except (KeyError, TypeError, ValueError):
                room = None
        self.pending_perception_events.append({
            "kind": event_kind,
            "object_id": int(object_id),
            "name": entity.get("name"),
            "object_type": object_type,
            "room": room,
            "position": serialized_position,
            "frame": int(self.num_frames or 0),
        })
        # Bound prompt growth while retaining substantially more than a single
        # navigation transition can discover in these scenes.
        self.pending_perception_events = self.pending_perception_events[-32:]

    def _holding_real_object(self):
        return any(item.get("type") is not None
                   for item in self.obs.get("held_objects", []))

    def _prepare_v34_perception_boundary(self):
        """Open a safe replanning boundary for relevant retained evidence.

        Low-level ``ongoing`` actions are never interrupted.  A newly observed
        goal target may stop blind room navigation/exploration after its
        current primitive terminates.  A newly observed bed may do the same
        while payload is held.  Other events remain queued until the active
        manipulation task ends naturally.
        """
        if not self._uses_v34_protocol() or not self.pending_perception_events:
            return
        if self.plan is None:
            return
        navigating = self.plan.startswith(("go to ", "explore "))
        if not navigating:
            return
        target_event = any(
            event.get("kind") == "new_target_evidence" and
            event.get("name") in self.goal_objects
            for event in self.pending_perception_events)
        bed_event = any(
            event.get("kind") == "new_bed_evidence"
            for event in self.pending_perception_events)
        if target_event or (bed_event and self._holding_real_object()):
            self.plan = None

    def _inject_v34_shared_bed(self):
        """Expose verified team bed evidence through the legacy bed list.

        The original transport executor intentionally remains byte-for-byte
        unchanged.  V3.4 reconciles shared symbolic knowledge one layer above
        it by adapting the compact decision-card fact into the same object
        record shape produced by local perception.
        """
        if not self._uses_shared_delivery_target_adapter():
            return
        if self._uses_v35_protocol():
            self._prune_v35_shared_beds()
        if self.object_list[2]:
            return
        shared_bed = self.LLM.shared_delivery_target()
        source_provenance = ((shared_bed or {}).get("episode_provenance")
                             if shared_bed else None)
        if (self._uses_v35_protocol() and
                source_provenance is not None and
                source_provenance != self.episode_provenance):
            # A provenance-bearing record from another episode is never safe
            # to turn into a navigation target. Records without provenance are
            # still accepted because current V3.4 cards predate this field.
            return
        shared_position = ((shared_bed or {}).get("position")
                           if shared_bed else None)
        if shared_position is None:
            return
        try:
            position = np.asarray(shared_position[:3], dtype=float)
        except (TypeError, ValueError, IndexError):
            return
        room = (shared_bed or {}).get("room")
        if room is None:
            room = self.env_api['belongs_to_which_room'](position)
        record = {
            "id": (shared_bed or {}).get("id"),
            "type": 2,
            "name": (shared_bed or {}).get("name") or "bed",
            "position": position,
            "knowledge_source": "shared_memory_board",
        }
        if self._uses_v35_protocol():
            record["episode_provenance"] = self.episode_provenance
            record["source_episode_provenance"] = source_provenance
        self.object_list[2].append(record)
        if room in self.object_per_room:
            self.object_per_room[room][2].append(record)

    def _bookkeep_invalid_object(self, object_id):
        """Keep an invalid target out of planning without claiming delivery."""
        if self.fix_lm_satisfied:
            if object_id not in self.force_ignore:
                self.force_ignore.append(object_id)
        else:
            # Original CoELA behavior, retained for baseline comparability.
            self.satisfied.append(object_id)

    def _bookkeep_opponent_held_object(self, object_id, object_name, object_type):
        """Temporarily remove a teammate-held object from the local map.

        The original implementation also appended the ID to `satisfied`, which
        made the prompt call it transported and filtered it forever. Corrected
        mode clears only the stale map entry. Current teammate-held IDs are
        already excluded by `oppo_holding_objects_id` and the memory ignore set;
        the object can therefore be rediscovered after it is released.
        """
        if self.fix_lm_satisfied or object_id not in self.satisfied:
            if not self.fix_lm_satisfied:
                # Original CoELA behavior, retained for baseline comparability.
                self.satisfied.append(object_id)
            self.object_info[object_id] = {
                "name": object_name,
                "id": object_id,
                "type": object_type,
            }
            self.object_map[np.where(self.id_map == object_id)] = 0
            self.id_map[np.where(self.id_map == object_id)] = 0

    def get_object_list(self):
        object_list = {0: [], 1: [], 2: []}
        self.object_per_room = {room: {0: [], 1: [], 2: []} for room in self.rooms_name}
        for object_type in [0, 1, 2]:
            obj_map_indices = np.where(self.object_map == object_type + 1)
            if obj_map_indices[0].shape[0] == 0:
                continue
            for idx in range(0, len(obj_map_indices[0])):
                i, j = obj_map_indices[0][idx], obj_map_indices[1][idx]
                id = self.id_map[i, j]
                if (id in self.satisfied or id in self.holding_objects_id or
                        id in self.oppo_holding_objects_id or
                        (self.fix_lm_satisfied and id in self.with_oppo) or
                        self.object_info[id] in object_list[object_type]):
                    continue
                object_list[object_type].append(self.object_info[id])
                room = self.env_api['belongs_to_which_room'](self.object_info[id]['position'])
                if room is None:
                    self.logger.warning(f"obj {self.object_info[id]} not in any room")
                    # raise Exception(f"obj not in any room")
                    continue
                self.object_per_room[room][object_type].append(self.object_info[id])
        self.object_list = object_list


    def get_new_object_list(self):
        self.visible_objects = self.obs['visible_objects']
        self.new_object_list = {0: [], 1: [], 2: []}
        for o_dict in self.visible_objects:
            if o_dict['id'] is None: continue
            self.color2id[o_dict['seg_color']] = o_dict['id']
            if (o_dict['id'] is None or o_dict['id'] in self.satisfied or
                    o_dict['id'] in self.with_character or
                    (self.fix_lm_satisfied and o_dict['id'] in self.with_oppo) or
                    o_dict['type'] == 4):
                continue
            position = self.cal_object_position(o_dict)
            if position is None:
                continue
            object_id = o_dict['id']
            new_obj = False
            if object_id not in self.object_info:
                self.object_info[object_id] = {}
                new_obj = True
            self.object_info[object_id]['id'] = object_id
            self.object_info[object_id]['type'] = o_dict['type']
            self.object_info[object_id]['name'] = o_dict['name']
            if o_dict['type'] == 3: # the agent
                if o_dict['id'] == self.opponent_agent_id:
                    position = self.cal_object_position(o_dict)
                    self.oppo_pos = position
                    if position is not None:
                        oppo_last_room = self.env_api['belongs_to_which_room'](position)
                        if oppo_last_room is not None:
                            self.oppo_last_room = oppo_last_room
                continue
            if (object_id in self.satisfied or object_id in self.with_character or
                    (self.fix_lm_satisfied and object_id in self.with_oppo)):
                continue
            self.object_info[object_id]['position'] = position
            if new_obj:
                # Unlike ``new_object_list``, this event isn't lost when two
                # objects quantize into the same semantic-map cell.
                self._queue_perception_event(object_id)
            if o_dict['type'] == 0:
                x, y, z = self.object_info[object_id]['position']

                i, j = self.pos2map(x, z)
                if self.object_map[i, j] == 0:
                    self.object_map[i, j] = 1
                    self.id_map[i, j] = object_id
                    if new_obj:
                        self.new_object_list[0].append(object_id)

            elif o_dict['type'] == 1:
                x, y, z = self.object_info[object_id]['position']
                i, j = self.pos2map(x, z)
                if self.object_map[i, j] == 0:
                    self.object_map[i, j] = 2
                    self.id_map[i, j] = object_id
                    if new_obj:
                        self.new_object_list[1].append(object_id)
            elif o_dict['type'] == 2:
                x, y, z = self.object_info[object_id]['position']
                i, j = self.pos2map(x, z)
                if self.object_map[i, j] == 0:
                    self.object_map[i, j] = 3
                    self.id_map[i, j] = object_id
                    if new_obj:
                        self.new_object_list[2].append(object_id)

    def color2id_fc(self, color):
        if color not in self.color2id:
            if (color != self.agent_color).any(): 
                return -100 # wall
            else: return self.agent_id # agent
        else: return self.color2id[color]

    def l2_distance(self, st, g):
        return ((st[0] - g[0]) ** 2 + (st[1] - g[1]) ** 2) ** 0.5

    def reach_target_pos(self, target_pos, threshold = 1.0):
        x, _, z = self.obs["agent"][:3]
        gx, _, gz = target_pos
        d = self.l2_distance((x, z), (gx, gz))
        if self.plan.startswith('transport'):
            if self.env_api['belongs_to_which_room'](np.array([x, 0, z])) != self.env_api['belongs_to_which_room'](np.array([gx, 0, gz])):
                return False
        return d < threshold

    def reset(self, obs, goal_objects = None, output_dir = None, env_api = None, rooms_name = None, agent_color = [-1, -1, -1], agent_id = 0, gt_mask = True, save_img = True):
        self.force_ignore = []
        self.agent_memory = AgentMemory(agent_id = agent_id, agent_color = agent_color, output_dir = output_dir, gt_mask=gt_mask, gt_behavior=True, env_api=env_api, constraint_type = None, map_size = self.map_size, scene_bounds = self._scene_bounds)
        self.invalid_count = 0
        self.obs = obs
        self.env_api = env_api
        self.agent_color = agent_color
        self.agent_id = agent_id
        self.rooms_name = rooms_name
        self.room_distance = 0
        assert type(goal_objects) == dict
        self.goal_objects = goal_objects
        self.oppo_pos = None
        goal_count = sum([v for k, v in goal_objects.items()])
        if output_dir is not None:
            self.output_dir = output_dir
        self.last_action = None
        self.id_map = np.zeros(self.map_size, np.int32)
        self.object_map = np.zeros(self.map_size, np.int32)

        self.object_info = {}
        self.object_list = {0: [], 1: [], 2: []}
        self.new_object_list = {0: [], 1: [], 2: []}
        self.pending_perception_events = []
        if self._uses_episode_isolation_protocol():
            # These views/cursors were missing from the legacy reset and could
            # therefore point into the preceding dataset episode until the
            # first fresh planner boundary.
            self.object_per_room = {}
            self.target_pos = None
            self.visible_objects = None
            self.pre_action = None
            self.local_step = 0
            self.episode_provenance = (
                self._episode_provenance_from_output_dir(output_dir))
            # The coordinator replaces the decision card before acting, but
            # clearing it here makes that ordering an invariant instead of an
            # implicit requirement and also covers direct-agent test paths.
            self.LLM.peer_decision_card = None
            self.LLM.pending_perception_events = []
            self.LLM.current_room = None
            self.LLM.rooms_explored = None
            self.LLM.object_list = None
            self.LLM.holding_objects = None
            self.LLM.obj_per_room = None
            self.LLM.allow_message_this_turn = True
        self.container_held = None
        self.holding_objects_id = []
        self.oppo_holding_objects_id = []
        self.with_character = []
        self.with_oppo = []
        self.oppo_last_room = None
        self.satisfied = []
        self.color2id = {}
        self.dropping_object = []
        self.steps = 0
        self.num_frames = 0
        # print(self.obs.keys())
        self.position = self.obs["agent"][:3]
        self.forward = self.obs["agent"][3:]
        self.current_room = self.env_api['belongs_to_which_room'](self.position)
        self.rotated = None
        self.explore_route_room = None
        self.explore_waypoints = []
        self.explore_waypoint_index = 0
        self.explore_scan_turns = 0
        if self._uses_episode_isolation_protocol():
            self.explore_count = 0
        self.rooms_explored = {}
        
        self.plan = None
        self._peer_consult_planning_boundary = False
        self._peer_consult_selected_plan = None
        self._peer_consult_commitment_transition = None
        self.action_history = [f"go to {self.current_room} at initial step"]
        self.dialogue_history = []
        self.gt_mask = gt_mask
        if self.gt_mask == True:
            self.detection_threshold = 5
        else:
            self.detection_threshold = 3
            from detection import init_detection
            # only here we need to use the detection model, other places we use the gt mask
            # so we put the import here
            self.detection_model = init_detection()
        self.navigation_threshold = 5
        # print(self.rooms_name)
        self.LLM.reset(self.rooms_name, self.goal_objects)
        self.save_img = save_img

    def _execute_room_navigation_plan(self):
        """Dispatch room navigation without changing published executors."""
        if self._uses_v4_protocol():
            return self.gotoroom()
        if self._uses_v34_protocol():
            return self.gotoroom_v34()
        if self.stable_task_protocol:
            return self.gotoroom_v33()
        return self.gotoroom()

    def _execute_room_exploration_plan(self):
        """Dispatch exploration while keeping V4 on original CoELA logic."""
        if self._uses_v4_protocol():
            return self.goexplore()
        if self.coverage_exploration:
            return self.goexplore_v33()
        return self.goexplore()

    def move(self, target_pos):
        self.local_step += 1
        action, path_len = self.agent_memory.move_to_pos(target_pos)
        return action

    def gotoroom(self):
        target_room = ' '.join(self.plan.split(' ')[2: 4])
        if target_room[-1] == ',': target_room = target_room[:-1]
        if self.debug:
            print(target_room)
        target_pos = self.env_api['center_of_room'](target_room)
        if self.current_room == target_room and self.room_distance == 0:
            self.plan = None
            return None
        # add an interruption if anything new happens
        if len(self.new_object_list[0]) + len(self.new_object_list[1]) + len(self.new_object_list[2]) > 0:
            self.action_history[-1] = self.action_history[-1].replace(self.plan, f'go to {self.current_room}')
            self.new_object_list = {0: [], 1: [], 2: []}
            self.plan = None
            return None
        return self.move(target_pos)

    def gotoroom_v33(self):
        """Stable V3.3 room transition using the unchanged path executor."""
        target_room = ' '.join(self.plan.split(' ')[2: 4])
        if target_room[-1] == ',': target_room = target_room[:-1]
        if self.debug:
            print(target_room)
        target_pos = self.env_api['center_of_room'](target_room)
        if self.current_room == target_room and self.room_distance == 0:
            self.plan = None
            return None
        # New objects are retained as evidence for the next decision boundary;
        # they no longer erase the current transition mid-route.
        return self.move(target_pos)

    def gotoroom_v34(self):
        """V3.4 transition; perception is reviewed at safe boundaries.

        The review happens centrally in ``act`` after a low-level primitive
        terminates.  Keeping this executor free of object-specific interrupts
        prevents the original target oscillation while ensuring evidence is
        neither discarded nor delayed indefinitely.
        """
        return self.gotoroom_v33()


    def goexplore(self):
        target_room = ' '.join(self.plan.split(' ')[-2:])
        # assert target_room == self.current_room, f"{target_room} != {self.current_room}"
        target_pos = self.env_api['center_of_room'](target_room)
        self.explore_count += 1
        dis_threshold = 1 + self.explore_count / 50
        if not self.reach_target_pos(target_pos, dis_threshold):
            return self.move(target_pos)
        if self.rotated is None:
            self.rotated = 0
        if self.rotated == 16:
            self.roatated = 0
            self.rooms_explored[target_room] = 'all'
            self.plan = None
            return None
        self.rotated += 1
        action = {"type": 1}
        return action

    def goexplore_v33(self):
        """Visit several room samples while reusing upstream navigation."""
        target_room = ' '.join(self.plan.split(' ')[-2:])
        if self.explore_route_room != target_room:
            waypoint_fn = self.env_api.get('room_waypoints')
            raw_waypoints = (waypoint_fn(target_room)
                             if waypoint_fn is not None else
                             [self.env_api['center_of_room'](target_room)])
            remaining = [np.asarray(value, dtype=float)
                         for value in raw_waypoints]
            ordered = []
            cursor = np.asarray(self.position, dtype=float)
            while remaining:
                index = min(
                    range(len(remaining)),
                    key=lambda value: np.linalg.norm(
                        remaining[value][[0, 2]] - cursor[[0, 2]]))
                cursor = remaining.pop(index)
                ordered.append(cursor)
            self.explore_route_room = target_room
            self.explore_waypoints = ordered
            self.explore_waypoint_index = 0
            self.explore_scan_turns = 0

        if self.explore_waypoint_index >= len(self.explore_waypoints):
            self.rooms_explored[target_room] = 'all'
            self.plan = None
            self.explore_route_room = None
            return None
        target_pos = self.explore_waypoints[self.explore_waypoint_index]
        if not self.reach_target_pos(target_pos, 1.0):
            return self.move(target_pos)
        if self.explore_scan_turns < 4:
            self.explore_scan_turns += 1
            return {"type": 1}
        self.explore_waypoint_index += 1
        self.explore_scan_turns = 0
        if self.explore_waypoint_index >= len(self.explore_waypoints):
            self.rooms_explored[target_room] = 'all'
            self.plan = None
            self.explore_route_room = None
            return None
        return self.move(self.explore_waypoints[
            self.explore_waypoint_index])

    def gograsp(self):
        target_object_id = int(self.plan.split(' ')[-1][1:-1])
        if target_object_id in self.holding_objects_id:
            self.logger.info(f"successful holding!")
            self.object_map[np.where(self.id_map == target_object_id)] = 0
            self.id_map[np.where(self.id_map == target_object_id)] = 0
            self.plan = None
            return None
        
        if self.target_pos is None:
            self.target_pos = copy.deepcopy(self.object_info[target_object_id]['position'])
        target_object_pos = self.target_pos

        if target_object_id not in self.object_info or target_object_id in self.with_oppo:
            if self.debug:
                self.logger.debug(f"grasp failed. object is not here any more!")
            self.plan = None
            return None
        if not self.reach_target_pos(target_object_pos):
            return self.move(target_object_pos)
        action = {"type": 3, "object": target_object_id, "arm": 'left' if self.obs["held_objects"][0]['id'] is None else 'right'}
        return action
    
    def goput(self):
        if len(self.holding_objects_id) == 0:
            self.plan = None
            self.with_character = [self.agent_id]
            return None
        if self.target_pos is None:
            self.target_pos = copy.deepcopy(self.object_list[2][0]['position'])
        target_pos = self.target_pos

        if not self.reach_target_pos(target_pos, 1.5):
            return self.move(target_pos)
        if self.obs["held_objects"][0]['type'] is not None:
            self.dropping_object += [self.obs["held_objects"][0]['id']]
            if self.obs["held_objects"][0]['type'] == 1:
                self.dropping_object += [x for x in self.obs["held_objects"][0]['contained'] if x is not None]
            return {"type": 5, "arm": "left"}
        else:
            self.dropping_object += [self.obs["held_objects"][1]['id']]
            if self.obs["held_objects"][1]['type'] == 1:
                self.dropping_object += [x for x in self.obs["held_objects"][1]['contained'] if x is not None]
            return {"type": 5, "arm": "right"}

    def putin(self):
        if len(self.holding_objects_id) == 1:
            self.logger.info("Successful putin")
            self.plan = None
            return None
        action = {"type": 4}
        return action
    
    def detect(self):
        detect_result = self.detection_model(self.obs['rgb'][..., [2, 1, 0]])['predictions'][0]
        obj_infos = []
        curr_seg_mask = np.zeros((self.obs['rgb'].shape[0], self.obs['rgb'].shape[1], 3)).astype(np.int32)
        curr_seg_mask.fill(-1)
        for i in range(len(detect_result['labels'])):
            if detect_result['scores'][i] < 0.3: continue
            mask = detect_result['masks'][:,:,i]
            label = detect_result['labels'][i]
            curr_info = self.env_api['get_id_from_mask'](mask = mask, name = self.detection_model.cls_to_name_map(label)).copy()
            if curr_info['id'] is not None:
                obj_infos.append(curr_info)
                curr_seg_mask[np.where(mask)] = curr_info['seg_color']
        curr_with_seg, curr_seg_flag = self.env_api['get_with_character_mask'](character_object_ids = self.with_character)
        curr_seg_mask = curr_seg_mask * (~ np.expand_dims(curr_seg_flag, axis = -1)) + curr_with_seg * np.expand_dims(curr_seg_flag, axis = -1)
        return obj_infos, curr_seg_mask

    def LLM_plan(self):
        # Some legacy/unit construction paths instantiate lm_agent without
        # running the modern __init__/reset sequence.  V3.4's event queue is
        # optional state for those paths, so treat an absent queue as empty.
        pending_events = copy.deepcopy(
            getattr(self, "pending_perception_events", []))
        self.LLM.pending_perception_events = pending_events
        result = self.LLM.run(self.num_frames, self.current_room, self.rooms_explored, self.obs['held_objects'],[self.object_info[x] for x in self.satisfied if x in self.object_info], self.object_list, self.object_per_room, self.action_history, self.dialogue_history, self.obs['oppo_held_objects'], self.oppo_last_room)
        # Consume only after the model call returned successfully; transport
        # or infrastructure exceptions therefore cannot silently lose events.
        processed_ids = {event.get("object_id") for event in pending_events}
        self.pending_perception_events = [
            event for event in getattr(
                self, "pending_perception_events", [])
            if event.get("object_id") not in processed_ids]
        return result

    def act(self, obs):
        # The V4 Harness monitors *planner* retries, not the low-level motion
        # commands emitted while one CoELA plan is executing.  Publish an
        # edge-triggered adapter marker and the selected semantic plan so the
        # coordinator still sees it when an executor immediately clears
        # ``self.plan`` (for example, an in-range grasp or a message action).
        self._peer_consult_planning_boundary = False
        self._peer_consult_selected_plan = None
        self._peer_consult_commitment_transition = None
        self.obs = obs.copy()
        self.obs['rgb'] = self.obs['rgb'].transpose(1, 2, 0)
        self.num_frames = obs['current_frames']
        self.steps += 1

        if not self.gt_mask:
            self.obs['visible_objects'], self.obs['seg_mask'] = self.detect()

        if obs['valid'] == False:
            if self.last_action is not None and 'object' in self.last_action:
                self.object_map[np.where(self.id_map == self.last_action['object'])] = 0
                self.id_map[np.where(self.id_map == self.last_action['object'])] = 0
                self._bookkeep_invalid_object(self.last_action['object'])
            self.invalid_count += 1
            self.plan = None
            assert self.invalid_count < 10, "invalid action for 10 times"
    
        if self.communication:
            for i in range(len(obs["messages"])):
                if obs["messages"][i] is not None:
                    self.dialogue_history.append(f"{self.agent_names[i]}: {copy.deepcopy(obs['messages'][i])}")
    
        self.position = self.obs["agent"][:3]
        self.forward = self.obs["agent"][3:]
        current_room = self.env_api['belongs_to_which_room'](self.position)
        if current_room is not None:
            self.current_room = current_room
        self.room_distance = self.env_api['get_room_distance'](self.position)
        if self.current_room not in self.rooms_explored or self.rooms_explored[self.current_room] != 'all':
            self.rooms_explored[self.current_room] = 'part'
        if self.agent_id not in self.with_character: self.with_character.append(self.agent_id) # DWH: buggy env, need to solve later.
        self.holding_objects_id = []
        self.with_oppo = []
        self.oppo_holding_objects_id = []
        for x in self.obs['held_objects']:
            if x['type'] == 0:
                self.holding_objects_id.append(x['id'])
                if x['id'] not in self.with_character: self.with_character.append(x['id']) # DWH: buggy env, need to solve later.
                # self.with_character.append(x['id'])
            elif x['type'] == 1:
                self.holding_objects_id.append(x['id'])
                if x['id'] not in self.with_character: self.with_character.append(x['id']) # DWH: buggy env, need to solve later.
                #self.with_character.append(x['id'])
                for y in x['contained']:
                    if y is None:
                        break
                    if y not in self.with_character: self.with_character.append(y)
                    #self.with_character.append(y)
        oppo_name = {}
        oppo_type = {}
        for x in self.obs['oppo_held_objects']:
            if x['type'] == 0:
                self.oppo_holding_objects_id.append(x['id'])
                self.with_oppo.append(x['id'])
                oppo_name[x['id']] = x['name']
                oppo_type[x['id']] = x['type']
            elif x['type'] == 1:
                self.oppo_holding_objects_id.append(x['id'])
                self.with_oppo.append(x['id'])
                oppo_name[x['id']] = x['name']
                oppo_type[x['id']] = x['type']
                for i, y in enumerate(x['contained']):
                    if y is None:
                        break
                    self.with_oppo.append(y)
                    oppo_name[y] = x['contained_name'][i]
                    oppo_type[y] = 0
        for obj in self.with_oppo:
            self._bookkeep_opponent_held_object(obj, oppo_name[obj], oppo_type[obj])
        if not self.obs['valid']: # invalid, the object is not there
            if self.last_action is not None and 'object' in self.last_action:
                self.object_map[np.where(self.id_map == self.last_action['object'])] = 0
                self.id_map[np.where(self.id_map == self.last_action['object'])] = 0
        if len(self.dropping_object) > 0 and self.obs['status'] == 1:
            self.logger.info(f"Drop object: {self.dropping_object}")
            if self.fix_lm_satisfied and not self.authoritative_satisfied:
                self.satisfied += [obj for obj in self.dropping_object
                                   if obj not in self.satisfied]
            elif not self.fix_lm_satisfied:
                self.satisfied += self.dropping_object
            self.dropping_object = []
            if len(self.holding_objects_id) == 0:
                self.logger.info("successful drop!")
                self.plan = None

        ignore_obstacles = []
        ignore_ids = []
        self.with_character = [self.agent_id]
        temp_with_oppo = []
        for x in self.obs["held_objects"]:
            if x is None or x["id"] is None:
                continue
            self.with_character.append(x["id"])
            if "contained" in x:
                for y in x["contained"]:
                    if y is not None:
                        self.with_character.append(y)

        for x in self.force_ignore:
            self.with_character.append(x)

        for x in self.obs["oppo_held_objects"]:
            if x is None or x["id"] is None:
                continue
            temp_with_oppo.append(x["id"])
            if "contained" in x:
                for y in x["contained"]:
                    if y is not None:
                        temp_with_oppo.append(y)

        ignore_obstacles = self.with_character + ignore_obstacles
        ignore_ids = self.with_character + ignore_ids
        ignore_ids = temp_with_oppo + ignore_ids
        ignore_ids += self.satisfied
        ignore_obstacles += self.satisfied

        self.agent_memory.update(
            obs, ignore_ids=ignore_ids, ignore_obstacles=ignore_obstacles, save_img = self.save_img
        )

        if self.obs['status'] == 0: # ongoing
            return {'type': 'ongoing'}

        self.get_new_object_list()
        print(self.new_object_list)
        self.get_object_list()
        self._inject_v34_shared_bed()
        self._prepare_v34_perception_boundary()

        info = {'satisfied': self.satisfied,
                'object_list': self.object_list,
                'new_object_list': self.new_object_list,
                'pending_perception_events': copy.deepcopy(
                    self.pending_perception_events),
                'current_room': self.current_room,
                'visible_objects': self.filtered(self.obs['visible_objects']),
                'obs': {k: v for k, v in self.obs.items() if k not in ['rgb', 'depth', 'seg_mask', 'camera_matrix', 'visible_objects']},
              }

        action = None
        lm_times = 0
        while action is None:
            if self.plan is None:
                self.target_pos = None
                if lm_times > 0:
                    print(info)
                if lm_times > 3:
                    raise Exception(f"retrying LM_plan too many times")
                plan, a_info = self.LLM_plan()
                if plan is None: # NO AVAILABLE PLANS! Explore from scratch!
                    print("No more things to do!")
                    plan = f"[wait]"
                if not plan_allowed_for_role(self.agent_role, plan):
                    raise PermissionError(
                        f"Role {self.agent_role!r} cannot execute plan {plan!r}")
                self.plan = plan
                if self._uses_v4_protocol():
                    self._peer_consult_planning_boundary = True
                    self._peer_consult_selected_plan = plan
                    self._peer_consult_commitment_transition = copy.deepcopy(
                        a_info.get("commitment_transition"))
                self.action_history.append(f"{'send a message' if plan.startswith('send a message:') else plan} at step {self.num_frames}")
                a_info.update({"Frames": self.num_frames})
                info.update({"LLM": a_info})
                lm_times += 1
            if self.plan.startswith('go to'):
                action = self._execute_room_navigation_plan()
            elif self.plan.startswith('explore'):
                self.explore_count = 0
                action = self._execute_room_exploration_plan()
            elif self.plan.startswith('go grasp'):
                action = self.gograsp()
            elif self.plan.startswith('put'):
                action = self.putin()
            elif self.plan.startswith('transport'):
                action = self.goput()
            #    self.with_character = [self.agent_id]
            elif self.plan.startswith('send a message:'):
                action = {"type": 6,
                          "message": ' '.join(self.plan.split(' ')[3:])}
                self.plan = None
            elif (self.plan.startswith('wait') or
                  self.plan.startswith('[wait]')):
                # The environment provides a one-frame no-op for both
                # embodiments.  This also makes the legacy human fallback
                # safe when no manipulation/navigation plan is available.
                action = {"type": 8, "delay": 1}
                self.plan = None
                break
            else:
                raise ValueError(f"unavailable plan {self.plan}")

        if action is not None and not action_allowed_for_role(
                self.agent_role, action):
            raise PermissionError(
                f"Role {self.agent_role!r} cannot execute action {action!r}")

        info.update({"action": action,
                     "plan": self.plan})
        if self.debug:
            self.logger.info(self.plan)
            self.logger.debug(info)
        self.last_action = action
        return action
