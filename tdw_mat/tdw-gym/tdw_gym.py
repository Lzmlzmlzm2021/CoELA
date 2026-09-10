import string
from typing import Optional

import gym
from gym.core import Env
import numpy as np
import os
import time
import copy

from tdw.replicant.arm import Arm
from tdw.tdw_utils import TDWUtils

from transport_challenge_multi_agent.transport_challenge import TransportChallenge
from collections import Counter
from tdw.replicant.action_status import ActionStatus
from tdw.replicant.image_frequency import ImageFrequency
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.output_data import OutputData, SegmentationColors, FieldOfView, Images
from tdw.scene_data.scene_bounds import SceneBounds
from tdw.add_ons.occupancy_map import OccupancyMap
from tdw.add_ons.object_manager import ObjectManager
from PIL import Image

import json
import pickle
from functools import partial
import signal
import psutil
from tenacity import retry, wait_fixed, retry_if_exception_type
from frame_budget import frame_budget_exhausted

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Function execution exceeded the timeout limit")

@retry(wait=wait_fixed(5), retry=retry_if_exception_type(TimeoutException))  # wait 5 seconds between retries
def might_fail_launch(launch, port = None):
    if port is not None:
        # The original shell pipeline only worked on Linux. Kill only a stale
        # TDW build whose command line contains this exact controller port.
        port_flag = f"-port {port}"
        for process in psutil.process_iter(["name", "cmdline"]):
            try:
                name = (process.info["name"] or "").lower()
                command_line = " ".join(process.info["cmdline"] or [])
                if name in ("tdw.exe", "tdw.x86_64") and port_flag in command_line:
                    print(f"Stopping stale TDW process {process.pid} on port {port}")
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except psutil.TimeoutExpired:
                        process.kill()
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                continue

    has_alarm = hasattr(signal, "SIGALRM") and hasattr(signal, "alarm")
    if has_alarm:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(600)
    try:
        print("Trying to launch tdw ...")
        return launch()
    finally:
        if has_alarm:
            signal.alarm(0)

class TDW(Env):
    def __init__(self, port = 1071, number_of_agents = 1, demo=False, rank=0, num_scenes = 0, train=False, \
                        screen_size = 512, exp = False, launch_build=True, gt_occupancy = False, gt_mask = True, enable_collision_detection = False, save_dir = 'results', max_frames = 3000, data_prefix = 'dataset/nips_dataset/', embodiments = None):
        self.messages = None
        self.data_prefix = data_prefix
        self.replicant_colors = None
        self.replicant_ids = None
        self.agent_colors = None
        self.names_mapping = None
        self.rooms_name = None
        self.action_buffer = None
        # A logical action can span multiple Unity frames and multiple buffered
        # Replicant sub-actions (for example reach_for -> grasp).  These tickets
        # make that lifecycle explicit to higher-level multi-agent controllers.
        self.action_epochs = None
        self.action_records = None
        self.scene_bounds = None
        self.goal_description = None
        self.object_manager = None
        self.occupancy_map = None
        self.gt_mask = gt_mask
        self.satisfied = None
        self.count = 0
        self.reach_threshold = 2
        self.number_of_agents = number_of_agents
        self.embodiments = list(embodiments) if embodiments is not None else ["replicant"] * number_of_agents
        if len(self.embodiments) != self.number_of_agents:
            raise ValueError("embodiments must have one entry per logical agent")
        if any(x not in ("replicant", "box") for x in self.embodiments):
            raise ValueError(f"Unsupported embodiments: {self.embodiments}")
        self.seed = None
        self.num_step = 0
        self.reward = 0
        self.done = False
        self.scene_info = None
        self.exp = exp
        self.success = False
        self.num_frames = 0
        self.data_id = rank     
        self.train = train
        self.port = port
        self.gt_occupancy = gt_occupancy
        self.screen_size = screen_size
        self.launch_build = launch_build
        self.enable_collision_detection = enable_collision_detection
        self.controller = None
        self.message_per_frame = 500
        rgb_space = gym.spaces.Box(0, 256,
                                 (3,
                                  self.screen_size,
                                  self.screen_size), dtype=np.int32)
        seg_space = gym.spaces.Box(0, 256, \
                                (self.screen_size, \
                                self.screen_size, \
                                3), dtype=np.int32)
        depth_space = gym.spaces.Box(0, 256, \
                                (self.screen_size, \
                                self.screen_size), dtype=np.int32)
        object_space = gym.spaces.Dict({
            'id': gym.spaces.Discrete(30),
            'type': gym.spaces.Discrete(4),
            'seg_color': gym.spaces.Box(0, 255, (3, ), dtype=np.int32),
            'name': gym.spaces.Text(max_length=100, charset=string.printable)
        })

        self.action_space_single = gym.spaces.Dict({
            'type': gym.spaces.Discrete(9), # 0-6 original actions; 8 is a timed wait
            'object': gym.spaces.Discrete(30),
            'arm': gym.spaces.Discrete(2),
            'message': gym.spaces.Text(max_length=1000, charset=string.printable)
        })
        
        self.hand_object_space = gym.spaces.Dict({
            'id': gym.spaces.Discrete(30),
            'type': gym.spaces.Discrete(4),
            'name': gym.spaces.Text(max_length=100, charset=string.printable),
            'contained': gym.spaces.Tuple(gym.spaces.Discrete(30) for _ in range(3)),
            'contained_name': gym.spaces.Tuple(gym.spaces.Text(max_length=100, charset=string.printable) for _ in range(3))
        })
        
        self.observation_space_single = gym.spaces.Dict({
            'rgb': rgb_space,
            'seg_mask': seg_space,
            'depth': depth_space,
            'agent': gym.spaces.Box(-30, 30, (6, ), dtype=np.float32),
            'held_objects': gym.spaces.Tuple((self.hand_object_space, self.hand_object_space)),
            'oppo_held_objects': gym.spaces.Tuple((self.hand_object_space, self.hand_object_space)),
            'visible_objects': gym.spaces.Tuple(object_space for _ in range(50)),
            'status': gym.spaces.Discrete(4),
            # ``status`` above is the legacy three-state projection used by
            # the published agents.  Preserve the originating ActionStatus so
            # navigation code can distinguish a collision from an unrelated
            # manipulation failure.
            'action_status': gym.spaces.Text(max_length=64,
                                             charset=string.printable),
            'action_status_code': gym.spaces.Discrete(
                max(value.value for value in ActionStatus) + 1),
            'obstacle_ids': gym.spaces.Sequence(
                gym.spaces.Discrete(2 ** 31)),
            'hit_wall': gym.spaces.Discrete(2),
            'valid': gym.spaces.Discrete(2),
            'action_id': gym.spaces.Discrete(2 ** 31),
            'action_type': gym.spaces.Text(max_length=32,
                                           charset=string.printable),
            'action_started_frame': gym.spaces.Box(
                low=-1, high=max_frames, shape=(), dtype=np.int32),
            'action_completed_frame': gym.spaces.Box(
                low=-1, high=max_frames, shape=(), dtype=np.int32),
            'action_buffer_length': gym.spaces.Discrete(16),
            'action_terminal': gym.spaces.Discrete(2),
            'FOV': gym.spaces.Box(0, 120, (1,), dtype=np.float32),
            'camera_matrix': gym.spaces.Box(-30, 30, (4, 4), dtype=np.float32),
            'messages': gym.spaces.Tuple(gym.spaces.Text(max_length=1000, charset=string.printable) for _ in range(2)),
            'current_frames': gym.spaces.Discrete(30),
        })

        self.observation_space = gym.spaces.Dict({
            str(i): self.observation_space_single for i in range(self.number_of_agents)
        })

        self.action_space = gym.spaces.Dict({
            str(i): self.observation_space_single for i in range(self.number_of_agents)
        })
        self.max_frame = max_frames
        action_log_dir = os.getenv("TDW_MAT_LOG_DIR", ".")
        os.makedirs(action_log_dir, exist_ok=True)
        self.f = open(os.path.join(action_log_dir, f'action{port}.log'), 'w')
        self.action_list = []
                    
        self.segmentation_colors = {}
        self.object_names = {}
        self.object_ids = {}
        self.object_categories = {}
        self.target_object_ids = []
        self.container_ids = []
        self.goal_position_id = None # The place to put the object
        self.fov = 0
        self.save_dir = save_dir

    def _agent_ids(self):
        """Return logical agent IDs, independent of their physical embodiment."""
        return range(self.number_of_agents)

    def _agent(self, agent_id):
        agents = getattr(self.controller, "agents", None)
        if agents is not None:
            return agents[agent_id]
        return self.controller.replicants[agent_id]

    def _is_replicant(self, agent_id):
        return self.embodiments[agent_id] == "replicant"

    @staticmethod
    def _empty_hand_observation():
        return {
            'id': None,
            'type': None,
            'name': None,
            'contained': [None, None, None],
            'contained_name': [None, None, None],
        }

    def _held_object_ids(self, agent_id):
        if not self._is_replicant(agent_id):
            return [None, None]
        held = self.controller.state.replicants.get(agent_id)
        if held is None:
            return [None, None]
        return list(held.values())

    def _agent_status(self, agent_id):
        return self._agent(agent_id).action.status

    @staticmethod
    def _action_diagnostics(action):
        """Return lossless, embodiment-neutral action diagnostics.

        Replicant actions expose ``status`` but don't retain overlap IDs,
        whereas ``BoxScoutAction`` additionally exposes ``obstacle_ids`` and
        ``hit_wall``. These fields are retained for logging and evaluation;
        the published CoELA planner does not add a collision-feedback cost
        overlay for either embodiment.
        """

        status = action.status
        status_name = status.name if isinstance(status, ActionStatus) else str(status)
        status_code = status.value if isinstance(status, ActionStatus) else int(status)
        obstacle_ids = tuple(sorted({
            int(object_id)
            for object_id in getattr(action, "obstacle_ids", ())
        }))
        hit_wall = bool(getattr(action, "hit_wall", False))
        return status_name, status_code, obstacle_ids, hit_wall
    
    def obs_filter(self, obs):
        if self.gt_mask:
            return obs
        else:
            new_obs = copy.deepcopy(obs)
            for agent in obs:
                new_obs[agent]['seg_mask'] = np.zeros_like(new_obs[agent]['seg_mask'])
                new_obs[agent]['visible_objects'] = []
                while len(new_obs[agent]['visible_objects']) < 50:
                    new_obs[agent]['visible_objects'].append({
                        'id': None,
                        'type': None,
                        'seg_color': None,
                        'name': None,
                    })
            return new_obs

    def get_object_type(self, id):
        if id in self.target_object_ids:
            return 0 # target object
        if id in self.container_ids:
            return 1 # container
        if self.object_categories[id] == 'bed':
            return 2 # goal position
        # 3: agent
        # 4: obstacle
        return 5 # unrelated object

    def get_with_character_mask(self, agent_id, character_object_ids):
        color_set = [self.segmentation_colors[id] for id in character_object_ids if id in self.segmentation_colors] + [self.agent_colors[id] for id in character_object_ids if self.agent_colors is not None and id in self.agent_colors and self.agent_colors[id] is not None]
        curr_with_seg = np.zeros_like(self.obs[str(agent_id)]['seg_mask'])
        curr_seg_flag = np.zeros((self.screen_size, self.screen_size), dtype = bool)
        for i in range(len(color_set)):
            color_pos = (self.obs[str(agent_id)]['seg_mask'] == np.array(color_set[i])).all(axis=2)
            curr_seg_flag = np.logical_or(curr_seg_flag, color_pos)
            curr_with_seg[color_pos] = color_set[i]
        return curr_with_seg, curr_seg_flag
        
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        output_dir: Optional[str] = None
    ):
        """
        reset the environment
        input:
            data_id: reset based on the data_id
        """
        # Changes it to always, since in each step, we need to get the image
        self._close_controller()
        # download_asset_bundles()
        check_tdw_version = os.environ.get(
            "TDW_MAT_SKIP_TDW_VERSION_CHECK", "0") != "1"
        self.controller = might_fail_launch(
            partial(
                TransportChallenge,
                port=self.port,
                check_version=check_tdw_version,
                launch_build=self.launch_build,
                screen_width=self.screen_size,
                screen_height=self.screen_size,
                image_frequency=ImageFrequency.always,
                png=True,
                image_passes=None,
                enable_collision_detection=self.enable_collision_detection,
                logger_dir=output_dir),
            port=self.port)
        print("Controller connected")
        self.success = False
        self.messages = [None for _ in range(self.number_of_agents)]
        self.reward = 0
        scene_info = options
        print(scene_info)
        self.satisfied = {}
        if output_dir is not None: self.save_dir = output_dir
        if scene_info is not None:
            scene = scene_info['scene']
            layout = scene_info['layout']
            if 'task' in scene_info:
                task = scene_info['task']
            else:
                task = None
        else: raise ValueError("No scene info assigned!")
        super().reset(seed=seed)
        self.seed = np.random.RandomState(seed)
        self.scene_info = scene_info
        
        # Now the scene is fixed, so num_containers and num_target_objects are not used anymore in new settings
        self.controller.start_floorplan_trial(scene=scene, layout=layout, replicants=self.number_of_agents, num_containers=4, num_target_objects=10,
                                   random_seed=seed, task = task, data_prefix = self.data_prefix,
                                   embodiments=self.embodiments)

        # Add a gt occupancy map. In the standard setting, we don't need this
        if self.gt_occupancy:
            self.occupancy_map = OccupancyMap()
            self.controller.add_ons.append(self.occupancy_map)
            self.occupancy_map.generate(cell_size=0.125, once = False)
        self.controller.communicate({"$type": "set_floorplan_roof",
                          "show": False})

        # Bright case   
        self.controller.communicate({"$type": "add_hdri_skybox", "name": "sky_white", "url": "https://tdw-public.s3.amazonaws.com/hdri_skyboxes/linux/2019.1/sky_white", "exposure": 2, "initial_skybox_rotation": 0, "sun_elevation": 90, "sun_initial_angle": 0, "sun_intensity": 1.25})
            
        # Set the field of view of the agent.
        for agent_id in self._agent_ids():
            self.controller.communicate({"$type": "set_field_of_view",
                            "avatar_id" : self._agent(agent_id).static.avatar_id, "field_of_view" : 90})
        self.fov = 90
        
        # Add a object manager for object position
        self.object_manager = ObjectManager()
        self.controller.add_ons.append(self.object_manager)

        data = self.controller.communicate({"$type": "send_segmentation_colors",
                          "show": False,
                          "frequency": "once"})
        
        # Show the occupancy map. In the standard setting, we don't need this
        if self.gt_occupancy:            
            self.occupancy_map.show()
            print(self.occupancy_map.occupancy_map)
            h, w = self.occupancy_map.occupancy_map.shape
            print(self.occupancy_map.occupancy_map.shape)

        # Make name easier to read
        names_mapping_path = f'./dataset/name_map.json'
        with open(names_mapping_path, 'r') as f: self.names_mapping = json.load(f)

        self.segmentation_colors = {}
        self.object_names = {}
        self.object_ids = {}
        self.object_categories = {}
        self.target_object_ids = self.controller.state.target_object_ids
        self.container_ids = self.controller.state.container_ids
        self.replicant_ids = [i for i in self._agent_ids() if self._is_replicant(i)]
        
        for i in range(len(data) - 1):
            r_id = OutputData.get_data_type_id(data[i])
            if r_id == "segm":
                segm = SegmentationColors(data[i])
                for j in range(segm.get_num()):
                    object_id = segm.get_object_id(j)
                    self.segmentation_colors[object_id] = segm.get_object_color(j)
                    self.object_names[object_id] = segm.get_object_name(j).lower()
                    if self.object_names[object_id] in self.names_mapping:
                        self.object_names[object_id] = self.names_mapping[self.object_names[object_id]]
                    self.object_categories[object_id] = segm.get_object_category(j)
                    if self.object_categories[object_id] == 'bed':
                        self.goal_position_id = object_id
        
        self.agent_colors = {i: getattr(self._agent(i).static, "segmentation_color", None)
                             for i in self._agent_ids()}
        # Backward-compatible name used by older helper code.
        self.replicant_colors = self.agent_colors

        self.containment_all = {}
        
        # check colors are different:
        for x in self.segmentation_colors.keys():
            for y in self.segmentation_colors.keys():
                if x != y: assert (self.segmentation_colors[x] != self.segmentation_colors[y]).any()

        self.num_step = 0
        self.num_frames = 0
        self.goal_description = {}
        for i in self.target_object_ids:
            if self.object_names[i] in self.goal_description:
                self.goal_description[self.object_names[i]] += 1
            else:
                self.goal_description[self.object_names[i]] = 1

        room_type_path = f'./dataset/room_types.json'
        with open(room_type_path, 'r') as f: room_types = json.load(f)
        
        self.rooms_name = {}
        #now return <room_type> (id) for each room.        
        if type(layout) == str: now_layout = int(layout[0])
        else: now_layout = int(layout)
        for i, rooms_name in enumerate(room_types[scene[0]][now_layout]):
            if rooms_name not in ['Kitchen', 'Livingroom', 'Bedroom', 'Office']:
                the_name = None
            else:
                the_name = f'<{rooms_name}> ({1000 * (i + 1)})'
            self.rooms_name[i] = the_name

        self.done = False
        self.action_buffer = [[] for _ in range(self.number_of_agents)]
        self.action_epochs = [0 for _ in range(self.number_of_agents)]
        self.action_records = [{
            'action_id': 0,
            'action_type': 'none',
            'started_frame': -1,
            'completed_frame': -1,
            'buffer_length': 0,
            'terminal': True,
            'valid': True,
            'status': 'success',
        } for _ in range(self.number_of_agents)]

        resp = self.controller.communicate([{"$type": "send_scene_regions"}])
        self.scene_bounds = SceneBounds(resp=resp)
        self.all_rooms = [self.rooms_name[i] for i in range(len(self.rooms_name)) if self.rooms_name[i] is not None]
        info = {
            'goal_description': self.goal_description,
            'rooms_name': self.all_rooms,
            'agent_colors': self.agent_colors,
        }
        env_api = [{
            'belongs_to_which_room': self.belongs_to_which_room,
            'center_of_room': self.center_of_room,
            'room_waypoints': self.room_waypoints,
            'check_pos_in_room': self.check_pos_in_room,
            'get_room_distance': self.get_room_distance,
            'get_id_from_mask': partial(self.get_id_from_mask, agent_id=i),
            'get_with_character_mask': partial(self.get_with_character_mask, agent_id=i),
        } for i in range(self.number_of_agents)]
        self.obs = self.get_obs()
        return self.obs_filter(self.obs), info, env_api

    def pos_to_2d_box_distance(self, px, py, rx1, ry1, rx2, ry2):
        if px < rx1:
            if py < ry1:
                return ((px - rx1) ** 2 + (py - ry1) ** 2) ** 0.5
            elif py > ry2:
                return ((px - rx1) ** 2 + (py - ry2) ** 2) ** 0.5
            else:
                return rx1 - px
        elif px > rx2:
            if py < ry1:
                return ((px - rx2) ** 2 + (py - ry1) ** 2) ** 0.5
            elif py > ry2:
                return ((px - rx2) ** 2 + (py - ry2) ** 2) ** 0.5
            else:
                return px - rx2
        else:
            if py < ry1:
                return ry1 - py
            elif py > ry2:
                return py - ry2
            else:
                return 0
    
    def belongs_to_which_room(self, pos):
        min_dis = 100000
        room = None
        for i, region in enumerate(self.scene_bounds.regions):
            distance = self.pos_to_2d_box_distance(pos[0], pos[2], region.x_min, region.z_min, region.x_max, region.z_max)
            if distance < min_dis and self.rooms_name[i] is not None:
                min_dis = distance
                room = self.rooms_name[i]
        return room
    
    def get_room_distance(self, pos):
        min_dis = 100000
        room = None
        for i, region in enumerate(self.scene_bounds.regions):
            distance = self.pos_to_2d_box_distance(pos[0], pos[2], region.x_min, region.z_min, region.x_max, region.z_max)
            if distance < min_dis and self.rooms_name[i] is not None:
                min_dis = distance
                room = self.rooms_name[i]
        return min_dis
    
    def center_of_room(self, room):
        assert type(room) == str
        for index, name in self.rooms_name.items():
            if name == room:
                room = index
        return self.scene_bounds.regions[room].center

    def room_waypoints(self, room):
        """Return inset coverage points for high-level room exploration.

        These are semantic room samples, not a replacement navigation path.
        AgentMemory still plans every movement and handles obstacles.
        """
        assert type(room) == str
        room_index = None
        for index, name in self.rooms_name.items():
            if name == room:
                room_index = index
                break
        if room_index is None:
            return [self.center_of_room(room)]
        region = self.scene_bounds.regions[room_index]
        center = np.asarray(region.center, dtype=float)
        x_values = (
            region.x_min + 0.28 * (region.x_max - region.x_min),
            region.x_max - 0.28 * (region.x_max - region.x_min),
        )
        z_values = (
            region.z_min + 0.28 * (region.z_max - region.z_min),
            region.z_max - 0.28 * (region.z_max - region.z_min),
        )
        points = [center]
        for x in x_values:
            for z in z_values:
                points.append(np.asarray([x, center[1], z], dtype=float))
        unique = []
        for point in points:
            if not any(np.linalg.norm(point[[0, 2]] - value[[0, 2]]) < 0.4
                       for value in unique):
                unique.append(point)
        return unique
    
    def check_pos_in_room(self, pos):
        if len(pos) == 3:
            for region in self.scene_bounds.regions:
                if region.is_inside(pos[0], pos[2]):
                    return True
        elif len(pos) == 2:
            for region in self.scene_bounds.regions:
                if region.is_inside(pos[0], pos[1]):
                    return True
        return False

    def map_status(self, status, buffer_len = 0):
        if status == ActionStatus.ongoing or buffer_len > 0:
            return 0
        elif status == ActionStatus.success or status == ActionStatus.still_dropping:
            return 1
        else: return 2

    def get_2d_distance(self, pos1, pos2):
        return np.linalg.norm(np.array(pos1[[0, 2]]) - np.array(pos2[[0, 2]]))

    def check_goal(self):
        r'''
        Check if the goal is achieved
        return: count, total, done
        '''
        place_pos = self.object_manager.transforms[self.goal_position_id].position
        count = 0
        for object_id in self.target_object_ids:
            pos = self.object_manager.transforms[object_id].position
            if (self.get_2d_distance(pos, place_pos) < 3 and self.belongs_to_which_room(pos) is not None and 'Bedroom' in self.belongs_to_which_room(pos)) or object_id in self.satisfied.keys():
                count += 1
                self.satisfied[object_id] = True
        return count, len(self.target_object_ids), count == len(self.target_object_ids)

    def get_delivered_objects(self):
        """Return evaluator-confirmed deliveries as ``object_id -> name``.

        ``self.satisfied`` is maintained exclusively by :meth:`check_goal`
        from TDW object transforms.  Exposing a copy gives coordination code
        an authoritative delivery signal without leaking target locations or
        other privileged simulator state to either planner.
        """
        return {
            int(object_id): self.object_names[object_id]
            for object_id in self.satisfied
            if object_id in self.object_names
        }

    def get_id_from_mask(self, agent_id, mask, name = None):
        r'''
        Get the object id from the mask
        '''
        seg_with_mask = (self.obs[str(agent_id)]['seg_mask'] * np.expand_dims(mask, axis = -1)).reshape(-1, 3)
        seg_with_mask = [tuple(x) for x in seg_with_mask]
        seg_counter = Counter(seg_with_mask)
        
        for seg in seg_counter:
            if seg == (0, 0, 0): continue
            if seg_counter[seg] / np.sum(mask) > 0.5:
                for i in range(len(self.obs[str(agent_id)]['visible_objects'])):
                    if self.obs[str(agent_id)]['visible_objects'][i]['seg_color'] == seg:
                        return self.obs[str(agent_id)]['visible_objects'][i]
        return {
                    'id': None,
                    'type': None,
                    'seg_color': None,
                    'name': None,
                }

    def _format_held_objects(self, held_objects):
        formatted = []
        for object_id in held_objects:
            if object_id is None:
                formatted.append(self._empty_hand_observation())
            elif self.get_object_type(object_id) == 0:
                formatted.append({
                    'id': object_id,
                    'type': 0,
                    'name': self.object_names[object_id],
                    'contained': [None, None, None],
                    'contained_name': [None, None, None],
                })
            else:
                contained = [x for x in self.containment_all.get(object_id, [])
                             if x not in held_objects and x in self.target_object_ids]
                contained = contained[:3]
                formatted.append({
                    'id': object_id,
                    'type': 1,
                    'name': self.object_names[object_id],
                    'contained': contained + [None] * (3 - len(contained)),
                    'contained_name': [self.object_names[x] for x in contained] +
                                      [None] * (3 - len(contained)),
                })
        return formatted

    def get_obs(self):
        for container_id, object_ids in self.controller.state.containment.items():
            self.containment_all.setdefault(container_id, [])
            for object_id in object_ids:
                if object_id not in self.containment_all[container_id]:
                    self.containment_all[container_id].append(object_id)

        obs = {str(i): {} for i in self._agent_ids()}
        visible_agents = {i: set() for i in self._agent_ids()}
        for agent_id in self._agent_ids():
            key = str(agent_id)
            dynamic = self._agent(agent_id).dynamic
            obs[key]['visible_objects'] = []
            if 'img' not in dynamic.images:
                raise RuntimeError(f"No image received for logical agent {agent_id}")

            rgb_image = dynamic.get_pil_image('img')
            id_image = dynamic.get_pil_image('id')
            obs[key]['rgb'] = np.array(rgb_image).transpose(2, 0, 1)
            obs[key]['seg_mask'] = np.array(id_image)
            colors = Counter(id_image.getdata())
            for object_id, color in self.segmentation_colors.items():
                segmentation_color = tuple(color)
                if segmentation_color in colors:
                    obs[key]['visible_objects'].append({
                        'id': object_id,
                        'type': self.get_object_type(object_id),
                        'seg_color': segmentation_color,
                        'name': self.object_names[object_id],
                    })
            for other_id, color in self.agent_colors.items():
                if color is None:
                    continue
                segmentation_color = tuple(color)
                if segmentation_color in colors:
                    visible_agents[agent_id].add(other_id)
                    obs[key]['visible_objects'].append({
                        'id': other_id,
                        'type': 3,
                        'seg_color': segmentation_color,
                        'name': 'agent',
                    })

            obs[key]['depth'] = np.flip(np.array(TDWUtils.get_depth_values(
                dynamic.get_pil_image('depth'), width=self.screen_size,
                height=self.screen_size)), 0)
            if dynamic.camera_matrix is None:
                raise RuntimeError(f"No camera matrix received for logical agent {agent_id}")
            obs[key]['camera_matrix'] = np.array(dynamic.camera_matrix).reshape((4, 4))
            x, y, z = dynamic.transform.position
            fx, fy, fz = dynamic.transform.forward
            obs[key]['agent'] = [x, y, z, fx, fy, fz]
            obs[key]['held_objects'] = self._format_held_objects(
                self._held_object_ids(agent_id))

            opponent_id = 1 - agent_id if self.number_of_agents == 2 else None
            if opponent_id is not None and opponent_id in visible_agents[agent_id]:
                opponent_held = self._held_object_ids(opponent_id)
            else:
                opponent_held = [None, None]
            obs[key]['oppo_held_objects'] = self._format_held_objects(opponent_held)

            while len(obs[key]['visible_objects']) < 50:
                obs[key]['visible_objects'].append({
                    'id': None, 'type': None, 'seg_color': None, 'name': None})
            obs[key]['visible_objects'] = obs[key]['visible_objects'][:50]
            obs[key]['FOV'] = self.fov
            action_status, action_status_code, obstacle_ids, hit_wall = \
                self._action_diagnostics(self._agent(agent_id).action)
            obs[key]['status'] = self.map_status(
                self._agent_status(agent_id), len(self.action_buffer[agent_id]))
            obs[key]['action_status'] = action_status
            obs[key]['action_status_code'] = action_status_code
            obs[key]['obstacle_ids'] = obstacle_ids
            obs[key]['hit_wall'] = hit_wall
            record = self.action_records[agent_id]
            obs[key]['action_id'] = record['action_id']
            obs[key]['action_type'] = record['action_type']
            obs[key]['action_started_frame'] = record['started_frame']
            obs[key]['action_completed_frame'] = record['completed_frame']
            obs[key]['action_buffer_length'] = len(
                self.action_buffer[agent_id])
            obs[key]['action_terminal'] = record['terminal']
            obs[key]['messages'] = [None for _ in self._agent_ids()]
            obs[key]['valid'] = True
            obs[key]['current_frames'] = self.num_frames
        return obs

    def get_info(self):
        #todo: add info needed
        return {}

    def add_name(self, inst):
        if type(inst) == int and inst in self.object_names:
            return f'{inst}_{self.object_names[inst]}'
        else:
            if type(inst) == dict:
                return {self.add_name(key): self.add_name(value) for key, value in inst.items()}
            elif type(inst) == list:
                return [self.add_name(item) for item in inst]
            else: raise NotImplementedError
    
    def add_name_and_empty(self, inst):
        for x in self.container_ids:
            if x not in inst:
                inst[x] = []
        return self.add_name(inst)

    def _communicate_action_frame(self, frames_this_step):
        """Advance TDW by one frame without crossing the episode budget.

        The environment used to check ``max_frame`` only after an entire
        logical action returned.  A logical action can span several Unity
        frames, so a step that started just below the limit could request one
        frame too many.  If the build exited at the configured limit, the
        controller then waited forever for a reply and Challenge never got a
        chance to persist ``result_episode.json``.

        ``None`` is an internal sentinel meaning that the caller must finish
        the current Gym step without another socket request.
        """
        if frame_budget_exhausted(
                self.num_frames, frames_this_step, self.max_frame):
            return None
        return self.controller.communicate([])

    def step(self, actions):
        '''
        Run one timestep of the environment's dynamics
        '''
        start = time.time()
        valid = [True for _ in range(self.number_of_agents)]
        # Receive actions
        for agent_id in self._agent_ids():
            action = actions[str(agent_id)]
            if action['type'] == 'ongoing': continue
            # otherwise we start an action directly
            self.action_buffer[agent_id] = []
            self.action_epochs[agent_id] += 1
            self.action_records[agent_id] = {
                'action_id': self.action_epochs[agent_id],
                'action_type': str(action['type']),
                'started_frame': self.num_frames,
                'completed_frame': -1,
                'buffer_length': 0,
                'terminal': False,
                'valid': True,
                'status': 'ongoing',
            }
            normalized_action = copy.deepcopy(action)
            if "arm" in normalized_action:
                if normalized_action['arm'] == 'left':
                    normalized_action['arm'] = Arm.left
                elif normalized_action['arm'] == 'right':
                    normalized_action['arm'] = Arm.right
            if action["type"] == 0:       # move forward 0.5m
                self.action_buffer[agent_id].append({**normalized_action, 'type': 'move_forward'})
            elif action["type"] == 1:     # turn left by 15 degree
                self.action_buffer[agent_id].append({**normalized_action, 'type': 'turn_left'})
            elif action["type"] == 2:     # turn right by 15 degree
                self.action_buffer[agent_id].append({**normalized_action, 'type': 'turn_right'})
            elif action["type"] == 3:     # go to and grasp object with arm
                if self._is_replicant(agent_id):
                    self.action_buffer[agent_id].append({**normalized_action, 'type': 'reach_for'})
                    self.action_buffer[agent_id].append({**normalized_action, 'type': 'grasp'})
                else:
                    valid[agent_id] = False
                    self.action_buffer[agent_id].append({'type': 'wait', 'delay': 1})
            elif action["type"] == 4:      # put in container
                if self._is_replicant(agent_id):
                    self.action_buffer[agent_id].append({**normalized_action, 'type': 'put_in'})
                else:
                    valid[agent_id] = False
                    self.action_buffer[agent_id].append({'type': 'wait', 'delay': 1})
            elif action["type"] == 5:      # drop held object in arm
                if self._is_replicant(agent_id):
                    self.action_buffer[agent_id].append({**normalized_action, 'type': 'drop'})
                else:
                    valid[agent_id] = False
                    self.action_buffer[agent_id].append({'type': 'wait', 'delay': 1})
            elif action["type"] == 6:      # send message
                self.action_buffer[agent_id].append({**normalized_action, 'type': 'send_message'})
            elif action["type"] == 8:      # wait for one or more physics frames
                self.action_buffer[agent_id].append({**normalized_action, 'type': 'wait'})
            else:
                raise ValueError(f"Invalid action type for logical agent {agent_id}: {action['type']}")

        # Do action here
        delay_frame_count = [0 for _ in range(self.number_of_agents)]
        finish = False
        num_frames = 0
        while not finish: # continue until any agent's action finishes
            for agent_id in self._agent_ids():
                if delay_frame_count[agent_id] > 0:
                    delay_frame_count[agent_id] -= 1
                    continue
                agent = self._agent(agent_id)
                if self._agent_status(agent_id) != ActionStatus.ongoing and len(self.action_buffer[agent_id]) == 0:
                    finish = True
                elif self._agent_status(agent_id) != ActionStatus.ongoing:
                    curr_action = self.action_buffer[agent_id].pop(0)
                    if curr_action['type'] == 'move_forward':       # move forward 0.5m
                        agent.move_forward()
                    elif curr_action['type'] == 'turn_left':     # turn left by 15 degree
                        agent.turn_by(angle = -15)
                    elif curr_action['type'] == 'turn_right':     # turn right by 15 degree
                        agent.turn_by(angle = 15)
                    elif curr_action['type'] == 'reach_for':     # go to and grasp object with arm
                        distance = self.get_2d_distance(agent.dynamic.transform.position, self.object_manager.transforms[int(curr_action["object"])].position)
                        if distance > self.reach_threshold:
                            valid[agent_id] = False
                            self.action_buffer[agent_id] = [] # the action is invalid
                        else: agent.move_to_position(self.object_manager.transforms[int(curr_action["object"])].position)
                    elif curr_action['type'] == 'grasp':
                        agent.grasp(int(curr_action["object"]), curr_action["arm"], relative_to_hand = False, axis = "yaw")
                    elif curr_action["type"] == 'put_in':      # put in container
                        agent.put_in()
                        held_objects = list(self.controller.state.replicants[agent_id].values())
                        if held_objects[0] is not None and held_objects[1] is not None:
                            container, target = None, None
                            if self.get_object_type(held_objects[0]) == 1:
                                container = held_objects[0]
                            else:
                                target = held_objects[0]
                            if self.get_object_type(held_objects[1]) == 1:
                                container = held_objects[1]
                            else:
                                target = held_objects[1]
                            if container is not None and target is not None:
                                if container in self.containment_all:
                                    if target not in self.containment_all[container]:
                                        self.containment_all[container].append(target)
                                else:
                                    self.containment_all[container] = [target]
                    elif curr_action["type"] == 'drop':      # drop held object in arm
                        agent.drop(curr_action['arm'], max_num_frames = 30)
                    elif curr_action["type"] == 'send_message':      # send message
                        self.messages[agent_id] = copy.deepcopy(curr_action['message'])
                        delay_frame_count[agent_id] = max((len(self.messages[agent_id]) - 1) // self.message_per_frame, 0)
                    elif curr_action["type"] == 'wait':
                        requested_frames = max(int(curr_action.get('delay', 1)), 1)
                        # This loop communicates once below; delay only the remaining frames.
                        delay_frame_count[agent_id] = requested_frames - 1
            if finish: break
            data = self._communicate_action_frame(num_frames)
            if data is None:
                # The frame budget is a hard episode boundary.  Return control
                # to Challenge so it can finalize the coordinator and atomically
                # write the episode result before the TDW process is closed.
                finish = True
                break
            for i in range(len(data) - 1):
                r_id = OutputData.get_data_type_id(data[i])
                if r_id == 'imag':
                    images = Images(data[i])
                    if (os.getenv("TDW_MAT_SAVE_TOPDOWN", "1") == "1"
                            and images.get_avatar_id() == "a"
                            and (self.num_frames + num_frames) % 1 == 0):
                        TDWUtils.save_images(images=images, filename= f"{self.num_frames + num_frames:05d}", output_directory = os.path.join(self.save_dir, 'top_down_image'))
            num_frames += 1

        self.num_frames += num_frames
        self.action_list.append(actions)
        goal_put, goal_total, self.success = self.check_goal()
        reward = 0
        for agent_id in self._agent_ids():
            action = actions[str(agent_id)]
            task_status = self._agent_status(agent_id)
            agent = self._agent(agent_id)
            record = self.action_records[agent_id]
            record['valid'] = valid[agent_id]
            record['status'] = (task_status.name if isinstance(
                task_status, ActionStatus) else str(task_status))
            record['buffer_length'] = len(self.action_buffer[agent_id])
            terminal = (task_status != ActionStatus.ongoing and
                        len(self.action_buffer[agent_id]) == 0)
            record['terminal'] = terminal
            if terminal and record['completed_frame'] < 0:
                record['completed_frame'] = self.num_frames
            self.f.write('step: {}, action: {}, time: {}, status: {}\n'
                    .format(self.num_step, action["type"],
                    time.time() - start,
                    task_status))
            container_info = self.add_name_and_empty(copy.deepcopy(self.controller.state.containment))
            self.f.write('position: {}, forward: {}, containment: {}, goal: {}, container: {}\n'.format(
                    agent.dynamic.transform.position,
                    agent.dynamic.transform.forward,
                    container_info, self.add_name(self.target_object_ids), self.add_name(self.container_ids)))
            self.f.flush()
            if task_status != ActionStatus.success and task_status != ActionStatus.ongoing:
                reward -= 0.1
            if not valid[agent_id]:
                reward -= 0.1
        
        self.num_step += 1        
        self.reward += reward
        done = False
        if self.num_frames >= self.max_frame or self.success:
            done = True
            self.done = True
        
        obs = self.get_obs()
        # add messages to obs
        if self.number_of_agents == 2:
            for agent_id in self._agent_ids():
                obs[str(agent_id)]['messages'] = copy.deepcopy(self.messages)
            self.messages = [None for _ in range(self.number_of_agents)]

        for agent_id in self._agent_ids():
            obs[str(agent_id)]['valid'] = valid[agent_id]
            obs[str(agent_id)]['current_frames'] = self.num_frames

        info = self.get_info()
        info['done'] = done
        info['num_frames_for_step'] = num_frames
        info['num_step'] = self.num_step
        if done:
            info['reward'] = self.reward

        self.obs = obs
        return self.obs_filter(self.obs), reward, done, info
     
    def render(self):
        return None
        
    def save_images(self, save_dir='./Images'):
        '''
        save images of current step, including rgb, depth and segmentation image
        '''
        os.makedirs(save_dir, exist_ok=True)
        for agent_id in self._agent_ids():
            save_path = os.path.join(save_dir, str(agent_id))
            os.makedirs(save_path, exist_ok=True)
            dynamic = self._agent(agent_id).dynamic
            img = dynamic.get_pil_image('img')
            depth = np.flip(np.array(TDWUtils.get_depth_values(dynamic.get_pil_image('depth'), width = self.screen_size, height = self.screen_size), dtype = np.float32), 0)
            depth_img = Image.fromarray(100 / depth).convert('RGB')
            seg = dynamic.get_pil_image('id')
            img.save(os.path.join(save_path, f'{self.num_step:04}_{self.num_frames:04}.png'))
            seg.save(os.path.join(save_path, f'{self.num_step:04}_{self.num_frames:04}_seg.png'))
            depth_img.save(os.path.join(save_path, f'{self.num_step:04}_{self.num_frames:04}_depth.png'))

    def close(self):
        print('close environment ...')
        self._close_controller()
        if getattr(self, "f", None) is not None and not self.f.closed:
            self.f.close()

    def _close_controller(self):
        if self.controller is None:
            return
        try:
            self.controller.communicate({"$type": "terminate"})
        except Exception as exc:
            print(f"Warning: couldn't send TDW terminate command: {exc}")
        try:
            self.controller.socket.close(linger=0)
        except Exception as exc:
            print(f"Warning: couldn't close TDW socket: {exc}")
        self.controller = None
    #    with open(os.path.join(self.save_dir, 'action.pkl'), 'wb') as f:
    #        d = {'scene_info': self.scene_info, 'actions': self.action_list}
    #        pickle.dump(d, f)
