"""A minimal mobile visual scout for the Human + Box TDW-MAT setting.

The :class:`BoxScout` deliberately exposes the small subset of the Replicant
interface that ``tdw_gym.py`` needs (``static``, ``dynamic``, ``action``,
``move_forward()`` and ``turn_by()``), but it is *not* a Replicant.  Its body
is TDW's native ``EmbodiedAvatar`` cube and therefore has a real collider.

Locomotion is intentionally simple and deterministic.  The cube is kinematic
and each linear move alternates between a Box-sized leading-edge overlap check
and a teleport of at most 0.1 m.  A blocked substep therefore leaves the scout
at its last safe pose, while a clear 0.5 m move takes five checks and five
teleports.  Communication belongs to the environment and is not handled in
this module.
"""

from dataclasses import dataclass, field
from io import BytesIO
from itertools import count
from math import ceil
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np
from PIL import Image

from tdw.add_ons.avatar_body import AvatarBody
from tdw.add_ons.embodied_avatar import EmbodiedAvatar
from tdw.object_data.transform import Transform
from tdw.output_data import (AvatarSegmentationColor, CameraMatrices, Images,
                             OutputData, Overlap)
from tdw.replicant.action_status import ActionStatus
from tdw.replicant.collision_detection import CollisionDetection
from tdw.replicant.image_frequency import ImageFrequency
from tdw.tdw_utils import TDWUtils


@dataclass
class BoxScoutAction:
    """Replicant-compatible public state for the current scout action."""

    name: str
    status: ActionStatus = ActionStatus.ongoing
    initialized: bool = True
    done: bool = False
    obstacle_ids: Tuple[int, ...] = field(default_factory=tuple)
    hit_wall: bool = False


@dataclass
class BoxScoutStatic:
    """The static fields used by TDW-MAT observation construction."""

    replicant_id: int
    avatar_id: str
    segmentation_color: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.uint8))
    # These empty dictionaries make accidental read-only Replicant-style
    # introspection harmless.  The scout has no body-part IDs or hands.
    body_parts: Dict = field(default_factory=dict)
    body_parts_by_id: Dict = field(default_factory=dict)
    hands: Dict = field(default_factory=dict)


class BoxScoutDynamic:
    """Latest transform, camera matrices, and image passes for a scout."""

    def __init__(self, transform: Transform, avatar_id: str) -> None:
        self.avatar_id: str = avatar_id
        self.transform: Transform = transform
        self.images: Dict[str, np.ndarray] = {}
        self.projection_matrix: Optional[np.ndarray] = None
        self.camera_matrix: Optional[np.ndarray] = None
        self.got_images: bool = False
        self.held_objects: Dict = {}
        self._image_extensions: Dict[str, str] = {}

    def update(self, resp: List[bytes], transform: Transform) -> None:
        """Update camera data while retaining the last image between captures."""

        self.transform = transform
        self.got_images = False
        for packet in resp[:-1]:
            data_type = OutputData.get_data_type_id(packet)
            if data_type == "imag":
                images = Images(packet)
                if images.get_avatar_id() != self.avatar_id:
                    continue
                self.got_images = True
                for index in range(images.get_num_passes()):
                    pass_mask = images.get_pass_mask(index)
                    image_data = images.get_image(index)
                    if pass_mask == "_depth":
                        image_data = TDWUtils.get_shaped_depth_pass(images=images,
                                                                    index=index)
                    pass_name = pass_mask[1:]
                    self.images[pass_name] = image_data
                    self._image_extensions[pass_name] = images.get_extension(index)
            elif data_type == "cama":
                matrices = CameraMatrices(packet)
                if matrices.get_avatar_id() == self.avatar_id:
                    self.projection_matrix = np.array(matrices.get_projection_matrix())
                    self.camera_matrix = np.array(matrices.get_camera_matrix())

    def get_pil_image(self, pass_mask: str = "img") -> Image.Image:
        """Return a PIL view of ``img``, ``id``, or ``depth`` image data."""

        if pass_mask not in self.images:
            raise KeyError(f"No {pass_mask!r} image has been captured for avatar "
                           f"{self.avatar_id!r}.")
        if pass_mask == "depth":
            return Image.fromarray(self.images[pass_mask])
        return Image.open(BytesIO(self.images[pass_mask].tobytes()))


class BoxScout(EmbodiedAvatar):
    """A cube-collider visual scout with Replicant-like navigation methods.

    ``position`` follows the Replicant convention: its ``y`` value is the
    floor-contact height. TDW's simple-body avatar interprets the root position
    as the bottom-center of its collider. With the default dimensions and
    camera offset, the optical center is approximately 0.40 m above the floor.

    Parameters
    ----------
    avatar_id:
        The TDW avatar/camera ID.  ``"1"`` replaces the second Replicant in the
        two-agent TDW-MAT setup.
    position:
        Ground-anchored initial world position.
    rotation:
        Initial Euler rotation or quaternion accepted by ``EmbodiedAvatar``.
    image_frequency:
        ``once`` captures on initialization and the final frame of successful
        actions; ``always`` captures every frame; ``never`` disables capture.
    """

    DEFAULT_WIDTH: float = 0.35
    DEFAULT_HEIGHT: float = 0.40
    DEFAULT_LENGTH: float = 0.70
    DEFAULT_MOVE_DISTANCE: float = 0.50
    DEFAULT_TURN_ANGLE: float = 15.0
    DEFAULT_PATH_SAMPLE_SPACING: float = 0.10
    DEFAULT_MOVE_FRAMES: int = 10
    DEFAULT_TURN_FRAMES: int = 2

    # Overlap query IDs are response tags only.  A process-wide counter avoids
    # collisions when more than one scout is instantiated.
    _OVERLAP_IDS = count(1_500_000_000)
    _SELF_PROBE_IDS = count(1_400_000_000)

    def __init__(
            self,
            avatar_id: str = "1",
            agent_id: Optional[int] = None,
            position: Optional[Union[Dict[str, float], np.ndarray]] = None,
            rotation: Optional[Union[Dict[str, float], np.ndarray]] = None,
            field_of_view: int = 90,
            color: Optional[Dict[str, float]] = None,
            width: float = DEFAULT_WIDTH,
            height: float = DEFAULT_HEIGHT,
            length: float = DEFAULT_LENGTH,
            move_distance: float = DEFAULT_MOVE_DISTANCE,
            turn_angle: float = DEFAULT_TURN_ANGLE,
            path_sample_spacing: float = DEFAULT_PATH_SAMPLE_SPACING,
            move_frames: int = DEFAULT_MOVE_FRAMES,
            turn_frames: int = DEFAULT_TURN_FRAMES,
            image_frequency: ImageFrequency = ImageFrequency.once,
            image_passes: Optional[List[str]] = None,
            target_framerate: int = 250,
            overlap_timeout_frames: int = 3,
            collision_clearance: float = 0.0) -> None:
        if width <= 0 or height <= 0 or length <= 0:
            raise ValueError("Box dimensions must be positive.")
        if move_distance <= 0 or turn_angle <= 0:
            raise ValueError("Move distance and turn angle must be positive.")
        if path_sample_spacing <= 0:
            raise ValueError("Path sample spacing must be positive.")
        if move_frames < 1 or turn_frames < 1:
            raise ValueError("Action frame counts must be at least one.")
        if overlap_timeout_frames < 1:
            raise ValueError("overlap_timeout_frames must be at least one.")
        if collision_clearance < 0:
            raise ValueError("collision_clearance can't be negative.")
        if not isinstance(image_frequency, ImageFrequency):
            raise TypeError("image_frequency must be an ImageFrequency value.")
        if agent_id is not None:
            if avatar_id != "1" and str(avatar_id) != str(agent_id):
                raise ValueError("avatar_id and agent_id refer to different agents.")
            avatar_id = str(agent_id)
        if image_passes is None:
            image_passes = ["_img", "_id", "_depth"]
        invalid_passes = [value for value in image_passes
                          if value not in Images.PASS_MASKS.values()]
        if invalid_passes:
            raise ValueError(f"Invalid TDW image passes: {invalid_passes}")
        if target_framerate < 1:
            raise ValueError("target_framerate must be positive.")

        self.width: float = float(width)
        self.height: float = float(height)
        self.length: float = float(length)
        self.move_distance: float = float(move_distance)
        self.turn_angle: float = float(turn_angle)
        self.path_sample_spacing: float = float(path_sample_spacing)
        self.move_frames: int = int(move_frames)
        self.turn_frames: int = int(turn_frames)
        self.image_frequency: ImageFrequency = image_frequency
        self.image_passes: List[str] = list(image_passes)
        self.target_framerate: int = int(target_framerate)
        self.overlap_timeout_frames: int = int(overlap_timeout_frames)
        self.collision_clearance: float = float(collision_clearance)

        ground_position = self._as_position(position)
        initial_rotation = self._as_rotation(rotation)
        self._body_initial_rotation: Dict[str, float] = initial_rotation
        if color is None:
            color = {"r": 0.08, "g": 0.30, "b": 0.80, "a": 1.0}

        super().__init__(avatar_id=str(avatar_id),
                         position=ground_position,
                         # ThirdPersonCameraBase interprets ``rotation`` as a
                         # sensor-local rotation.  Rotate the cube body below
                         # instead, so body.forward, collider, and camera agree.
                         rotation=None,
                         field_of_view=field_of_view,
                         color=color,
                         body=AvatarBody.cube,
                         scale_factor={"x": self.width,
                                       "y": self.height,
                                       "z": self.length},
                         mass=20,
                         dynamic_friction=0.3,
                         static_friction=0.3,
                         bounciness=0.0,
                         drag=10,
                         angular_drag=10)

        self.replicant_id: int = int(avatar_id) if str(avatar_id).isdigit() else -1
        self.static: BoxScoutStatic = BoxScoutStatic(
            replicant_id=self.replicant_id, avatar_id=self.avatar_id)
        self.dynamic: BoxScoutDynamic = BoxScoutDynamic(
            transform=self.transform, avatar_id=self.avatar_id)
        self.action: BoxScoutAction = BoxScoutAction(
            name="idle", status=ActionStatus.success, done=True)
        self.collision_detection: CollisionDetection = CollisionDetection(
            previous_was_same=False)
        self.obstacle_ids: Set[int] = set()
        # TDW 1.11.23 reports an A_Simple_Body avatar's visible primitive as a
        # normal object in send_overlap_box output.  Its numeric object ID is
        # assigned by the build and can change when other avatars are created,
        # so discover it per instance instead of hard-coding an ID.
        self.self_object_id: Optional[int] = None
        self._self_probe_query_id: Optional[int] = None
        self._self_probe_command: Optional[dict] = None
        self._self_probe_wait_frames: int = 0

        self._phase: str = "idle"
        self._move_start: Optional[np.ndarray] = None
        self._move_target: Optional[np.ndarray] = None
        self._move_safe_position: Optional[np.ndarray] = None
        self._pending_move_position: Optional[np.ndarray] = None
        self._move_step: int = 0
        self._move_steps: int = 0
        self._turn_step_angle: float = 0.0
        self._turn_steps_remaining: int = 0
        self._overlap_commands: Dict[int, dict] = {}
        self._overlap_results: Set[int] = set()
        self._overlap_wait_frames: int = 0

    def get_initialization_commands(self) -> List[dict]:
        """Create the cube, its collider, and its visual observation stream."""

        commands = super().get_initialization_commands()
        commands.extend([
            {"$type": "set_pass_masks",
             "pass_masks": self.image_passes,
             "avatar_id": self.avatar_id},
            # Deterministic discrete locomotion uses non-physics substeps;
            # the kinematic body retains its collider but can't push task items.
            {"$type": "set_avatar_kinematic_state",
             "is_kinematic": True,
             "use_gravity": False,
             "avatar_id": self.avatar_id},
            {"$type": "set_avatar_collision_detection_mode",
             "mode": "continuous_speculative",
             "avatar_id": self.avatar_id},
            self._get_initial_body_rotation_command(),
            {"$type": "translate_sensor_container_by",
             # The sensor's default local y is 0.5.  Avatar scaling also
             # scales this local offset, so another +0.5 places the camera at
             # one full body height (0.40 m for the default Box).
             "move_by": {"x": 0.0, "y": 0.5, "z": 0.0},
             "avatar_id": self.avatar_id},
            {"$type": "send_avatar_segmentation_colors",
             "ids": [self.avatar_id],
             "frequency": "once"},
        ])
        if self.image_frequency == ImageFrequency.always:
            commands.extend(self._capture_commands(frequency="always"))
        elif self.image_frequency == ImageFrequency.once:
            commands.extend(self._capture_commands(frequency="once"))
        return commands

    def on_send(self, resp: List[bytes]) -> None:
        """Cache TDW output and advance the asynchronous action state machine."""

        super().on_send(resp=resp)
        self.dynamic.update(resp=resp, transform=self.transform)
        self._cache_segmentation_color(resp=resp)

        self._consume_self_overlap_probe(resp=resp)
        if (self.initialized and self.self_object_id is None and
                self._self_probe_query_id is None):
            self._queue_self_overlap_probe()

        if self._phase == "checking_move":
            self._consume_overlap_results(resp=resp)
        elif self._phase == "moving":
            self._complete_move_step()
            if self._move_step >= self._move_steps:
                self._finish_action(ActionStatus.success)
            elif self.collision_detection.avoid:
                self._phase = "checking_move"
                self._queue_next_move_overlap_check()
            else:
                self._queue_next_move_step()
        elif self._phase == "turning":
            self._turn_steps_remaining -= 1
            if self._turn_steps_remaining <= 0:
                self._finish_action(ActionStatus.success)
            else:
                self._queue_turn_step()

        self.dynamic.transform = self.transform
        self.is_moving = self.action.status == ActionStatus.ongoing

    def move_forward(self, distance: float = DEFAULT_MOVE_DISTANCE) -> None:
        """Move forward through collision-checked substeps."""

        self.move_by(distance=abs(float(distance)))

    def move_backward(self, distance: float = DEFAULT_MOVE_DISTANCE) -> None:
        """Move backward through collision-checked substeps."""

        self.move_by(distance=-abs(float(distance)))

    def move_by(self, distance: float) -> None:
        """Begin an asynchronous, collision-checked linear movement."""

        self._require_idle()
        distance = float(distance)
        if abs(distance) < 1e-8:
            self.action = BoxScoutAction(name="move_by",
                                         status=ActionStatus.success,
                                         done=True)
            return
        forward = np.array(self.transform.forward, dtype=np.float64)
        forward[1] = 0.0
        norm = float(np.linalg.norm(forward))
        if norm < 1e-8:
            self.action = BoxScoutAction(name="move_by",
                                         status=ActionStatus.failed_to_move,
                                         done=True)
            return
        forward /= norm
        self._move_start = np.array(self.transform.position, dtype=np.float64)
        self._move_target = self._move_start + forward * distance
        self._move_safe_position = np.array(self._move_start, copy=True)
        self._pending_move_position = None
        # Collision-checked movement spends two communicate() calls per
        # substep: one overlap response and one teleport response.  Preserve
        # the configured action duration as a lower bound while never making a
        # checked substep longer than path_sample_spacing.
        duration_divisor = 2 if self.collision_detection.avoid else 1
        duration_steps = int(ceil(self.move_frames * abs(distance) /
                                  self.move_distance / duration_divisor))
        spacing_steps = (int(ceil(abs(distance) /
                                  self.path_sample_spacing))
                         if self.collision_detection.avoid else 1)
        self._move_steps = max(1, duration_steps, spacing_steps)
        self._move_step = 0
        self.action = BoxScoutAction(name="move_by")

        if not self.collision_detection.avoid:
            self._queue_next_move_step()
            return
        if self.self_object_id is None:
            raise RuntimeError("Box collider self-ID hasn't been calibrated yet.")
        self._phase = "checking_move"
        self._queue_next_move_overlap_check()

    def turn_by(self, angle: float) -> None:
        """Turn in place over multiple frames (positive is clockwise)."""

        self._require_idle()
        angle = float(angle)
        if abs(angle) < 1e-8:
            self.action = BoxScoutAction(name="turn_by",
                                         status=ActionStatus.success,
                                         done=True)
            return
        num_steps = max(1, int(ceil(self.turn_frames *
                                    abs(angle) / self.turn_angle)))
        self._turn_step_angle = angle / num_steps
        self._turn_steps_remaining = num_steps
        self.action = BoxScoutAction(name="turn_by")
        self._phase = "turning"
        self._queue_turn_step()

    def set_obstacle_ids(self, object_ids: Iterable[int]) -> None:
        """Record task items that must never be pushed or crossed.

        The overlap policy already treats every detected non-excluded object as
        an obstacle (including furniture).  This explicit set is retained for
        integration diagnostics and guarantees that task targets/containers
        aren't accidentally added to ``exclude_objects`` later.
        """

        self.obstacle_ids = {int(object_id) for object_id in object_ids}
        if self.obstacle_ids:
            self.collision_detection.exclude_objects = [
                object_id for object_id in
                self.collision_detection.exclude_objects
                if object_id not in self.obstacle_ids]

    def reset(self,
              position: Optional[Union[Dict[str, float], np.ndarray]] = None,
              rotation: Optional[Union[Dict[str, float], np.ndarray]] = None) -> None:
        """Reset Python-side state so this add-on can initialize in a new scene."""

        ground_position = self._as_position(position)
        self.position = ground_position
        self._rotation = None
        self._body_initial_rotation = self._as_rotation(rotation)
        self.initialized = False
        self.commands.clear()
        self.transform.position = np.zeros(3)
        self.transform.rotation = np.zeros(4)
        self.transform.forward = np.zeros(3)
        self.dynamic = BoxScoutDynamic(transform=self.transform,
                                       avatar_id=self.avatar_id)
        self.static.segmentation_color = np.zeros(3, dtype=np.uint8)
        self.action = BoxScoutAction(name="idle",
                                     status=ActionStatus.success,
                                     done=True)
        self.self_object_id = None
        self._self_probe_query_id = None
        self._self_probe_command = None
        self._self_probe_wait_frames = 0
        self._clear_action_state()

    @property
    def half_extents(self) -> Dict[str, float]:
        """Oriented overlap-box half extents, including optional clearance."""

        return {"x": self.width / 2.0 + self.collision_clearance,
                "y": self.height / 2.0,
                "z": self.length / 2.0 + self.collision_clearance}

    def _queue_next_move_overlap_check(self) -> None:
        """Check only the volume newly occupied by the next safe substep."""

        assert self._move_safe_position is not None
        candidate = self._get_next_move_position()
        step_vector = candidate - self._move_safe_position
        step_distance = float(np.linalg.norm(step_vector))
        if step_distance < 1e-8:
            self._finish_action(ActionStatus.failed_to_move)
            return
        direction = step_vector / step_distance

        # The current pose is already known to be safe.  For a straight move,
        # the only newly occupied volume is the thin slab extending from the
        # Box's leading face through the next substep.  This matches the
        # Replicant's per-frame gate while using the Box's true footprint.
        center = (self._move_safe_position +
                  direction * (self.length / 2.0 + step_distance / 2.0))
        # send_overlap_box expects the volume center whereas the simple avatar
        # transform is reported at the collider's bottom-center.
        center[1] += self.height / 2.0
        half_extents = {
            "x": self.width / 2.0 + self.collision_clearance,
            "y": self.height / 2.0,
            "z": step_distance / 2.0 + self.collision_clearance,
        }

        self._overlap_commands.clear()
        self._overlap_results.clear()
        self._overlap_wait_frames = 0
        self._pending_move_position = candidate
        query_id = next(self._OVERLAP_IDS)
        command = {"$type": "send_overlap_box",
                   "id": query_id,
                   "half_extents": half_extents,
                   "rotation": TDWUtils.array_to_vector4(
                       np.array(self.transform.rotation)),
                   "position": TDWUtils.array_to_vector3(center)}
        self._overlap_commands[query_id] = command
        self.commands.append(command)

    def _consume_overlap_results(self, resp: List[bytes]) -> None:
        obstacle_ids: Set[int] = set()
        hit_wall = False
        for packet in resp[:-1]:
            if OutputData.get_data_type_id(packet) != "over":
                continue
            overlap = Overlap(packet)
            query_id = overlap.get_id()
            if query_id not in self._overlap_commands:
                continue
            self._overlap_results.add(query_id)
            if overlap.get_env() and overlap.get_walls():
                hit_wall = True
            if self.collision_detection.objects:
                obstacle_ids.update(
                    int(object_id) for object_id in overlap.get_object_ids()
                    if self._is_blocking_object_id(int(object_id)))

        if hit_wall or obstacle_ids:
            self.action.obstacle_ids = tuple(sorted(obstacle_ids))
            self.action.hit_wall = hit_wall
            self._finish_action(ActionStatus.detected_obstacle)
            return

        missing = set(self._overlap_commands).difference(self._overlap_results)
        if not missing:
            self._begin_move()
            return

        self._overlap_wait_frames += 1
        if self._overlap_wait_frames >= self.overlap_timeout_frames:
            self._finish_action(ActionStatus.failure)
            return
        # A missing response is unusual, but retrying only the missing requests
        # avoids silently treating an unchecked segment as free space.
        self.commands.extend(self._overlap_commands[query_id]
                             for query_id in sorted(missing))

    def _queue_self_overlap_probe(self) -> None:
        """Request the build-assigned object ID of this avatar's own collider."""

        center = np.array(self.transform.position, dtype=np.float64)
        center[1] += self.height / 2.0
        query_id = next(self._SELF_PROBE_IDS)
        command = {"$type": "send_overlap_box",
                   "id": query_id,
                   # This volume is wholly inside the Box.  More than one hit
                   # means the spawn itself is interpenetrating an obstacle.
                   "half_extents": {"x": 0.01, "y": 0.01, "z": 0.01},
                   "rotation": {"x": 0.0, "y": 0.0,
                                "z": 0.0, "w": 1.0},
                   "position": TDWUtils.array_to_vector3(center)}
        self._self_probe_query_id = query_id
        self._self_probe_command = command
        self._self_probe_wait_frames = 0
        self.commands.append(command)

    def _consume_self_overlap_probe(self, resp: List[bytes]) -> None:
        """Resolve the exact collider ID returned for this SimpleBodyAvatar."""

        if self._self_probe_query_id is None:
            return
        for packet in resp[:-1]:
            if OutputData.get_data_type_id(packet) != "over":
                continue
            overlap = Overlap(packet)
            if overlap.get_id() != self._self_probe_query_id:
                continue
            object_ids = sorted({int(value)
                                 for value in overlap.get_object_ids()})
            if overlap.get_env() or overlap.get_walls():
                raise RuntimeError(
                    "Box spawn intersects the TDW environment during self-ID calibration.")
            if len(object_ids) != 1:
                raise RuntimeError(
                    "Box self-ID calibration expected exactly one collider; "
                    f"received {object_ids}.")
            self.self_object_id = object_ids[0]
            self._self_probe_query_id = None
            self._self_probe_command = None
            self._self_probe_wait_frames = 0
            return

        self._self_probe_wait_frames += 1
        if self._self_probe_wait_frames >= self.overlap_timeout_frames:
            raise RuntimeError("TDW didn't return the Box self-ID overlap probe.")
        if self._self_probe_command is not None:
            self.commands.append(dict(self._self_probe_command))

    def _is_blocking_object_id(self, object_id: int) -> bool:
        """Return True for real obstacles, excluding only this Box collider."""

        return (object_id != self.self_object_id and
                object_id not in self.collision_detection.exclude_objects)

    def _begin_move(self) -> None:
        """Queue the already-checked substep without advancing safe state."""

        assert self._pending_move_position is not None
        self._phase = "moving"
        self._overlap_commands.clear()
        self._overlap_results.clear()
        self._overlap_wait_frames = 0
        self._queue_pending_move_step()

    def _queue_next_move_step(self) -> None:
        """Queue the next unchecked substep when avoidance is disabled."""

        self._pending_move_position = self._get_next_move_position()
        self._phase = "moving"
        self._queue_pending_move_step()

    def _queue_pending_move_step(self) -> None:
        assert self._pending_move_position is not None
        self.commands.append({
            "$type": "teleport_avatar_to",
            "position": TDWUtils.array_to_vector3(
                self._pending_move_position),
            "avatar_id": self.avatar_id,
        })
        if self._move_step + 1 == self._move_steps:
            self.commands.extend(self._final_frame_capture_commands())

    def _complete_move_step(self) -> None:
        """Commit a completed teleport as the new last-safe pose."""

        assert self._pending_move_position is not None
        self._move_safe_position = np.array(
            self._pending_move_position, copy=True)
        self._pending_move_position = None
        self._move_step += 1

    def _get_next_move_position(self) -> np.ndarray:
        assert self._move_start is not None and self._move_target is not None
        if self._move_step >= self._move_steps:
            return np.array(self._move_target, copy=True)
        alpha = (self._move_step + 1) / self._move_steps
        return self._move_start + ((self._move_target - self._move_start) * alpha)

    def _queue_turn_step(self) -> None:
        self.commands.append({"$type": "rotate_avatar_by",
                              "angle": self._turn_step_angle,
                              "axis": "yaw",
                              "is_world": True,
                              "avatar_id": self.avatar_id})
        if self._turn_steps_remaining == 1:
            self.commands.extend(self._final_frame_capture_commands())

    def _finish_action(self, status: ActionStatus) -> None:
        self.action.status = status
        self.action.done = True
        self._clear_action_state()

    def _clear_action_state(self) -> None:
        self._phase = "idle"
        self._move_start = None
        self._move_target = None
        self._move_safe_position = None
        self._pending_move_position = None
        self._move_step = 0
        self._move_steps = 0
        self._turn_step_angle = 0.0
        self._turn_steps_remaining = 0
        self._overlap_commands.clear()
        self._overlap_results.clear()
        self._overlap_wait_frames = 0

    def _require_idle(self) -> None:
        if self.action.status == ActionStatus.ongoing:
            raise RuntimeError(f"BoxScout {self.avatar_id!r} is already executing "
                               f"{self.action.name!r}.")

    def _cache_segmentation_color(self, resp: List[bytes]) -> None:
        for packet in resp[:-1]:
            # TDW 1.11.x emits avsc for AvatarSegmentationColor.  Accept the
            # alternate spelling used by some builds as well.
            if OutputData.get_data_type_id(packet) not in {"avsc", "avsg"}:
                continue
            segmentation = AvatarSegmentationColor(packet)
            if segmentation.get_id() == self.avatar_id:
                self.static.segmentation_color = np.array(
                    segmentation.get_segmentation_color(), dtype=np.uint8)

    def _capture_commands(self, frequency: str) -> List[dict]:
        return [{"$type": "send_images",
                 "ids": [self.avatar_id],
                 "frequency": frequency},
                {"$type": "send_camera_matrices",
                 "ids": [self.avatar_id],
                 "frequency": frequency}]

    def _final_frame_capture_commands(self) -> List[dict]:
        if self.image_frequency == ImageFrequency.once:
            return self._capture_commands(frequency="once")
        return []

    def _get_initial_body_rotation_command(self) -> dict:
        if "w" in self._body_initial_rotation:
            return {"$type": "rotate_avatar_to",
                    "rotation": self._body_initial_rotation,
                    "avatar_id": self.avatar_id}
        return {"$type": "rotate_avatar_to_euler_angles",
                "euler_angles": self._body_initial_rotation,
                "avatar_id": self.avatar_id}

    @staticmethod
    def _as_position(
            value: Optional[Union[Dict[str, float], np.ndarray]]) -> Dict[str, float]:
        if value is None:
            return {"x": 0.0, "y": 0.0, "z": 0.0}
        if isinstance(value, np.ndarray):
            if value.shape != (3,):
                raise ValueError(f"Expected a 3-vector position, got {value.shape}.")
            return TDWUtils.array_to_vector3(value.astype(float))
        if isinstance(value, dict) and {"x", "y", "z"}.issubset(value):
            return {axis: float(value[axis]) for axis in ("x", "y", "z")}
        raise TypeError(f"Invalid BoxScout position: {value!r}")

    @staticmethod
    def _as_rotation(
            value: Optional[Union[Dict[str, float], np.ndarray]]) -> Dict[str, float]:
        if value is None:
            return {"x": 0.0, "y": 0.0, "z": 0.0}
        if isinstance(value, np.ndarray):
            if value.shape not in {(3,), (4,)}:
                raise ValueError(f"Expected a 3- or 4-vector rotation, got "
                                 f"{value.shape}.")
            keys: Iterable[str] = ("x", "y", "z") if value.shape == (3,) \
                else ("x", "y", "z", "w")
            return {key: float(component)
                    for key, component in zip(keys, value)}
        if isinstance(value, dict) and {"x", "y", "z"}.issubset(value):
            keys = ("x", "y", "z", "w") if "w" in value else ("x", "y", "z")
            return {key: float(value[key]) for key in keys}
        raise TypeError(f"Invalid BoxScout rotation: {value!r}")


# ``TransportChallenge`` used this descriptive name while the implementation
# was being integrated.  Keep it as a stable public alias.
BoxScoutAvatar = BoxScout
