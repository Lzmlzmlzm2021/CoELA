import unittest
from unittest.mock import patch

import numpy as np

from tdw.replicant.action_status import ActionStatus
from tdw.replicant.image_frequency import ImageFrequency

import transport_challenge_multi_agent.box_scout as box_scout_module
from transport_challenge_multi_agent.box_scout import BoxScoutAvatar


class _FakeOverlap:
    def __init__(self, query_id, object_ids=(), env=False, walls=False):
        self._query_id = query_id
        self._object_ids = list(object_ids)
        self._env = env
        self._walls = walls

    def get_id(self):
        return self._query_id

    def get_object_ids(self):
        return self._object_ids

    def get_env(self):
        return self._env

    def get_walls(self):
        return self._walls


class BoxScoutTests(unittest.TestCase):
    def _ready_scout(self, **kwargs) -> BoxScoutAvatar:
        scout = BoxScoutAvatar(agent_id=1, **kwargs)
        scout.transform.position = np.array([1.0, 0.0, 2.0], dtype=float)
        scout.transform.rotation = np.array([0.0, 0.0, 0.0, 1.0],
                                            dtype=float)
        scout.transform.forward = np.array([0.0, 0.0, 1.0], dtype=float)
        scout.self_object_id = 49_744_125
        return scout

    @staticmethod
    def _commands_of_type(scout, command_type):
        return [command for command in scout.commands
                if command["$type"] == command_type]

    def _respond_to_current_probe(self, scout, object_ids=(), env=False,
                                  walls=False):
        self.assertEqual(len(scout._overlap_commands), 1)
        query_id = next(iter(scout._overlap_commands))
        packet = _FakeOverlap(query_id=query_id,
                              object_ids=object_ids,
                              env=env,
                              walls=walls)
        # Simulate the controller draining the queued probe before on_send().
        scout.commands.clear()
        with patch.object(box_scout_module.OutputData,
                          "get_data_type_id", return_value="over"), \
                patch.object(box_scout_module, "Overlap",
                             side_effect=lambda value: value):
            scout.on_send([packet, b""])

    def _apply_queued_teleport(self, scout):
        teleports = self._commands_of_type(scout, "teleport_avatar_to")
        self.assertEqual(len(teleports), 1)
        position = teleports[0]["position"]
        scout.transform.position = np.array(
            [position[axis] for axis in ("x", "y", "z")], dtype=float)
        # Simulate the controller draining the teleport before on_send().
        scout.commands.clear()
        scout.on_send([b""])

    def test_initializes_native_cube_and_camera(self) -> None:
        scout = BoxScoutAvatar(agent_id=1,
                               position={"x": 1, "y": 0, "z": 2})
        commands = scout.get_initialization_commands()
        by_type = {}
        for command in commands:
            by_type.setdefault(command["$type"], []).append(command)

        self.assertEqual(by_type["change_avatar_body"][0]["body_type"], "Cube")
        self.assertEqual(by_type["scale_avatar"][0]["scale_factor"],
                         {"x": 0.35, "y": 0.4, "z": 0.7})
        self.assertEqual(by_type["set_pass_masks"][-1]["pass_masks"],
                         ["_img", "_id", "_depth"])
        self.assertEqual(by_type["teleport_avatar_to"][0]["position"]["y"],
                         0.0)
        self.assertEqual(by_type["translate_sensor_container_by"][0]
                         ["move_by"]["y"], 0.5)
        self.assertTrue(by_type["set_avatar_kinematic_state"][0]
                        ["is_kinematic"])

    def test_move_starts_with_one_box_specific_thin_gate(self) -> None:
        scout = self._ready_scout()
        scout.move_forward()
        overlap_commands = self._commands_of_type(scout, "send_overlap_box")

        self.assertEqual(scout.action.status, ActionStatus.ongoing)
        self.assertEqual(len(overlap_commands), 1)
        # Current front face is z=2.35; the first 0.1 m gate spans
        # z=2.35..2.45 and is centered at z=2.40.
        self.assertAlmostEqual(overlap_commands[0]["position"]["z"], 2.4)
        self.assertAlmostEqual(
            overlap_commands[0]["half_extents"]["x"], 0.175)
        self.assertAlmostEqual(
            overlap_commands[0]["half_extents"]["y"], 0.2)
        self.assertAlmostEqual(
            overlap_commands[0]["half_extents"]["z"], 0.05)
        self.assertEqual(overlap_commands[0]["position"]["y"], 0.2)

    def test_clear_half_meter_move_alternates_five_gates_and_teleports(self):
        scout = self._ready_scout(image_frequency=ImageFrequency.always)
        scout.move_forward()

        gate_centers = []
        teleport_targets = []
        for step in range(5):
            gates = self._commands_of_type(scout, "send_overlap_box")
            self.assertEqual(len(gates), 1)
            self.assertEqual(self._commands_of_type(
                scout, "teleport_avatar_to"), [])
            gate_centers.append(gates[0]["position"]["z"])

            self._respond_to_current_probe(scout)
            teleports = self._commands_of_type(scout, "teleport_avatar_to")
            self.assertEqual(len(teleports), 1)
            self.assertEqual(self._commands_of_type(
                scout, "send_overlap_box"), [])
            teleport_targets.append(teleports[0]["position"]["z"])
            self.assertEqual(scout.action.status, ActionStatus.ongoing)

            self._apply_queued_teleport(scout)
            if step < 4:
                self.assertEqual(scout.action.status, ActionStatus.ongoing)

        np.testing.assert_allclose(gate_centers,
                                   [2.4, 2.5, 2.6, 2.7, 2.8])
        np.testing.assert_allclose(teleport_targets,
                                   [2.1, 2.2, 2.3, 2.4, 2.5])
        self.assertEqual(scout.action.status, ActionStatus.success)
        self.assertTrue(scout.action.done)
        self.assertAlmostEqual(scout.transform.position[2], 2.5)

    def test_object_hit_preserves_last_safe_pose_and_metadata(self):
        scout = self._ready_scout()
        scout.move_forward()

        for _ in range(2):
            self._respond_to_current_probe(scout)
            self._apply_queued_teleport(scout)
        self.assertAlmostEqual(scout.transform.position[2], 2.2)

        self._respond_to_current_probe(scout, object_ids=[831, 702])

        self.assertEqual(scout.action.status,
                         ActionStatus.detected_obstacle)
        self.assertTrue(scout.action.done)
        self.assertEqual(scout.action.obstacle_ids, (702, 831))
        self.assertFalse(scout.action.hit_wall)
        self.assertAlmostEqual(scout.transform.position[2], 2.2)
        self.assertEqual(self._commands_of_type(
            scout, "teleport_avatar_to"), [])

    def test_failed_move_can_be_retried_like_published_replicant_navigation(self):
        scout = self._ready_scout()
        scout.move_forward()
        self._respond_to_current_probe(scout, object_ids=[831])
        self.assertEqual(scout.action.status,
                         ActionStatus.detected_obstacle)
        failed_position = scout.transform.position.copy()

        # The planner has no added blocked-corridor/forced-turn overlay. If the
        # published AgentMemory selects type 0 again, the Box must retry the
        # same physical primitive instead of silently changing the action.
        scout.move_forward()
        self.assertEqual(scout.action.status, ActionStatus.ongoing)
        self.assertEqual(len(self._commands_of_type(
            scout, "send_overlap_box")), 1)
        self.assertEqual(self._commands_of_type(
            scout, "teleport_avatar_to"), [])
        np.testing.assert_array_equal(scout.transform.position,
                                      failed_position)

    def test_thin_wall_hit_stops_before_first_teleport(self):
        scout = self._ready_scout()
        scout.move_forward()

        self._respond_to_current_probe(scout, env=True, walls=True)

        self.assertEqual(scout.action.status,
                         ActionStatus.detected_obstacle)
        self.assertTrue(scout.action.hit_wall)
        self.assertEqual(scout.action.obstacle_ids, ())
        np.testing.assert_allclose(scout.transform.position, [1.0, 0.0, 2.0])
        self.assertEqual(self._commands_of_type(
            scout, "teleport_avatar_to"), [])

    def test_self_id_in_gate_does_not_block_substep(self):
        scout = self._ready_scout()
        scout.move_forward()

        self._respond_to_current_probe(
            scout, object_ids=[scout.self_object_id])

        self.assertEqual(scout.action.status, ActionStatus.ongoing)
        self.assertEqual(len(self._commands_of_type(
            scout, "teleport_avatar_to")), 1)

    def test_backward_move_uses_rear_gate_and_reaches_half_meter(self):
        scout = self._ready_scout()
        scout.move_backward()

        first_gate = self._commands_of_type(scout, "send_overlap_box")[0]
        # Current rear face is z=1.65; the first backward gate spans
        # z=1.55..1.65 and is centered at z=1.60.
        self.assertAlmostEqual(first_gate["position"]["z"], 1.6)
        self.assertAlmostEqual(first_gate["half_extents"]["z"], 0.05)

        for _ in range(5):
            self._respond_to_current_probe(scout)
            self._apply_queued_teleport(scout)

        self.assertEqual(scout.action.status, ActionStatus.success)
        self.assertAlmostEqual(scout.transform.position[2], 1.5)

    def test_turn_has_nonzero_multiframe_cost(self) -> None:
        scout = self._ready_scout(turn_frames=2)
        scout.turn_by(15)
        self.assertEqual(scout.action.status, ActionStatus.ongoing)

        # Simulate the response to each queued turn command.  A single trailing
        # packet is TDW's normal response terminator, so no parser is invoked.
        scout.commands.clear()
        scout.on_send([b""])
        self.assertEqual(scout.action.status, ActionStatus.ongoing)
        self.assertEqual(len([command for command in scout.commands
                              if command["$type"] == "rotate_avatar_by"]), 1)
        scout.commands.clear()
        scout.on_send([b""])
        self.assertEqual(scout.action.status, ActionStatus.success)

    def test_once_images_are_requested_on_final_motion_frame(self) -> None:
        scout = self._ready_scout(image_frequency=ImageFrequency.once)
        scout.move_forward()
        for step in range(5):
            self._respond_to_current_probe(scout)
            command_types = [command["$type"] for command in scout.commands]
            self.assertIn("teleport_avatar_to", command_types)
            if step < 4:
                self.assertNotIn("send_images", command_types)
                self.assertNotIn("send_camera_matrices", command_types)
            else:
                self.assertIn("send_images", command_types)
                self.assertIn("send_camera_matrices", command_types)
            self._apply_queued_teleport(scout)

    def test_always_images_remain_enabled_for_every_frame(self) -> None:
        scout = BoxScoutAvatar(agent_id=1,
                               image_frequency=ImageFrequency.always)
        commands = scout.get_initialization_commands()
        image_commands = [command for command in commands
                          if command["$type"] in {
                              "send_images", "send_camera_matrices"}]

        self.assertEqual(len(image_commands), 2)
        self.assertTrue(all(command["frequency"] == "always"
                            for command in image_commands))

    def test_task_items_cannot_be_excluded_from_collision_checks(self) -> None:
        scout = self._ready_scout()
        scout.collision_detection.exclude_objects.extend([10, 11, 12])
        scout.set_obstacle_ids([10, 11])
        self.assertEqual(scout.obstacle_ids, {10, 11})
        self.assertEqual(scout.collision_detection.exclude_objects, [12])

    def test_self_probe_uses_a_tiny_volume_inside_the_box(self) -> None:
        scout = self._ready_scout()
        scout.self_object_id = None
        scout.initialized = True
        scout.on_send([b""])
        probes = [command for command in scout.commands
                  if command["$type"] == "send_overlap_box"]
        self.assertEqual(len(probes), 1)
        self.assertEqual(probes[0]["half_extents"],
                         {"x": 0.01, "y": 0.01, "z": 0.01})
        self.assertEqual(probes[0]["position"],
                         {"x": 1.0, "y": 0.2, "z": 2.0})

    def test_collision_filter_excludes_only_the_calibrated_self_id(self) -> None:
        scout = self._ready_scout()
        scout.self_object_id = 49_744_125
        scout.collision_detection.exclude_objects.append(123)
        self.assertFalse(scout._is_blocking_object_id(49_744_125))
        self.assertFalse(scout._is_blocking_object_id(123))
        self.assertTrue(scout._is_blocking_object_id(1791249))


if __name__ == "__main__":
    unittest.main()
