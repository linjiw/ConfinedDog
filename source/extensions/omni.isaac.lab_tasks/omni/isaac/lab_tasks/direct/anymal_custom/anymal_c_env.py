# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations


import torch


import omni.isaac.lab.envs.mdp as mdp
import omni.isaac.lab.sim as sim_utils
# import prims
# import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensor, ContactSensorCfg, RayCaster, RayCasterCfg, patterns
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass

import omni.isaac.core.utils.prims as prim_utils

##
# Pre-defined configs
##
from omni.isaac.lab_assets.anymal import ANYMAL_C_CFG  # isort: skip
from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


from omni.isaac.lab.envs import ViewerCfg
from typing import Literal

import numpy as np




@configclass
class EventCfg:
   """Configuration for randomization."""


   physics_material = EventTerm(
	   func=mdp.randomize_rigid_body_material,
	   mode="startup",
	   params={
		   "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
		   "static_friction_range": (0.8, 0.8),
		   "dynamic_friction_range": (0.6, 0.6),
		   "restitution_range": (0.0, 0.0),
		   "num_buckets": 64,
	   },
   )


   add_base_mass = EventTerm(
	   func=mdp.randomize_rigid_body_mass,
	   mode="startup",
	   params={
		   "asset_cfg": SceneEntityCfg("robot", body_names="base"),
		   "mass_distribution_params": (-5.0, 5.0),
		   "operation": "add",
	   },
   )




@configclass
class AnymalCFlatEnvCfg(DirectRLEnvCfg):
   # env
   episode_length_s = 20
   decimation = 4
   action_scale = 0.5
   num_actions = 12
   num_observations = 48
   num_states = 0


   # simulation
   sim: SimulationCfg = SimulationCfg(
	   dt=1 / 200,
	   render_interval=decimation,
	   disable_contact_processing=True,
	   physics_material=sim_utils.RigidBodyMaterialCfg(
		   friction_combine_mode="multiply",
		   restitution_combine_mode="multiply",
		   static_friction=1.0,
		   dynamic_friction=1.0,
		   restitution=0.0,
	   ),
   )
#    terrain = TerrainImporterCfg(
#        prim_path="/World/ground",
#        terrain_type="plane",
#        collision_group=-1,
#        physics_material=sim_utils.RigidBodyMaterialCfg(
#            friction_combine_mode="multiply",
#            restitution_combine_mode="multiply",
#            static_friction=1.0,
#            dynamic_friction=1.0,
#            restitution=0.0,
#        ),
#        debug_vis=False,
#    )
   terrain = TerrainImporterCfg(
		prim_path="/World/ground",
		terrain_type="height_field",
		# height_field_folder="/home/joey/IsaacLab/test_height_fields",
		ground_folder="/home/orbit/Downloads/IsaacLab/new_test_height_fields_ground",
		ceiling_folder="/home/orbit/Downloads/IsaacLab/new_test_height_fields_ceiling",
		grid_size=(32, 32),
		terrain_size=(8.0, 8.0),
		collision_group=-1,
		physics_material=sim_utils.RigidBodyMaterialCfg(
			friction_combine_mode="multiply",
			restitution_combine_mode="multiply",
			static_friction=1.0,
			dynamic_friction=1.0,
		),
		visual_material=sim_utils.MdlFileCfg(
			mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
			project_uvw=True,
		),
		debug_vis=False,
	)

   # scene
   scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=8.0, replicate_physics=True)


   # events
   events: EventCfg = EventCfg()


   # robot
   robot: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="/World/envs/env_.*/Robot")
   contact_sensor: ContactSensorCfg = ContactSensorCfg(
	   prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
   )


   # reward scales
   lin_vel_reward_scale = 1
   yaw_rate_reward_scale = 0.5
   z_vel_reward_scale = -2.0
   ang_vel_reward_scale = -0.05
   joint_torque_reward_scale = -2.5e-5
   joint_accel_reward_scale = -2.5e-7
   action_rate_reward_scale = -0.01
   feet_air_time_reward_scale = 0.5
   undersired_contact_reward_scale = -1.0
   flat_orientation_reward_scale = -5.0
   x_vel_reward_scale = 0



@configclass
class AnymalCRoughEnvCfg(AnymalCFlatEnvCfg):
   # env
   num_observations = 235

   # terrain = TerrainImporterCfg(
   #     prim_path="/World/ground",
   #     terrain_type="generator",
   #     terrain_generator=ROUGH_TERRAINS_CFG,
   #     max_init_terrain_level=9,
   #     collision_group=-1,
   #     physics_material=sim_utils.RigidBodyMaterialCfg(
   #         friction_combine_mode="multiply",
   #         restitution_combine_mode="multiply",
   #         static_friction=1.0,
   #         dynamic_friction=1.0,
   #     ),
   #     visual_material=sim_utils.MdlFileCfg(
   #         mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
   #         project_uvw=True,
   #     ),
   #     debug_vis=False,
   # )
   viewer: ViewerCfg = ViewerCfg(
		eye=(600.0, 150.0, 0.5),  # Adjust these values to get a good view
		lookat=(600.0, -150.0, 0.0),  # Look at the center of the terrain, slightly above ground
		origin_type="env",
		env_index=1,  # This will set the camera relative to the first environment
	)
   terrain = TerrainImporterCfg(
		prim_path="/World/ground",
		terrain_type="height_field",
		# height_field_folder="/home/joey/IsaacLab/test_height_fields",
		# ground_folder="/home/orbit/Downloads/IsaacLab/new_test_height_fields_ground",
		# ceiling_folder="/home/orbit/Downloads/IsaacLab/new_test_height_fields_ceiling",
		ground_folder="/home/orbit/Downloads/IsaacLab/test_height_fields_ground",
		ceiling_folder="/home/orbit/Downloads/IsaacLab/test_height_fields_ceiling",
		grid_size=(32, 32),
		terrain_size=(32.0, 32.0),
		collision_group=-1,
		physics_material=sim_utils.RigidBodyMaterialCfg(
			friction_combine_mode="multiply",
			restitution_combine_mode="multiply",
			static_friction=1.0,
			dynamic_friction=1.0,
		),
		visual_material=sim_utils.MdlFileCfg(
			mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
			project_uvw=True,
		),
		debug_vis=False,
	)
  


   # we add a height scanner for perceptive locomotion
   height_scanner = RayCasterCfg(
	   prim_path="/World/envs/env_.*/Robot/base",
	   offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
	   attach_yaw_only=True,
	   pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
	   debug_vis=False,
	   mesh_prim_paths=["/World/ground"],
   )


   # reward scales (override from flat config)
   flat_orientation_reward_scale = 0.0
   time_out_penalty_scale = 0
   died_penalty_scale = 0




class AnymalCEnv(DirectRLEnv):
   cfg: AnymalCFlatEnvCfg | AnymalCRoughEnvCfg


   def __init__(self, cfg: AnymalCFlatEnvCfg | AnymalCRoughEnvCfg, render_mode: str | None = None, **kwargs):
	   super().__init__(cfg, render_mode, **kwargs)


	   # Joint position command (deviation from default joint positions)
	   self._actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)
	   self._previous_actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)


	   # X/Y linear velocity and yaw angular velocity commands
	   self._commands = torch.zeros(self.num_envs, 3, device=self.device)
	   self.start_position = torch.zeros(self.num_envs, 3, device=self.device)

	   # Logging
	   self._episode_sums = {
		   key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
		   for key in [
			   "track_lin_vel_xy_exp",
			   "track_ang_vel_z_exp",
			   "lin_vel_z_l2",
			   "ang_vel_xy_l2",
			   "dof_torques_l2",
			   "dof_acc_l2",
			   "action_rate_l2",
			   "feet_air_time",
			   "undesired_contacts",
			   "flat_orientation_l2",
			   "x_vel_l2",
			   "time_out_penalty",
			   "died_penalty",
		   ]
	   }
	   # Get specific body indices
	   self._base_id, _ = self._contact_sensor.find_bodies("base")
	   self._feet_ids, _ = self._contact_sensor.find_bodies(".*FOOT")
	   self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(".*THIGH")
	   self.rewards_dict = {}

   def _setup_scene(self):
	   self._robot = Articulation(self.cfg.robot)
	   self.scene.articulations["robot"] = self._robot
	   self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
	   self.scene.sensors["contact_sensor"] = self._contact_sensor
	   if isinstance(self.cfg, AnymalCRoughEnvCfg):
		   # we add a height scanner for perceptive locomotion
		   self._height_scanner = RayCaster(self.cfg.height_scanner)
		   self.scene.sensors["height_scanner"] = self._height_scanner
	   self.cfg.terrain.num_envs = self.scene.cfg.num_envs
	   self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
	   self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
	   # clone, filter, and replicate
	   self.scene.clone_environments(copy_from_source=False)
	   self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
	   # add lights
	   dome_light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.8, 0.8, 0.8))
	   dome_light_cfg.func("/World/Light", dome_light_cfg)

	   # Add distant lights from different directions
	   distant_light_cfg = sim_utils.DomeLightCfg(
		   intensity=100000.0,
		   color=(1.0, 1.0, 1.0),
		#    angle=0.53,
	   )
	   distant_light_height = 0.5
	   # Light from above
	   distant_light_cfg.func("/World/DistantLight1", distant_light_cfg, 
							  translation=(0, 0, distant_light_height), 
							  orientation=(0.7071, 0, 0, 0.7071))

	   # Light from +X direction
	   distant_light_cfg.func("/World/DistantLight2", distant_light_cfg, 
							  translation=(10, 0, distant_light_height), 
							  orientation=(0.5, 0.5, -0.5, 0.5))

	   # Light from -X direction
	   distant_light_cfg.func("/World/DistantLight3", distant_light_cfg, 
							  translation=(-10, 0, distant_light_height), 
							  orientation=(0.5, -0.5, 0.5, 0.5))

	   # Light from +Y direction
	   distant_light_cfg.func("/World/DistantLight4", distant_light_cfg, 
							  translation=(0, 10, distant_light_height), 
							  orientation=(0.5, 0.5, 0.5, -0.5))

	   cylinder_light_cfg = sim_utils.CylinderLightCfg(
		   intensity=500.0,
		   color=(1.0, 1.0, 0.9),
		   length=7.0,
		   radius=0.1,
	   )

	   # Add multiple cylinder lights
	   for i in range(5):
		   cylinder_light_cfg.func(f"/World/CylinderLight{i}", cylinder_light_cfg,
								   translation=(i*2 - 4, 0, 0.5))

	   # Add lights for each terrain
	#    self._setup_terrain_lights()




	#    # Add distant lights from different directions
	#    distant_light_cfg = sim_utils.DistantLightCfg(
	#        intensity=1000.0,
	#        color=(1.0, 1.0, 1.0),
	#        angle=0.53,
	#    )
	   
	#    # Light from above
	#    distant_light_cfg.func("/World/DistantLight1", distant_light_cfg)

	#    # Light from the side
	#    distant_light_cfg.func("/World/DistantLight2", distant_light_cfg)

	#    # Light from another side
	#    distant_light_cfg.func("/World/DistantLight3", distant_light_cfg)



   def _setup_terrain_lights(self):
	   dome_light_cfg = sim_utils.DomeLightCfg(
		   intensity=3000.0,  # Adjust this value as needed
		   color=(1.0, 1.0, 1.0),
		   exposure=0.0  # Adjust this if you need more or less brightness
	   )

	   distant_light_cfg = sim_utils.DistantLightCfg(
		   intensity=1000.0,
		   color=(1.0, 1.0, 1.0),
		   angle=0.53,
	   )

	   # Calculate the number of environments in each dimension
	   num_envs_x = 32
	   num_envs_y = 32

	   # Calculate the spacing between environments
	   env_spacing_x = 32.0
	   env_spacing_y = 32.0

	   for i in range(num_envs_x):
		   for j in range(num_envs_y):
			   # Calculate the position for this environment's light
			   x = i * env_spacing_x + env_spacing_x / 2
			   y = j * env_spacing_y + env_spacing_y / 2
			   z = 0.5  # Height of 0.5 meters

			   # Create a unique name for each light
			   dome_light_name = f"/World/EnvDomeLight_{i}_{j}"

			   # Spawn the dome light
			   dome_light_cfg.func(
				   dome_light_name, 
				   dome_light_cfg,
				   translation=(x, y, z)
			   )

			   # Add distant lights in different directions
			   directions = [
				   ("Top", (0, 0, 1), (0.7071, 0, 0, 0.7071)),
				   ("Bottom", (0, 0, -1), (-0.7071, 0, 0, 0.7071)),
				   ("Front", (1, 0, 0), (0.5, 0.5, -0.5, 0.5)),
				   ("Back", (-1, 0, 0), (0.5, -0.5, 0.5, 0.5)),
				   ("Left", (0, 1, 0), (0.5, 0.5, 0.5, -0.5)),
				   ("Right", (0, -1, 0), (0.5, -0.5, -0.5, 0.5))
			   ]

			   for direction, translation_offset, orientation in directions:
				   distant_light_name = f"/World/EnvDistantLight_{i}_{j}_{direction}"
				   distant_light_cfg.func(
					   distant_light_name,
					   distant_light_cfg,
					   translation=(x + translation_offset[0], y + translation_offset[1], z + translation_offset[2]),
					   orientation=orientation
				   )

	   # Add a global ambient light
	   global_dome_light_cfg = sim_utils.DomeLightCfg(
		   intensity=2000.0,  # Lower intensity for ambient light
		   color=(0.8, 0.8, 1.0),  # Slightly blue tint for ambient light
		   exposure=0.0
	   )
	   global_dome_light_cfg.func("/World/GlobalLight", global_dome_light_cfg)


   def _pre_physics_step(self, actions: torch.Tensor):
	   self._actions = actions.clone()
	   self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos


   def _apply_action(self):
	   self._robot.set_joint_position_target(self._processed_actions)

   
   def _get_observations(self) -> dict:
	   self._previous_actions = self._actions.clone()
	   height_data = None
	   if isinstance(self.cfg, AnymalCRoughEnvCfg):
		   height_data = (
			   self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2] - 0.5
		   ).clip(-1.0, 1.0)
	   obs = torch.cat(
		   [
			   tensor
			   for tensor in (
				   self._robot.data.root_lin_vel_b, # shape (num_envs, 3)
				   self._robot.data.root_ang_vel_b, # shape (num_envs, 3)
				   self._robot.data.projected_gravity_b, # shape (num_envs, 3)
				   self._commands, # shape (num_envs, 3), commands position in obs = [..., 6:9]
				   self._robot.data.joint_pos - self._robot.data.default_joint_pos, # shape (num_envs, 18)
				   self._robot.data.joint_vel, # shape (num_envs, 18)
				   height_data, # shape (num_envs, 16)
				   self._actions,
			   )
			   if tensor is not None
		   ],
		   dim=-1,
	   )
	   observations = {"policy": obs}
	   return observations


   def _get_rewards(self) -> torch.Tensor:
	   # add reward to encourage x velocity
	   x_vel_reward = self._robot.data.root_lin_vel_b[:, 0]
	   lin_vel_error = torch.sum(torch.square(self._commands[:, 1:2] - self._robot.data.root_lin_vel_b[:, 1:2]), dim=1)
	   lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25) 
	   # yaw rate tracking
	   yaw_rate_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
	   yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)

	   # print linear velocity and error
	#    print(f"Linear velocity: {self._robot.data.root_lin_vel_b[:, :2]}")
	#    print(f"Linear velocity error: {lin_vel_error}")
	#    # print yaw rate and error
	#    print(f"Yaw rate: {self._robot.data.root_ang_vel_b[:, 2]}")
	#    print(f"Yaw rate error: {yaw_rate_error}")
	   # z velocity tracking
	   z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
	   # angular velocity x/y
	   ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
	   # joint torques
	   joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
	   # joint acceleration
	   joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
	   # action rate
	   action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
	   # feet air time
	   first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
	   last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
	   air_time = torch.sum((last_air_time - 0.5) * first_contact, dim=1) * (
		   torch.norm(self._commands[:, :2], dim=1) > 0.1
	   )
	   # undersired contacts
	   net_contact_forces = self._contact_sensor.data.net_forces_w_history
	   is_contact = (
		   torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0] > 1.0
	   )
	   contacts = torch.sum(is_contact, dim=1)
	   # flat orientation
	   flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)
	   time_out = self.episode_length_buf >= self.max_episode_length - 1
	   net_contact_forces = self._contact_sensor.data.net_forces_w_history
	   died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1)
	   # add time out penalty and died penalty
	   time_out_penalty = time_out * self.cfg.time_out_penalty_scale
	   died_penalty = died * self.cfg.died_penalty_scale

	   rewards = {
		   "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt, #shape (num_envs,)
		   "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt, #shape (num_envs,)
		   "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt, #shape (num_envs,)
		   "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt, #shape (num_envs,)
		   "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt, #shape (num_envs,)
		   "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt, #shape (num_envs,)
		   "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt, #shape (num_envs,)
		   "feet_air_time": air_time * self.cfg.feet_air_time_reward_scale * self.step_dt, #shape (num_envs,)
		   "undesired_contacts": contacts * self.cfg.undersired_contact_reward_scale * self.step_dt, #shape (num_envs,)
		   "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt, #shape (num_envs,)
		   "x_vel_l2": x_vel_reward * self.cfg.x_vel_reward_scale * self.step_dt, #shape (num_envs,)
		   "time_out_penalty": time_out_penalty, #shape (num_envs,)
		   "died_penalty": died_penalty, #shape (num_envs,)
	   }
	   self.rewards_dict = rewards
	   reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
	
	
		
	
	   # Logging
	#    for key, value in rewards.items():
	# 	   self._episode_sums[key] += value
	   return reward


   def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
	#    print(f"Episode length: {self.episode_length_buf}, max episode length: {self.max_episode_length}")
	   time_out = self.episode_length_buf >= self.max_episode_length - 1
	   net_contact_forces = self._contact_sensor.data.net_forces_w_history
	   died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1)
	   return died, time_out




   def replace_terrain(self, new_height_field_folder: str, grid_size: tuple[int, int], terrain_size: tuple[float, float]):
	   """Replace the current terrain with a new terrain from a different height field folder."""
	   # Delete the existing terrain
	   self._terrain.delete_terrain("terrain")


	   # Ensure the ground prim is deleted
	   prim_utils.delete_prim("/World/ground")


	   # Create a new terrain configuration
	   new_terrain_cfg = TerrainImporterCfg(
		   prim_path="/World/ground",
		   terrain_type="height_field",
		   height_field_folder=new_height_field_folder,
		   grid_size=grid_size,
		   terrain_size=terrain_size,
		   collision_group=-1,
		   physics_material=sim_utils.RigidBodyMaterialCfg(
			   friction_combine_mode="multiply",
			   restitution_combine_mode="multiply",
			   static_friction=1.0,
			   dynamic_friction=1.0,
			   restitution=0.0,
		   ),
		   visual_material=self.cfg.terrain.visual_material,
		   debug_vis=False,
		   num_envs=self.num_envs,
		   env_spacing=self.cfg.terrain.env_spacing,
	   )


	   # Update the terrain configuration
	   self._terrain.cfg = new_terrain_cfg


	   # Import the new terrain
	   self._terrain.import_height_field_folder(
		   new_height_field_folder,
		   self.cfg.terrain.grid_size,
		   self.cfg.terrain.terrain_size
	   )


	   # # Reconfigure environment origins for the new terrain
	   # self._terrain.configure_env_origins()


	   # Reset all robots
	   self._reset_idx(None)  # Passing None resets all environments


	   print(f"Terrain replaced with new height fields from {new_height_field_folder} and all robots reset.")


   def replace_terrain_with_ground_plane(self):
	   """Replace the current terrain with a flat ground plane and reset all robots."""
	   # Delete the existing terrain
	   self._terrain.delete_terrain("terrain")


	   # Ensure the ground prim is deleted
	   import omni.isaac.core.utils.prims as prim_utils
	   prim_utils.delete_prim("/World/ground")


	   # Create a new ground plane configuration
	   ground_plane_cfg = TerrainImporterCfg(
		   prim_path="/World/ground",
		   terrain_type="plane",
		   collision_group=-1,
		   physics_material=sim_utils.RigidBodyMaterialCfg(
			   friction_combine_mode="multiply",
			   restitution_combine_mode="multiply",
			   static_friction=1.0,
			   dynamic_friction=1.0,
			   restitution=0.0,
		   ),
		   debug_vis=False,
		   num_envs=self.num_envs,
		   env_spacing=self.cfg.terrain.env_spacing,
	   )


	   # Update the terrain configuration
	   self._terrain.cfg = ground_plane_cfg


	   # Import the new ground plane
	   self._terrain.import_ground_plane("terrain")


	   # Reconfigure environment origins for the flat ground
	   self._terrain.configure_env_origins()


	   # Reset all robots
	   self._reset_idx(None)  # Passing None resets all environments


	   print("Terrain replaced with a flat ground plane and all robots reset.")








   def _reset_idx(self, env_ids: torch.Tensor | None):
	   if env_ids is None or len(env_ids) == self.num_envs:
		   env_ids = self._robot._ALL_INDICES
	   self._robot.reset(env_ids)
	   super()._reset_idx(env_ids)
	#    if len(env_ids) == self.num_envs:
	# 	   # Spread out the resets to avoid spikes in training when many environments reset at a similar time
	# 	#    for i in range(1000):
	# 	# 	   print(f"Resetting episode_length_buf")
			   
	# 	   self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length - 1))
	   self._actions[env_ids] = 0.0
	   self._previous_actions[env_ids] = 0.0
	   # Sample new commands
	#    self._commands[env_ids] = torch.zeros_like(self._commands[env_ids])
	#    self._commands[env_ids, 0] = -0.3  # Set to maximum forward velocity
	   # self._commands[env_ids, 1] = 0.0  # No lateral velocity (already 0 from zeros_like)
	   # self._commands[env_ids, 2] = 0.0  # No yaw rate (already 0 from zeros_like)
	   # what is current command space
	   # add print statement to explain command generation
	#    print(f"Commands generated for {len(env_ids)} environments: {self._commands[0]}")
	   # Reset robot state
	   joint_pos = self._robot.data.default_joint_pos[env_ids]
	   joint_vel = self._robot.data.default_joint_vel[env_ids]
	   # default_root_state = self._robot.data.default_root_state[env_ids]
	   # if self._terrain.env_origins is not None:
	   #     default_root_state[:, :3] += self._terrain.env_origins[env_ids]
	   # else:
	   #     # If env_origins is None, assume a flat ground at z=0
	   #     default_root_state[:, 2] = 0.0
	   initial_height = 0.0
	   default_root_state = self._robot.data.default_root_state[env_ids]
	   if self._terrain.terrain_origins is not None:
		   default_root_state[:, :3] += self._terrain.terrain_origins.view(-1, 3)[env_ids]
	   else:
		   # If terrain_origins is None, assume a flat ground at z=0
		   default_root_state[:, 2] = 0.0
	   default_root_state[:, 2] += initial_height
	   default_root_state[:, 0] += 8
	   self.start_position = default_root_state[:, :3]


	   self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
	   self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
	   self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


	   extras = dict()
	   for key in self._episode_sums.keys():
		   episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
		   extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
		   self._episode_sums[key][env_ids] = 0.0
	   self.extras["log"] = dict()
	   self.extras["log"].update(extras)
	   extras = dict()
	   extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
	   extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
	   self.extras["log"].update(extras)

@configclass
class AnymalConfinedEnvCfg(AnymalCRoughEnvCfg):
	# env
	episode_length_s = 20  # Longer episodes for more complex navigation
	num_observations = 235 + 3 + 1 # Increased for additional sensors
	num_actions = 12 + 1 # Same as before
	# Still using height scanner
	viewer: ViewerCfg = ViewerCfg(
		eye=(600.0, 150.0, 0.5),  # Adjust these values to get a good view
		lookat=(600.0, -150.0, 0.0),  # Look at the center of the terrain, slightly above ground
		origin_type="env",
		env_index=1,  # This will set the camera relative to the first environment
	)
	terrain = TerrainImporterCfg(
		prim_path="/World/ground",
		terrain_type="height_field",
		# height_field_folder="/home/joey/IsaacLab/test_height_fields",
		# ground_folder="/home/orbit/Downloads/IsaacLab/new_test_height_fields_ground",
		# ceiling_folder="/home/orbit/Downloads/IsaacLab/new_test_height_fields_ceiling",
		ground_folder="/home/orbit/Downloads/IsaacLab/test_height_fields_ground_0.25",
		ceiling_folder="/home/orbit/Downloads/IsaacLab/test_height_fields_ceiling_0.25",
		grid_size=(32, 32),
		terrain_size=(32.0, 32.0),
		collision_group=-1,
		physics_material=sim_utils.RigidBodyMaterialCfg(
			friction_combine_mode="multiply",
			restitution_combine_mode="multiply",
			static_friction=1.0,
			dynamic_friction=1.0,
		),
		visual_material=sim_utils.MdlFileCfg(
			mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
			project_uvw=True,
		),
		debug_vis=False,
	)



	# we add a height scanner for perceptive locomotion
	height_scanner = RayCasterCfg(
		prim_path="/World/envs/env_.*/Robot/base",
		offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
		attach_yaw_only=True,
		pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
		debug_vis=False,
		mesh_prim_paths=["/World/ground"],
	)
	# Reward scales
	goal_reward_scale = 100.0
	progress_reward_scale = 3.0
	collision_penalty_scale = -5.0
	smooth_motion_reward_scale = 0.1
	exploration_reward_scale = 0.5
	fall_penalty_scale = -300.0
	stability_reward_scale = 0.5
class AnymalConfinedEnv(AnymalCEnv):
	cfg: AnymalConfinedEnvCfg

	def __init__(self, cfg: AnymalConfinedEnvCfg, render_mode: str | None = None, **kwargs):
		super().__init__(cfg, render_mode, **kwargs)
		# self._height_scanner = RayCaster(self.cfg.height_scanner)
		# self.scene.sensors["height_scanner"] = self._height_scanner
		# self._goal_positions = torch.zeros(self.num_envs, 3, device=self.device)
		# robot starts at (8, 0, 0)
		# goal is at (-8, 0, 0)
		# we only care the robot passes in the x direction
		# Initialize goal positions with the correct size

		# Modify the action space
		self.cfg.num_actions = 13  # 12 joint positions + 3 planar velocities
		self._actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)
		self._previous_actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)

		self.str_statistics = ""
		self._goal_positions = torch.zeros(self.num_envs, 3, device=self.device) + self._terrain.env_origins
		self._start_positions = torch.zeros(self.num_envs, 3, device=self.device) + self._terrain.env_origins
		self._previous_positions = torch.zeros(self.num_envs, 3, device=self.device) + self._terrain.env_origins
		self._previous_positions[:, 0] = 8.0
		# Set initial values
		self._start_positions[:, 0] += 8.0
		self._goal_positions[:, 0] += -8.0
		# self._set_random_goals()
		
		# Add these lines for tracking success rate
		self.max_success_rate = 0.0
		self.sum_total_dones = 0
		self.sum_total_successes = 0
		self.sum_total_timeouts = 0
		self.sum_total_died = 0
		self.success_rate_lst = []
		self.total_dones = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
		self.total_successes = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
		self.total_timeouts = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
		self.total_died = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

	# def _reset_robot(self):
	# 	# Set robot to the start position
	# 	self._robot.set_root_pos(self._start_positions)
	# 	# Set random goal positions for each environment
	# 	pass
		# self._goal_positions[:, 0] = torch.rand(self.num_envs, device=self.device) * 14 +  torch.ones(self.num_envs, device=self.device)# X: 1 to 15
		# self._goal_positions[:, 1] = torch.rand(self.num_envs, device=self.device) * 14 + torch.ones(self.num_envs, device=self.device)  # Y: 1 to 15
		# self._goal_positions[:, 2] = torch.ones(self.num_envs, device=self.device) * 0.5  # Fixed height


	def _pre_physics_step(self, actions: torch.Tensor):
		self._actions = actions.clone()
		
		# Split actions into joint positions and desired velocities
		joint_actions = self._actions[:, :12]
		velocity_actions = self._actions[:, 12:]
		# scale velocity to (0,1)		
		velocity_actions = (velocity_actions - 1) / 2
		# Process joint actions
		self._processed_joint_actions = self.cfg.action_scale * joint_actions + self._robot.data.default_joint_pos
		
		# Process velocity actions (scale if necessary)
		# self._commands[:, 0] = velocity_actions.squeeze()
		# print 1st 2 commands
		# print(f"Commands: {self._commands[:2]}")
		# print commands statistics, min, max, mean, median, std
		# print(f"Commands shape: {self._commands.shape}")
		# print(f"Commands min: {torch.min(self._commands).item()}, max: {torch.max(self._commands).item()}, mean: {torch.mean(self._commands).item()}, median: {torch.median(self._commands).item()}, std: {torch.std(self._commands).item()}")


	def _apply_action(self):
		# Apply joint position targets
		self._robot.set_joint_position_target(self._processed_joint_actions)
		
		# Apply velocity commands
		# self._robot.set_world_velocity(self._processed_velocity_actions)


	def _get_observations(self) -> dict:
		base_obs = super()._get_observations()
		height_scanner_data = self._height_scanner.data.ray_hits_w.view(self.num_envs, -1)
		
		# Print goal positions, start positions, and current positions for the first five environments
		# num_to_print = min(5, self.num_envs)
		# print("\nPositions for the first", num_to_print, "environments:")
		# for i in range(num_to_print):
		# # 	print(f"Env {i}:")
		# # 	print(f"  Goal position:   {self._goal_positions[i].tolist()}")
		# # 	print(f"  Start position:  {self._start_positions[i].tolist()}")
		# # 	print(f"  Current position: {self._robot.data.root_pos_w[i].tolist()}")
		# # 	# print env origin
		# 	print(f"  Env origin: {self._terrain.env_origins[i].tolist()}")
		# print()  # Add an extra newline for better readability
		
		goal_relative = self._goal_positions - self._robot.data.root_pos_w
		
		# Print shape of each part of the observation
		# print(f"Height scanner data shape: {height_scanner_data.shape}")
		# print(f"Goal relative shape: {goal_relative.shape}")
		
		# Remove commands from base_obs
		base_obs_tensor = base_obs["policy"]
		# base_obs_without_commands = torch.cat([
		# 	base_obs_tensor[:, :6],  # Keep root_lin_vel_b and root_ang_vel_b
		# 	base_obs_tensor[:, 9:]   # Skip commands and keep the rest
		# ], dim=1)

		# print(f"base_obs_without_commands shape: {base_obs_without_commands.shape}")
		# print(f"goal_relative shape: {goal_relative.shape}")
		
		# Concatenate observations
		# obs = torch.cat([base_obs_without_commands, goal_relative], dim=-1)
		# print(f"base_obs_tensor shape: {base_obs_tensor.shape}")
		# print(f"goal_relative shape: {goal_relative.shape}")
		obs = torch.cat([base_obs_tensor, goal_relative], dim=-1)
		# print(f"Final observation shape: {obs.shape}")
		
		return {"policy": obs}

	def _get_rewards(self) -> torch.Tensor:
		super()._get_rewards()
		base_reward = self.rewards_dict
		# print(f"base_reward: {base_reward}")
		# Remove specific rewards from base_reward
		# rewards_to_remove = [
		# 	# "track_lin_vel_xy_exp",
		# 	# "track_ang_vel_z_exp",
		# 	# "lin_vel_z_l2"
		# ]
		
		# for reward_name in rewards_to_remove:
		# 	if reward_name in base_reward:
		# 		# del self._episode_sums[reward_name]
		# 		del base_reward[reward_name]
		
		# Recalculate base_reward without the removed components
		# base_reward = sum(reward for name, reward in self._episode_sums.items() if name not in rewards_to_remove)
		# Goal-based reward
		sum_base_reward = sum(base_reward.values())
		distance_to_goal = torch.norm(self._goal_positions - self._robot.data.root_pos_w, dim=1)
		x_distance_to_goal = torch.abs(self._goal_positions[:, 0] - self._robot.data.root_pos_w[:, 0])
		goal_reached = x_distance_to_goal < 0.5
		# goal reache reward only take effect when goal is reached, add 20 reward when goal is reached
		goal_reward = torch.where(goal_reached, 
								  torch.ones_like(distance_to_goal) * self.cfg.goal_reward_scale, 
								  torch.zeros_like(distance_to_goal))
		# print if any goal is reached and print the index
		goal_reached_indices = torch.where(goal_reached)[0]
		# if len(goal_reached_indices) > 0:
		# 	print(f"Goal reached at indices: {goal_reached_indices}")
		# else:
		# 	print("Goal not reached")	

		# Calculate distance to goal at current and previous step
		current_distance = torch.norm(self._goal_positions - self._robot.data.root_pos_w, dim=1)
		previous_distance = torch.norm(self._goal_positions - self._previous_positions, dim=1)
		
		# Calculate progress reward
		progress_reward = (previous_distance - current_distance) * self.cfg.progress_reward_scale
		
		# Update previous positions for next step
		self._previous_positions = self._robot.data.root_pos_w.clone()

		# # Collision penalty
		# collision = torch.any(self._height_scanner.data.ray_hits_w < 0.2, dim=1)
		# collision_penalty = collision.float() * self.cfg.collision_penalty_scale

		# # Smooth motion reward
		# smooth_motion_reward = -torch.sum(torch.abs(self._actions - self._previous_actions), dim=1) * self.cfg.smooth_motion_reward_scale

		# # Exploration reward (based on new areas visited)
		# exploration_reward = self._compute_exploration_reward() * self.cfg.exploration_reward_scale

		# print each rewards shape
		# print(f"sum_base_reward shape: {sum_base_reward.shape}")
		# print(f"goal_reward shape: {goal_reward.shape}")
		# print(f"collision_penalty shape: {collision_penalty.shape}")
		# print(f"smooth_motion_reward shape: {smooth_motion_reward.shape}")
		# print(f"exploration_reward shape: {exploration_reward.shape}")
		# print avg progress reward	
		# stability_reward = self._compute_stability_reward() * self.cfg.stability_reward_scale
		avg_progress_reward = torch.mean(progress_reward)
		# print(f"Avg progress reward: {avg_progress_reward:.4f} avg_balance_reward: {torch.mean(sum_base_reward):.4f}")
		total_reward = sum_base_reward + goal_reward + progress_reward
		for key, value in base_reward.items():
			self._episode_sums[key] += value
		return total_reward

	# def _compute_exploration_reward(self):
	# 	# Implement logic to reward exploring new areas
	# 	# This could involve maintaining a visited map and rewarding new cell visits
	# 	pass
	# 	return 0.0

	# def _get_base_orientation(self):
	#     # Extract quaternion orientation
	#     quat = self._robot.data.root_quat_w  # Shape: (num_envs, 4)

	#     # Convert quaternion to roll and pitch
	#     # Roll (rotation around x-axis)
	#     sinr_cosp = 2.0 * (quat[:, 0] * quat[:, 1] + quat[:, 2] * quat[:, 3])
	#     cosr_cosp = 1.0 - 2.0 * (quat[:, 1] * quat[:, 1] + quat[:, 2] * quat[:, 2])
	#     roll = torch.atan2(sinr_cosp, cosr_cosp)

	#     # Pitch (rotation around y-axis)
	#     sinp = 2.0 * (quat[:, 0] * quat[:, 2] - quat[:, 3] * quat[:, 1])
	#     pitch = torch.where(
	#         torch.abs(sinp) >= 1,
	#         torch.sign(sinp) * torch.tensor(3.14159 / 2, device=self.device),
	# 		torch.asin(sinp)
	# 	)

	# 	return roll, pitch  # Each tensor has shape (num_envs,)

	# def _compute_stability_reward(self):
	#     roll, pitch = self._get_base_orientation()
		
	#     # Convert to degrees for easier interpretation
	#     roll_deg = torch.rad2deg(roll)
	#     pitch_deg = torch.rad2deg(pitch)

	#     # Calculate stability based on roll and pitch
	#     # The exp(-x^2) function peaks at 0 and falls off quickly as x increases
	# 	roll_stability = torch.exp(-(roll_deg / 15)**2)  # 15 degrees as characteristic scale
	# 	pitch_stability = torch.exp(-(pitch_deg / 15)**2)

	# 	# Combine roll and pitch stability
	# 	stability = (roll_stability + pitch_stability) / 2

	#     return stability

	def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
		died, time_out = super()._get_dones()
		
		# Check if goal is reached in x direction only
		x_distance_to_goal = torch.abs(self._goal_positions[:, 0] - self._robot.data.root_pos_w[:, 0])
		goal_reached = x_distance_to_goal < 0.5

		# Update total counts
		# self.total_dones += 1
		# total dones is goal reached or timeout or died
		single_successes = goal_reached.long().sum().item()
		single_timeouts = time_out.long().sum().item()
		single_died = died.long().sum().item()
  
  
		# self.total_dones += single_successes + single_timeouts + single_died
		self.sum_total_dones += single_successes + single_timeouts + single_died
		self.sum_total_successes += single_successes
		self.sum_total_timeouts += single_timeouts
		self.sum_total_died += single_died
		# print single pass rate for died, timeout, success
		# print(f"died rate: {died.long().sum().item()/self.total_dones.sum().item()}")
		# single_dones = (goal_reached | time_out | died).long() + 1

		# print(f"total_dones : {self.sum_total_dones}, single successes: {single_successes}, single timeouts: {single_timeouts}, single died: {single_died}")
		# print(f"single pass rate for died: {single_died.sum().item()/single_dones.sum().item()}, timeout: {single_timeouts.sum().item()/single_dones.sum().item()}, success: {single_successes.sum().item()/single_dones.sum().item()}")

		# # Calculate rates
		# def safe_divide(a, b):
		# 	return torch.where(b > 0, a.float() / b.float(), torch.zeros_like(a, dtype=torch.float))

		# success_rate = safe_divide(self.total_successes, self.total_dones)
		# timeout_rate = safe_divide(self.total_timeouts, self.total_dones)
		# died_rate = safe_divide(self.total_died, self.total_dones)
		# terminated_rate = safe_divide(self.total_successes + self.total_died, self.total_dones)

		print(f"self.sum_total_dones: {self.sum_total_dones}")
		if self.sum_total_dones >= self.num_envs * 3:
			# sum_total_dones = self.total_dones.sum().item()
			# sum_total_successes = self.total_successes.sum().item()
			# sum_total_timeouts = self.total_timeouts.sum().item()
			# sum_total_died = self.total_died.sum().item()
			# print final statistics before reset
			print(f"Final statistics before reset:")
			print(f"  Success rate:    {self.sum_total_successes/self.sum_total_dones:.2%} (Count: {self.sum_total_successes}, Total: {self.sum_total_dones})")
			print(f"  Timeout rate:    {self.sum_total_timeouts/self.sum_total_dones:.2%} (Count: {self.sum_total_timeouts}, Total: {self.sum_total_dones})")
			print(f"  Died rate:       {self.sum_total_died/self.sum_total_dones:.2%} (Count: {self.sum_total_died}, Total: {self.sum_total_dones})")
			self.max_success_rate = max(self.max_success_rate, self.sum_total_successes/self.sum_total_dones)
			print(f"Max success rate: {self.max_success_rate:.2%}")
			self.success_rate_lst.append(self.sum_total_successes/self.sum_total_dones)
			# print statistics of success rate list, count, min, max, mean, median, std
			print(f"Success rate list count: {len(self.success_rate_lst)}")
			print(f"Success rate list min: {min(self.success_rate_lst):.2%}, max: {max(self.success_rate_lst):.2%}, mean: {np.mean(self.success_rate_lst):.2%}, median: {np.median(self.success_rate_lst):.2%}, std: {np.std(self.success_rate_lst):.2%}")
			self.str_statistics = f"Success rate list count: {len(self.success_rate_lst)}, min: {min(self.success_rate_lst):.2%}, max: {max(self.success_rate_lst):.2%}, mean: {np.mean(self.success_rate_lst):.2%}, median: {np.median(self.success_rate_lst):.2%}, std: {np.std(self.success_rate_lst):.2%}"
   # if len(self.success_rate_lst) > 10, do not update statistics
			# if len(self.success_rate_lst) > 10:
			# 	print("Success rate list is too long, do not update statistics")
			# else:
			# 	self.max_success_rate = max(self.max_success_rate, np.mean(self.success_rate_lst))
   # reset sum_total_dones, sum_total_successes, sum_total_timeouts, sum_total_died
			self.sum_total_dones = 0
			self.sum_total_successes = 0
			self.sum_total_timeouts = 0
			self.sum_total_died = 0
   # # print(f"  Terminated rate: {terminated_rate.mean().item():.2%} (Count: {(self.total_successes + self.total_died).sum().item()}, Total: {self.total_dones.sum().item()})")
			# self.total_dones = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
			# self.total_successes = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
			# self.total_timeouts = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
			# self.total_died = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
		else:
				# Print status
			# Print status
			# print(f"In progress: total dones {self.total_dones.sum().item()}")
			pass
			# avg_total_dones = self.total_dones.float().mean().item()
			# print(f"Average rates (based on {avg_total_dones:.1f} average dones), new died {died.long().sum().item()}, new timeout {time_out.long().sum().item()}")
			# print(f"  Success rate:    {success_rate.mean().item():.2%} (Count: {self.total_successes.sum().item()}, Total: {self.total_dones.sum().item()})")
			# print(f"  Timeout rate:    {timeout_rate.mean().item():.2%} (Count: {self.total_timeouts.sum().item()}, Total: {self.total_dones.sum().item()})")
			# print(f"  Died rate:       {died_rate.mean().item():.2%} (Count: {self.total_died.sum().item()}, Total: {self.total_dones.sum().item()})")
			# print(f"  Terminated rate: {terminated_rate.mean().item():.2%} (Count: {(self.total_successes + self.total_died).sum().item()}, Total: {self.total_dones.sum().item()})")

		terminated = died | goal_reached
		truncated = time_out
		print(f"Max success rate: {self.max_success_rate:.2%}")
		print(f"Statistics: {self.str_statistics}")

		return terminated, truncated
	# def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
	# 	died, time_out = super()._get_dones()
		
	# 	# Check if goal is reached in x direction only
	# 	x_distance_to_goal = torch.abs(self._goal_positions[:, 0] - self._robot.data.root_pos_w[:, 0])
	# 	goal_reached = x_distance_to_goal < 0.5

	# 	# Update total counts
	# 	# self.total_dones += 1
	# 	# total dones is goal reached or timeout or died
	# 	self.total_dones += (goal_reached | time_out | died).long()
	# 	self.total_successes += goal_reached.long()
	# 	self.total_timeouts += time_out.long()
	# 	self.total_died += died.long()

	# 	# Calculate rates
	# 	def safe_divide(a, b):
	# 		return torch.where(b > 0, a.float() / b.float(), torch.zeros_like(a, dtype=torch.float))

	# 	success_rate = safe_divide(self.total_successes, self.total_dones)
	# 	timeout_rate = safe_divide(self.total_timeouts, self.total_dones)
	# 	died_rate = safe_divide(self.total_died, self.total_dones)
	# 	terminated_rate = safe_divide(self.total_successes + self.total_died, self.total_dones)

	# 	# Print status
	# 	avg_total_dones = self.total_dones.float().mean().item()
	# 	print(f"Average rates (based on {avg_total_dones:.1f} average dones), new died {died.long().sum().item()}, new timeout {time_out.long().sum().item()}")
	# 	print(f"  Success rate:    {success_rate.mean().item():.2%} (Count: {self.total_successes.sum().item()}, Total: {self.total_dones.sum().item()})")
	# 	print(f"  Timeout rate:    {timeout_rate.mean().item():.2%} (Count: {self.total_timeouts.sum().item()}, Total: {self.total_dones.sum().item()})")
	# 	print(f"  Died rate:       {died_rate.mean().item():.2%} (Count: {self.total_died.sum().item()}, Total: {self.total_dones.sum().item()})")
	# 	print(f"  Terminated rate: {terminated_rate.mean().item():.2%} (Count: {(self.total_successes + self.total_died).sum().item()}, Total: {self.total_dones.sum().item()})")

	# 	terminated = died | goal_reached
	# 	truncated = time_out

	# 	return terminated, truncated
	def get_dones(self):
		return self._get_dones()
	def _reset_idx(self, env_ids: torch.Tensor | None):
		super()._reset_idx(env_ids)
		if env_ids is None:
			env_ids = torch.arange(self.num_envs, device=self.device)
		# self._set_random_goals()
		# Reset exploration map if implemented
		# reset previous positions for env_ids
		self._previous_positions[env_ids] = self._start_positions[env_ids]

		# reset total_dones, total_successes, total_timeouts, total_died if sum of total_dones is greater than 3000
