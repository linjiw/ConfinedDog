# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import numpy as np

import omni.isaac.lab.envs.mdp as mdp
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import RayCaster, RayCasterCfg, patterns
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab_assets.f1_car import F1_CAR_CFG

from omni.isaac.lab.envs import ViewerCfg

@configclass
class F1TenthEnvCfg(DirectRLEnvCfg):
	# env
	episode_length_s = 60.0
	decimation = 4
	action_scale = 1.0
	num_actions = 2  # steering and velocity
	num_observations = 3  # 180 laser scan points + 3 for velocity
	num_states = 0
	progress_reward_scale = 1.0
	
	# simulation
	sim: SimulationCfg = SimulationCfg(
		dt=1 / 100,
		render_interval=decimation,
		physics_material=sim_utils.RigidBodyMaterialCfg(
			static_friction=0.5,
			dynamic_friction=0.5,
			restitution=0.0,
		),
	)
	
	# scene
	scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=8.0, replicate_physics=True)
	
	# Map configuration
	map_size = (256, 256)
	track_width = 20
	num_maps = 10
	map_folder = "/path/to/map/folder"

	# terrain = TerrainImporterCfg(
	# 	prim_path="/World/ground",
	# 	terrain_type="height_field",
	# 	# height_field_folder="/home/joey/IsaacLab/test_height_fields",
	# 	ground_folder="/home/joey/IsaacLab/new_test_height_fields_ground",
	# 	ceiling_folder="/home/joey/IsaacLab/new_test_height_fields_ceiling",
	# 	grid_size=(32, 32),
	# 	terrain_size=(4.0, 4.0),
	# 	collision_group=-1,
	# 	physics_material=sim_utils.RigidBodyMaterialCfg(
	# 		friction_combine_mode="multiply",
	# 		restitution_combine_mode="multiply",
	# 		static_friction=1.0,
	# 		dynamic_friction=1.0,
	# 	),
	# 	visual_material=sim_utils.MdlFileCfg(
	# 		mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
	# 		project_uvw=True,
	# 	),
	# 	debug_vis=False,
	# )

	terrain = TerrainImporterCfg(
		prim_path="/World/ground",
		terrain_type="plane",
		collision_group=-1,
		physics_material=sim_utils.RigidBodyMaterialCfg(
			friction_combine_mode="average",
			restitution_combine_mode="average",
			static_friction=1.0,
			dynamic_friction=1.0,
			restitution=0.0,
		),
		debug_vis=False,
	)
	# terrain = TerrainImporterCfg(
	# 	prim_path="/World/ground",
	# 	terrain_type="height_field",
	# 	# height_field_folder="/home/joey/IsaacLab/test_height_fields",
	# 	ground_folder="/home/orbit/Downloads/IsaacLab/test_height_fields_ground",
	# 	ceiling_folder="/home/orbit/Downloads/IsaacLab/test_height_fields_ceiling",
	# 	grid_size=(32, 32),
	# 	terrain_size=(8.0, 8.0),
	# 	collision_group=-1,
	# 	physics_material=sim_utils.RigidBodyMaterialCfg(
	# 		friction_combine_mode="multiply",
	# 		restitution_combine_mode="multiply",
	# 		static_friction=1.0,
	# 		dynamic_friction=1.0,
	# 	),
	# 	visual_material=sim_utils.MdlFileCfg(
	# 		mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
	# 		project_uvw=True,
	# 	),
	# 	debug_vis=False,
	# )
	
	# F1TENTH car
	# car model path: /home/joey/IsaacLab/source/extensions/omni.isaac.lab_assets/data/Vehicles/F1Car/f1_robot.usd
	robot: ArticulationCfg = F1_CAR_CFG.replace(prim_path="/World/envs/env_.*/F1Car")
	

	
	# viewer
	viewer: ViewerCfg = ViewerCfg(
		eye=(10.0, 10.0, 5.0),
		lookat=(0.0, 0.0, 0.0),
		origin_type="env",
		env_index=0,
	)

class F1TenthEnv(DirectRLEnv):
	cfg: F1TenthEnvCfg

	def __init__(self, cfg: F1TenthEnvCfg, render_mode: str | None = None, **kwargs):
		super().__init__(cfg, render_mode, **kwargs)

		self._actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)
		self._previous_actions = torch.zeros(self.num_envs, self.cfg.num_actions, device=self.device)

		# Logging
		self._episode_sums = {
			key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
			for key in [
				"progress",
				"collision",
				"steering",
			]
		}

	def _setup_scene(self):

		
		# Create the Articulation after spawning
		self._robot = Articulation(self.cfg.robot)
		self.scene.articulations["robot"] = self._robot
		# self._lidar = RayCaster(self.cfg.lidar)
		# self.scene.sensors["lidar"] = self._lidar
		
		self.cfg.terrain.num_envs = self.scene.cfg.num_envs
		self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
		self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
		
		self.scene.clone_environments(copy_from_source=False)
		self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
		
		# add lights
		light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
		light_cfg.func("/World/Light", light_cfg)

	def _pre_physics_step(self, actions: torch.Tensor):
		self._actions = actions.clone()
		self._processed_actions = self.cfg.action_scale * self._actions

	def _apply_action(self):
		steering = self._processed_actions[:, 0]
		velocity = self._processed_actions[:, 1]
		# Apply steering and velocity to the car
		# You'll need to implement this based on your car model

	def _get_observations(self) -> dict:
		self._previous_actions = self._actions.clone()
		
		# lidar_data = self._lidar.data.ray_hits_w[..., :2].reshape(self.num_envs, -1)
		velocity = self._robot.data.root_lin_vel_b
		
		# obs = torch.cat([lidar_data, velocity], dim=-1)
		obs = velocity
		observations = {"policy": obs}
		return observations

	def _get_rewards(self) -> torch.Tensor:
		# Implement your reward function here
		# This could include progress along the track, penalties for collisions, etc.
		rewards = torch.zeros(self.num_envs, device=self.device)
		
		# Example: reward for forward progress
		progress = self._robot.data.root_lin_vel_b[:, 0]  # Assuming x is forward
		rewards += progress * self.cfg.progress_reward_scale
		
		# Example: penalty for collision
		collision = self._check_collision()  # You need to implement this
		rewards -= collision * 10.0  # Large penalty for collision
		
		# Example: small penalty for steering to encourage smooth driving
		steering_penalty = torch.abs(self._actions[:, 0]) * 0.1
		rewards -= steering_penalty
		
		# Logging
		self._episode_sums["progress"] += progress
		self._episode_sums["collision"] += collision
		self._episode_sums["steering"] += steering_penalty
		
		return rewards

	def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
		time_out = self.episode_length_buf >= self.max_episode_length - 1
		collision = self._check_collision()  # You need to implement this
		return collision, time_out

	def _check_collision(self) -> torch.Tensor:
		# Implement collision detection here
		# This could use the LiDAR data or other sensors
		# For now, we'll just return a dummy value
		return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

	def _reset_idx(self, env_ids: torch.Tensor | None):
		pass
		# if env_ids is None:
		# 	env_ids = self._robot._ALL_INDICES
		
		# self._robot.reset(env_ids)
		# super()._reset_idx(env_ids)

		# initial_height = 0.0
		# default_root_state = self._robot.data.default_root_state[env_ids]
		# if self._terrain.terrain_origins is not None:
		# 	default_root_state[:, :3] += self._terrain.terrain_origins.view(-1, 3)[env_ids]
		# else:
		# 	# If terrain_origins is None, assume a flat ground at z=0
		# 	default_root_state[:, 2] = 0.0
		# default_root_state[:, 2] += initial_height
		
		# self._actions[env_ids] = 0.0
		# self._previous_actions[env_ids] = 0.0
		
		# # Reset car position and orientation
		# default_root_state = self._robot.data.default_root_state[env_ids]
		# self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
		# self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
		
		# # Log episode statistics
		# extras = dict()
		# for key in self._episode_sums.keys():
		# 	episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
		# 	extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
		# 	self._episode_sums[key][env_ids] = 0.0
		# self.extras["log"] = dict()
		# self.extras["log"].update(extras)