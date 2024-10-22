# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the F1 Car."""

from __future__ import annotations

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Configuration
##

F1_CAR_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/F1Car",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/orbit/Downloads/IsaacLab/source/extensions/omni.isaac.lab_assets/data/Vehicles/F1Car/f1_car.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.05),  # Slightly above ground to prevent initial penetration
        joint_pos={
            "a_car_1_left_steering_hinge_joint": 0.0,
            "a_car_1_right_steering_hinge_joint": 0.0,
            "a_car_1_left_front_wheel_joint": 0.0,
            "a_car_1_right_front_wheel_joint": 0.0,
            "a_car_1_left_rear_wheel_joint": 0.0,
            "a_car_1_right_rear_wheel_joint": 0.0,
        },
    ),
    actuators={
        "steering": ImplicitActuatorCfg(
            joint_names_expr=[".*_steering_hinge_joint"],
            stiffness=1000.0,
            damping=100.0,
        ),
        "drive": ImplicitActuatorCfg(
            joint_names_expr=[".*_wheel_joint"],
            stiffness=0.0,
            damping=10.0,
        ),
    },
)
"""Configuration for the F1 Car."""