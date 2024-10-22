# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
F1TENTH racing environment.
"""

import gymnasium as gym

from . import agents
from .f1_env import F1TenthEnv, F1TenthEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-F1TENTH-Direct-v0",
    entry_point="omni.isaac.lab_tasks.direct.f1tenth:F1TenthEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": F1TenthEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_f1tenth_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:F1TenthPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_f1tenth_ppo_cfg.yaml",
    },
)