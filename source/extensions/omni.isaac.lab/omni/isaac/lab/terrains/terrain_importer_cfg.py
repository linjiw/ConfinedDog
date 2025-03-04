# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING, Literal, Tuple

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.utils import configclass

from .terrain_importer import TerrainImporter

if TYPE_CHECKING:
    from .terrain_generator_cfg import TerrainGeneratorCfg


@configclass
class TerrainImporterCfg:
    """Configuration for the terrain manager."""

    class_type: type = TerrainImporter
    """The class to use for the terrain importer.

    Defaults to :class:`omni.isaac.lab.terrains.terrain_importer.TerrainImporter`.
    """

    collision_group: int = -1
    """The collision group of the terrain. Defaults to -1."""

    prim_path: str = MISSING
    """The absolute path of the USD terrain prim.

    All sub-terrains are imported relative to this prim path.
    """

    num_envs: int = MISSING
    """The number of environment origins to consider."""

    terrain_type: Literal["generator", "usd", "plane", "height_field", "constrained_space", "f1tenth"] = "plane"
    """The type of terrain to import. Defaults to "plane".

    Available options are "plane", "usd", "height_field", "constrained_space", and "f1tenth".
    """

    ceiling_height: float | None = None
    """The height of the ceiling. Required if terrain_type is 'constrained_space'."""

    ground_folder: str | None = None
    """The folder containing ground height field .npy files. Required if terrain_type is 'height_field'."""

    ceiling_folder: str | None = None
    """The folder containing ceiling height field .npy files. Required if terrain_type is 'height_field'."""

    height_field_folder: str | None = None
    """The folder containing height field .npy files. Required if terrain_type is 'height_field'."""

    grid_size: Tuple[int, int] = (1, 1)
    """The size of the grid for arranging terrains. Required if terrain_type is 'height_field'."""

    terrain_size: Tuple[float, float] = (25.6, 25.6)
    """The size of each terrain piece. Required if terrain_type is 'height_field'."""

    terrain_generator: TerrainGeneratorCfg | None = None
    """The terrain generator configuration.

    Only used if ``terrain_type`` is set to "generator".
    """

    usd_path: str | None = None
    """The path to the USD file containing the terrain.

    Only used if ``terrain_type`` is set to "usd".
    """

    env_spacing: float | None = None
    """The spacing between environment origins when defined in a grid. Defaults to None.

    Note:
      This parameter is used only when the ``terrain_type`` is ``"plane"`` or ``"usd"``.
    """

    visual_material: sim_utils.VisualMaterialCfg | None = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.065, 0.0725, 0.080)
    )
    """The visual material of the terrain. Defaults to a dark gray color material.

    The material is created at the path: ``{prim_path}/visualMaterial``. If `None`, then no material is created.

    .. note::
        This parameter is used only when the ``terrain_type`` is ``"generator"``.
    """

    physics_material: sim_utils.RigidBodyMaterialCfg = sim_utils.RigidBodyMaterialCfg()
    """The physics material of the terrain. Defaults to a default physics material.

    The material is created at the path: ``{prim_path}/physicsMaterial``.

    .. note::
        This parameter is used only when the ``terrain_type`` is ``"generator"`` or ``"plane"``.
    """

    max_init_terrain_level: int | None = None
    """The maximum initial terrain level for defining environment origins. Defaults to None.

    The terrain levels are specified by the number of rows in the grid arrangement of
    sub-terrains. If None, then the initial terrain level is set to the maximum
    terrain level available (``num_rows - 1``).

    Note:
      This parameter is used only when sub-terrain origins are defined.
    """

    debug_vis: bool = False
    """Whether to enable visualization of terrain origins for the terrain. Defaults to False."""

    map_folder: str | None = None
    """The folder containing F1TENTH map files (for 'f1tenth' terrain type)."""
