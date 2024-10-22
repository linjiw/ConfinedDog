# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import torch
import trimesh
from typing import TYPE_CHECKING, Tuple
import os
import warp
from pxr import UsdGeom

import omni.isaac.lab.sim as sim_utils
import omni.isaac.core.utils.prims as prim_utils

from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
from omni.isaac.lab.utils.warp import convert_to_warp_mesh

from .terrain_generator import TerrainGenerator
from .trimesh.utils import make_plane
from .utils import create_prim_from_mesh
from .height_field.utils import convert_height_field_to_mesh
# from .map_generator import height_field_to_mesh


if TYPE_CHECKING:
    from .terrain_importer_cfg import TerrainImporterCfg


class TerrainImporter:
    r"""A class to handle terrain meshes and import them into the simulator.

    We assume that a terrain mesh comprises of sub-terrains that are arranged in a grid with
    rows ``num_rows`` and columns ``num_cols``. The terrain origins are the positions of the sub-terrains
    where the robot should be spawned.

    Based on the configuration, the terrain importer handles computing the environment origins from the sub-terrain
    origins. In a typical setup, the number of sub-terrains (:math:`num\_rows \times num\_cols`) is smaller than
    the number of environments (:math:`num\_envs`). In this case, the environment origins are computed by
    sampling the sub-terrain origins.

    If a curriculum is used, it is possible to update the environment origins to terrain origins that correspond
    to a harder difficulty. This is done by calling :func:`update_terrain_levels`. The idea comes from game-based
    curriculum. For example, in a game, the player starts with easy levels and progresses to harder levels.
    """

    meshes: dict[str, trimesh.Trimesh]
    """A dictionary containing the names of the meshes and their keys."""
    warp_meshes: dict[str, warp.Mesh]
    """A dictionary containing the names of the warp meshes and their keys."""
    terrain_origins: torch.Tensor | None
    """The origins of the sub-terrains in the added terrain mesh. Shape is (num_rows, num_cols, 3).

    If None, then it is assumed no sub-terrains exist. The environment origins are computed in a grid.
    """
    env_origins: torch.Tensor
    """The origins of the environments. Shape is (num_envs, 3)."""

    def __init__(self, cfg: TerrainImporterCfg):
        """Initialize the terrain importer.

        Args:
            cfg: The configuration for the terrain importer.

        Raises:
            ValueError: If input terrain type is not supported.
            ValueError: If terrain type is 'generator' and no configuration provided for ``terrain_generator``.
            ValueError: If terrain type is 'usd' and no configuration provided for ``usd_path``.
            ValueError: If terrain type is 'usd' or 'plane' and no configuration provided for ``env_spacing``.
        """
        # store inputs
        self.cfg = cfg
        self.device = sim_utils.SimulationContext.instance().device  # type: ignore

        # create a dict of meshes
        self.meshes = dict()
        self.warp_meshes = dict()
        self.env_origins = None
        self.terrain_origins = None
        # private variables
        self._terrain_flat_patches = dict()

        # auto-import the terrain based on the config
        if self.cfg.terrain_type == "generator":
            # check config is provided
            if self.cfg.terrain_generator is None:
                raise ValueError("Input terrain type is 'generator' but no value provided for 'terrain_generator'.")
            # generate the terrain
            terrain_generator = TerrainGenerator(cfg=self.cfg.terrain_generator, device=self.device)
            self.import_mesh("terrain", terrain_generator.terrain_mesh)
            # configure the terrain origins based on the terrain generator
            self.configure_env_origins(terrain_generator.terrain_origins)
            # refer to the flat patches
            self._terrain_flat_patches = terrain_generator.flat_patches
        elif self.cfg.terrain_type == "usd":
            # check if config is provided
            if self.cfg.usd_path is None:
                raise ValueError("Input terrain type is 'usd' but no value provided for 'usd_path'.")
            # import the terrain
            self.import_usd("terrain", self.cfg.usd_path)
            # configure the origins in a grid
            self.configure_env_origins()
        elif self.cfg.terrain_type == "plane":
            # load the plane
            self.import_ground_plane("terrain")
            # configure the origins in a grid
            self.configure_env_origins()
        elif self.cfg.terrain_type == "height_field":
            # check if config is provided
            if self.cfg.ground_folder is None or self.cfg.ceiling_folder is None:
                raise ValueError("Input terrain type is 'height_field' but no value provided for 'height_field_folder'.")
            # import the height field terrain
            self.import_height_field_folder(self.cfg.ground_folder, self.cfg.ceiling_folder, self.cfg.grid_size, self.cfg.terrain_size)
            # configure_env_origins is called within import_height_field_folder
        # write elif for constrained space
        elif self.cfg.terrain_type == "constrained_space":
            # check if config is provided
            if self.cfg.ground_folder is None or self.cfg.ceiling_folder is None:
                raise ValueError("Input terrain type is 'constrained_space' but no value provided for 'ground_folder' or 'ceiling_folder'.")
            # import the constrained space terrain
            self.import_constrained_space(self.cfg.ground_folder, self.cfg.ceiling_folder, self.cfg.grid_size, self.cfg.terrain_size, self.cfg.ceiling_height)
            # configure_env_origins is called within import_constrained_space
        elif self.cfg.terrain_type == "f1tenth":
            # check if config is provided
            if self.cfg.map_folder is None:
                raise ValueError("Input terrain type is 'f1tenth' but no value provided for 'map_folder'.")
            # import the F1TENTH maps
            self.import_f1tenth_maps(self.cfg.map_folder, self.cfg.grid_size, self.cfg.terrain_size)
            # configure_env_origins is called within import_f1tenth_maps


        else:
            raise ValueError(f"Terrain type '{self.cfg.terrain_type}' not available.")

        # set initial state of debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

    """
    Properties.
    """

    @property
    def has_debug_vis_implementation(self) -> bool:
        """Whether the terrain importer has a debug visualization implemented.

        This always returns True.
        """
        return True

    @property
    def flat_patches(self) -> dict[str, torch.Tensor]:
        """A dictionary containing the sampled valid (flat) patches for the terrain.

        This is only available if the terrain type is 'generator'. For other terrain types, this feature
        is not available and the function returns an empty dictionary.

        Please refer to the :attr:`TerrainGenerator.flat_patches` for more information.
        """
        return self._terrain_flat_patches

    """
    Operations - Visibility.
    """

    def set_debug_vis(self, debug_vis: bool) -> bool:
        """Set the debug visualization of the terrain importer.

        Args:
            debug_vis: Whether to visualize the terrain origins.

        Returns:
            Whether the debug visualization was successfully set. False if the terrain
            importer does not support debug visualization.

        Raises:
            RuntimeError: If terrain origins are not configured.
        """
        # create a marker if necessary
        if debug_vis:
            if not hasattr(self, "origin_visualizer"):
                self.origin_visualizer = VisualizationMarkers(
                    cfg=FRAME_MARKER_CFG.replace(prim_path="/Visuals/TerrainOrigin")
                )
                if self.terrain_origins is not None:
                    self.origin_visualizer.visualize(self.terrain_origins.reshape(-1, 3))
                elif self.env_origins is not None:
                    self.origin_visualizer.visualize(self.env_origins.reshape(-1, 3))
                else:
                    raise RuntimeError("Terrain origins are not configured.")
            # set visibility
            self.origin_visualizer.set_visibility(True)
        else:
            if hasattr(self, "origin_visualizer"):
                self.origin_visualizer.set_visibility(False)
        # report success
        return True

    """
    Operations - Import.
    """

    def import_ground_plane(self, key: str, size: tuple[float, float] = (2.0e6, 2.0e6)):
        """Add a plane to the terrain importer.

        Args:
            key: The key to store the mesh.
            size: The size of the plane. Defaults to (2.0e6, 2.0e6).

        Raises:
            ValueError: If a terrain with the same key already exists.
        """
        # check if key exists
        if key in self.meshes:
            raise ValueError(f"Mesh with key {key} already exists. Existing keys: {self.meshes.keys()}.")
        # create a plane
        mesh = make_plane(size, height=0.0, center_zero=True)
        # store the mesh
        self.meshes[key] = mesh
        # create a warp mesh
        device = "cuda" if "cuda" in self.device else "cpu"
        self.warp_meshes[key] = convert_to_warp_mesh(mesh.vertices, mesh.faces, device=device)

        # get the mesh
        ground_plane_cfg = sim_utils.GroundPlaneCfg(physics_material=self.cfg.physics_material, size=size)
        ground_plane_cfg.func(self.cfg.prim_path, ground_plane_cfg)

    def import_mesh(self, key: str, mesh: trimesh.Trimesh):
        """Import a mesh into the simulator.

        The mesh is imported into the simulator under the prim path ``cfg.prim_path/{key}``. The created path
        contains the mesh as a :class:`pxr.UsdGeom` instance along with visual or physics material prims.

        Args:
            key: The key to store the mesh.
            mesh: The mesh to import.

        Raises:
            ValueError: If a terrain with the same key already exists.
        """
        # check if key exists
        if key in self.meshes:
            raise ValueError(f"Mesh with key {key} already exists. Existing keys: {self.meshes.keys()}.")
        # store the mesh
        self.meshes[key] = mesh
        # create a warp mesh
        device = "cuda" if "cuda" in self.device else "cpu"
        self.warp_meshes[key] = convert_to_warp_mesh(mesh.vertices, mesh.faces, device=device)

        # get the mesh
        mesh = self.meshes[key]
        mesh_prim_path = self.cfg.prim_path + f"/{key}"
        # import the mesh
        create_prim_from_mesh(
            mesh_prim_path,
            mesh,
            visual_material=self.cfg.visual_material,
            physics_material=self.cfg.physics_material,
        )

    def import_usd(self, key: str, usd_path: str):
        """Import a mesh from a USD file.

        We assume that the USD file contains a single mesh. If the USD file contains multiple meshes, then
        the first mesh is used. The function mainly helps in registering the mesh into the warp meshes
        and the meshes dictionary.

        Note:
            We do not apply any material properties to the mesh. The material properties should
            be defined in the USD file.

        Args:
            key: The key to store the mesh.
            usd_path: The path to the USD file.

        Raises:
            ValueError: If a terrain with the same key already exists.
        """
        # add mesh to the dict
        if key in self.meshes:
            raise ValueError(f"Mesh with key {key} already exists. Existing keys: {self.meshes.keys()}.")
        # add the prim path
        cfg = sim_utils.UsdFileCfg(usd_path=usd_path)
        cfg.func(self.cfg.prim_path + f"/{key}", cfg)

        # traverse the prim and get the collision mesh
        # THINK: Should the user specify the collision mesh?
        mesh_prim = sim_utils.get_first_matching_child_prim(
            self.cfg.prim_path + f"/{key}", lambda prim: prim.GetTypeName() == "Mesh"
        )
        # check if the mesh is valid
        if mesh_prim is None:
            raise ValueError(f"Could not find any collision mesh in {usd_path}. Please check asset.")
        # cast into UsdGeomMesh
        mesh_prim = UsdGeom.Mesh(mesh_prim)
        # store the mesh
        vertices = np.asarray(mesh_prim.GetPointsAttr().Get())
        faces = np.asarray(mesh_prim.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        self.meshes[key] = trimesh.Trimesh(vertices=vertices, faces=faces)
        # create a warp mesh
        device = "cuda" if "cuda" in self.device else "cpu"
        self.warp_meshes[key] = convert_to_warp_mesh(vertices, faces, device=device)

    """
    Operations - Origins.
    """

    def configure_env_origins(self, origins: np.ndarray | None = None):
        """Configure the origins of the environments based on the added terrain.

        Args:
            origins: The origins of the sub-terrains. Shape is (num_rows, num_cols, 3).
        """
        # decide whether to compute origins in a grid or based on curriculum
        if origins is not None:
            # convert to numpy
            if isinstance(origins, np.ndarray):
                origins = torch.from_numpy(origins)
            # store the origins
            self.terrain_origins = origins.to(self.device, dtype=torch.float)
            # compute environment origins
            self.env_origins = self._compute_env_origins_curriculum(self.cfg.num_envs, self.terrain_origins)
        else:
            self.terrain_origins = None
            # check if env spacing is valid
            if self.cfg.env_spacing is None:
                raise ValueError("Environment spacing must be specified for configuring grid-like origins.")
            # compute environment origins
            self.env_origins = self._compute_env_origins_grid(self.cfg.num_envs, self.cfg.env_spacing)

    def update_env_origins(self, env_ids: torch.Tensor, move_up: torch.Tensor, move_down: torch.Tensor):
        """Update the environment origins based on the terrain levels."""
        # check if grid-like spawning
        if self.terrain_origins is None:
            return
        # update terrain level for the envs
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # robots that solve the last level are sent to a random one
        # the minimum level is zero
        self.terrain_levels[env_ids] = torch.where(
            self.terrain_levels[env_ids] >= self.max_terrain_level,
            torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
            torch.clip(self.terrain_levels[env_ids], 0),
        )
        # update the env origins
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    """
    Internal helpers.
    """

    def _compute_env_origins_curriculum(self, num_envs: int, origins: torch.Tensor) -> torch.Tensor:
        """Compute the origins of the environments defined by the sub-terrains origins."""
        # extract number of rows and cols
        num_rows, num_cols = origins.shape[:2]
        # maximum initial level possible for the terrains
        if self.cfg.max_init_terrain_level is None:
            max_init_level = num_rows - 1
        else:
            max_init_level = min(self.cfg.max_init_terrain_level, num_rows - 1)
        # store maximum terrain level possible
        self.max_terrain_level = num_rows
        # define all terrain levels and types available
        self.terrain_levels = torch.randint(0, max_init_level + 1, (num_envs,), device=self.device)
        self.terrain_types = torch.div(
            torch.arange(num_envs, device=self.device),
            (num_envs / num_cols),
            rounding_mode="floor",
        ).to(torch.long)
        # create tensor based on number of environments
        env_origins = torch.zeros(num_envs, 3, device=self.device)
        env_origins[:] = origins[self.terrain_levels, self.terrain_types]
        return env_origins

    def _compute_env_origins_grid(self, num_envs: int, env_spacing: float) -> torch.Tensor:
        """Compute the origins of the environments in a grid based on configured spacing."""
        # create tensor based on number of environments
        env_origins = torch.zeros(num_envs, 3, device=self.device)
        # create a grid of origins
        num_rows = np.ceil(num_envs / int(np.sqrt(num_envs)))
        num_cols = np.ceil(num_envs / num_rows)
        ii, jj = torch.meshgrid(
            torch.arange(num_rows, device=self.device), torch.arange(num_cols, device=self.device), indexing="ij"
        )
        env_origins[:, 0] = -(ii.flatten()[:num_envs] - (num_rows - 1) / 2) * env_spacing
        env_origins[:, 1] = (jj.flatten()[:num_envs] - (num_cols - 1) / 2) * env_spacing
        env_origins[:, 2] = 0.0
        return env_origins

    def import_constrained_space(self, ground_folder: str, ceiling_folder: str, grid_size: Tuple[int, int], terrain_size: Tuple[float, float], ceiling_height: float):
        """
        Import height field .npy files for both ground and ceiling, convert them to terrain meshes,
        and arrange them in a grid to create a constrained space.

        Args:
            ground_folder (str): Path to the folder containing ground height field .npy files.
            ceiling_folder (str): Path to the folder containing ceiling height field .npy files.
            grid_size (Tuple[int, int]): Number of rows and columns in the grid.
            terrain_size (Tuple[float, float]): Size of each terrain piece (width, length).
            ceiling_height (float): Height at which to place the ceiling terrain.

        Returns:
            None
        """
        # Import ground terrain
        ground_mesh = self._import_height_field_folder(ground_folder, grid_size, terrain_size)

        # Import ceiling terrain
        ceiling_mesh = self._import_height_field_folder(ceiling_folder, grid_size, terrain_size)

        # Move ceiling terrain to the specified height and invert it
        ceiling_transform = np.eye(4)
        ceiling_transform[2, 3] = ceiling_height
        ceiling_transform[2, 2] = -1  # Invert the z-axis
        ceiling_mesh.apply_transform(ceiling_transform)

        # Combine ground and ceiling meshes
        combined_mesh = trimesh.util.concatenate([ground_mesh, ceiling_mesh])

        # Import the combined mesh
        self.import_mesh("terrain", combined_mesh)



        # Configure environment origins
        num_envs = grid_size[0] * grid_size[1]
        env_origins = torch.zeros((num_envs, 3), device=self.device)
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                env_index = i * grid_size[1] + j
                env_origins[env_index, 0] = i * terrain_size[0] + terrain_size[0] / 2
                env_origins[env_index, 1] = j * terrain_size[1] + terrain_size[1] / 2

        self.env_origins = env_origins

        # Configure environment origins
        # self._configure_env_origins(grid_size, terrain_size)

    def _import_height_field_folder(self, folder_path: str, grid_size: Tuple[int, int], terrain_size: Tuple[float, float]) -> trimesh.Trimesh:
        """
        Import height field .npy files from a folder, convert them to terrain meshes,
        and arrange them in a grid.

        Args:
            folder_path (str): Path to the folder containing height field .npy files.
            grid_size (Tuple[int, int]): Number of rows and columns in the grid.
            terrain_size (Tuple[float, float]): Size of each terrain piece (width, length).

        Returns:
            trimesh.Trimesh: The combined mesh of all terrain pieces.
        """
        # ... (existing code from import_height_field_folder)
        # Instead of importing the mesh, return the combined_mesh
        # Get list of .npy files in the folder
        height_field_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
        
        if len(height_field_files) < grid_size[0] * grid_size[1]:
            raise ValueError(f"Not enough height field files in {folder_path} to fill the grid.")

        # Shuffle the files to randomize placement
        np.random.shuffle(height_field_files)

        # Create a grid of terrains
        grid_terrains = []
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                file_index = i * grid_size[1] + j
                height_field = np.load(os.path.join(folder_path, height_field_files[file_index]))
                
                # Convert height field to mesh
                mesh = self._height_field_to_mesh(height_field, terrain_size)
                
                # Position the mesh in the grid
                transform = np.eye(4)
                transform[0, 3] = i * terrain_size[0]
                transform[1, 3] = j * terrain_size[1]
                mesh.apply_transform(transform)
                
                grid_terrains.append(mesh)

        # Combine all meshes
        combined_mesh = trimesh.util.concatenate(grid_terrains)
        return combined_mesh

    def _configure_env_origins(self, grid_size: Tuple[int, int], terrain_size: Tuple[float, float]):
        """Configure environment origins for the constrained space."""
        num_envs = grid_size[0] * grid_size[1]
        env_origins = torch.zeros((num_envs, 3), device=self.device)
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                env_index = i * grid_size[1] + j
                env_origins[env_index, 0] = i * terrain_size[0] + terrain_size[0] / 2
                env_origins[env_index, 1] = j * terrain_size[1] + terrain_size[1] / 2

        self.env_origins = env_origins
        self.terrain_origins = env_origins.reshape(grid_size[0], grid_size[1], 3)


    def delete_terrain(self, key: str):
        """Delete an imported terrain.

        This function removes the terrain from the simulator and cleans up associated data structures.

        Args:
            key: The key of the terrain to delete.

        Raises:
            KeyError: If the terrain with the given key does not exist.
        """
        # Check if the terrain exists
        if key not in self.meshes:
            raise KeyError(f"Terrain with key '{key}' does not exist.")

        # Remove the terrain from the simulator
        terrain_prim_path = f"{self.cfg.prim_path}/{key}"
        prim_utils.delete_prim(terrain_prim_path)

        # Remove the terrain from our data structures
        del self.meshes[key]
        del self.warp_meshes[key]

        # If we're deleting the main terrain, reset related attributes
        if key == "terrain":
            self.terrain_origins = None
            self.env_origins = None
            self.terrain_levels = None
            self.terrain_types = None
            self._terrain_flat_patches = dict()

        # Reset visualization if necessary
        try:
            if hasattr(self, "origin_visualizer"):
                self.origin_visualizer.clear()
                del self.origin_visualizer
        except:
            pass

        print(f"Terrain '{key}' has been deleted.")

    # def import_height_field_folder(self, folder_path: str, grid_size: Tuple[int, int], terrain_size: Tuple[float, float], env_spacing: float):
    #     """
    #     Import height field .npy files from a folder, convert them to terrain meshes,
    #     and arrange them in a grid.

    #     Args:
    #         folder_path (str): Path to the folder containing height field .npy files.
    #         grid_size (Tuple[int, int]): Number of rows and columns in the height field grid (e.g., (16, 16)).
    #         terrain_size (Tuple[float, float]): Overall size of the terrain (e.g., (16.0, 4.0) or (32.0, 8.0)).
    #         env_spacing (float): Spacing between environment origins.

    #     Returns:
    #         None
    #     """
    #     # Calculate the size of each terrain piece
    #     piece_size = (terrain_size[0] / grid_size[0], terrain_size[1] / grid_size[1])

    #     # Get list of .npy files in the folder
    #     height_field_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
        
    #     if len(height_field_files) < grid_size[0] * grid_size[1]:
    #         raise ValueError(f"Not enough height field files in {folder_path} to fill the grid.")

    #     # Shuffle the files to randomize placement
    #     np.random.shuffle(height_field_files)

    #     # Create a grid of terrains
    #     grid_terrains = []
    #     for i in range(grid_size[0]):
    #         for j in range(grid_size[1]):
    #             file_index = i * grid_size[1] + j
    #             height_field = np.load(os.path.join(folder_path, height_field_files[file_index]))
                
    #             # Convert height field to mesh
    #             mesh = self._height_field_to_mesh(height_field, piece_size)
                
    #             # Position the mesh in the grid
    #             transform = np.eye(4)
    #             transform[0, 3] = i * piece_size[0]
    #             transform[1, 3] = j * piece_size[1]
    #             mesh.apply_transform(transform)
                
    #             grid_terrains.append(mesh)

    #     # Combine all meshes
    #     ground_mesh = trimesh.util.concatenate(grid_terrains)

    #     # Create ceiling mesh
    #     ceiling_mesh = ground_mesh.copy()
    #     ceiling_transform = np.eye(4)
    #     ceiling_transform[2, 3] = 2.8
    #     ceiling_transform[2, 2] = -1  # Invert the z-axis
    #     ceiling_mesh.apply_transform(ceiling_transform)

    #     # Combine ground and ceiling meshes
    #     combined_mesh = trimesh.util.concatenate([ground_mesh, ceiling_mesh])

    #     # Import the combined mesh
    #     self.import_mesh("terrain", combined_mesh)

    #     # Configure environment origins
    #     num_envs = int((terrain_size[0] / env_spacing) * (terrain_size[1] / env_spacing))
    #     env_origins = torch.zeros((num_envs, 3), device=self.device)
        
    #     env_grid_size = (int(terrain_size[0] / env_spacing), int(terrain_size[1] / env_spacing))
    #     for i in range(env_grid_size[0]):
    #         for j in range(env_grid_size[1]):
    #             env_index = i * env_grid_size[1] + j
    #             env_origins[env_index, 0] = i * env_spacing + env_spacing / 2
    #             env_origins[env_index, 1] = j * env_spacing + env_spacing / 2

    #     self.env_origins = env_origins
    #     self.terrain_origins = env_origins.reshape(env_grid_size[0], env_grid_size[1], 3)

    #     print(f"Terrain imported with size {terrain_size}, grid size {grid_size}, and {num_envs} environment origins.")

    def import_height_field_folder(self, ground_folder: str, ceiling_folder: str, grid_size: Tuple[int, int], terrain_size: Tuple[float, float]):
        """
        Import height field .npy files from ground and ceiling folders, convert them to terrain meshes,
        and arrange them in a grid.

        Args:
            ground_folder (str): Path to the folder containing ground height field .npy files.
            ceiling_folder (str): Path to the folder containing ceiling height field .npy files.
            grid_size (Tuple[int, int]): Number of rows and columns in the grid.
            terrain_size (Tuple[float, float]): Size of each terrain piece (width, length).

        Returns:
            None
        """
        # Get list of .npy files in the ground folder
        height_field_files = [f for f in os.listdir(ground_folder) if f.endswith('.npy')]
        
        if len(height_field_files) < grid_size[0] * grid_size[1]:
            raise ValueError(f"Not enough height field files in {ground_folder} to fill the grid.")

        horizontal_scale = 1.0
        vertical_scale = 0.9
        slope_threshold = 0.2

        # Create a grid of ground terrains
        grid_terrains = []
        print(f"terrain size: {terrain_size}")
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                file_index = i * grid_size[1] + j
                height_field = np.load(os.path.join(ground_folder, height_field_files[file_index]))
                
                # Convert height field to mesh
                vertices, triangles  = convert_height_field_to_mesh(height_field, horizontal_scale, vertical_scale, slope_threshold)
                mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
                
                # Position the mesh in the grid
                transform = np.eye(4)
                transform[0, 3] = i * terrain_size[0]
                transform[1, 3] = j * terrain_size[1]
                mesh.apply_transform(transform)
                
                grid_terrains.append(mesh)

        # Combine all ground meshes
        ground_mesh = trimesh.util.concatenate(grid_terrains)

        # Get list of .npy files in the ceiling folder
        ceiling_files = [f for f in os.listdir(ceiling_folder) if f.endswith('.npy')]
        if len(ceiling_files) < grid_size[0] * grid_size[1]:
            raise ValueError(f"Not enough height field files in {ceiling_folder} to fill the grid.")

        # Create a grid of ceiling terrains
        ceiling_grid_terrains = []
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                file_index = i * grid_size[1] + j
                height_field = np.load(os.path.join(ceiling_folder, ceiling_files[file_index]))
                
                # Convert height field to mesh
                vertices, triangles = convert_height_field_to_mesh(height_field, horizontal_scale, vertical_scale, slope_threshold)
                mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
                
                # Position the mesh in the grid
                transform = np.eye(4)
                transform[0, 3] = i * terrain_size[0]
                transform[1, 3] = j * terrain_size[1]
                mesh.apply_transform(transform)

                ceiling_grid_terrains.append(mesh)
        
        # Combine all ceiling meshes
        ceiling_mesh = trimesh.util.concatenate(ceiling_grid_terrains)

        # Invert the z-axis and raise the ceiling mesh
        ceiling_transform = np.eye(4)
        ceiling_transform[2, 3] = 0.7 #0.7
        ceiling_transform[2, 2] = -1  # Invert the z-axis
        ceiling_mesh.apply_transform(ceiling_transform)

        # Import the ground and ceiling meshes
        self.import_mesh("ground", ground_mesh)
        self.import_mesh("ceiling", ceiling_mesh)
        
        # Configure environment origins
        # terrain_size = (8,8)
        num_envs = grid_size[0] * grid_size[1]
        env_origins = torch.zeros((num_envs, 3), device=self.device)
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                env_index = i * grid_size[1] + j
                env_origins[env_index, 0] = i * terrain_size[0] + terrain_size[0] / 2
                env_origins[env_index, 1] = j * terrain_size[1] + terrain_size[1] / 2

        self.env_origins = env_origins
        self.terrain_origins = env_origins.reshape(grid_size[0], grid_size[1], 3)

    def _height_field_to_mesh(self, height_field: np.ndarray, terrain_size: Tuple[float, float], base_height = -0.5) -> trimesh.Trimesh:
        """
        Convert a height field to a trimesh.

        Args:
            height_field (np.ndarray): 2D array representing the height field.
            terrain_size (Tuple[float, float]): Size of the terrain (width, length).

        Returns:
            trimesh.Trimesh: The resulting mesh.
        """
        # rows, cols = height_field.shape
        # vertices = []
        # faces = []

        # for i in range(rows):
        #     for j in range(cols):
        #         x = j * terrain_size[0] / (cols - 1)
        #         y = i * terrain_size[1] / (rows - 1)
        #         z = height_field[i, j]
        #         vertices.append([x, y, z])

        # vertices = np.array(vertices)

        # for i in range(rows - 1):
        #     for j in range(cols - 1):
        #         v0 = i * cols + j
        #         v1 = v0 + 1
        #         v2 = (i + 1) * cols + j
        #         v3 = v2 + 1
        #         faces.extend([[v0, v2, v1], [v1, v2, v3]])

        # faces = np.array(faces)

        # mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            
        # # Calculate vertex normals
        # mesh.vertex_normals


        # return mesh
    # Create top surface vertices
        rows, cols = height_field.shape
        vertices = []
        faces = []
        for i in range(rows):
            for j in range(cols):
                x = j * terrain_size[0] / (cols - 1)
                y = i * terrain_size[1] / (rows - 1)
                z = height_field[i, j]
                vertices.append([x, y, z])

        # Create bottom surface vertices
        for i in range(rows):
            for j in range(cols):
                x = j * terrain_size[0] / (cols - 1)
                y = i * terrain_size[1] / (rows - 1)
                z = base_height
                vertices.append([x, y, z])

        vertices = np.array(vertices)

        # Create top surface faces
        for i in range(rows - 1):
            for j in range(cols - 1):
                v0 = i * cols + j
                v1 = v0 + 1
                v2 = (i + 1) * cols + j
                v3 = v2 + 1
                faces.extend([[v0, v2, v1], [v1, v2, v3]])

        # Create bottom surface faces (inverted)
        bottom_offset = rows * cols
        for i in range(rows - 1):
            for j in range(cols - 1):
                v0 = bottom_offset + i * cols + j
                v1 = v0 + 1
                v2 = bottom_offset + (i + 1) * cols + j
                v3 = v2 + 1
                faces.extend([[v0, v1, v2], [v1, v3, v2]])

        # Create side faces to connect top and bottom
        for i in range(rows):
            v_top = i * cols
            v_bottom = bottom_offset + i * cols
            v_top_next = ((i + 1) % rows) * cols
            v_bottom_next = bottom_offset + ((i + 1) % rows) * cols
            faces.extend([[v_top, v_bottom, v_top_next], [v_bottom, v_bottom_next, v_top_next]])

        for j in range(cols):
            v_top = j
            v_bottom = bottom_offset + j
            v_top_next = (j + 1) % cols
            v_bottom_next = bottom_offset + (j + 1) % cols
            faces.extend([[v_top, v_top_next, v_bottom], [v_top_next, v_bottom_next, v_bottom]])

        faces = np.array(faces)

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # Ensure consistent face winding and normals
        mesh.fix_normals()

        return mesh

    def import_f1tenth_maps(self, map_folder: str, grid_size: Tuple[int, int], terrain_size: Tuple[float, float]):
        """
        Import F1TENTH map .npy files from a folder, convert them to terrain meshes,
        and arrange them in a grid.

        Args:
            map_folder (str): Path to the folder containing F1TENTH map .npy files.
            grid_size (Tuple[int, int]): Number of rows and columns in the grid.
            terrain_size (Tuple[float, float]): Size of each terrain piece (width, length).

        Returns:
            None
        """
        # Get list of .npy files in the map folder
        map_files = [f for f in os.listdir(map_folder) if f.endswith('.npy')]
        
        if len(map_files) < grid_size[0] * grid_size[1]:
            raise ValueError(f"Not enough map files in {map_folder} to fill the grid.")

        horizontal_scale = terrain_size[0] / 256  # Assuming 256x256 maps
        vertical_scale = 0.1  # 10cm height scale

        # Create a grid of terrains
        grid_terrains = []
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                file_index = i * grid_size[1] + j
                height_field = np.load(os.path.join(map_folder, map_files[file_index]))
                
                # Convert height field to mesh
                mesh = height_field_to_mesh(height_field, horizontal_scale, vertical_scale)
                
                # Position the mesh in the grid
                transform = np.eye(4)
                transform[0, 3] = i * terrain_size[0]
                transform[1, 3] = j * terrain_size[1]
                mesh.apply_transform(transform)
                
                grid_terrains.append(mesh)

        # Combine all meshes
        combined_mesh = trimesh.util.concatenate(grid_terrains)

        # Import the combined mesh
        self.import_mesh("f1tenth_terrain", combined_mesh)
        
        # Configure environment origins
        num_envs = grid_size[0] * grid_size[1]
        env_origins = torch.zeros((num_envs, 3), device=self.device)
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                env_index = i * grid_size[1] + j
                env_origins[env_index, 0] = i * terrain_size[0] + terrain_size[0] / 2
                env_origins[env_index, 1] = j * terrain_size[1] + terrain_size[1] / 2

        self.env_origins = env_origins
        self.terrain_origins = env_origins.reshape(grid_size[0], grid_size[1], 3)