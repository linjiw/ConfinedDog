# import os
# import numpy as np
# import torch
# # from omni.isaac.lab.terrains import TerrainImporter, TerrainImporterCfg
# # from omni.isaac.lab_tasks.direct.anymal_c.anymal_c_env import AnymalCEnv, AnymalCRoughEnvCfg

# def generate_random_height_fields(folder_path, num_fields, size):
#     """Generate random height field files for testing."""
#     os.makedirs(folder_path, exist_ok=True)
#     for i in range(num_fields):
#         height_field = np.random.uniform(0, 0.1, size)
#         np.save(os.path.join(folder_path, f"height_field_{i}.npy"), height_field)

# height_field_folder = "test_height_fields_plane"
# generate_random_height_fields(height_field_folder, 256, (32, 32))
import os
import numpy as np

def generate_random_height_fields(folder_path, num_fields, size, height_range=(0, 0.1)):
    """Generate random height field files for testing."""
    os.makedirs(folder_path, exist_ok=True)
    for i in range(num_fields):
        height_field = np.random.uniform(height_range[0], height_range[1], size)
        np.save(os.path.join(folder_path, f"height_field_{i}.npy"), height_field)

# Generate ground terrain
difficulty = 0.15
ground_folder = f"test_height_fields_ground_{difficulty}"
generate_random_height_fields(ground_folder, 1024, (32, 32), (0, difficulty))

# Generate ceiling terrain (inverted height field)
ceiling_folder = f"test_height_fields_ceiling_{difficulty}"
generate_random_height_fields(ceiling_folder, 1024, (32, 32), (-difficulty, 0))