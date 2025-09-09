import os, cv2
import numpy as np
import json
import pickle
from PIL import Image
import trimesh
from training.deformers.smplx import SMPL
import torch

pose_params = torch.zeros([1, 72], dtype=torch.float32)

# Set axis-angle rotation vectors.
# In the SMPL model, pose parameters 4 and 5 represent the left arm rotation,
# and pose parameters 7 and 8 represent the right arm rotation.
# In the basic SMPL model, these parameters correspond to axis-angle rotations in radians.
# Typically, the Y-axis points downward; we define "downward along the side of the body" as positive.
# Therefore:
#   - If the arm moves backward (away from the torso), the rotation is positive.
#   - If the arm moves forward (toward the torso), the rotation is negative.

# Convert degrees to radians using numpy
angle_left = np.radians(-45)
angle_right = np.radians(52)

# Assuming the torso faces forward, and the joint rotation axes are along the lateral direction (Y-axis),
# we need to set the pose parameters of the left and right shoulders accordingly.
left_shoulder_idx = 16*3+2
right_shoulder_idx = 17*3+2

# Example (commented out):
# left_shoulder_x_idx = 16*3
# right_shoulder_x_idx = 17*3
# pose_params[0, left_shoulder_x_idx] = np.radians(-90)
# pose_params[0, right_shoulder_x_idx] = np.radians(-90)
# Note: The sign convention may need adjustment depending on your specific model.

# Set shoulder rotation parameters
pose_params[0, left_shoulder_idx] = angle_left
pose_params[0, right_shoulder_idx] = angle_right

angle_feet = np.radians(3)

# In the SMPL pose parameter vector, the z-axis rotation components of the leg joints
# are at the following indices (may vary depending on the specific model you use):
left_leg_idx = 1*3+2   # z-axis component of the first leg joint
right_leg_idx = 2*3+2  # z-axis component of the second leg joint

# To slightly spread the feet in the XY plane:
# - The left foot should rotate outward to the left.
# - The right foot should rotate outward to the right.
# This requires rotation around the z-axis.
# Note: The sign convention may need to be adjusted depending on your model’s coordinate system.
pose_params[0, left_leg_idx] = angle_feet
pose_params[0, right_leg_idx] = -angle_feet

# Remove the first 3 parameters (global rotation) and keep the body pose parameters
pose_params = pose_params[:, 3:]
print('pose_params', pose_params)
print('pose_params.shape', pose_params.shape)

# Example parameter parsing function (currently commented out):
# def parse_params(data):
#     print(data)
#     params = {}
#     params['scale'] = np.array([1], dtype=np.float32)
#     params['transl'] = np.array(data['transl'], dtype=np.float32) / 2
#     params['transl'] = np.array([0., 0., 0.], dtype=np.float32)
#     params['global_orient'] = np.array(data['global_orient'], dtype=np.float32)
#     params['body_pose'] = np.array(data['body_pose'], dtype=np.float32)
#     params['betas'] = np.array(data['betas'], dtype=np.float32)
#
#     params['body_pose'] = torch.from_numpy(params['body_pose'].reshape(1, 69))
#     params['global_orient'] = torch.from_numpy(params['global_orient'].reshape(1, 3))
#     params['transl'] = torch.from_numpy(params['transl'].reshape(1, 3))
#     params['scale'] = torch.from_numpy(params['scale'].reshape(1, 1))
#     params['betas'] = torch.from_numpy(params['betas'].reshape(1, 10))
#
#     return params

if __name__ == "__main__":

    body_model = SMPL('./training/deformers/smplx/SMPLX', gender='neutral')

    # Generate SMPL mesh with specified pose parameters
    smpl_outputs = body_model(
        betas=torch.zeros(1, 10),
        body_pose=pose_params,
        global_orient=torch.zeros(1, 3),
        transl=torch.zeros(1, 3),
    )

    smpl_v = smpl_outputs['vertices'].clone().reshape(-1, 3)
    mesh = trimesh.Trimesh(smpl_v, body_model.faces)

    # Export mesh as .obj file
    mesh.export('smpl.obj')

