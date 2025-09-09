import os, cv2
import numpy as np
import json
import pickle
from PIL import Image
import trimesh
from training.deformers.smplx import SMPL
import torch



pose_params = torch.zeros([1, 72], dtype=torch.float32)

# 设置轴角的旋转向量。在SMPL模型中，第4，5个pose参数代表左手臂的旋转，第7，8个代表右手臂的旋转
# 在基本的SMPL模型，这两个参数分别对应轴角旋转，单位是弧度，通常Y轴向下，我们以身体侧面的向下为正方向
# 因此，如果手臂向后（从身体向外）移动，旋转将为正；如果手臂向前（向身体中心）移动，旋转将为负

# 使用numpy将角度从度转换为弧度
angle_left = np.radians(-45)
angle_right = np.radians(52)

# 假设模型躯干朝前，且关节旋转轴是沿着人的侧面方向（Y轴），需要相应地设置左右肩的pose参数
left_shoulder_idx = 16*3+2
right_shoulder_idx = 17*3+2

# left_shoulder_x_idx = 16*3
# right_shoulder_x_idx = 17*3
# pose_params[0, left_shoulder_x_idx] = np.radians(-90)  # 再次强调，这里的符号可能需要您根据您的模型进行实际测试和调整
# pose_params[0, right_shoulder_x_idx] = np.radians(-90)

# 设置旋转参数
pose_params[0, left_shoulder_idx] = angle_left
pose_params[0, right_shoulder_idx] = angle_right

# pose_params = torch.from_numpy(pose_params[3:]).reshape(1, 69)
# print('pose_params.shape', pose_params.shape)
angle_feet = np.radians(3)

# 腿部关节在 SMPL pose 参数向量中的 z 轴旋转坐标
# 下面的索引可能需要根据您使用的具体模型来调整
left_leg_idx = 1*3+2  # 第一个关节的 z 分量
right_leg_idx = 2*3+2  # 第二个关节的 z 分量

# 为了让脚在xy平面内略微张开，左脚应该向左旋转，右脚应该向右旋转
# 这需要绕 z 轴进行旋转
# 符号可能需要试验调整以匹配您的模型坐标系和实际效果
pose_params[0, left_leg_idx] = angle_feet
pose_params[0, right_leg_idx] = -angle_feet

pose_params = pose_params[:, 3:]
print('pose_params', pose_params)
print('pose_params.shape', pose_params.shape)


# def parse_params(data):
#     print(data)
#     params = {}
#     # params['cam'] = np.array(data[0], dtype=np.float32)
#     params['scale'] = np.array([1], dtype=np.float32)
#
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
# smpl_outputs = body_model(betas=params["betas"],
#                                   body_pose=params["body_pose"],
#                                   global_orient=params["global_orient"],
#                                   # transl=params["transl"],
#                                   transl=torch.zeros(1, 3),
#                                   )
    smpl_outputs = body_model(betas=torch.zeros(1, 10),
                                  body_pose=pose_params,
                                  global_orient=torch.zeros(1, 3),
                                  # transl=params["transl"],
                                  transl=torch.zeros(1, 3),
                                  )

    smpl_v = smpl_outputs['vertices'].clone().reshape(-1, 3)
# print('smpl_v', smpl_v.shape)
# print(body_model.faces.shape)
    mesh = trimesh.Trimesh(smpl_v, body_model.faces)

    mesh.export('smpl.obj')
