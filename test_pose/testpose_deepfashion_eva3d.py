
import os, cv2
import numpy as np
from PIL import Image
import pickle
import trimesh
from training.deformers.smplx import SMPL
import torch

def all_file(file_dir):
    L=[]
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            extend = os.path.splitext(file)[1]
            if extend == '.png' or extend == '.jpg' or extend == '.jpeg' or extend == '.JPG':
                L.append(os.path.join(root, file))
    return L

def parse_params(data):
    print(data)
    params = {}
    params['camera_translation'] = data['camera_translation']

    params['scale'] = np.array(1)
    params['transl'] = data['camera_translation']
    # params['transl'] = np.array([0., 0., 0.])
    params['global_orient'] = data['global_orient']
    params['body_pose'] = data['body_pose']
    params['betas'] = data['betas']

    params['body_pose'] = torch.from_numpy(params['body_pose'].reshape(1, 69))
    params['global_orient'] = torch.from_numpy(params['global_orient'].reshape(1, 3))
    params['transl'] = torch.from_numpy(params['transl'].reshape(1, 3))
    params['scale'] = torch.from_numpy(params['scale'].reshape(1, 1))
    params['betas'] = torch.from_numpy(params['betas'].reshape(1, 10))

    return params

def project_scene(mesh, img, cam, color=[255, 0, 0]):

    # Expand cam attributes
    K, P = cam
    P_inv = np.linalg.inv(P)

    # Project mesh vertices into 2D
    p3d_h = np.hstack((mesh.vertices, np.ones((mesh.vertices.shape[0], 1))))
    p2d_h = (K @ P_inv[:3, :] @ p3d_h.T).T
    p2d = p2d_h[:, :-1] / p2d_h[:, -1:]

    # Draw p2d to image
    img_proj = np.array(img)
    p2d = np.clip(p2d, 0, img.width - 1).astype(np.uint32)
    img_proj[p2d[:, 1], p2d[:, 0]] = color

    return Image.fromarray(img_proj.astype(np.uint8))


if __name__ == "__main__":
    
    root = '/data/vdd/zhongyuhe/workshop/dataset/human_syn_2/images_padding'

    img_paths = sorted(all_file(root))
     save_dir = 'tmp_mesh/deepfashion_eva3d'
    
    os.makedirs(save_dir, exist_ok=True)
    
    smpl_path = './EVA3D/datasets/DeepFashion/smpl.pkl'
    with open(smpl_path, 'rb') as fp:
        smpl_dist = pickle.load(fp)
    # print(smpl_dist.shape)  # (8037, 111)

    idx = 0
    for img_name in sorted(os.listdir(root)):
        idx = idx + 1
        if idx > 10:
            continue
        # img_path = (img_name.split('.')[-2])[4:]
        img_path = img_name.split('.')[-2]
        params = parse_params(smpl_dist[img_path])

        for key in params.keys():
            print(key, params[key].shape)
            print(params[key])

        cam_trans = params['camera_translation']
        # cam_data = params['cam']
        # cam2world_matrix = cam_data[:16].reshape(4, 4)

        # intrinsics = cam_data[16:25].reshape(3, 3)
        # fx = float(intrinsics[0, 0])
        # fy = float(intrinsics[1, 1])
        # cx = float(intrinsics[0, 2])
        # cy = float(intrinsics[1, 2])
        #
        # print('fx, fy, cx, cy: ', fx, fy, cx, cy)
        # orig_img_size = 512
        # intrinsics = np.array(
        #     [[fx * orig_img_size, 0.00000000e+00, cx * orig_img_size],
        #      [0.00000000e+00, fy * orig_img_size * 2, cy * orig_img_size],
        #      [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
        # )

        # focal_length = 2500
        # focal_length = 3000

        focal_length = 2500
        w = 512
        h = 512
        intrinsics = np.array(
            [[focal_length, 0.00000000e+00, 0.5 * w],
             [0.00000000e+00, focal_length, 0.5 * h],
             [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
        )

        # cam2world_matrix = np.array(
        #     [[1., 0., 0., cam_trans[0][0]],
        #      [0., 1., 0., cam_trans[0][1]],
        #      [0., 0., 1., cam_trans[0][2]],
        #       [0., 0., 0., 1.]]
        # )

        cam2world_matrix = np.array(
            [[1., 0., 0., 0.],
              [0., 1., 0., 0.],
              [0., 0., 1., 0.],
              [0., 0., 0., 1.]]
        )
        print('cam2world_matrix', cam2world_matrix)
        print('intrinsics', intrinsics)
        # load camera

        cam = (intrinsics, cam2world_matrix)


        # # load smpl mesh
        # mesh_path = os.path.join(root, '%05d_flipx.obj'%idx)
        # mesh = trimesh.load(mesh_path, process=False, maintain_order=True)
        # vert = mesh.vertices

        body_model = SMPL('./training/deformers/smplx/SMPLX', gender='neutral')
        smpl_outputs = body_model(betas=params["betas"],
                                  body_pose=params["body_pose"],
                                  global_orient=params["global_orient"],
                                  transl=params["transl"],
                                  scale=params["scale"] if "scale" in params.keys() else None)
        smpl_v = smpl_outputs['vertices'].clone().reshape(-1, 3)
        print('smpl_v', smpl_v.shape)
        print(body_model.faces.shape)
        mesh = trimesh.Trimesh(smpl_v, body_model.faces)

        # load img
        # img_path = img_paths[idx]

        img_path2 = os.path.join(root, img_name)
        img = Image.open(img_path2)
        # img = img.crop((0, 0, 512, 512))
        img.save(os.path.join(save_dir, f'{img_name}'))

        # # fill img with white
        # img = np.array(img)
        # img[:, :, 0] = 255
        # img[:, :, 1] = 255
        # img[:, :, 2] = 255
        # img = Image.fromarray(img)

        # Project the mesh on each image
        img_proj = project_scene(mesh, img, cam)
        img_proj.save(os.path.join(save_dir, f'{img_path}_smpl.png'))
        print('save %s' % os.path.join(save_dir, f'{img_path}_smpl.png'))















