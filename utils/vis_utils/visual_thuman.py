import os, cv2
import numpy as np
import json
import pickle
from PIL import Image
import trimesh
from training.deformers.smplx import SMPL
import torch


def all_file(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            extend = os.path.splitext(file)[1]
            if extend == '.png' or extend == '.jpg' or extend == '.jpeg' or extend == '.JPG':
                L.append(os.path.join(root, file))
    return L


def parse_params(data):
    print(data)
    params = {}
    # params['cam'] = np.array(data[0], dtype=np.float32)
    params['scale'] = np.array([1], dtype=np.float32)

    params['transl'] = np.array(data['transl'], dtype=np.float32) / 2
    params['transl'] = np.array([0., 0., 0.], dtype=np.float32)
    params['global_orient'] = np.array(data['global_orient'], dtype=np.float32)
    params['body_pose'] = np.array(data['body_pose'], dtype=np.float32)
    params['betas'] = np.array(data['betas'], dtype=np.float32)

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
    # root = '/data/qingyao/neuralRendering/mycode/pretrainedModel/EVA3D-main/datasets/DeepFashion/images_padding'
    # root = '/data/vdd/zhongyuhe/workshop/dataset/human_syn_2/images_padding'
    # root = '/data/vdd/zhongyuhe/workshop/dataset/motion/A_pose/image_0'
    root = '/data/vdd/zhongyuhe/workshop/dataset/human_base/'
    # root = '/data/vdd/zhongyuhe/workshop/dataset/human_test_512/images'
    img_paths = sorted(all_file(root))
    # save_dir = '/data/vdd/zhongyuhe/workshop/dataset/human_test_512/visual'
    save_dir = './tmp_test'
    os.makedirs(save_dir, exist_ok=True)

    # path = '/data/vdd/zhongyuhe/workshop/dataset/human_syn_2/dataset.json'
    path = './data/data_7.json'
    # path = '/data/vdd/zhongyuhe/workshop/tools/Multiview-Avatar-main/dataset/dataset.json'

    with open(path, 'r') as fp:
        dp_pose_dist = json.load(fp)
    # print(dp_pose_dist.shape)  # (8037, 111)

    idx = 0
    for file in sorted(os.listdir(root)):
        # file_name = 'images/' + file
        idx = idx + 1
        file_name = 'img_000001_' + str(idx) + '.png'
        if idx > 7:
            continue
        params = parse_params(dp_pose_dist[file_name])

        for key in params.keys():
            print(key, params[key].shape)
            print(params[key])

        # cam_data = params['cam']
        # cam2world_matrix = cam_data[:16].reshape(4, 4)
        # intrinsics = cam_data[16:25].reshape(3, 3)
        # fx = float(intrinsics[0, 0])
        # fy = float(intrinsics[1, 1])
        # cx = float(intrinsics[0, 2])
        # cy = float(intrinsics[1, 2])

        cam2world_matrix = np.array(
            [[1., 0., 0., 0.],
             [0., 1., 0., 0.],
             [0., 0., 1., -100.],
             [0., 0., 0., 1.]], dtype=np.float32
        )
        # focal_length = 2500
        # print('fx, fy, cx, cy: ', fx, fy, cx, cy)
        orig_img_size = 512
        fx = 50
        fy = 50
        cx = 0.5
        cy = 0.5

        intrinsics = np.array(
            [[fx * orig_img_size, 0.00000000e+00, cx * orig_img_size],
             [0.00000000e+00, fy * orig_img_size, cy * orig_img_size + 40],
             [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
        )

        # fx=2500/512 cx'=0.5
        # intrinsics_origin = np.array(
        #             [[5, 0.00000000e+00, 0.5],
        #              [0.00000000e+00, 5, 0.5],
        #              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
        #         )

        print('cam2world_matrix', cam2world_matrix)
        print('intrinsics', intrinsics)

        cam = (intrinsics, cam2world_matrix)

        # # load smpl mesh
        # mesh_path = os.path.join(root, '%05d_flipx.obj'%idx)
        # mesh = trimesh.load(mesh_path, process=False, maintain_order=True)
        # vert = mesh.vertices

        body_model = SMPL('./training/deformers/smplx/SMPLX', gender='neutral')
        smpl_outputs = body_model(betas=params["betas"],
                                  body_pose=params["body_pose"],
                                  global_orient=params["global_orient"],
                                  # transl=params["transl"],
                                  transl=torch.zeros(1, 3),
                                  scale=params["scale"] if "scale" in params.keys() else None)

        smpl_v = smpl_outputs['vertices'].clone().reshape(-1, 3)
        # print('smpl_v', smpl_v.shape)
        # print(body_model.faces.shape)
        mesh = trimesh.Trimesh(smpl_v, body_model.faces)

        # load img
        img_path = os.path.join(root, file)
        img = Image.open(img_path)
        # img = img.crop((0, 0, 512, 512))
        # img.save(os.path.join(save_dir, f'{file}'))

        # # fill img with white
        # img = np.array(img)
        # img[:, :, 0] = 255
        # img[:, :, 1] = 255
        # img[:, :, 2] = 255
        # img = Image.fromarray(img)

        # Project the mesh on each image
        img_proj = project_scene(mesh, img, cam)
        file_name = file.split('.')[-2]
        img_proj.save(os.path.join(save_dir, f'{file_name}_smpl.png'))
        print('save %s' % os.path.join(save_dir, f'{file_name}_smpl.png'))
