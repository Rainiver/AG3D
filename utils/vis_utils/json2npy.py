import os, cv2
import numpy as np
import json
import torch



if __name__ == "__main__":

    pose_data = []
    # json_path = '/data/vdd/zhongyuhe/workshop/dataset/human_syn_2/dataset.json'
    json_path = '/data/vdd/zhongyuhe/workshop/tools/Multiview-Avatar-main/dataset/dataset.json'
    with open(json_path, 'r') as fp:
        json_data = json.load(fp)

    pose_path = '/data/vdd/zhongyuhe/workshop/AG3D/data/gen_human_2.npy'


    idx = 0
    for key in json_data['labels']:
        pose_data.append([])
        for i in range(2):
            for item in json_data['labels'][key][i]:
                pose_data[idx].append(item)

        idx = idx + 1


    pose_data = np.array(pose_data, dtype=np.float32)
    np.save(pose_path, pose_data)

