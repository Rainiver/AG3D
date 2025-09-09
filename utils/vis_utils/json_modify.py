import sys
import os
import pickle
import json
import numpy as np





json_path = '/data/vdd/zhongyuhe/workshop/dataset/human_syn_2/dataset.json'
with open(json_path, 'rb') as fp:
    json_data = json.load(fp)

# info = json_data['labels']
for key in json_data['labels']:
    if (key.split('.')[-2])[-1] == '7':
        json_data['labels'][key][1][:4] = json_data['labels']['images_padding/img_000000_1.png'][1][:4]
        json_data['labels'][key][1][7:] = json_data['labels']['images_padding/img_000000_1.png'][1][7:]





with open("/data/vdd/zhongyuhe/workshop/dataset/human_syn_2/dataset_2.json", 'w') as write_f:
    json.dump(json_data, write_f, indent=4, ensure_ascii=False)

