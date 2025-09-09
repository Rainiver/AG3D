import cv2
import os
from PIL import Image, ImagePath
import numpy


def rescale(im, target_height, target_width):
    height, width = im.shape[:2]  # 取彩色图片的长、宽。

    ratio_h = height / target_height
    ration_w = width / target_width

    ratio = max(ratio_h, ration_w)

    # 缩小图像  resize(...,size)--size(width，height)
    size = (int(width / ratio), int(height / ratio))
    shrink = cv2.resize(im, size, interpolation=cv2.INTER_AREA)  # 双线性插值
    BLACK = [0, 0, 0]
    WHITE = [255, 255, 255]

    a = (target_width - int(width / ratio)) / 2
    b = (target_height - int(height / ratio)) / 2

    constant = cv2.copyMakeBorder(shrink, int(b), int(b), int(a), int(a), cv2.BORDER_CONSTANT, value=WHITE)
    constant = cv2.resize(constant, (target_width, target_height), interpolation=cv2.INTER_AREA)
    return constant

def crop_image(image_dir, mask_dir, output_path, size):   # image_dir 批量处理图像文件夹 size 裁剪后的尺寸

    for filename in sorted(os.listdir(image_dir)):
        if 'img' in filename:
            idx = (filename.split('.')[-2])[4:]  # 去掉img_
            seg_path = os.path.join(mask_dir, 'mask_' + idx + '.png')
            image = cv2.imread(os.path.join(image_dir, filename))
            seg_map = cv2.imread(seg_path)
            _, seg_map = cv2.threshold(seg_map, 0, 255, cv2.THRESH_BINARY)
            seg_map = rescale(seg_map, 512, 512)
            black_back = cv2.bitwise_and(image, seg_map)
            white_mask = 255 - seg_map
            image_new = black_back + white_mask
            output_filename = output_path + filename
            cv2.imwrite(output_filename, image_new)





if __name__ == "__main__":

    image_dir = "/data/vdd/zhongyuhe/workshop/dataset/human_syn_2/images_padding/"
    output_path = "/data/vdd/zhongyuhe/workshop/dataset/human_syn_2/images_padding/"
    mask_dir = '/data/vdd/zhongyuhe/workshop/dataset/human_syn/segm/'
    size = 512
    crop_image(image_dir, mask_dir, output_path, size)