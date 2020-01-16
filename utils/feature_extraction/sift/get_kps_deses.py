# author: 龚潇颖(Xiaoying Gong)
# date： 2019/10/4 22:38  
# IDE：PyCharm 
# des:
# input(s)：
# output(s)：

import cv2
import utils

def img_resize_fixed(img_1, img_2, resize_w, resize_h):
    img_1 = cv2.resize(img_1, (resize_w, resize_h))
    img_2 = cv2.resize(img_2, (resize_w, resize_h))
    return img_1, img_2, resize_w, resize_h

def get_kps_deses(img_path_1, img_path_2, resize_w, resize_h):
    """
    :param img_path_1:
    :param img_path_2:
    :return:
    """
    # 图像路径
    img_1 = cv2.imread(img_path_1)[:, :, [2, 1, 0]]
    img_2 = cv2.imread(img_path_2)[:, :, [2, 1, 0]]
    # resize
    img_1, img_2, resize_w, resize_h = img_resize_fixed(img_1, img_2, resize_w, resize_h)
    # 获得kps与deses
    kp_1, kp_2, des_1, des_2 = utils.sift_matching.get_kps_deses(img_1, img_2)
    return kp_1, kp_2, des_1, des_2, img_1, img_2, resize_w, resize_h
