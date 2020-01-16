#author: 龚潇颖
#date:2019_8_27
#des:用于提取sift的匹配，输出为一个 n * 2的矩阵，每一行为一个点的坐标

import cv2
import numpy as np

def get_matches_exist_kp(des1, des2, sift_threshold):
    good_match = get_good_match(des1, des2, sift_threshold)
    match_index = np.zeros((len(good_match), 2), dtype=np.int)
    for i in range(len(good_match)):
        index1 = good_match[i].queryIdx
        index2 = good_match[i].trainIdx
        match_index[i] = np.array([index1, index2])
    return match_index

# 得到key_points与descriptors
def get_kps_deses(img1, img2):
    kp1, des1 = sift_kp(img1)
    kp2, des2 = sift_kp(img2)
    kp1, index1 = repeat_removal(kp1)
    kp2, index2 = repeat_removal(kp2)
    des1 = np.array(des1)[index1]
    des2 = np.array(des2)[index2]
    return kp1, kp2, des1, des2


def get_matches(img1, img2, sift_threshold):
    kp1, des1 = sift_kp(img1)
    kp2, des2 = sift_kp(img2)
    _, index1 = repeat_removal(kp1)
    _, index2 = repeat_removal(kp2)
    kp1 = np.array(kp1)[index1]
    kp2 = np.array(kp2)[index2]
    des1 = np.array(des1)[index1]
    des2 = np.array(des2)[index2]
    good_match = get_good_match(des1, des2, sift_threshold)
    matching_points_1, matching_points_2, des_1, des_2, match_index = get_matching_points(kp1, kp2, des1, des2, good_match)
    return matching_points_1, matching_points_2, des_1, des_2, match_index


# 去重返回值为不重复的值的下标
def repeat_removal(kp):
    temp = np.zeros([len(kp), 2])
    for i in range(len(kp)):
        temp[i] = kp[i].pt
    temp, index = np.unique(temp, return_index=True, axis=0)
    return temp, index


# 得到在预匹配过后筛选的点,matching_points是一个n乘以2的二维矩阵，第一例为x坐标，第二例为y坐标
# match_index为一个n*2的矩阵，在预匹配后用于记录点的对应关系
def get_matching_points(kp1, kp2, des1, des2, good_match):
    matching_points_1 = np.zeros((len(good_match), 2))
    matching_points_2 = np.zeros((len(good_match), 2))
    des_1 = np.zeros((len(good_match), 128))
    des_2 = np.zeros((len(good_match), 128))
    match_index = np.zeros((len(good_match), 2))
    for i in range(len(good_match)):
        index1 = good_match[i].queryIdx
        index2 = good_match[i].trainIdx
        matching_points_1[i][0] = kp1[index1].pt[0]
        matching_points_1[i][1] = kp1[index1].pt[1]
        matching_points_2[i][0] = kp2[index2].pt[0]
        matching_points_2[i][1] = kp2[index2].pt[1]
        des_1[i] = des1[index1]
        des_2[i] = des2[index2]
        match_index[i] = np.array([index1, index2])
    return matching_points_1, matching_points_2, des_1, des_2, match_index

# 此方法不返回kp
def get_matching_points_v2(kp1, kp2, des1, des2, good_match):
    match_index = np.zeros((len(good_match), 2))
    for i in range(len(good_match)):
        index1 = good_match[i].queryIdx
        index2 = good_match[i].trainIdx
        match_index[i] = np.array([index1, index2])
    return match_index


# 得到关键点
def sift_kp(image):
    #gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_image = image
    sift = cv2.xfeatures2d_SIFT.create()
    kp, des = sift.detectAndCompute(gray_image, None)
    return kp, des


# 做匹配
def get_good_match(des1, des2, sift_threshold):
    bf = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < sift_threshold * n.distance:
            good.append(m)
    return good

