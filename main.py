# author: 龚潇颖(Xiaoying Gong)
# date： 2020/8/17 16:09  
# IDE：PyCharm 
# des:
# input(s)：
# output(s)：
import utils
import scipy.io as io
import matplotlib.pyplot as plt
import ATPS

img_r = "./data/img/56-r.jpg"
img_s = "./data/img/56-l.jpg"
kp_s, kp_r, des_s, des_r, img_s, img_r, resize_w, resize_h = \
    utils.get_kps_deses.get_kps_deses(img_s, img_r, utils.constants.resize_img_w, utils.constants.resize_img_h)
pre_match_09_index = utils.sift_matching.get_matches_exist_kp(des_s, des_r, 0.9)
pre_match_07_index = utils.sift_matching.get_matches_exist_kp(des_s, des_r, 0.6)

kp_s_all = kp_s[pre_match_09_index[:, 0]]
kp_r_all = kp_r[pre_match_09_index[:, 1]]

kp_s_inlier = kp_s[pre_match_07_index[:, 0]]
kp_r_inlier = kp_r[pre_match_07_index[:, 1]]

io.savemat("./result/" + str(56) + ".mat",
           {'kp_s_inlier': kp_s_inlier, 'kp_r_inlier': kp_r_inlier, 'kp_s': kp_s_all, 'kp_r': kp_r_all})

transformed_kp_s_all = ATPS.ATPS(kp_s_inlier, kp_r_inlier, kp_s_all, 10)
print(transformed_kp_s_all)
plt.scatter(kp_r_all[:, 0], kp_r_all[:, 1], c='r', s=10)
plt.scatter(transformed_kp_s_all[:, 0], transformed_kp_s_all[:, 1], c='b', s=5)
plt.show()
