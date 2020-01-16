# author: 龚潇颖(Xiaoying Gong)
# date： 2020/1/9 17:02  
# IDE：PyCharm 
# des: This is a implementation for ATPS
# input(s)：
# output(s)：
import numpy as np
import matplotlib.pyplot as plt
import utils
# des:K_ij = U(u) = u^2 log u     the radial basis function of ATPS
#     u_ij = ||x_i - x_j||

def radial_basis_function(x, y):
    N = len(x)
    M = len(y)

    x_1 = x[:, 0]
    x_2 = x[:, 1]
    x_tile = np.array([x_1.reshape([N, 1]), x_2.reshape([N, 1])]) # reshape to 3-D
    x_tile = np.tile(x_tile, (1, 1, M))

    y = y.T
    y_1 = y[0, :]
    y_2 = y[1, :]
    y_tile = np.array([y_1.reshape([1, M]), y_2.reshape([1, M])]) # reshape to 3-D
    y_tile = np.tile(y_tile, (1, N, 1))

    u = np.sqrt(np.sum((x_tile - y_tile) ** 2, axis=0)) # u is the Euclidean distance between x_tile and y_tile
    r_b_f = (u**2)*np.log(u + 0.00000001)
    return r_b_f

# des(s): using ATPS to1
# inputs(s): ita
# output(s):
def calculate_transformation_coefficient(fixed_points, mov_points, ita):
    mov_N = len(mov_points)
    K = radial_basis_function(mov_points, mov_points)
    H = np.hstack((np.ones([mov_N, 1]), mov_points))
    part_1_row_1 = np.hstack((K + ita * np.ones([mov_N, 1]), H))
    part_1_row_2 = np.hstack((H.T, np.zeros([3, 3])))
    part_1 = np.vstack((part_1_row_1, part_1_row_2))
    part_1_inv = np.linalg.pinv(part_1)
    part_2 = np.vstack((fixed_points, np.zeros([3, 2])))
    transform_coefficient = np.dot(part_1_inv, part_2)
    return transform_coefficient


# des(s): using ATPS to1
# inputs(s): ita
# output(s):
def transforming_points(mov_points, mov_point_inliers, transformation_coefficient):
    mov_N = len(mov_points)
    K = radial_basis_function(mov_points, mov_point_inliers)
    H = np.hstack((np.ones([mov_N, 1]), mov_points))
    transformed_mov_points = np.dot(np.hstack((K, H)), transformation_coefficient)
    return transformed_mov_points


if __name__ == "__main__":
    import utils
    img_r = "./data/img/81-r.jpg"
    img_s = "./data/img/81-l.jpg"
    kp_s, kp_r, des_s, des_r, img_s, img_r, resize_w, resize_h = \
        utils.get_kps_deses.get_kps_deses(img_s, img_r, utils.constants.resize_img_w, utils.constants.resize_img_h)
    pre_match_09_index = utils.sift_matching.get_matches_exist_kp(des_s, des_r, 0.9)
    pre_match_07_index = utils.sift_matching.get_matches_exist_kp(des_s, des_r, 0.7)

    kp_s_all = kp_s[pre_match_09_index[:, 0]]
    kp_r_all = kp_r[pre_match_09_index[:, 1]]

    kp_s_inlier = kp_s[pre_match_07_index[:, 0]]
    kp_r_inlier = kp_r[pre_match_07_index[:, 1]]

    for i in range(10):
        plt.scatter(kp_r_all[:, 0], kp_r_all[:, 1], c='r', s=10)

        transformation_coefficient = \
            calculate_transformation_coefficient(kp_r_inlier, kp_s_inlier, 0.5)

        kp_s_all = \
            transforming_points(kp_s_all,  kp_s_inlier, transformation_coefficient)

        plt.scatter(kp_s_all[:, 0], kp_s_all[:, 1], c='b', s=5)
        _, delete_index = utils.utils.set_difference_2d(pre_match_09_index, pre_match_07_index)
        kp_s_inlier = kp_s_all[delete_index]
        print(kp_s_inlier)
        plt.show()

        plt.close()
