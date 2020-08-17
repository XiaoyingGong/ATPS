# author: 龚潇颖(Xiaoying Gong)
# date： 2020/8/17 11:06  
# IDE：PyCharm 
# des:
# input(s)：
# output(s)：
import numpy as np
def cal_kernel(X, Y):
    X_num, X_d = X.shape
    Y_num, Y_d = Y.shape
    diff = np.tile(X.reshape([-1, 1, X_d]), [1, Y_num, 1]) - \
               np.tile(Y.reshape([1, -1, Y_d]), [X_num, 1, 1])
    dist = np.sum(diff**2, axis=2)
    dist = np.maximum(dist, 2.2204e-16) # 避免0出现
    K = dist * np.log(np.sqrt(dist))
    return K

def ATPS(inlier_X, inlier_Y, grid, eta):
    inlier_num, d = inlier_X.shape
    grid_num, _ = grid.shape
    K = cal_kernel(inlier_X, inlier_X)
    H = np.hstack((np.ones([inlier_num, 1]), inlier_X))
    L = np.vstack((np.hstack((K + eta * np.identity(inlier_num), H)), np.hstack((H.T, np.zeros([3, 3])))))
    R = np.vstack((inlier_Y, np.zeros([3, 2])))
    theta = np.dot(np.linalg.pinv(L), R)

    K_2 = cal_kernel(grid, inlier_X)
    H_2 = np.hstack((np.ones([grid_num, 1]), grid))
    transformed_grid = np.dot(np.hstack((K_2, H_2)), theta)
    return transformed_grid

