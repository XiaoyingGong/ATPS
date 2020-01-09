# author: 龚潇颖(Xiaoying Gong)
# date： 2020/1/7 20:49  
# IDE：PyCharm 
# des: normalizing 2D points to have zero mean and unit standard variation
# input(s)：
# output(s)：
import numpy as np
import matplotlib.pyplot as plt

# input(s)：N * 2 points_1; M * 2 points_2
# output(s)：N * 2 n_points_1; M * 2 n_points_2; and scale parameter
# 这种方法给予原始数据的均值（mean）和标准差（standard deviation）进行数据的标准化
# x* = （x - μ） / sigma    sigma is standard deviation
def normalization(points_1, points_2):
    N = len(points_1)
    M = len(points_2)
    n_mean_1 = np.mean(points_1, axis=0)
    n_mean_2 = np.mean(points_2, axis=0)
    n_points_1 = points_1 - np.tile(n_mean_1, (N, 1))
    n_points_2 = points_2 - np.tile(n_mean_2, (M, 1))
    n_sigma_1 = np.sqrt(np.sum(n_points_1 ** 2) / N)
    n_sigma_2 = np.sqrt(np.sum(n_points_2 ** 2) / M)
    n_points_1 = n_points_1 / n_sigma_1
    n_points_2 = n_points_2 / n_sigma_2
    return n_points_1, n_points_2, {"n_mean_1": n_mean_1, "n_mean_2": n_mean_2,
                                    "n_sigma_1": n_sigma_1, "n_sigma_2": n_sigma_2}

# input(s)：N * 2 n_points_1; M * 2 n_points_2; normalization parameter
# output(s)：denormalized points; N * 2 points_1; M * 2 points_2;
def denormalization(n_points_1, n_points_2, scale_param):
    n_mean_1 = scale_param["n_mean_1"]
    n_mean_2 = scale_param["n_mean_2"]
    n_sigma_1 = scale_param["n_sigma_1"]
    n_sigma_2 = scale_param["n_sigma_2"]
    n_points_1 = n_points_1 * n_sigma_1
    n_points_2 = n_points_2 * n_sigma_2
    points_1 = n_points_1 + n_mean_1
    points_2 = n_points_2 + n_mean_2
    return points_1, points_2

if __name__ == '__main__':
    a = np.array([[1, 1], [4, 4], [1, 3]])
    b = np.array([[1, 1], [3, 3]])
    n_a, n_b, param = normalization(a, b)
    plt.scatter(a[:, 0], a[:, 1])
    plt.scatter(n_a[:, 0], n_a[:, 1])
    plt.show()
    print(n_a)
    print(n_b)
    print(param)
    p_1, p_2 = denormalization(n_a, n_b, param)
    print(p_1)
    print(p_2)
