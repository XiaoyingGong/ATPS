# author: 龚潇颖(Xiaoying Gong)
# date： 2020/1/9 17:02  
# IDE：PyCharm 
# des: This is a implementation for ATP
# input(s)：
# output(s)：
import numpy as np

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
    r_b_f = (u**2)*np.log(u)
    return r_b_f

# des(s): using ATPS to1
# inputs(s): ita
# output(s):
def transform_points(fixed_points, mov_points):
    transformed_mov_points = 0
    transformation_param = 0
    return transformed_mov_points, transformation_param

x = np.array([[1, 3], [2, 4]])
y = np.array([[4, 5], [5, 6], [3, 5]])

radial_basis_function(x, y)
