# author: 龚潇颖(Xiaoying Gong)
# date： 2019/11/23 14:41  
# IDE：PyCharm 
# des:
# input(s)：
# output(s)：
import numpy as np
# 去掉重复的对应，便于构建邻域
def non_reapted_neighbors(pre_match_index):
    pre_index_1 = pre_match_index[:, 0]
    pre_index_2 = pre_match_index[:, 1]
    new_pre_index_2, reserved = np.unique(pre_index_2, return_index=True)
    new_pre_index_1 = pre_index_1[reserved]
    pre_match_index_non_repeated = np.hstack((np.resize(new_pre_index_1, [len(new_pre_index_1), 1]),
                                              np.resize(new_pre_index_2, [len(new_pre_index_2), 1])))
    return pre_match_index_non_repeated

def non_reapted_neighbors_with_descriptor(pre_match_index, des_1, des_2):
    pre_index_1 = pre_match_index[:, 0]
    pre_index_2 = pre_match_index[:, 1]
    new_pre_index_2, reserved = np.unique(pre_index_2, return_index=True)
    new_pre_index_1 = pre_index_1[reserved]
    des_non_repeated_1 = des_1[reserved]
    des_non_repeated_2 = des_2[reserved]
    pre_match_index_non_repeated = np.hstack((np.resize(new_pre_index_1, [len(new_pre_index_1), 1]),
                                              np.resize(new_pre_index_2, [len(new_pre_index_2), 1])))
    return pre_match_index_non_repeated, des_non_repeated_1, des_non_repeated_2

# 去掉相同2d坐标
def unique_2d(arr):
    return np.array(list(set([tuple(t) for t in arr])))


def set_difference_2d(larger_set, less_set):
    """
    用于二维集合的差集操作
    :param larger_set: 一个较大的集合
    :param less_set: 一个较小的集合
    :return: 较大的集合去掉较小的集合
    """
    less_set_len = less_set.shape[0]
    delete_index = np.array([], dtype=np.int)
    for i in range(less_set_len):
       mask = larger_set == less_set[i]
       index = np.where(np.all(mask, axis=1))[0]
       delete_index = np.append(delete_index, index)
    result = np.delete(larger_set, delete_index, axis=0)
    return result, delete_index

