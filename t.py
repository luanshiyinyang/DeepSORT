"""
Author: Zhou Chen
Date: 2020/4/29
Desc: desc
"""
import numpy as np


def _cosine_distance(a, b, data_is_normalized=False):
    """
    计算两点的余弦距离
    Parameters
    ----------
    a
    b
    data_is_normalized 若为True则认为a和b已经是单位长度向量，否则会将其先单位化

    Returns
    -------

    """
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)