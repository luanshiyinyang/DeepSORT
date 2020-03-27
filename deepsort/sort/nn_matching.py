import numpy as np


def _pdist(a, b):
    """
    计算点距
    Parameters
    ----------
    a
    b

    Returns
    -------

    """
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    r2 = np.clip(r2, 0., float(np.inf))
    return r2


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


def _nn_euclidean_distance(x, y):
    """
    欧氏距离
    Parameters
    ----------
    x
    y

    Returns
    -------

    """
    distances = _pdist(x, y)
    return np.maximum(0.0, distances.min(axis=0))


def _nn_cosine_distance(x, y):
    distances = _cosine_distance(x, y)
    return distances.min(axis=0)


class NearestNeighborDistanceMetric(object):

    def __init__(self, metric, matching_threshold, budget=None):
        """

        Parameters
        ----------
        metric "euclidean" or "cosine"
        matching_threshold 匹配阈值，大于此认为无效匹配
        budget 如果不是None，则将每个类的样本最多固定为这个数字。当达到budget大小时，删除最老的样本。
        """

        if metric == "euclidean":
            self._metric = _nn_euclidean_distance
        elif metric == "cosine":
            self._metric = _nn_cosine_distance
        else:
            raise ValueError("Invalid metric; must be either 'euclidean' or 'cosine'")
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}

    def partial_fit(self, features, targets, active_targets):
        """
        使用新数据更新距离指标
        Parameters
        ----------
        features M维的N个特征
        targets 关联目标Id的数组
        active_targets 场景中当前存在的目标列表

        Returns
        -------

        """
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget:]
        self.samples = {k: self.samples[k] for k in active_targets}

    def distance(self, features, targets):
        cost_matrix = np.zeros((len(targets), len(features)))
        for i, target in enumerate(targets):
            cost_matrix[i, :] = self._metric(self.samples[target], features)
        return cost_matrix
