class TrackState:
    """
    轨迹状态
    """
    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    包含一个轨迹的所有信息
    """

    def __init__(self, mean, covariance, track_id, n_init, max_age,
                 feature=None):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1  # 命中次数
        self.time_since_update = 0

        self.state = TrackState.Tentative  # 创建时的状态为Tentative
        self.features = []
        if feature is not None:
            self.features.append(feature)

        self._n_init = n_init
        self._max_age = max_age

    def to_tlwh(self):
        """
        当前目标位置，格式转换
        Returns
        -------

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kf):
        """
        使用卡尔曼滤波进行状态预测
        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.time_since_update += 1  # 每次预测自增1

    def update(self, kf, detection):
        """
        进行相关矩阵和数据的更新
        """
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        self.features.append(detection.feature)

        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """
        该轨迹是否为tentative（临时存在）
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """
        该轨迹是否确认
        """
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """
        该轨迹是否删除
        """
        return self.state == TrackState.Deleted
