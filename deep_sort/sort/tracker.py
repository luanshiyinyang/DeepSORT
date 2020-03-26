import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track


class Tracker:
    """
    多目标跟踪器实现
    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=70, n_init=3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1

    def predict(self):
        """
        状态预测
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        """
        状态更新
        """
        # 级联匹配
        matches, unmatched_tracks, unmatched_detections = self._match(detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            # 成功匹配的要用检测结果更新对于track的参数
            # 包括
            #   更新卡尔曼滤波一系列运动变量、命中次数以及重置time_since_update
            #   检测的深度特征保存到track的特征集中
            #   连续命中三帧，将track状态由tentative改为confirmed

            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            # 未成功匹配的track
            #   若未经过confirm则删除
            #   若已经confirm但连续max_age帧未匹配到检测结果也删除
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            # 未匹配的检测，为其创建新的track
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        # 更新已经确认的track的特征集
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):
        """
        跟踪结果和检测结果的匹配
        :param detections:
        :return:
        """

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        # 将track分为确认track和未确认track
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # 将确认的track和检测结果进行级联匹配（使用外观特征）
        matches_a, unmatched_tracks_a, unmatched_detections = linear_assignment.matching_cascade(
            gated_metric, self.metric.matching_threshold, self.max_age,
            self.tracks, detections, confirmed_tracks)

        # 将上一步未成功匹配的track和未确认的track组合到一起形成iou_track_candidates于还没有匹配结果的检测结果进行IOU匹配
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        # 计算两两之间的iou，再通过1-iou得到cost matrix
        matches_b, unmatched_tracks_b, unmatched_detections = linear_assignment.min_cost_matching(
            iou_matching.iou_cost, self.max_iou_distance, self.tracks,
            detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b  # 组合获得当前所有匹配结果
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        """
        初始化新的跟踪器，对应新的检测结果
        :param detection:
        :return:
        """
        # 初始化卡尔曼
        mean, covariance = self.kf.initiate(detection.to_xyah())
        # 创建新的跟踪器
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature))
        # id自增
        self._next_id += 1
