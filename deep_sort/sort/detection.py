import numpy as np


class Detection(object):
    """
    检测基类
    """

    def __init__(self, tlwh, confidence, feature):
        """

        :param tlwh: bbox (x, y, w, h)
        :param confidence: 置信度
        :param feature: 特征向量
        """
        self.tlwh = np.asarray(tlwh, dtype=np.float)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)

    def to_tlbr(self):
        """
        转换bbox为(top left bottom right)的格式即(minx miny maxx maxy)_
        :return:
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """
        转换bbox为(center x, center y, aspect ration, height)
        :return:
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret
