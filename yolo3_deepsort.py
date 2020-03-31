import os
import cv2
import time
import argparse
import torch

from detector import build_detector
from deepsort import build_tracker
from utils.draw_bbox import draw_boxes
from utils.parse_config import parse_config

current_path = os.path.dirname(__file__)


class VideoTracker(object):
    def __init__(self, config, arguments, video_path=None):
        self.cfg = config
        self.args = arguments
        self.video_fps = 60  # 默认输出视频FPS为60
        if video_path is not None:
            self.args.video_path = video_path
        is_use_cuda = self.args.use_cuda and torch.cuda.is_available()
        if not is_use_cuda:
            print("Running programme in cpu")
        else:
            print("Running programme in gpu")

        if self.args.display:
            # 创建可视化窗口
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        self.vdo = cv2.VideoCapture()
        self.detector = build_detector(self.cfg, use_cuda=is_use_cuda)
        self.deepsort = build_tracker(self.cfg, use_cuda=is_use_cuda)

    def __enter__(self):
        self.vdo.open(self.args.video_path)
        self.video_fps = self.vdo.get(cv2.CAP_PROP_FPS)
        print("input video fps", self.video_fps)
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.args.output_path:
            # 视频写入时尽量保证和原视频FPS一致
            writer_encoder = cv2.VideoWriter_fourcc(*"XVID") if self.args.output_type == "avi" else cv2.VideoWriter_fourcc(*"X264")
            self.writer = cv2.VideoWriter(self.args.output_path, writer_encoder, self.video_fps, (self.im_width, self.im_height))
        assert self.vdo.isOpened()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)
        self.vdo.release()
        self.writer.release()
        cv2.destroyAllWindows()

    def run(self):
        idx_frame = 0  # 帧序列号
        fps_list = []
        while self.vdo.grab():
            # 循环取帧图像
            idx_frame += 1
            start = time.time()
            _, ori_im = self.vdo.retrieve()  # 解码并返回一帧图像
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)

            # 目标检测
            bbox_xywh, cls_confidence, cls_ids = self.detector(im)
            if bbox_xywh is not None:
                # 取出所有类别id为0的检测框，该类别id对应行人，具体可以查看yolo配置文件中的coco.names文件查看
                mask = (cls_ids == 0)
                bbox_xywh = bbox_xywh[mask]
                cls_confidence = cls_confidence[mask]
                bbox_xywh[:, 2:] *= 1.2  # 等比扩大检测框的宽度和高度，防止过小
                # 跟踪
                outputs = self.deepsort.update(bbox_xywh, cls_confidence, im)

                # 绘制跟踪结果框
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    ori_im = draw_boxes(ori_im, bbox_xyxy, identities)

            end = time.time()
            fps = 1 / (end - start)
            print("frame index: {}, spend time: {:.03f}s, fps: {:.03f}".format(idx_frame, end - start, fps))
            fps_list.append(fps)

            if self.args.display:
                cv2.imshow("test", ori_im)
                cv2.waitKey(1)
            if idx_frame % self.args.frame_interval == 0:
                # 按照间隔写入视频，并非每一帧都写入
                if self.args.output_path:
                    self.writer.write(ori_im)

        print(sum(fps_list) / idx_frame)

    def run_with_limit(self, frame_limit=200, save_path=None):
        if save_path:
            self.args.output_path = save_path
        idx_frame = 0
        result_path = []  # 存放预览的跟踪结果图片
        while self.vdo.grab() and idx_frame < frame_limit * self.args.frame_interval:
            idx_frame += 1
            start = time.time()
            _, ori_im = self.vdo.retrieve()
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)

            bbox_xywh, cls_confidence, cls_ids = self.detector(im)
            if bbox_xywh is not None:
                mask = (cls_ids == 0)

                bbox_xywh = bbox_xywh[mask]
                bbox_xywh[:, 2:] *= 1.2
                cls_confidence = cls_confidence[mask]

                outputs = self.deepsort.update(bbox_xywh, cls_confidence, im)

                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    ori_im = draw_boxes(ori_im, bbox_xyxy, identities)

            end = time.time()
            print("frame index: {}, spend time: {:.03f}s, fps: {:.03f}".format(idx_frame, end - start, 1 / (end - start)))

            if self.args.display:
                cv2.imshow("test", ori_im)
                cv2.waitKey(1)
            if idx_frame % self.args.frame_interval == 0:
                if self.args.output_path:
                    self.writer.write(ori_im)
                    file_path = os.path.join(save_path, '{}.png'.format(idx_frame))
                    result_path.append(os.path.split(file_path)[-1])  # 只返回文件名，不包含完整路径，这是为了配合Django的静态文件设置
                    cv2.imwrite(file_path, ori_im)
        return result_path


def parse_arguments():
    """
    解析命令行脚本参数
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default='TownCentreXVID.avi')  # 进行跟踪的源视频
    parser.add_argument("--config_detection", type=str, default="./configs/yolov3.yml")  # yolo3检测配置文件
    parser.add_argument("--config_deepsort", type=str, default="./configs/deepsort.yml")  # deepsort跟踪配置文件
    parser.add_argument("--frame_interval", type=int, default=1)  # 输出视频帧间隔
    parser.add_argument("--show_window", dest="display", default=False)  # 是否视频控制台显示
    parser.add_argument("--show_width", type=int, default=800)  # 输出视频宽度
    parser.add_argument("--show_height", type=int, default=600)  # 输出视频高度
    parser.add_argument("--output_path", type=str, default="./result/result.avi")  # 输出视频保存路径
    parser.add_argument("--use_cuda", action="store_true", default=True)  # 是否使用GPU
    parser.add_argument("--output_type", type=str, default="avi")
    return parser.parse_args()


class Argument(object):
    def __init__(self, video_path):
        """
        模块调用参数，与上面的命令行参数选择其一，防止模块化不能调用命令行参数
        :param video_path:
        """
        self.video_path = video_path  # 输入视频路径
        self.config_detector = os.path.join(current_path, 'configs/yolov3.yml')  # 检测器配置文件
        self.config_deepsort = os.path.join(current_path, 'configs/deepsort.yml')  # deepsort算法配置文件
        self.display_window = False  # 默认API调用模式不显示opencv窗口
        self.frame_interval = 1  # 输出帧间隔默认为1，此种情况下若输出视频与输入视频FPS为相等，则输出视频与输入视频等时长
        self.display_width = 800  # 输出视频宽度
        self.display_height = 600  # 输出视频高度
        self.output_path = os.path.join(current_path, 'result/result.avi')  # 输出视频文件路径
        self.output_type = "avi"
        self.use_cuda = True  # 是否使用GPU


if __name__ == "__main__":
    args = parse_arguments()
    cfg = parse_config()
    cfg.merge_from_file(args.config_detection)  # 获取检测配置文件
    cfg.merge_from_file(args.config_deepsort)  # 获取deepsort算法配置文件

    with VideoTracker(cfg, args) as vdo_trk:
        vdo_trk.run()
