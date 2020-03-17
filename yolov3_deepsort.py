import os
import cv2
import time
import argparse
import torch

from detector import build_detector
from deep_sort import build_tracker
from utils.draw import draw_boxes
from utils.parser import get_config
current_path = os.path.dirname(__file__)


class VideoTracker(object):
    def __init__(self, cfg, args, video_path=None):
        self.cfg = cfg
        self.args = args
        if video_path:
            self.args.video_path = video_path
        is_use_cuda = args.use_cuda and torch.cuda.is_available()
        if not is_use_cuda:
            print("Running in cpu mode!")

        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        self.vdo = cv2.VideoCapture()
        self.detector = build_detector(cfg, use_cuda=is_use_cuda)
        self.deepsort = build_tracker(cfg, use_cuda=is_use_cuda)
        self.class_names = self.detector.class_names

    def __enter__(self):
        self.vdo.open(self.args.video_path)
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.args.save_path:
            self.writer = cv2.VideoWriter(self.args.save_path, cv2.VideoWriter_fourcc(*'XVID'), 60, (self.im_width, self.im_height))

        assert self.vdo.isOpened()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def run(self):
        idx_frame = 0
        while self.vdo.grab():
            idx_frame += 1
            if idx_frame % self.args.frame_interval:
                continue

            start = time.time()
            _, ori_im = self.vdo.retrieve()
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)

            # do detection
            bbox_xywh, cls_conf, cls_ids = self.detector(im)
            if bbox_xywh is not None:
                # select person class
                mask = cls_ids == 0

                bbox_xywh = bbox_xywh[mask]
                bbox_xywh[:, 3:] *= 1.2  # bbox dilation just in case bbox too small
                cls_conf = cls_conf[mask]

                # do tracking
                outputs = self.deepsort.update(bbox_xywh, cls_conf, im)

                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    ori_im = draw_boxes(ori_im, bbox_xyxy, identities)

            end = time.time()
            print("time: {:.03f}s, fps: {:.03f}".format(end - start, 1 / (end - start)))

            if self.args.display:
                cv2.imshow("test", ori_im)
                cv2.waitKey(1)

            if self.args.save_path:
                self.writer.write(ori_im)

    def run_with_limit(self, frame_limit=10, saved_path=None):
        idx_frame = 0
        paths = []
        while self.vdo.grab() and idx_frame < frame_limit:
            idx_frame += 1
            if idx_frame % self.args.frame_interval:
                continue

            start = time.time()
            _, ori_im = self.vdo.retrieve()
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)

            # do detection
            bbox_xywh, cls_conf, cls_ids = self.detector(im)
            if bbox_xywh is not None:
                # select person class
                mask = cls_ids == 0

                bbox_xywh = bbox_xywh[mask]
                bbox_xywh[:, 3:] *= 1.2  # bbox dilation just in case bbox too small
                cls_conf = cls_conf[mask]

                # do tracking
                outputs = self.deepsort.update(bbox_xywh, cls_conf, im)

                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    ori_im = draw_boxes(ori_im, bbox_xyxy, identities)

            end = time.time()
            print("time: {:.03f}s, fps: {:.03f}".format(end - start, 1 / (end - start)))

            if self.args.display:
                cv2.imshow("test", ori_im)
                cv2.waitKey(1)

            if self.args.save_path:
                self.writer.write(ori_im)
            file_path = os.path.join(saved_path, '{}.png'.format(idx_frame))
            paths.append(os.path.split(file_path)[-1])  # 只返回文件名，不包含完整路径，这是为了配合Django的静态文件设置

            cv2.imwrite(file_path, ori_im)
        return paths


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default='TownCentreXVID.avi')
    parser.add_argument("--config_detection", type=str, default="./configs/yolov3.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="./demo/demo.avi")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=False)
    return parser.parse_args()


class Argument(object):
    def __init__(self, video_path):
        self.video_path = video_path
        self.config_detection = os.path.join(current_path, 'configs/yolov3.yaml')
        self.config_deepsort = os.path.join(current_path, 'configs/deep_sort.yaml')
        self.ignore_display = True
        self.display = False
        self.frame_interval = 1
        self.display_width = 800
        self.display_height = 600
        self.save_path = os.path.join(current_path, 'demo/demo.avi')
        self.cpu = False
        self.use_cuda = False


if __name__ == "__main__":
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_deepsort)

    with VideoTracker(cfg, args) as vdo_trk:
        vdo_trk.run()
