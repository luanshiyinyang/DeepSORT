"""
Author: Zhou Chen
Date: 2020/3/17
Desc: desc
"""
from django.shortcuts import render
import os
import uuid
from .settings import MEDIA_ROOT, STATIC_ROOT
import sys
sys.path.append("../")
import yolov3_deepsort


def upload(request):
    if request.method == 'POST':
        files = request.FILES['video']
        if len(files) > 0:
            if not os.path.exists(MEDIA_ROOT):
                # 若不存在媒体存储目录
                os.mkdir(MEDIA_ROOT)
            video = files
            extension = os.path.splitext(video.name)[1]
            # 重命名文件
            file_name = '{}{}'.format(uuid.uuid4(), extension)
            file_path = '{}/{}'.format(MEDIA_ROOT, file_name)
            # 保存文件到本机
            with open(file_path, 'wb') as f:
                for c in video.chunks():
                    f.write(c)
            # 保存本机之后调用模型
            # args = yolov3_deepsort.Argument(file_path)
            # cfg = yolov3_deepsort.get_config()
            # cfg.merge_from_file(args.config_detection)
            # cfg.merge_from_file(args.config_deepsort)
            # with yolov3_deepsort.VideoTracker(cfg, args, file_path) as vdo_trk:
            #     images = vdo_trk.run_with_limit(30, saved_path=STATIC_ROOT + '/images/')
            images = ["{}.png".format(i) for i in range(1, 31)]
            return render(request, 'show.html', {'images': images})
        else:
            return render(request, 'upload.html')
    return render(request, 'upload.html')