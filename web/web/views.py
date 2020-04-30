"""
Author: Zhou Chen
Date: 2020/3/17
Desc: desc
"""
from django.shortcuts import render
import re
import mimetypes
from wsgiref.util import FileWrapper
from django.http import StreamingHttpResponse
import os
import uuid
from .settings import MEDIA_ROOT, STATIC_ROOT
import sys
sys.path.append("../")
import yolo3_deepsort


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
            # 视频保存本机之后调用模型

            """args = yolo3_deepsort.Argument(file_path)
            cfg = yolo3_deepsort.get_config()
            cfg.merge_from_file(args.config_detection)
            cfg.merge_from_file(args.config_deepsort)
            with yolo3_deepsort.VideoTracker(cfg, args, file_path) as vdo_trk:
                images = vdo_trk.run_with_limit(30, save_path=STATIC_ROOT + '/images/')
            """
            return render(request, 'show_video.html', {'filename': 'result.mp4'})
        else:
            return render(request, 'upload.html')
    return render(request, 'upload.html')


def file_iterator(file_name, chunk_size=8192, offset=0, length=None):
    with open(file_name, "rb") as f:
        f.seek(offset, os.SEEK_SET)
        remaining = length
        while True:
            bytes_length = chunk_size if remaining is None else min(remaining, chunk_size)
            data = f.read(bytes_length)
            if not data:
                break
            if remaining:
                remaining -= len(data)
            yield data


def stream_video(request):
    path = request.GET.get('path')
    path = os.path.join("static", "videos", path)
    range_header = request.META.get('HTTP_RANGE', '').strip()
    range_re = re.compile(r'bytes\s*=\s*(\d+)\s*-\s*(\d*)', re.I)
    range_match = range_re.match(range_header)
    size = os.path.getsize(path)
    content_type, encoding = mimetypes.guess_type(path)
    content_type = content_type or 'application/octet-stream'
    if range_match:
        first_byte, last_byte = range_match.groups()
        first_byte = int(first_byte) if first_byte else 0
        last_byte = first_byte + 1024 * 1024 * 8  # 8M 每片,响应体最大体积
        if last_byte >= size:
            last_byte = size - 1
        length = last_byte - first_byte + 1
        resp = StreamingHttpResponse(file_iterator(path, offset=first_byte, length=length), status=206,
                                     content_type=content_type)
        resp['Content-Length'] = str(length)
        resp['Content-Range'] = 'bytes %s-%s/%s' % (first_byte, last_byte, size)
    else:
        # 不是以视频流方式的获取时，以生成器方式返回整个文件，节省内存
        resp = StreamingHttpResponse(FileWrapper(open(path, 'rb')), content_type=content_type)
        resp['Content-Length'] = str(size)
    resp['Accept-Ranges'] = 'bytes'
    return resp