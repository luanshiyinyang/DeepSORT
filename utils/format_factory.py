"""
Author: Zhou Chen
Date: 2020/4/2
Desc:  进行视频格式和编码的转换，需要安装ffmpeg包并加入当前环境的环境变量
"""
from ffmpy3 import FFmpeg


def avi2mp4(source_path: str, target_path:str):
    print("start transformation")
    ff = FFmpeg(
        inputs={source_path: '-f avi'},
        outputs={target_path: '-f mp4'}
    )
    print(ff.cmd)
    ff.run()
    print("finish transformation")


if __name__ == '__main__':
    # 测试脚本
    avi2mp4("../result/result.avi", "../result/result.mp4")