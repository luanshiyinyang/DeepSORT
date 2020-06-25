# 基于pytorch实现DeepSORT多目标跟踪
> 该仓库参考 [ZQPei的项目](https://github.com/ZQPei/deep_sort_pytorch)，我在其基础上进行了一些优化。

## 环境配置
基于Python3.6并在虚拟环境下安装如下几个核心包即可，具体见[requirements](./requirements.txt)文件即可。

- pytorch>=1.0
- numpy
- scipy

## 运行脚本
使用如下命令对视频进行跟踪。

`python yolo3_deepsort.py --video_path ../data/TownCenter.avi`

使用如下命令，打开默认摄像头，实时跟踪。

`python yolo3_deepsort_camera.py`


## 跟踪结果
在示例视频上跟踪效果如下图。

![](./assets/demo.gif)





