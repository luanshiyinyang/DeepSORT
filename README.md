# DeepSORT multi-object detection implementation by pytorch
> This project refer to [ZQPei's repo](https://github.com/ZQPei/deep_sort_pytorch)，i did some optimization.


## environment
install several following packages, more info in [requirements](./requirements.txt).

- pytorch>=1.0
- numpy
- scipy


## run demo
use next code in terminal to run tracking in a video file.

`python yolo3_deepsort.py --video_path ../data/TownCenter.avi`

use next code in terminal to run tracking in your camera capture.

`python yolo3_deepsort_camera.py`

## results
sample result as following picture.

![](./assets/demo.gif)