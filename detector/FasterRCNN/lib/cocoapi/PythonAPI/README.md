please download pycocotools pythonAPI in this folder and install it.

## 方法1
`pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI`

使用上述命令在虚拟环境下编译安装该模块，需要事先安装Visual C++ Build Tools。

## 方法2
从[指定地址](https://github.com/philferriere/cocoapi.g)下载Pycocotools的PythonAPI模块并且在本目录解压安装至当前虚拟环境。这是一个对官方cocoapi修改的可在Windows下运行的cocoapi工具包，需要安装Visual C++ Build Tools。

使用下述命令来安装该Python包。

`python setup.py build_ext --inplace`

`python setup.py build_ext install`