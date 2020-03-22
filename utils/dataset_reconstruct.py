"""
Author: Zhou Chen
Date: 2020/3/1
Desc: desc
"""
import os
import shutil
import re
import tqdm


def reconstruct_market1501(source_path, generate_path):
    """
    重构MARKET数据集为不同的行人在不同的文件夹下（MARS数据集就是这种格式，无需重构）
    """
    img_names = os.listdir(source_path)
    pattern = re.compile(r'([-\d]+)_c(\d)')
    for img_name in tqdm.tqdm(img_names):
        if '.jpg' not in img_name:
            continue
        # pid: 每个人的标签编号 1
        # _  : 摄像头号 2
        pid, _ = map(int, pattern.search(img_name).groups())
        # 去掉没用的图片
        if pid == 0 or pid == -1:
            # 不处理的无用图片
            continue
        target_folder = os.path.join(generate_path, str(pid))
        if not os.path.exists(target_folder):
            os.mkdir(target_folder)
        shutil.copy(os.path.join(source_path, img_name), os.path.join(target_folder, img_name))


if __name__ == '__main__':
    src_dir = r'data/Market-1501-v15.09.15/'
    target_dir = r'data/Market-generated/'
    reconstruct_market1501(src_dir, target_dir)