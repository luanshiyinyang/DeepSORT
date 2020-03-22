import os
import glob
import shutil
import random
import tqdm


data_folder = '../dataset/MARS/'
train_folder = '../dataset/MARS-generated/bbox_train/'
test_folder = '../dataset/MARS-generated/bbox_test/'


def check_folder():
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    if not os.path.exists(train_folder):
        os.mkdir(train_folder)
    if not os.path.exists(test_folder):
        os.mkdir(test_folder)


def split_dataset():
    """
    划分训练集和测试集
    :return:
    """
    raw_data_folder = '../dataset/MARS/'
    categories = os.listdir(raw_data_folder)
    label_list = []
    for category in tqdm.tqdm(categories):
        label = categories.index(category)
        label_list.append(label)
        category_folder = os.path.join(raw_data_folder, category)
        files = glob.glob(category_folder + '/*.jpg')
        random.shuffle(files)
        train_size = int(0.8 * len(files))
        test_size = int(0.2 * len(files))
        train_files = files[:train_size]
        test_files = files[train_size:]
        out_path = os.path.join(train_folder, str(label))
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        for img in train_files:
            shutil.copy(img, os.path.join(out_path, os.path.split(img)[-1]))
        out_path = os.path.join(test_folder, str(label))
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        for img in test_files:
            shutil.copy(img, os.path.join(out_path, os.path.split(img)[-1]))


if __name__ == '__main__':
    check_folder()
    split_dataset()
