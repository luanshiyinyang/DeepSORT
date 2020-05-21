import torch
import torchvision.transforms as transforms
import numpy as np
import cv2

from .model import Net


class Extractor(object):
    def __init__(self, model_path, use_cuda=True):
        self.net = Net(reid=True, num_classes=1261)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)['net_dict']
        self.net.load_state_dict(state_dict)
        print("Loaded weights from {}.".format(model_path))
        self.net.to(self.device)
        self.size = (64, 128)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _preprocess(self, im_crops):
        """
        特征提取器的图像预处理
        归一到0-1
        调整图像大小
        图像标准化
        Torch张量化
        :param im_crops: 一个batch的RGB图像（单图需要放在列表中）
        :return:
        """
        def _resize(im, size):
            return cv2.resize(im.astype(np.float32)/255., size)

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
        return im_batch

    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.net(im_batch)
        return features.cpu().numpy()


def test():
    def cosine(a, b, data_is_normalized=False):
        if not data_is_normalized:
            a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
            b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
        return np.dot(a, b.T)

    img1 = cv2.cvtColor(cv2.resize(cv2.imread("1.jpg"), (64, 128)), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.resize(cv2.imread("2.jpg"), (64, 128)), cv2.COLOR_BGR2RGB)
    img3 = cv2.cvtColor(cv2.resize(cv2.imread("3.jpg"), (64, 128)), cv2.COLOR_BGR2RGB)
    imgs = [img1, img2, img3]
    extractor = Extractor("checkpoint/ckpt.t7")
    feature = extractor(imgs)
    a = feature[0]
    b = feature[1]
    c = feature[2]
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.title(" ")
    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    fig.suptitle("Cosine similarity:" + str(cosine(a.reshape(1, -1), b.reshape(1, -1), True)[0][0]) + "\n")
    plt.title(" ")
    plt.savefig("true.png")
    plt.show()

    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.title(" ")
    plt.subplot(1, 2, 2)
    plt.imshow(img3)
    fig.suptitle("Cosine similarity:" + str(cosine(a.reshape(1, -1), c.reshape(1, -1), True)[0][0]) + "\n")
    plt.title(" ")
    plt.savefig("false.png")
    plt.show()


if __name__ == '__main__':
    # 测试提取器
    test()





