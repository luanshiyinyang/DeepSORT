import argparse
import os
import time
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torchvision

from model import Net

# 命令行参数配置
parser = argparse.ArgumentParser(description="Train on MARS")
parser.add_argument("--data-dir", default='/SISDC_GPFS/Home_SE/jiangm-jnu/xiaf-jnu/zhouchen/dataset/MARS-generated/', type=str)  # 修改为自己的数据集目录
parser.add_argument("--gpu-id", default=0, type=int)
parser.add_argument("--lr", default=0.1, type=float)
parser.add_argument("--interval", '-i', default=20, type=int)
parser.add_argument('--resume', '-r', action='store_true', default=False)
args = parser.parse_args()

# 确定训练设备
device = "cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu"

if torch.cuda.is_available():
    cudnn.benchmark = True  # 对固定的网络结构优化

# 数据载入
root = args.data_dir
train_dir = os.path.join(root, "bbox_train")
test_dir = os.path.join(root, "bbox_test")
# 图像预处理
transform_train = torchvision.transforms.Compose([
    torchvision.transforms.Resize((128, 64)),  # 如果采用Market数据集这一步可以删去，Mars必须要这一步
    torchvision.transforms.RandomCrop((128, 64), padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize((128, 64)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

trainloader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(train_dir, transform=transform_train),
    batch_size=128, shuffle=True
)

testloader = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(test_dir, transform=transform_test),
    batch_size=128, shuffle=True
)

num_classes = len(trainloader.dataset.classes)

# net definition
start_epoch = 0
net = Net(num_classes=num_classes)

if args.resume:
    # 是否使用预训练参数
    assert os.path.isfile("./checkpoint/ckpt.t7"), "Error: no checkpoint file found!"
    print('Loaded pretrained weights from checkpoint file')
    checkpoint = torch.load("./checkpoint/ckpt.t7")  # 该字典含有net_dict，acc，epoch三个键
    net_dict = checkpoint['net_dict']
    net.load_state_dict(net_dict)

net.to(device)

# 使用交叉熵和SGD
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), args.lr, momentum=0.9, weight_decay=5e-4)
best_acc = 0.0


# train function for each epoch
def train(epoch):
    print("Epoch{}".format(epoch + 1))
    print("Training...")
    net.train()
    training_loss = 0.
    train_loss = 0.
    correct = 0
    total = 0
    interval = args.interval
    start = time.time()
    for idx, (inputs, labels) in enumerate(trainloader):
        # 前向传播
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 计算指标
        training_loss += loss.item()
        train_loss += loss.item()
        correct += outputs.max(dim=1)[1].eq(labels).sum().item()
        total += labels.size(0)

        if (idx + 1) % interval == 0:
            # 固定step输出一次信息
            end = time.time()
            print("[Progress:{:.1f}%] time:{:.2f}s Loss:{:.5f}  Acc:{:.3f}%".format(
                100. * (idx + 1) / len(trainloader), end - start, training_loss / interval,
                100. * correct / total
            ))
            training_loss = 0.0
            start = time.time()

    return train_loss / len(trainloader), 1. - correct / total


def test(epoch):
    global best_acc
    print("Testing...")
    net.eval()
    test_loss = 0.
    correct = 0
    total = 0
    start = time.time()
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            correct += outputs.max(dim=1)[1].eq(labels).sum().item()
            total += labels.size(0)

        end = time.time()
        print("[progress:{:.1f}%]time:{:.2f}s Loss:{:.5f} Correct:{}/{} Acc:{:.3f}%".format(
            100. * (idx + 1) / len(testloader), end - start, test_loss / len(testloader), correct, total,
            100. * correct / total
        ))

    # 保存训练参数
    acc = 100. * correct / total
    if acc > best_acc:
        # 始终保留最好的参数，如果过拟合，则不保留参数
        best_acc = acc
        print("Saving parameters to checkpoint/ckpt.t7")
        checkpoint = {
            'net_dict': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(checkpoint, './checkpoint/ckpt.t7')

    return test_loss / len(testloader), 1. - correct / total


# 绘制训练曲线
x_epoch = []
record = {'train_loss': [], 'train_err': [], 'test_loss': [], 'test_err': []}
fig = plt.figure(figsize=(18, 6))
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="err")


def draw_curve(epoch, train_loss, train_err, test_loss, test_err):
    global record
    record['train_loss'].append(train_loss)
    record['train_err'].append(train_err)
    record['test_loss'].append(test_loss)
    record['test_err'].append(test_err)

    x_epoch.append(epoch)
    ax0.plot(x_epoch, record['train_loss'], 'bo-', label='training')
    ax0.plot(x_epoch, record['test_loss'], 'ro-', label='validation')
    ax1.plot(x_epoch, record['train_err'], 'bo-', label='training')
    ax1.plot(x_epoch, record['test_err'], 'ro-', label='validation')
    if epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig("train.png")


def lr_decay():
    # 设置学习率衰减
    global optimizer
    for params in optimizer.param_groups:
        params['lr'] *= 0.1
        lr = params['lr']
        print("Learning rate adjusted to {}".format(lr))


def main():
    # 训练50轮即达到饱和
    for epoch in range(50):
        train_loss, train_err = train(epoch)
        test_loss, test_err = test(epoch)
        draw_curve(epoch, train_loss, train_err, test_loss, test_err)
        if (epoch + 1) % 20 == 0:
            lr_decay()


if __name__ == '__main__':
    main()
