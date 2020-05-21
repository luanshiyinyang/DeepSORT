"""
Author: Zhou Chen
Date: 2020/5/21
Desc: desc
"""
import matplotlib.pyplot as plt


def parse_txt(filepath="log_train.txt"):
    loss_list = []
    with open(filepath, 'r', encoding="utf8") as f:
        line = f.readline().strip()
        while line:
            loss = float(line.split(" ")[2].split(":")[-1])
            loss_list.append(loss)
            line = f.readline().strip()
    return loss_list


def draw_his(loss):
    plt.figure()
    plt.plot(list(range(len(loss))), loss)
    plt.savefig('loss.png')
    plt.show()


if __name__ == '__main__':
    rst = parse_txt()
    draw_his(rst)