import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils


class MnistClassifier(torch.nn.Module):
    def __init__(self):
        super(MnistClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = x.view(-1, 784)
        return self.network(x)


class Runner:
    def __init__(self, args):
        # set seed and config
        self.set_seeds(args.seed)
        self.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
        self.model_path = f'./run1_net_{args.optim}.pth'
        self.num_epochs = args.epochs
        # log
        self.writer = SummaryWriter("./run1")
        # get data
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
        train_dataset = datasets.MNIST(root="./data/", transform=transform, train=True, download=True)
        test_dataset = datasets.MNIST(root='./data/', transform=transform, train=False, download=True)
        self.train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=32)
        self.test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=32)
        print(len(train_dataset))
        print(len(test_dataset))
        # model
        net = MnistClassifier()
        self.model = net.to(self.device)

        # optimizer and loss
        self.criterion = nn.CrossEntropyLoss()
        optim_dict = {
            'Adagrad': optim.Adagrad(net.parameters(), lr=0.001, lr_decay=0, weight_decay=0,
                                     initial_accumulator_value=0),
            'SGD': optim.SGD(net.parameters(), lr=0.001, momentum=0.9),
            'Adadelta': optim.Adadelta(net.parameters(), lr=0.001, rho=0.9, eps=1e-06, weight_decay=0),
            'RMSprop': optim.RMSprop(net.parameters(), lr=0.001, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0,
                                     centered=False),
            "Adam": optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0),
            'AdamW': optim.AdamW(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0),
            'Rprop': optim.Rprop(net.parameters(), lr=0.01, etas=(0.5, 1.2), step_sizes=(1e-06, 50)),
            'Adamax': optim.Adamax(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        }
        self.optimizer = optim_dict.get(args.optim)

    def set_seeds(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def train(self):
        train_acc = []
        train_loss = []
        log_freq = 10
        step = 0
        running_loss = 0
        for epoch in range(self.num_epochs):
            for i, data in enumerate(self.train_loader):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                # 输出loss
                running_loss += loss.item()
                if i % log_freq == 0:
                    # 训练曲线绘制
                    _, predicted = torch.max(outputs.data, 1)
                    total = labels.size(0)
                    correct = (predicted == labels).sum().item()
                    train_acc.append(100 * correct / total)
                    train_loss.append(loss.item())
                    print("[{} Epoch][{} Step] loss: {:.6f}".format(epoch + 1, step, running_loss / log_freq))
                    self.writer.add_scalar("loss/step", running_loss / log_freq, step)
                    self.writer.add_scalar("acc/step", 100 * correct / total,  step)
                    running_loss = 0.0
                step += 1
        print("Train Finished")
        # 保存模型
        torch.save(self.model.state_dict(), self.model_path)
        # draw
        train_iters = range(len(train_acc))
        self.draw_train_process("training", train_iters, train_loss, train_acc, "training loss", "training acc")

    def test(self):
        # 在测试集上整体的准确率
        self.model.load_state_dict(torch.load(self.model_path))
        correct = 0
        total = 0
        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.model(inputs)
                predicted = outputs.argmax(1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print("Acc:%.2f%%" % (100 * correct / total))

    def getOneImg(self):
        # 生成一张图片
        oneimg, label = self.train_loader.dataset[1]
        oneimg = oneimg.numpy().transpose(1, 2, 0)
        std = [0.5]
        mean = [0.5]
        oneimg = oneimg * std + mean
        oneimg.resize(28, 28)
        plt.imshow(oneimg)
        plt.show()

    def getOneBatchImgs(self):
        # 输出一个batch的图片
        images, labels = next(iter(self.train_loader))
        img = utils.make_grid(images)
        img = img.numpy().transpose(1, 2, 0)
        std = [0.5]
        mean = [0.5]
        img = img * std + mean
        for i in range(64):
            print(labels[i], end=" ")
            if i % 8 == 0:
                print(end="\n")
        plt.imshow(img)
        plt.show()

    def draw_train_process(self, title, iters, costs, accs, label_cost, label_acc):
        plt.title(title, fontsize=24)
        plt.xlabel('iter', fontsize=20)
        plt.ylabel('acc(\%)', fontsize=20)
        plt.plot(iters, costs, color='red', label=label_cost)
        plt.plot(iters, accs, color='green', label=label_acc)
        plt.legend()
        plt.grid()
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train")
    parser.add_argument('--optim', default='AdamW', type=str)
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    runner = Runner(args)
    runner.getOneImg()
    runner.getOneBatchImgs()
    runner.train()
    runner.test()
