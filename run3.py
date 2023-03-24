import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image


class AE(nn.Module):
    def __init__(self, in_dim=784, h_dim=400):
        super(AE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(h_dim, in_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out


class DeepAE(nn.Module):
    def __init__(self, in_dim=784, h_dim=400):
        super(DeepAE, self).__init__()

        self.encoder = nn.Sequential(nn.Linear(in_dim, 128),
                                     nn.ReLU(True),
                                     nn.Linear(128, 64),
                                     nn.ReLU(True),
                                     nn.Linear(64, h_dim)
                                     )
        self.decoder = nn.Sequential(
            nn.Linear(h_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, in_dim),
            nn.Sigmoid())

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out


class Runner:
    def __init__(self, args):
        # set seed and config
        self.set_seeds(args.seed)
        self.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
        self.model_path = f'./run3_net_{args.optim}.pth'
        self.num_epochs = args.epochs
        # log
        self.writer = SummaryWriter("./run3")
        # get data
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
        train_dataset = datasets.MNIST(root="./data/", transform=transform, train=True, download=True)
        test_dataset = datasets.MNIST(root='./data/', transform=transform, train=False, download=True)
        self.train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=32)
        self.test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=32)
        print(len(train_dataset))
        print(len(test_dataset))
        # model
        net = AE(in_dim=28 * 28, h_dim=30)
        # net = DeepAE(in_dim=28 * 28, h_dim=30)
        self.model = net.to(self.device)

        # optimizer and loss
        # self.criterion = nn.BCELoss()
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
        # save fixed inputs for debugging
        fixed_x, _ = next(iter(self.train_loader))
        save_image(Variable(fixed_x).data.cpu(), './data/real_images.png')
        self.writer.add_images("images", fixed_x.data.cpu(), 1)
        fixed_x = fixed_x.view(fixed_x.size(0), -1).to(self.device)
        log_freq = 100
        step = 0
        running_loss = 0.0
        for epoch in range(self.num_epochs):
            for i, data in enumerate(self.train_loader):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                inputs = inputs.view(inputs.size(0), -1)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, inputs)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if step % log_freq == 0:
                    # 训练曲线绘制
                    print("[{} Epoch][{} Step] loss: {:.6f}".format(epoch + 1, step, running_loss / log_freq))
                    self.writer.add_scalar("loss/step", running_loss / log_freq, step)
                    running_loss = 0.0
                step += 1
            # save the reconstructed images
            reconst_images = self.model(fixed_x)
            reconst_images = reconst_images.view(reconst_images.size(0), 1, 28, 28)
            save_image(reconst_images.data.cpu(), './data/reconst_images_%d.png' % (epoch + 1))
            self.writer.add_images('images', reconst_images.data.cpu(), epoch + 2)

        print("Train Finished")
        # 保存模型
        torch.save(self.model.state_dict(), self.model_path)

    def test(self):
        # 在测试集上整体的准确率
        self.model.load_state_dict(torch.load(self.model_path))
        total_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                inputs = inputs.view(inputs.size(0), -1)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, inputs)
                total_loss += loss.item()
        print(f"Test Loss: {total_loss / len(self.test_loader)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train")
    parser.add_argument('--optim', default='SGD', type=str)
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()
    runner = Runner(args)
    runner.train()
    runner.test()
