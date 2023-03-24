import argparse
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image


class VAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=128, hidden_dims=None):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3,
                      kernel_size=3, stride=2, padding=1),
            nn.Tanh()
        )
        self.weights_init()

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.zero_()

    def encode(self, input, label=None):
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = logvar.mul(0.5).exp_()
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, label=None):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var


class CVAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=128, hidden_dims=None, num_classes=10, img_size=32):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.embed_class = nn.Linear(num_classes, img_size * img_size)
        self.embed_data = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        in_channels += 1  # To account for the extra label channel
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim + num_classes, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3,
                      kernel_size=3, stride=2, padding=1),
            nn.Sigmoid()
        )

    def encode(self, input):

        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = logvar.mul(0.5).exp_()
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, label=None):
        embedded_class = self.embed_class(label)
        embedded_class = embedded_class.view(-1, self.img_size, self.img_size).unsqueeze(1)
        embedded_input = self.embed_data(input)
        input = torch.cat([embedded_input, embedded_class], dim=1)
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        z = torch.cat((z, label), dim=1)
        return self.decode(z), mu, log_var


class Runner:
    def __init__(self, args):
        # set seed and config
        self.set_seeds(args.seed)
        self.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
        self.model_path = f'./run4_net_{args.model}.pth'
        self.num_epochs = args.epochs
        # log
        self.writer = SummaryWriter(f"./run4/{args.model}")
        # get data
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = datasets.CIFAR10(root="./data/", transform=transform, train=True, download=True)
        test_dataset = datasets.CIFAR10(root='./data/', transform=transform, train=False, download=True)
        self.train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=32)
        self.test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=32)
        print(len(train_dataset))
        print(len(test_dataset))
        # model
        if args.model == "cvae":
            net = CVAE()
        else:
            net = VAE()
        self.model = net.to(self.device)
        # optimizer
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

    def criterion(self, recon_x, x, mu, logvar):
        """
        损失函数BCE+KLD
        :param recon_x: 重构图像
        :param x: 原图像
        :param mu: 特征均值
        :param logvar: 特征方差
        :return:
        """
        re_loss = F.mse_loss(recon_x, x, reduction="sum") / x.size(0)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())

        return re_loss, kl_loss

    def idx2onehot(self, data):
        labels = torch.zeros((data.shape[0], 10))
        labels = labels.scatter_(1, data.unsqueeze(1), 1)
        return labels

    def train(self):
        # 获取一个batch图像，观察训练过程
        fixed_loader = DataLoader(self.train_loader.dataset, batch_size=32, shuffle=False, num_workers=32)
        fixed_x, fixed_y = next(iter(fixed_loader))

        save_image(Variable(fixed_x).data.cpu(), './data/real_images.png')
        self.writer.add_images("images/real", fixed_x.data.cpu(), 0)
        for i in range(4):
            self.writer.add_image(f"image_real/{i}", fixed_x[i].data.cpu(), 0)
        fixed_y = self.idx2onehot(fixed_y)
        fixed_x, fixed_y = fixed_x.to(self.device), fixed_y.to(self.device)

        # Train
        for epoch in range(1, self.num_epochs + 1):
            running_loss = 0.0
            re_losses = 0.0
            kl_losses = 0.0
            for i, data in enumerate(self.train_loader):
                labels = self.idx2onehot(data[1])
                inputs, labels = data[0].to(self.device), labels.to(self.device)
                output, mu, logvar = self.model(inputs, labels)
                re_loss, kl_loss = self.criterion(output, inputs, mu, logvar)
                loss = re_loss + 0.000025 * kl_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # 输出loss
                running_loss += loss.item()
                re_losses += re_loss.item()
                kl_losses += kl_loss.item()
            # 记录损失
            print("[{} Epoch] loss: {:.6f} re_loss: {:.6f} kl_loss: {:.6f}".format(
                epoch,
                running_loss / len(self.train_loader),
                re_losses / len(self.train_loader),
                kl_losses / len(self.train_loader))
            )
            self.writer.add_scalar("loss/total_loss", running_loss / len(self.train_loader), epoch)
            self.writer.add_scalar("loss/re_loss", re_losses / len(self.train_loader), epoch)
            self.writer.add_scalar("loss/kl_loss", kl_losses / len(self.train_loader), epoch)
            if epoch % 5 == 0:
                # save the reconstructed images
                reconst_images, mu, logvar = self.model(fixed_x, fixed_y)
                reconst_images = reconst_images.view(reconst_images.size(0), 3, 32, 32)
                save_image(reconst_images.data.cpu(), './data/reconst_images_%d.png' % (epoch))
                self.writer.add_images('images', reconst_images.data.cpu(), epoch)
                for i in range(4):
                    self.writer.add_image(f"image/{i}", reconst_images[i].data.cpu().data.cpu(), epoch)

        print("Train Finished")
        # 保存模型
        torch.save(self.model.state_dict(), self.model_path)

    def test(self):
        # 在测试集上整体的准确率
        self.model.load_state_dict(torch.load(self.model_path))
        total_loss = 0.0
        re_losses = 0.0
        kl_losses = 0.0
        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                labels = self.idx2onehot(data[1])
                inputs, labels = data[0].to(self.device), labels.to(self.device)
                output, mu, logvar = self.model(inputs, labels)
                re_loss, kl_loss = self.criterion(output, inputs, mu, logvar)
                loss = re_loss + kl_loss
                total_loss += loss.item()
                re_losses += re_loss.item()
                kl_losses += kl_loss.item()
        print(
            f"Test Loss: {total_loss / len(self.test_loader)} Re_loss:{re_losses / len(self.test_loader)} KL_loss: {kl_losses / len(self.test_loader)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train")
    parser.add_argument('--optim', default='AdamW', type=str)
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--model', default='cvae', type=str)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    runner = Runner(args)
    runner.train()
    runner.test()
