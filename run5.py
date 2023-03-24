import argparse
import os
import random
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch import optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


class Tokenizer():
    def __init__(self):
        self.vocab = []
        with open('vocab.txt', 'r', encoding='utf-8') as f:
            self.vocab = [line.strip(' \n') for line in f.readlines()]
        self.vocab_dict = {char: i for i, char in enumerate(self.vocab)}
        self.UNK = self.vocab_dict['[UNK]']
        self.SOS = self.vocab_dict['[SOS]']
        self.EOS = self.vocab_dict['[EOS]']
        self.max_seq_length = 15

    def encode(self, txt):
        txt = txt.lower()
        if len(txt) > self.max_seq_length:
            txt = txt[:self.max_seq_length - 2]

        return [self.SOS] + [self.vocab_dict.get(char, self.UNK) for char in txt] + [self.EOS]

    def decode(self, idxs):
        return [self.vocab[idx] for idx in idxs]

    @property
    def vocab_size(self):
        return len(self.vocab)


class NameDatasets(Dataset):
    def __init__(self, data_path="/home/jiangzheng/data/name"):
        super(NameDatasets, self).__init__()
        self.data = []
        for file in os.listdir(data_path):
            with open(os.path.join(data_path, file), 'r', encoding='utf-8') as f:
                names = [line.strip(' \n') for line in f.readlines()]
                self.data.extend(names)

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


class NameGeneratorNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, embedding_size=128, num_layers=1):
        super(NameGeneratorNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, inputs, hiddens):
        inputs = self.embedding(inputs)
        outputs, hiddens = self.lstm(inputs, hiddens)
        outputs = self.fc(outputs)
        outputs = self.softmax(outputs)
        return outputs, hiddens

    def init_hiddens(self, batch_size, device=None):
        hiddens = (torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device),
                   torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device))
        return hiddens


def collate(batch, tokenizer):
    inputs, targets = [], []
    for txt in batch:
        txt = tokenizer.encode(txt)
        inputs.append(torch.tensor(txt[:-1]))  # 输入数据
        targets.append(torch.tensor(txt[1:]))  # 标签数据
    inputs = pad_sequence(inputs, batch_first=True)
    targets = pad_sequence(targets, batch_first=True)
    return inputs, targets


class Runner:

    def __init__(self, args):
        # set seed and config
        self.set_seeds(args.seed)
        self.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
        self.model_path = f'./net_{args.optim}.pth'
        # log
        self.writer = SummaryWriter("./run5")
        # get data
        train_datasets = NameDatasets()
        self.tokenizer = Tokenizer()

        collate_fn = partial(collate, tokenizer=self.tokenizer)
        self.train_loader = DataLoader(train_datasets, batch_size=64, shuffle=True, num_workers=32,
                                       collate_fn=collate_fn)

        # model
        net = NameGeneratorNet(input_size=self.tokenizer.vocab_size, hidden_size=256, embedding_size=128,
                               output_size=self.tokenizer.vocab_size)
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

    def criterion(self, inputs, targets):
        inputs = inputs.permute(0, 2, 1)
        return F.nll_loss(inputs, targets)

    def train(self):
        # Train
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, (inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                hiddens = self.model.init_hiddens(inputs.size(0), device=self.device)
                outputs, hiddens = self.model(inputs, hiddens)
                loss = self.criterion(outputs, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # 输出loss
                running_loss += loss.item()
            print("[{} Epoch] loss: {:.6f}".format(epoch + 1, running_loss / len(self.train_loader)))
            self.writer.add_scalar("loss/step", running_loss / len(self.train_loader), epoch)

        print("Train Finished")
        # 保存模型
        torch.save(self.model.state_dict(), self.model_path)

    def predict(self, txt, max_length=100, topk=5):
        # 在测试集上整体的准确率
        self.model.load_state_dict(torch.load(self.model_path))
        txt = self.tokenizer.encode(txt)[:-1]
        hiddens = self.model.init_hiddens(1, device=self.device)
        latter_chars = []
        for i in range(max_length):
            if i < len(txt):
                inputs = torch.LongTensor(1, 1).fill_(txt[i]).to(self.device)
            else:
                inputs = s[:, :, 0].to(self.device)
            with torch.no_grad():
                pred, hiddens = self.model(inputs, hiddens)
                prob, s = torch.topk(pred.exp(), topk, largest=True, sorted=True) # 最大可能字符
                if s[:, :, 0].item() == self.tokenizer.EOS:
                    break
                latter_chars.append(s)
        latter_chars = torch.cat(latter_chars, dim=1).squeeze(0).tolist()
        return latter_chars


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train")
    parser.add_argument('--optim', default='AdamW', type=str)
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    num_epochs = 100
    args = parser.parse_args()
    runner = Runner(args)
    # runner.train()
    txt = "Alle"
    latter_chars = runner.predict(txt)
    print(f"开始生成：{txt}..")
    for i,chars in enumerate(latter_chars):
        print(f"Time[{i}] Top5:" + " ".join(runner.tokenizer.decode(chars)))
    print(f"Final name:{txt + ''.join(runner.tokenizer.decode(list(zip(*latter_chars))[0]))}")
