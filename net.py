import torch
import torch.nn as nn
import numpy as np
from torchvision import models
from loss import comput_hard_triplet
from text_preprocess import flatten
import pdb


class ImNet(models.ResNet):
    def __init__(self):
        super().__init__(block=models.resnet.Bottleneck, layers=[3, 4, 6, 3])

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        y = x.view(x.size(0), -1)
        return y


class TxNet(nn.Module):
    def __init__(self, args, vocab):
        super().__init__()
        self.embed = nn.Embedding(vocab.vectors.shape[0], vocab.vectors.shape[1])
        self.embed.weight.data.copy_(vocab.vectors)
        self.lstm = nn.LSTM(input_size=vocab.vectors.shape[1], hidden_size=args.lstm_hidden,
                            num_layers=args.lstm_layer, dropout=args.lstm_dropout, batch_first=True)
        self.bn = nn.BatchNorm1d(args.lstm_hidden)
        self.fc = nn.Linear(args.lstm_hidden, args.feat_dim)

    def forward(self, x):
        label = flatten([[i + 2] * 5 for i in range(int(x.shape[0] / 5))])
        label = torch.tensor(label).cuda()
        x_len = (1 - (x == 1).int()).sum(dim=1)
        x_len, index = torch.sort(x_len, descending=True)
        label = label.index_select(0, index)
        x = self.embed(x)
        x = x.index_select(0, index)

        x = nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=True)
        y, h = self.lstm(x)
        y = h[0][0, :, :]
        y = self.bn(y)
        y = self.fc(y)
        return y, label


class Net(nn.Module):
    def __init__(self, args, vocab):
        super().__init__()
        self.ImNet = ImNet()
        self.TxNet = TxNet(args=args, vocab=vocab)

    def forward(self, x, mode):
        X_img = x[0]  # [batch_size, 3(rgb), 224, 224]
        X_cap = x[1]  # [batch_size, 5, cap_len]
        if mode == 'text':
            X_cap = X_cap.view(-1, X_cap.shape[2])
            y, label = self.TxNet(X_cap)
            self.label = label
        else:
            X_cap = X_cap.view(-1, X_cap.shape[2])
            y_cap, label = self.TxNet(X_cap)
            y_img = self.ImNet(X_img)
            y = [y_cap, y_img]
        self.y = y
        return y

    def loss(self, mode, epoch):
        # TODO: loss function
        if mode == 'text':
            loss, acc = comput_hard_triplet(self.y, self.label, epoch)
        else:
            loss = self.y[0][0, 0, 1] - self.y[0][0, 0, 0]
        return loss, acc
