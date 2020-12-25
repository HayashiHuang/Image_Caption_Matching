import torch.nn as nn
import torch
import numpy as np
from torchvision import models
from loss import comput_hard_triplet
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
    def __init__(self, vocab, hidden_size=2048, emb_dim=300, n_layer=1, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(len(vocab), emb_dim)
        self.embed.weight.data.copy_(vocab.vectors)
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=hidden_size,
                            num_layers=n_layer, dropout=dropout, batch_first=True)

    def forward(self, x):
        x = self.embed(x)
        y, _ = self.lstm(x)
        y = y[:, -1, :]
        return y


class Net(nn.Module):
    def __init__(self, vocab, feature_size):
        super().__init__()
        self.ImNet = ImNet()
        self.TxNet = TxNet(vocab=vocab, hidden_size=feature_size)
        self.feature_size = feature_size

    def forward(self, x, mode):
        X_img = x[0]  # [batch_size, 3(rgb), 224, 224]
        X_cap = x[1]  # [batch_size, 5, cap_len]
        if mode == 'text':
            X_cap = X_cap.view(-1, X_cap.shape[2])
            y = self.TxNet(X_cap)
        else:
            X_cap = X_cap.view(-1, X_cap.shape[2])
            y_cap = self.TxNet(X_cap)
            y_img = self.ImNet(X_img)
            y = [y_cap, y_img]
        self.y = y
        return y

    def loss(self, mode):
        # TODO: loss function
        if mode == 'text':
            label = [i for i in range(int(self.y.shape[0]/5))]
            label = torch.tensor(label).view(len(label), 1).cuda()
            label = label.repeat(1, 5).view(-1)
            loss, acc = comput_hard_triplet(self.y, label)
        else:
            loss = self.y[0][0, 0, 1] - self.y[0][0, 0, 0]
        return loss, acc
