import torch
import torch.nn as nn
import torch.nn.functional as F

class MPred(nn.Module):
    def __init__(self):
        super(MPred, self).__init__()

    def forward(self, x, img_size=416):
        x = x.cpu()
        batches = x.shape[0]
        cell_num = x.shape[1]
        for i in range(batches):
            for j in range(cell_num):
                out = x[i, j, :]
                out[0] = torch.sigmoid(out[0])
                out[1] = torch.sigmoid(out[1])
                out[2] = torch.exp(out[2])/img_size
                out[3] = torch.exp(out[3])/img_size
                out[4] = torch.sigmoid(out[4])
        return x

class MLoss(nn.Module):
    def __init__(self):
        super(MLoss, self).__init__()

    def forward(self, x, y, img_size=416, thresh=0.5, alpha=0.1):
        x = x.cpu()
        y = y.cpu()
        batches = x.shape[0]
        cell_num = x.shape[1]
        loss = torch.zeros(x.shape[0], x.shape[2]).cpu()
        for i in range(batches):
            for j in range(cell_num):
                out = x[i, j, :]
                out[0] = torch.sigmoid(out[0])
                out[1] = torch.sigmoid(out[1])
                out[2] = torch.exp(out[2])/img_size
                out[3] = torch.exp(out[3])/img_size
                out[4] = torch.sigmoid(out[4])
                label = y[i, j, :]
                if label[4].item() > thresh:
                    loss[i, :] += torch.pow((label - out), 2)
                else:
                    loss[i, 4] += alpha * torch.pow((label[4] - out[4]), 2)
        return loss.mean()
