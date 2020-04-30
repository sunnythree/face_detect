import torch
import torch.nn as nn
import torch.nn.functional as F


class MLoss(nn.Module):
    def __init__(self):
        super(MLoss, self).__init__()

    def forward(self, x, y, thresh=0.5, alpha=0.1):
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
                out[2] = torch.exp(out[2])
                out[3] = torch.exp(out[3])
                out[4] = torch.sigmoid(out[4])
                label = y[i, j, :]
                if label[4].item() > thresh:
                    loss[i, :] += alpha * torch.pow((label - out), 2)
                else:
                    loss[i, :] += torch.pow((label - out), 2)
        return loss.mean()
