import torch
import torch.nn as nn
import torch.nn.functional as F

class MPred(nn.Module):
    def __init__(self):
        super(MPred, self).__init__()

    def forward(self, x, img_size=416):
        x = x.cpu()
        batches = x.shape[0]
        for i in range(batches):
            out = x[i, :, 0:3]
            out = torch.sigmoid(out)
            out = x[i, :, 3:5]
            out = torch.exp(out)/img_size
        return x

class MLoss(nn.Module):
    def __init__(self):
        super(MLoss, self).__init__()

    def forward(self, x, y, img_size=416, thresh=0.5, alpha=0.1):
        batches = x.shape[0]
        cell_num = x.shape[1]
        for i in range(batches):
            out = x[i, :, 0:3]
            out = torch.sigmoid(out)
            out = x[i, :, 3:5]
            out = torch.exp(out)/img_size
        outs = []
        labels = []
        for i in range(batches):
            for j in range(cell_num):
                label = y[i, j, :]
                if label[0].item() > thresh:
                    outs.append(x[i, j, :])
                    labels.append(label)
        outs_tensor = torch.cat(outs)
        labels_tensor = torch.cat(labels)
        diff = torch.pow((labels_tensor-outs_tensor), 2) - alpha * torch.pow(outs_tensor, 2)

        diff_bg = alpha * torch.pow(x[:, :, 0], 2)
        return diff.sum() + diff_bg.sum()
