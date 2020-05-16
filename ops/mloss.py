import torch
import torch.nn as nn
import torch.nn.functional as F


class MLoss(nn.Module):
    def __init__(self):
        super(MLoss, self).__init__()

    def forward(self, x, y, thresh=0.5, alpha=0.1):
        batches = x.shape[0]
        cell_num = x.shape[1]

        outs = []
        labels = []
        label_tensor = y
        for i in range(batches):
            for j in range(cell_num):
                label = y[i, j, :]
                if label[0].item() > thresh:
                    outs.append(x[i, j, :])
                    labels.append(label)
        outs_tensor = torch.stack(outs)
        labels_tensor = torch.stack(labels)
        diff = torch.pow((labels_tensor-outs_tensor), 2)
        diff_c = alpha * torch.pow(outs_tensor[:, 0], 2)
        diff_bg = alpha * torch.pow(x[:, :, 0], 2)
        return diff.sum() + diff_bg.sum() - diff_c.sum()
