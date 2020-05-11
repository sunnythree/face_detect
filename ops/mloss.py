import torch
import torch.nn as nn
import torch.nn.functional as F

class MPred(nn.Module):
    def __init__(self):
        super(MPred, self).__init__()

    def forward(self, x, img_size=416):
        x = x.cpu()
        out1 = torch.sigmoid(x[:, :, 0:3])
        out2 = torch.exp(x[:, :, 3:5])/img_size
        out = torch.cat([out1, out2], dim=2)
        return out

class MLoss(nn.Module):
    def __init__(self):
        super(MLoss, self).__init__()

    def forward(self, x, y, img_size=416, thresh=0.5, alpha=0.1):
        batches = x.shape[0]
        cell_num = x.shape[1]

        out1 = torch.sigmoid(x[:, :, 0:3])
        out2 = torch.exp(x[:, :, 3:5])/img_size
        out = torch.cat([out1, out2], dim=2)

        outs = []
        labels = []
        for i in range(batches):
            for j in range(cell_num):
                label = y[i, j, :]
                if label[0].item() > thresh:
                    outs.append(out[i, j, :])
                    labels.append(label)
        outs_tensor = torch.stack(outs)
        labels_tensor = torch.stack(labels)
        diff = torch.pow((labels_tensor-outs_tensor), 2)
        diff_c = alpha * torch.pow(outs_tensor[:, 0], 2)
        diff_bg = alpha * torch.pow(out[:, :, 0], 2)
        return diff.sum() + diff_bg.sum() - diff_c.sum()
