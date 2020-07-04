import torch
import torch.nn as nn
import torch.nn.functional as F


class MLoss(nn.Module):
    def __init__(self):
        super(MLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, x, y, img_size=416, thresh=0.5, alpha=1.0):
        batches = x.shape[0]
        cell_num = x.shape[1]

        outs = []
        labels = []
        bgs = []
        face_num = 0.0
        for i in range(batches):
            for j in range(cell_num):
                label = y[i, j, :]
                if label[0].item() > thresh:
                    face_num += 1
                    outs.append(x[i, j, :])
                    labels.append(label)
                else:
                    bgs.append(x[i, j, 0])
        if len(outs) == 0:
            return None
        outs_tensor = torch.stack(outs)
        labels_tensor = torch.stack(labels)
        bg_tensor = torch.stack(bgs)
        diff_box = (1.0 + 1.0 / face_num) * self.mse_loss(outs_tensor[:, 1:5], labels_tensor[:, 1:5])
        diff_c = (1.0 + 1.0 / face_num) * self.bce_loss(outs_tensor[:, 0], labels_tensor[:, 0])
        diff_bg = alpha * self.bce_loss(bg_tensor, torch.zeros(bg_tensor.shape).cuda())
        diff = diff_box + diff_c + diff_bg
        return diff.sum()
