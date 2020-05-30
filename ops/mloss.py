import torch
import torch.nn as nn
import torch.nn.functional as F


class MLoss(nn.Module):
    def __init__(self):
        super(MLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, x, y, img_size=416, thresh=0.5, alpha=0.5):
        batches = x.shape[0]
        cell_num = x.shape[1]

        outs = []
        labels = []
        face_num = 0.0
        for i in range(batches):
            for j in range(cell_num):
                label = y[i, j, :]
                if label[0].item() > thresh:
                    face_num += 1
                    outs.append(x[i, j, :])
                    labels.append(label)
        if len(outs) == 0:
            return None
        outs_tensor = torch.stack(outs)
        labels_tensor = torch.stack(labels)
        diff_box = (1.0 + 1.0 / face_num) * self.mse_loss(outs_tensor[:, 1:5], labels_tensor[:, 1:5])
        diff_c = (1.0 + 1.0 / face_num) * self.bce_loss(outs_tensor[:, 0], labels_tensor[:, 0])
        all_confidence = x[:, :, 0]
        diff_bg = alpha * self.bce_loss(all_confidence, torch.zeros(all_confidence.shape).cuda())
        diff = diff_box + diff_c + diff_bg
        return diff.sum()
