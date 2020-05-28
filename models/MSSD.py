'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class Mish(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        x = x * (torch.tanh(F.softplus(x)))
        return x


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.mish = Mish()
        self.stride = stride

        self.conv4 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.mish(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.mish(out)

        out = self.conv3(out)
        out = self.bn3(out)

        #down sample
        residual = self.conv4(residual)
        residual = self.bn4(residual)

        out += residual
        out = self.mish(out)

        return out

class BaseBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=0, active=True):
        super(BaseBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.active = active
        if active:
            self.mish = Mish()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        if self.active:
            x = self.mish(x)
        return x

def get_m_index(data):
    indexs = []
    for i in range(len(data)):
        if data[i] == 'M':
            indexs.append(i)
    return indexs





class MPred(nn.Module):
    def __init__(self):
        super(MPred, self).__init__()

    def forward(self, x1, x2, x3, img_size=416, anchor=[12, 36, 108]):
        out1_cxy = torch.sigmoid(x1[:, 0:3, :])
        out1_wh = anchor[0]*torch.exp(x1[:, 3:5, :])/img_size
        out1 = torch.cat([out1_cxy, out1_wh], dim=1)

        out2_cxy = torch.sigmoid(x2[:, 0:3, :])
        out2_wh = anchor[1]*torch.exp(x2[:, 3:5, :])/img_size
        out2 = torch.cat([out2_cxy, out2_wh], dim=1)

        out3_cxy = torch.sigmoid(x3[:, 0:3, :])
        out3_wh = anchor[2]*torch.exp(x3[:, 3:5, :])/img_size
        out3 = torch.cat([out3_cxy, out3_wh], dim=1)
        return out1, out2, out3

class MSSD(nn.Module):
    def __init__(self):
        super(MSSD, self).__init__()
        self.b1 = BaseBlock(3, 64, 5, 1, 2)
        self.down_sample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bx1 = Bottleneck(64, 128, 1)
        self.bx1_1 = Bottleneck(128, 256, 1)
        self.bx2 = Bottleneck(256, 128, 1)
        self.bx2_1 = Bottleneck(128, 128, 1)
        self.bx3 = Bottleneck(128, 128, 1)
        self.bx4 = Bottleneck(128, 128, 1)
        self.bx5 = Bottleneck(128, 128, 1)

        self.o1 = BaseBlock(128, 5, 3, 1, 1, active=False)
        self.o2 = BaseBlock(128, 5, 3, 1, 1, active=False)
        self.o3 = BaseBlock(128, 5, 3, 1, 1, active=False)

        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear')

        self.pred = MPred()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # vgg to get feature
        x = self.b1(x)                         # 416
        x = self.down_sample(x)                # 208
        x = self.bx1(x)                        # 208
        x = self.bx1_1(x)                      # 208
        x = self.down_sample(x)                # 104
        x = self.bx2(x)                        # 104
        x = self.bx2_1(x)                      # 104
        x1 = self.down_sample(x)               # 52
        x1 = self.bx3(x1)                      # 52
        x2 = self.down_sample(x1)              # 26
        x2 = self.bx4(x2)                      # 26
        x3 = self.down_sample(x2)              # 13
        x3 = self.bx5(x3)

        u2 = self.up_sample(x3)
        u1 = self.up_sample(x2)

        o1 = self.o1(x3)                       # 13
        o2 = self.o2(x2+u2)                    # 26
        o3 = self.o3(x1+u1)                    # 52

        o1 = o1.view(o1.shape[0], o1.shape[1], o1.shape[2] * o1.shape[3])
        o2 = o2.view(o2.shape[0], o2.shape[1], o2.shape[2] * o2.shape[3])
        o3 = o3.view(o3.shape[0], o3.shape[1], o3.shape[2] * o3.shape[3])
        out = torch.cat(self.pred(o3, o2, o1), dim=2)
        out = out.permute(0, 2, 1)
        return out


def test():
    net = MSSD()
    x = torch.randn(2, 3, 416, 416)
    y = net(x)
    print(y.size())

# test()
