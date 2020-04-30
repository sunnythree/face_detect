'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


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
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

        self.conv4 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        #down sample
        residual = self.conv4(residual)
        residual = self.bn4(residual)

        out += residual
        out = self.relu(out)

        return out

class BaseBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=0):
        super(BaseBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x

def get_m_index(data):
    indexs = []
    for i in range(len(data)):
        if data[i] == 'M':
            indexs.append(i)
    return indexs


class MSSD(nn.Module):
    def __init__(self):
        super(MSSD, self).__init__()
        self.b1 = BaseBlock(3, 64, 5, 1, 2)
        self.down_sample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bx1 = Bottleneck(64, 128, 1)
        self.bx2 = Bottleneck(128, 256, 1)
        self.bx3 = Bottleneck(256, 128, 1)
        self.bx4 = Bottleneck(128, 64, 1)
        self.o1 = BaseBlock(64, 5, 3, 1, 1)
        self.up_sample = nn.Upsample(scale_factor=2)
        self.o2 = BaseBlock(5, 5, 3, 1, 1)
        self.o3 = BaseBlock(5, 5, 3, 1, 1)
        self.ox1 = BaseBlock(128, 5, 1, 1)
        self.ox2 = BaseBlock(256, 5, 1, 1)

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
        x = self.down_sample(x)                # 104
        x1 = self.bx2(x)                       # 104
        x1 = self.down_sample(x1)                # 52
        x2 = self.bx3(x1)                      # 52
        x2 = self.down_sample(x2)                # 26
        x3 = self.bx4(x2)                      # 26
        x3 = self.down_sample(x3)                # 13
        o1 = self.o1(x3)                       # 13
        ox1 = self.ox1(x2)                     # 26
        o2 = self.up_sample(o1)                # 26
        o2 = o2+ox1                            # 26
        o2 = self.o2(o2)                       # 26
        ox2 = self.ox2(x1)                     # 26
        o3 = self.up_sample(o2)                # 52
        o3 = o3+ox2                            # 52
        o3 = self.o3(o3)                       # 52
        o1 = o1.view(o1.shape[0], o1.shape[1], o1.shape[2] * o1.shape[3])
        o2 = o2.view(o2.shape[0], o2.shape[1], o2.shape[2] * o2.shape[3])
        o3 = o3.view(o3.shape[0], o3.shape[1], o3.shape[2] * o3.shape[3])
        out = torch.cat((o1, o2, o3), dim=2)
        out = out.permute(0, 2, 1)
        return out




def test():
    net = MSSD('VGG11')
    x = torch.randn(2, 3, 416, 416)
    y = net(x)
    print(y.size())

# test()
