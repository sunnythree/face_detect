'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = None
        if in_planes != planes:
            self.downsample = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BaseBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1):
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


class MSSD(nn.Module):
    def __init__(self):
        super(MSSD, self).__init__()
        self.down_sample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up_sample = nn.Upsample(scale_factor=2)
        # feature
        self.bx = nn.Conv2d(3, 64,  kernel_size=5,  stride=1, padding=2)
        self.bx1 = BasicBlock(64, 128)
        self.bx2 = BasicBlock(128, 256)
        self.bx3 = BasicBlock(256, 512)
        self.bx4 = BaseBlock(512, 512)
        self.bx5 = BaseBlock(512, 512)
        self.bx6 = BaseBlock(512, 512)
        self.bx7 = BaseBlock(512, 512)


        # bbox output
        self.out1 = BaseBlock(512, 5, kernel_size=1, stride=1, padding=0)
        self.out2 = BaseBlock(512, 5, kernel_size=1, stride=1, padding=0)
        self.out3 = BaseBlock(512, 5, kernel_size=1, stride=1, padding=0)
        self.out4 = BaseBlock(512, 5, kernel_size=1, stride=1, padding=0)
        self.out5 = BaseBlock(512, 5, kernel_size=1, stride=1, padding=0)
        self.out6 = BaseBlock(512, 5, kernel_size=1, stride=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # get feature map
        x = self.bx(x)   #512
        x = self.down_sample(x)  # 256
        x = self.bx1(x)  # 256
        x = self.down_sample(x)  # 128
        x = self.bx2(x)  # 128
        x = self.down_sample(x)  # 64
        x1 = self.bx3(x)  # 64
        x2 = self.down_sample(x1)  # 32
        x2 = self.bx4(x2)  # 32
        x3 = self.down_sample(x2)  # 16
        x3 = self.bx5(x3)  # 16
        x4 = self.down_sample(x3)  # 8
        x4 = self.bx6(x4)  # 8
        x5 = self.down_sample(x4)  # 4
        x5 = self.bx7(x5)  # 4
        x6 = self.down_sample(x5)  # 2

        u1 = self.up_sample(x2)
        u2 = self.up_sample(x3)
        u3 = self.up_sample(x4)
        u4 = self.up_sample(x5)
        u5 = self.up_sample(x6)


        # output
        out1 = self.out1(x1+u1)
        out2 = self.out2(x2+u2)
        out3 = self.out3(x3+u3)
        out4 = self.out4(x4+u4)
        out5 = self.out5(x5+u5)
        out6 = self.out6(x6)

        out1 = out1.view(out1.shape[0], out1.shape[1], out1.shape[2] * out1.shape[3])
        out2 = out2.view(out2.shape[0], out2.shape[1], out2.shape[2] * out2.shape[3])
        out3 = out3.view(out3.shape[0], out3.shape[1], out3.shape[2] * out3.shape[3])
        out4 = out4.view(out4.shape[0], out4.shape[1], out4.shape[2] * out4.shape[3])
        out5 = out5.view(out5.shape[0], out5.shape[1], out5.shape[2] * out5.shape[3])
        out6 = out6.view(out6.shape[0], out6.shape[1], out6.shape[2] * out6.shape[3])

        out = torch.cat((out1, out2, out3, out4, out5, out6), dim=2)
        out = out.permute(0, 2, 1)
        out = torch.sigmoid(out)
        return out


def test():
    net = MSSD()
    x = torch.randn(2, 3, 512, 512)
    y = net(x)
    print(y.size())

# test()
