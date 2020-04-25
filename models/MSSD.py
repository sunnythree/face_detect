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


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1)
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
    def __init__(self, vgg_name):
        super(MSSD, self).__init__()
        self.vgg_name = vgg_name
        self.layers = self._make_layers(cfg[vgg_name])
        self.m_indexs = get_m_index(cfg[vgg_name])
        self.feature_map = self._make_feature_map(cfg[vgg_name])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # vgg to get feature
        outs = []
        for layer in self.layers:
            x = layer(x)
            outs.append(x)
        out1 = self.feature_map[0](outs[self.m_indexs[2]])
        out2 = self.feature_map[1](outs[self.m_indexs[3]])
        out3 = self.feature_map[2](outs[self.m_indexs[4]])
        out1 = out1.view(out1.shape[0], out1.shape[1], out1.shape[2] * out1.shape[3])
        out2 = out2.view(out2.shape[0], out2.shape[1], out2.shape[2] * out2.shape[3])
        out3 = out3.view(out3.shape[0], out3.shape[1], out3.shape[2] * out3.shape[3])
        out = torch.cat((out1, out2, out3), dim=2)
        batches = out.split(split_size=1, dim=0)
        all = []
        for batch in batches:
            x = batch[0, 0, :]
            x = torch.sigmoid(x)
            y = batch[0, 1, :]
            y = torch.sigmoid(y)
            w = batch[0, 2, :]
            w = torch.exp(w)
            h = batch[0, 3, :]
            h = torch.exp(h)
            c = batch[0, 4, :]
            c = torch.sigmoid(c)
            all.append(torch.stack([x, y, w, h, c]))
        out = torch.stack(all)
        out = out.permute(0, 2, 1)
        return out

    def _make_feature_map(self, cfg):
        feature_map = []
        for i in range(3):
            feature_map.append(nn.Sequential(
                    nn.Conv2d(cfg[self.m_indexs[i+2]-1], 5, kernel_size=3, padding=1),
                    nn.BatchNorm2d(5),
                    nn.ReLU()))
        return nn.ModuleList(feature_map)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Sequential(
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU())]
                in_channels = x
        return nn.ModuleList(layers)


def test():
    net = MSSD('VGG11')
    x = torch.randn(2, 3, 416, 416)
    y = net(x)
    print(y.size())

# test()
