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
        for i in range(out.shape[0]):
            x = out[i, 0, :]
            tmp = torch.sigmoid(x)
            for j in range(x.shape[0]):
                x[j] = tmp[j]
            y = out[i, 1, :]
            tmp = torch.sigmoid(y)
            for j in range(y.shape[0]):
                y[j] = tmp[j]
            w = out[i, 2, :]
            tmp = torch.exp(w)
            for j in range(w.shape[0]):
                w[j] = tmp[j]
            h = out[i, 3, :]
            tmp = torch.exp(h)
            for j in range(h.shape[0]):
                h[j] = tmp[j]
            c = out[i, 4, :]
            tmp = torch.sigmoid(c)
            for j in range(x.shape[0]):
                c[j] = tmp[j]
        return out

    def _make_feature_map(self, cfg):
        feature_map = []
        for i in range(3):
            feature_map.append(nn.Sequential(nn.Conv2d(cfg[self.m_indexs[i + 2] - 1], 5,
                                                       kernel_size=3, padding=1),
                                             nn.BatchNorm2d(5),
                                             nn.ReLU()))
        return feature_map

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Sequential(nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                                         nn.BatchNorm2d(x),
                                         nn.ReLU(inplace=True))]
                in_channels = x
        return layers


def test():
    net = MSSD('VGG11')
    x = torch.randn(2, 3, 416, 416)
    y = net(x)
    print(y.size())

# test()
