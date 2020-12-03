import torch
from torch import nn
# import tensorrt as trt


class Convolutional(nn.Module):
    def __init__(self, in_ch, out_ch, ks, stride, padding, bn=True):
        super(Convolutional, self).__init__()
        self.out_ch = out_ch
        self.ks = ks
        self.stride = stride
        self.padding = padding
        self.bn = bn
        self.conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=ks, stride=stride, padding=padding,
                              bias=not bn)
        self.bn = nn.BatchNorm2d(num_features=out_ch, momentum=0.03, eps=1e-4) if bn else None
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, input):
        x = self.conv(input)
        if self.bn:
            x = self.bn(x)
        x = self.act(x)
        return x





class Residual(nn.Module):
    def __init__(self, in_ch, out_ch, use_weight=False):
        super(Residual, self).__init__()
        self.use_weight = use_weight
        self.weight = nn.Parameter(torch.Tensor([0., 0.]).to(torch.float32), requires_grad=True) if use_weight else None
        mid_ch = int(in_ch // 2)
        self.conv_1 = Convolutional(in_ch=in_ch, out_ch=mid_ch, ks=1, stride=1, padding=0)
        self.conv_2 = Convolutional(in_ch=mid_ch, out_ch=out_ch, ks=3, stride=1, padding=1)

    def forward(self, input):
        x = self.conv_1(input)
        x = self.conv_2(x)
        if self.use_weight:
            w = torch.sigmoid(self.weight)
            input = input * w[0]
            x = x * w[1]
        x = x + input
        return x


class ConvolutionalSet(nn.Module):
    def __init__(self, in_ch,out_ch):
        super(ConvolutionalSet, self).__init__()
        mid_ch = out_ch
        self.conv_1 = Convolutional(in_ch=in_ch, out_ch=mid_ch, ks=1, stride=1, padding=0)
        self.conv_2 = Convolutional(mid_ch, in_ch, 3, 1, 1)
        self.conv_3 = Convolutional(in_ch, mid_ch, 1, 1, 0)
        self.conv_4 = Convolutional(mid_ch, in_ch, 3, 1, 1)
        self.conv_5 = Convolutional(in_ch, out_ch, 1, 1, 0)

    def forward(self, input):
        x = self.conv_1(input)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        return x
