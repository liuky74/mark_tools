import torch
from torch import nn
from .custom_layer import Convolutional, Residual

class ChannelConcat(nn.Module):
    def __init__(self,duration=12,out_channel=96):
        super(ChannelConcat, self).__init__()
        if not (out_channel%3==0):
            raise Exception("wrong")
        single_conv_out_channel = int(out_channel/3)
        self.conv_1 = nn.Conv2d(duration, single_conv_out_channel, 3, 1, 1)
        self.conv_2 = nn.Conv2d(duration, single_conv_out_channel, 3, 1, 1)
        self.conv_3 = nn.Conv2d(duration, single_conv_out_channel, 3, 1, 1)
        self.bn = nn.BatchNorm2d(num_features=out_channel, momentum=0.03, eps=1e-4)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        res = []
        res.append(self.conv_1(x[:, 0, :, :, :]))
        res.append(self.conv_2(x[:, 1, :, :, :]))
        res.append(self.conv_3(x[:, 2, :, :, :]))
        x = torch.cat(res,dim = 1)
        x = self.bn(x)
        x = self.act(x)
        return x


class DarkNet(nn.Module):
    def __init__(self, in_ch=3,duration=1,num_cls=10,bone_net=False):
        super(DarkNet, self).__init__()
        self.bone = bone_net
        if duration ==1:
            self.basic_conv = Convolutional(in_ch=in_ch, out_ch=32, ks=3, stride=1, padding=1)
            self.basic_pool = Convolutional(32, 64, 3, 2, 1)
        else:
            self.basic_conv = ChannelConcat(duration=duration, out_channel=96)
            self.basic_pool = Convolutional(96, 64, 3, 2, 1)

        self.block_1 = Residual(in_ch=64, out_ch=64, use_weight=False)
        self.pool_1 = Convolutional(64, 128, 3, 2, padding=1)

        self.block_2 = nn.ModuleList([Residual(128, 128, True) for _ in range(2)])
        self.pool_2 = Convolutional(128, 256, 3, 2, padding=1)

        self.block_3 = nn.ModuleList([Residual(256, 256, False) for _ in range(8)])
        self.pool_3 = Convolutional(256, 512, 3, 2, padding=1)

        self.block_4 = nn.ModuleList([Residual(512, 512, False) for _ in range(8)])
        self.pool_4 = Convolutional(512, 1024, 3, 2, padding=1)

        self.block_5 = nn.ModuleList([Residual(1024, 1024, False) for _ in range(4)])

        if not self.bone:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc1 = nn.Linear(1024, 128)
            self.fc1_act = nn.ReLU(inplace=True)
            self.fc2 = nn.Linear(128, num_cls)
            self.fc2_act = nn.Softmax(dim=-1)

    def forward(self, input):
        source = []
        x = self.basic_conv(input)
        x = self.basic_pool(x)
        # block1
        x = self.block_1(x)
        x = self.pool_1(x)
        # block2
        for layer in self.block_2:
            x = layer(x)
        x = self.pool_2(x)
        # block3
        for layer in self.block_3:
            x = layer(x)
        source.append(x)
        x = self.pool_3(x)
        # block4
        for layer in self.block_4:
            x = layer(x)
        source.append(x)
        x = self.pool_4(x)
        # block5
        for layer in self.block_5:
            x = layer(x)
        source.append(x)
        if self.bone:
            return source
        else:
            x = self.avg_pool(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = self.fc1_act(x)
            x = self.fc2(x)
            x = self.fc2_act(x)
            return x


if __name__ == '__main__':
    import numpy as np

    model = DarkNet(in_ch=3,duration=None,num_cls=5,bone_net=False)
    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))
    input_data = torch.Tensor(np.random.randn(1, 3, 416, 416))
    torch.onnx.export(model=model,
                      args=input_data,
                      f="darknet.onnx",
                      export_params=True,
                      verbose=True,)
    model(input_data)


    torch.nn.MaxPool2d()