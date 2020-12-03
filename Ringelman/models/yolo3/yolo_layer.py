from torch import nn
import torch
from .dark_net import DarkNet
from .custom_layer import ConvolutionalSet, Convolutional


class YoloLayer(nn.Module):
    def __init__(self, in_ch, in_up_ch=None, out_up_ch=None, up_sample=True,up_sample_size=None, num_cls=80, anchor_num=3):
        """
        :param in_ch: sources中输入的feature map的in channel
        :param in_up_ch: 如果有来自下层上采样后的feature map需要融合，则需要指明该feature map的channel
        :param out_up_ch: 上采样操作时卷积的out channel
        :param up_sample: 是否需要上采样
        :param up_sample_size: 上采样后的图像大小，启用ONNX时需要指定
        :param num_cls: 分类数
        :param anchor_num: 每个像素中心需要生成多少个anchor box
        """
        super(YoloLayer, self).__init__()
        if in_up_ch is None:
            in_up_ch = 0
        mid_ch = in_ch // 2
        self.conv_set = ConvolutionalSet(in_ch=in_ch + in_up_ch, out_ch=mid_ch)
        self.up = up_sample
        self.anchor_num = anchor_num
        self.num_cls = num_cls
        if self.up:
            self.up_conv = Convolutional(mid_ch, out_up_ch, ks=1, stride=1, padding=0)
            if up_sample_size is None:
                self.up_sample = nn.Upsample(scale_factor=2)
            else:
                self.up_sample = nn.Upsample(size=up_sample_size)
        self.conv_1 = Convolutional(mid_ch, in_ch, 3, 1, 1)
        self.conv_2 = nn.Conv2d(in_ch, (5 + num_cls) * anchor_num, 1, 1, 0)

    def forward(self, input, up_feature=None):
        if not up_feature is None:
            input = torch.cat([input, up_feature], axis=1)
        x = self.conv_set(input)
        pred_x = self.conv_1(x)
        pred_x = self.conv_2(pred_x)
        pred_x = pred_x.view(pred_x.shape[0], self.anchor_num, 5 + self.num_cls, pred_x.shape[2],
                             pred_x.shape[3]).permute(0, 3, 4, 1, 2).contiguous()
        if self.up:  # 对当前feature map上采样，然后与上一层feature map融合
            up_x = self.up_conv(x)
            up_x = self.up_sample(up_x)
            # up_x = torch.cat([up_x, up_feature], axis=1)
            return pred_x, up_x
        else:
            return pred_x


class YoLoV3(nn.Module):
    def __init__(self, cfg):
        super(YoLoV3, self).__init__()

        self.bone_net = DarkNet(bone_net=True, in_ch=cfg["in_channel"], duration=cfg["duration"])
        self.yolo_layer_1 = YoloLayer(in_ch=1024, in_up_ch=None, out_up_ch=256, up_sample=True,up_sample_size=None,
                                      num_cls=cfg["num_class"], anchor_num=len(cfg["anchors"]))
        self.yolo_layer_2 = YoloLayer(in_ch=512, in_up_ch=256, out_up_ch=128, up_sample=True,up_sample_size=None,
                                      num_cls=cfg["num_class"], anchor_num=len(cfg["anchors"]))
        self.yolo_layer_3 = YoloLayer(in_ch=256, in_up_ch=128, out_up_ch=None, up_sample=False,
                                      num_cls=cfg["num_class"], anchor_num=len(cfg["anchors"]))

    def forward(self, input):
        source = self.bone_net(input)
        pred_3, up_3 = self.yolo_layer_1(source[2])  # 最后一层feature map开始向上采样
        pred_2, up_2 = self.yolo_layer_2(source[1], up_3)
        pred_1 = self.yolo_layer_3(source[0], up_2)

        return pred_1, pred_2, pred_3


if __name__ == '__main__':
    model = YoLoV3(duration=8,num_cls=3)
    state_dict = torch.load("../../weights/snaps/Base416_D8_BCE_yolov3_C3_E160.snap")
    model.load_state_dict(state_dict["model"])
    model.eval()
    input_data = torch.randn(4, 3,8, 416, 416)
    res = model(input_data)
    torch.onnx.export(model=model,
                      args=input_data,
                      f="yolo3.onnx",
                      export_params=True,
                      verbose=True,opset_version=9)
    print('end')
