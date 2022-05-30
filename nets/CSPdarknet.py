import torch
import torch.nn as nn
import math


class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Focus(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):  # 640, 640, 3 => 320, 320, 12     320, 320, 12 => 320, 320, 64
        # 每隔一个像素点取一个值
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


class Conv(nn.Module):  # SiLU卷积函数
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class ConvX(nn.Module):  # LeakyReLU卷积（卷积核为3）
    def __init__(self, in_planes, out_planes, kernel=3, stride=1):
        super(ConvX, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=kernel//2, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class CatBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super(CatBottleneck, self).__init__()
        assert block_num > 1, print("block number should be larger than 1.")
        self.conv_list = nn.ModuleList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(nn.Conv2d(out_planes // 2, out_planes // 2, kernel_size=3, stride=2,
                                                     padding=1, groups=out_planes // 2, bias=False),
                                           nn.BatchNorm2d(out_planes // 2),)
            self.skip = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            stride = 1
            # i=0:STDC1  ->  layer=4 j=[0,1,2,3]
            # i=1:STDC2  ->  layer=5 j=[0,1,2,3,4]
            # i=2:STDC3  ->  layer=2 j=[0,1,2]
        for idx in range(block_num):  # idx=0,1,2 block_num=3
            if idx == 0:
                self.conv_list.append(ConvX(in_planes, out_planes // 2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(ConvX(out_planes // 2, out_planes // 2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(ConvX(out_planes // 2, out_planes // 4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvX(out_planes // int(math.pow(2, idx)), out_planes // int(math.pow(2, idx + 1))))
            else:
                self.conv_list.append(ConvX(out_planes // int(math.pow(2, idx)), out_planes // int(math.pow(2, idx))))

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)
        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)
        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)
        out = torch.cat(out_list, dim=1)
        return out


class CSPDarknet(nn.Module):
    def __init__(self, base=32, layers=[4, 5, 3], block_num=4):
        super().__init__()
        block = CatBottleneck
        '''x2:Focus; x4:CBL; x3、x4、x5:STDC模块'''
        self.features = self._make_layer(base, layers, block_num, block)
        self.x2 = nn.Sequential(self.features[:1])
        self.x4 = nn.Sequential(self.features[1:2])
        self.x8 = nn.Sequential(self.features[2:6])
        self.x16 = nn.Sequential(self.features[6:11])
        self.x32 = nn.Sequential(self.features[11:])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))  # 正态分布的均值和方差
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)  # 卷积核参数 全填充1
                m.bias.data.zero_()

    def _make_layer(self, base, layers, block_num, block):  # STDC2模块(4个block)
        features = []  # base=64
        features += [Focus(3, base, k=3)]        # focus:32个k=3*3,s=2 (3->64)
        features += [ConvX(base, base*2, 3, 2)]  # CBL:32个k=3*3,s=2 (64->128)
        for i, layer in enumerate(layers):
            # i=0:STDC1  ->  layer=4 j=[0,1,2,3]
            # i=1:STDC2  ->  layer=5 j=[0,1,2,3,4]
            # i=2:STDC3  ->  layer=2 j=[0,1,2]
            for j in range(layer):
                if i == 0 and j == 0:  # 第1次 STDC1
                    features.append(block(base*2, base*4, block_num, 2))
                elif j == 0:           # 第1次 STDC2,STDC3
                    features.append(block(base*int(math.pow(2, i+1)), base*int(math.pow(2, i+2)), block_num, 2))
                else:                  # 第2、3、4次 STDC1,STDC2,STDC3
                    features.append(block(base*int(math.pow(2, i+2)), base*int(math.pow(2, i+2)), block_num, 1))
        return nn.Sequential(*features)

    def forward(self, x):
        feat2 = self.x2(x)
        feat4 = self.x4(feat2)
        feat8 = self.x8(feat4)     # 80, 80, 256
        feat16 = self.x16(feat8)   # 40, 40, 512
        feat32 = self.x32(feat16)  # 20, 20, 1024
        return feat8, feat16, feat32


def cspdarknet():
    model = CSPDarknet()
    return model

# from thop import clever_format, profile
# from torchsummary import summary
# if __name__ == "__main__":
#     input_shape = [640, 640]
#     anchors_mask = [[3, 4, 5], [0, 1, 2]]
#     num_classes = 1
#     backbone = 'cspdarknet'
#     phi = 's'
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     m = cspdarknet().to(device)
#     summary(m, (3, input_shape[0], input_shape[1]))
#
#     dummy_input = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
#     flops, params = profile(m.to(device), (dummy_input,), verbose=False)
#     flops = flops
#     flops, params = clever_format([flops, params], "%.3f")
#     print('Total GFLOPS: %s' % flops)
#     print('Total params: %s' % params)
