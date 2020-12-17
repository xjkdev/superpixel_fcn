import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_
from .model_util import *
from train_util import *

# define the function includes in import *
__all__ = [
    'SpixelNet1l', 'SpixelNet1l_bn',
    'SpixelNet1l_CBAM', 'SpixelNet1l_CBAM_bn'
]


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class Down(nn.Module):
    def __init__(self, in_planes, out_planes,
                 kernel_size=3, stridea=2, strideb=1, batchNorm=True):
        super(Down, self).__init__()
        self.batchNorm = batchNorm
        self.conv_a = conv(self.batchNorm, in_planes, out_planes, kernel_size,
                           stridea)
        self.conv_b = conv(self.batchNorm, out_planes, out_planes, kernel_size,
                           strideb)
        self.ca = ChannelAttention(out_planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        out = self.conv_b(self.conv_a(x))
        out = self.ca(out) * out
        out = self.sa(out) * out
        return out


class SpixelNet(nn.Module):
    expansion = 1

    def __init__(self, batchNorm=True):
        super(SpixelNet, self).__init__()

        self.batchNorm = batchNorm
        self.assign_ch = 9
        self.conv0a = conv(self.batchNorm, 3, 16, kernel_size=3)
        self.conv0b = conv(self.batchNorm, 16, 16, kernel_size=3)

        self.conv1a = conv(self.batchNorm, 16, 32, kernel_size=3, stride=2)
        self.conv1b = conv(self.batchNorm, 32, 32, kernel_size=3)

        self.conv2a = conv(self.batchNorm, 32, 64, kernel_size=3, stride=2)
        self.conv2b = conv(self.batchNorm, 64, 64, kernel_size=3)

        self.conv3a = conv(self.batchNorm, 64, 128, kernel_size=3, stride=2)
        self.conv3b = conv(self.batchNorm, 128, 128, kernel_size=3)

        self.conv4a = conv(self.batchNorm, 128, 256, kernel_size=3, stride=2)
        self.conv4b = conv(self.batchNorm, 256, 256, kernel_size=3)

        self.deconv3 = deconv(256, 128)
        self.conv3_1 = conv(self.batchNorm, 256, 128)
        self.pred_mask3 = predict_mask(128, self.assign_ch)

        self.deconv2 = deconv(128, 64)
        self.conv2_1 = conv(self.batchNorm, 128, 64)
        self.pred_mask2 = predict_mask(64, self.assign_ch)

        self.deconv1 = deconv(64, 32)
        self.conv1_1 = conv(self.batchNorm, 64, 32)
        self.pred_mask1 = predict_mask(32, self.assign_ch)

        self.deconv0 = deconv(32, 16)
        self.conv0_1 = conv(self.batchNorm, 32, 16)
        self.pred_mask0 = predict_mask(16, self.assign_ch)

        self.softmax = nn.Softmax(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x):
        out1 = self.conv0b(self.conv0a(x))  # 5*5
        out2 = self.conv1b(self.conv1a(out1))  # 11*11
        out3 = self.conv2b(self.conv2a(out2))  # 23*23
        out4 = self.conv3b(self.conv3a(out3))  # 47*47
        out5 = self.conv4b(self.conv4a(out4))  # 95*95

        out_deconv3 = self.deconv3(out5)
        concat3 = torch.cat((out4, out_deconv3), 1)
        out_conv3_1 = self.conv3_1(concat3)

        out_deconv2 = self.deconv2(out_conv3_1)
        concat2 = torch.cat((out3, out_deconv2), 1)
        out_conv2_1 = self.conv2_1(concat2)

        out_deconv1 = self.deconv1(out_conv2_1)
        concat1 = torch.cat((out2, out_deconv1), 1)
        out_conv1_1 = self.conv1_1(concat1)

        out_deconv0 = self.deconv0(out_conv1_1)
        concat0 = torch.cat((out1, out_deconv0), 1)
        out_conv0_1 = self.conv0_1(concat0)
        mask0 = self.pred_mask0(out_conv0_1)
        prob0 = self.softmax(mask0)

        return prob0

    def weight_parameters(self):
        return [param for name, param in self.named_parameters()
                if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters()
                if 'bias' in name]


class SpixelNetCBAM(nn.Module):
    expansion = 1

    def __init__(self, batchNorm=True):
        super(SpixelNetCBAM, self).__init__()

        self.batchNorm = batchNorm
        self.assign_ch = 9

        self.down0 = Down(3, 16, 3, 1, 1, self.batchNorm)
        self.down1 = Down(16, 32, 3, 2, 1, self.batchNorm)
        self.down2 = Down(32, 64, 3, 2, 1, self.batchNorm)
        self.down3 = Down(64, 128, 3, 2, 1, self.batchNorm)
        self.down4 = Down(128, 256, 3, 2, 1, self.batchNorm)

        self.deconv3 = deconv(256, 128)
        self.conv3_1 = conv(self.batchNorm, 256, 128)
        self.pred_mask3 = predict_mask(128, self.assign_ch)

        self.deconv2 = deconv(128, 64)
        self.conv2_1 = conv(self.batchNorm, 128, 64)
        self.pred_mask2 = predict_mask(64, self.assign_ch)

        self.deconv1 = deconv(64, 32)
        self.conv1_1 = conv(self.batchNorm, 64, 32)
        self.pred_mask1 = predict_mask(32, self.assign_ch)

        self.deconv0 = deconv(32, 16)
        self.conv0_1 = conv(self.batchNorm, 32, 16)
        self.pred_mask0 = predict_mask(16, self.assign_ch)

        self.softmax = nn.Softmax(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x):
        out1 = self.down0(x)  # 5*5
        out2 = self.down1(out1)  # 11*11
        out3 = self.down2(out2)  # 23*23
        out4 = self.down3(out3)  # 47*47
        out5 = self.down4(out4)  # 95*95

        out_deconv3 = self.deconv3(out5)
        concat3 = torch.cat((out4, out_deconv3), 1)
        out_conv3_1 = self.conv3_1(concat3)

        out_deconv2 = self.deconv2(out_conv3_1)
        concat2 = torch.cat((out3, out_deconv2), 1)
        out_conv2_1 = self.conv2_1(concat2)

        out_deconv1 = self.deconv1(out_conv2_1)
        concat1 = torch.cat((out2, out_deconv1), 1)
        out_conv1_1 = self.conv1_1(concat1)

        out_deconv0 = self.deconv0(out_conv1_1)
        concat0 = torch.cat((out1, out_deconv0), 1)
        out_conv0_1 = self.conv0_1(concat0)
        mask0 = self.pred_mask0(out_conv0_1)
        prob0 = self.softmax(mask0)

        return prob0

    def weight_parameters(self):
        return [param for name, param in self.named_parameters()
                if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters()
                if 'bias' in name]


def SpixelNet1l(data=None):
    # Model without  batch normalization
    model = SpixelNet(batchNorm=False)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model


def SpixelNet1l_bn(data=None):
    # model with batch normalization
    model = SpixelNet(batchNorm=True)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model


def SpixelNet1l_CBAM(data=None):
    # Model without  batch normalization
    model = SpixelNetCBAM(batchNorm=False)
    if data is not None:
        if data['arch'] == 'SpixelNet1l':
            print("use migration model")
            model.load_state_dict(migrate_state_dict(data['state_dict']))
        else:
            model.load_state_dict(data['state_dict'])
    return model


def SpixelNet1l_CBAM_bn(data=None):
    # model with batch normalization
    model = SpixelNetCBAM(batchNorm=True)
    if data is not None:
        if data['arch'] == 'SpixelNet1l_bn':
            print("use migration model")
            model.load_state_dict(migrate_state_dict(data['state_dict']))
        else:
            model.load_state_dict(data['state_dict'])
    return model
#
