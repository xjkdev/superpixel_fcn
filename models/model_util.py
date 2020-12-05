import torch
import torch.nn as nn
import re
import collections
# *************************** my functions ****************************


def predict_param(in_planes, channel=3):
    return nn.Conv2d(in_planes, channel, kernel_size=3,
                     stride=1, padding=1, bias=True)


def predict_mask(in_planes, channel=9):
    return nn.Conv2d(in_planes, channel, kernel_size=3,
                     stride=1, padding=1, bias=True)


def predict_feat(in_planes, channel=20, stride=1):
    return nn.Conv2d(in_planes, channel, kernel_size=3,
                     stride=stride, padding=1, bias=True)


def predict_prob(in_planes, channel=9):
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            channel,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True),
        nn.Softmax(1)
    )
# ***********************************************************************


def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=(
                    kernel_size - 1) // 2,
                bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=(
                    kernel_size - 1) // 2,
                bias=True),
            nn.LeakyReLU(0.1)
        )


def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_planes,
            out_planes,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=True),
        nn.LeakyReLU(0.1)
    )


def migrate_state_dict(state_dict):
    result = collections.OrderedDict()
    for key in state_dict:
        if m := re.match(r'conv(\d)([ab])\.([01])\.([a-z_]+)', key):
            newkey = 'down{0}.conv_{1}.{2}.{3}'.format(*m.groups())
            print(key, '->', newkey)
            result[newkey] = state_dict[key]
        else:
            result[key] = state_dict[key]
    return result
