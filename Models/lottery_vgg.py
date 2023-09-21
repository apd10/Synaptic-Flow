# Based on code taken from https://github.com/facebookresearch/open_lth

# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch.nn.functional as F
from Layers import layers

class ConvModulePT(nn.Module):
    """A single convolutional module in a VGG network."""

    def __init__(self, in_filters, out_filters):
        super(ConvModulePT, self).__init__()
        self.conv = nn.Conv2d(in_filters, out_filters, kernel_size=3, padding=1)

    def forward(self, x):
        return F.relu(self.conv(x))

class ConvBNModulePT(nn.Module):
    """A single convolutional module in a VGG network."""

    def __init__(self, in_filters, out_filters):
        super(ConvBNModulePT, self).__init__()
        self.conv = nn.Conv2d(in_filters, out_filters, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_filters)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class ConvModule(nn.Module):
    """A single convolutional module in a VGG network."""

    def __init__(self, in_filters, out_filters):
        super(ConvModule, self).__init__()
        self.conv = layers.Conv2d(in_filters, out_filters, kernel_size=3, padding=1)

    def forward(self, x):
        return F.relu(self.conv(x))

class ConvBNModule(nn.Module):
    """A single convolutional module in a VGG network."""

    def __init__(self, in_filters, out_filters):
        super(ConvBNModule, self).__init__()
        self.conv = layers.Conv2d(in_filters, out_filters, kernel_size=3, padding=1)
        self.bn = layers.BatchNorm2d(out_filters)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class VGG_PT(nn.Module):
    """A VGG-style neural network designed for CIFAR-10."""

    def __init__(self, plan, conv, num_classes=10, dense_classifier=False):
        super(VGG_PT, self).__init__()
        layer_list = []
        filters = 3

        for spec in plan:
            if spec == 'M':
                layer_list.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layer_list.append(conv(filters, spec))
                filters = spec

        self.layers = nn.Sequential(*layer_list)        

        self.fc = nn.Linear(512, num_classes)
        if dense_classifier:
            self.fc.do_not_roast = True

        self._initialize_weights()

    def forward(self, x):
        x = self.layers(x)
        x = nn.AvgPool2d(x.shape[-1])(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class VGG_PT_WIDTH(nn.Module):
    """A VGG-style neural network designed for CIFAR-10."""

    def __init__(self, plan, conv, num_classes=10, dense_classifier=False):
        super(VGG_PT_WIDTH, self).__init__()
        layer_list = []
        filters = 3

        for spec in plan:
            if spec == 'M':
                layer_list.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layer_list.append(conv(filters, spec))
                filters = spec

        self.layers = nn.Sequential(*layer_list)        

        self.fc = nn.Linear(plan[-1], num_classes)
        if dense_classifier:
            self.fc.do_not_roast = True

        self._initialize_weights()

    def forward(self, x):
        x = self.layers(x)
        x = nn.AvgPool2d(x.shape[-1])(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
class VGG(nn.Module):
    """A VGG-style neural network designed for CIFAR-10."""

    def __init__(self, plan, conv, num_classes=10, dense_classifier=False):
        super(VGG, self).__init__()
        layer_list = []
        filters = 3

        for spec in plan:
            if spec == 'M':
                layer_list.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layer_list.append(conv(filters, spec))
                filters = spec

        self.layers = nn.Sequential(*layer_list)        

        self.fc = layers.Linear(512, num_classes)
        if dense_classifier:
            self.fc = nn.Linear(512, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.layers(x)
        x = nn.AvgPool2d(x.shape[-1])(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (layers.Linear, nn.Linear, layers.Conv2d)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, layers.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def _plan(num):
    if num == 11.32:
        plan = [32, 'M', 64, 'M', 128, 128 , 'M', 256, 256, 'M', 256, 256]
    elif num == 11.16:
        plan = [16, 'M', 32, 'M', 64, 64 , 'M', 128,128 , 'M', 128, 128]
    elif num == 11.8:
        plan = [8, 'M', 16, 'M', 32, 32 , 'M', 64, 64, 'M', 64, 64]
    elif num == 11.4:
        plan = [4, 'M', 8, 'M', 16, 16 , 'M', 32, 32, 'M', 32, 32]
    elif num == 11.2:
        plan = [2, 'M', 4, 'M', 8, 8 , 'M', 16, 16, 'M', 16, 16]
    elif num == 11:
        plan = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512]
    elif num == 13:
        plan = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512]
    elif num == 16:
        plan = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
    elif num == 19:
        plan = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
    else:
        raise ValueError('Unknown VGG model: {}'.format(num))
    return plan

def _pt_vgg(arch, plan, conv, num_classes, dense_classifier, pretrained):
    model = VGG_PT(plan, conv, num_classes, dense_classifier)
    if pretrained:
        pretrained_path = 'Models/pretrained/{}-lottery.pt'.format(arch)
        pretrained_dict = torch.load(pretrained_path)
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

def _pt_vgg_width(arch, plan, conv, num_classes, dense_classifier, pretrained):
    model = VGG_PT_WIDTH(plan, conv, num_classes, dense_classifier)
    if pretrained:
        pretrained_path = 'Models/pretrained/{}-lottery.pt'.format(arch)
        pretrained_dict = torch.load(pretrained_path)
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

def _vgg(arch, plan, conv, num_classes, dense_classifier, pretrained):
    model = VGG(plan, conv, num_classes, dense_classifier)
    if pretrained:
        pretrained_path = 'Models/pretrained/{}-lottery.pt'.format(arch)
        pretrained_dict = torch.load(pretrained_path)
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

def vgg11(input_shape, num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(11)
    return _vgg('vgg11_bn', plan, ConvModule, num_classes, dense_classifier, pretrained)

def pt_vgg11(input_shape, num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(11)
    return _pt_vgg('vgg11_bn', plan, ConvModulePT, num_classes, dense_classifier, pretrained)

def vgg11_bn(input_shape, num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(11)
    return _vgg('vgg11_bn', plan, ConvBNModule, num_classes, dense_classifier, pretrained)

def pt_vgg11_bn(input_shape, num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(11)
    return _pt_vgg('vgg11_bn', plan, ConvBNModulePT, num_classes, dense_classifier, pretrained)

def pt_vgg11_2_bn(input_shape, num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(11.2)
    return _pt_vgg_width('vgg11_bn', plan, ConvBNModulePT, num_classes, dense_classifier, pretrained)

def pt_vgg11_4_bn(input_shape, num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(11.4)
    return _pt_vgg_width('vgg11_bn', plan, ConvBNModulePT, num_classes, dense_classifier, pretrained)

def pt_vgg11_8_bn(input_shape, num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(11.8)
    return _pt_vgg_width('vgg11_bn', plan, ConvBNModulePT, num_classes, dense_classifier, pretrained)

def pt_vgg11_16_bn(input_shape, num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(11.16)
    return _pt_vgg_width('vgg11_bn', plan, ConvBNModulePT, num_classes, dense_classifier, pretrained)

def pt_vgg11_32_bn(input_shape, num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(11.32)
    return _pt_vgg_width('vgg11_bn', plan, ConvBNModulePT, num_classes, dense_classifier, pretrained)

def vgg13(input_shape, num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(13)
    return _vgg('vgg13_bn', plan, ConvModule, num_classes, dense_classifier, pretrained)

def vgg13_bn(input_shape, num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(13)
    return _vgg('vgg13_bn', plan, ConvBNModule, num_classes, dense_classifier, pretrained)

def vgg16(input_shape, num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(16)
    return _vgg('vgg16_bn', plan, ConvModule, num_classes, dense_classifier, pretrained)

def vgg16_bn(input_shape, num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(16)
    return _vgg('vgg16_bn', plan, ConvBNModule, num_classes, dense_classifier, pretrained)

def vgg19(input_shape, num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(19)
    return _vgg('vgg19_bn', plan, ConvModule, num_classes, dense_classifier, pretrained)

def vgg19_bn(input_shape, num_classes, dense_classifier=False, pretrained=False):
    plan = _plan(19)
    return _vgg('vgg19_bn', plan, ConvBNModule, num_classes, dense_classifier, pretrained)
