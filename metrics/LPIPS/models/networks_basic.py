from __future__ import absolute_import

import sys
sys.path.append('..')
sys.path.append('.')
import torch.nn as nn
from collections import namedtuple
import torch
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_tensor(in_feat, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat**2, dim=1)).view(in_feat.size()[0], 1, in_feat.size()[2],
                                                                in_feat.size()[3])
    return in_feat/(norm_factor.expand_as(in_feat)+eps)


class alexnet(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(alexnet, self).__init__()
        alexnet_pretrained_features = models.alexnet(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(2):
            self.slice1.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(10, 12):
            self.slice5.add_module(str(x), alexnet_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        alexnet_outputs = namedtuple("AlexnetOutputs", ['relu1', 'relu2', 'relu3', 'relu4', 'relu5'])
        out = alexnet_outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5)

        return out


# Learned perceptual metric
# use PNetLin
class PNetLin(nn.Module):
    def __init__(self, pnet_type='alex', use_dropout=True, use_gpu=False, version='0.1'):
        super(PNetLin, self).__init__()

        self.use_gpu = use_gpu
        self.pnet_type = pnet_type  # net always alex
        self.version = version

        net_type = alexnet
        self.chns = [64, 192, 384, 256, 256]

        self.net = [net_type(pretrained=True, requires_grad=True), ]

        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        #  生成一个1*3*1*1的Variable，每个通道的值分别为-.030, -.088, -.188
        self.shift = torch.autograd.Variable(torch.Tensor([-.030, -.088, -.188]).view(1, 3, 1, 1))
        self.scale = torch.autograd.Variable(torch.Tensor([.458, .448, .450]).view(1, 3, 1, 1))

        if(use_gpu):
            self.net[0].to(device)
            self.shift = self.shift.to(device)
            self.scale = self.scale.to(device)
            self.lin0.to(device)
            self.lin1.to(device)
            self.lin2.to(device)
            self.lin3.to(device)
            self.lin4.to(device)

    def forward(self, in0, in1):
        # 每个通道分别对位相减-.030, -.088, -.188，再除以.458, .448, .450
        in0_sc = (in0 - self.shift.expand_as(in0))/self.scale.expand_as(in0)
        in1_sc = (in1 - self.shift.expand_as(in0))/self.scale.expand_as(in0)

        in0_input = in0_sc
        in1_input = in1_sc

        outs0 = self.net[0].forward(in0_input)
        outs1 = self.net[0].forward(in1_input)

        feats0 = {}
        feats1 = {}
        diffs = [0]*len(outs0)

        for (kk, out0) in enumerate(outs0):
            feats0[kk] = normalize_tensor(outs0[kk])
            feats1[kk] = normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk]-feats1[kk])**2

            # diffs[kk] = (outs0[kk]-outs1[kk])**2

        val1 = torch.mean(torch.mean(self.lin0.model(diffs[0]), dim=3), dim=2)
        val2 = torch.mean(torch.mean(self.lin1.model(diffs[1]), dim=3), dim=2)
        val3 = torch.mean(torch.mean(self.lin2.model(diffs[2]), dim=3), dim=2)
        val4 = torch.mean(torch.mean(self.lin3.model(diffs[3]), dim=3), dim=2)
        val5 = torch.mean(torch.mean(self.lin4.model(diffs[4]), dim=3), dim=2)

        val = val1 + val2 + val3 + val4 + val5
        val_out = val.view(val.size()[0], val.size()[1], 1, 1)

        val_out2 = [val1, val2, val3, val4, val5]

        return val_out, val_out2
        # return [val1, val2, val3, val4, val5]


class NetLinLayer(nn.Module):
    ''' A single linear layer which does a 1x1 conv '''
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()

        layers = [nn.Dropout(), ]
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Network', net)
    print('Total number of parameters: %d' % num_params)
