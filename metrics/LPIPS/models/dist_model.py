from __future__ import absolute_import

import sys
sys.path.append('..')
sys.path.append('.')
import torch
import numpy as np
import os

from . import networks_basic as networks

import sys
from torch.autograd import Variable
sys.path.insert(1, './LPIPS/')

# LPIPS Model
class DistModel():

    def initialize(self, model='net-lin', net='·', use_gpu=True, printNet=False, version='0.1'):
        '''
        INPUTS
            model - ['net-lin'] for linearly calibrated network  线性校准网络
                    ['net'] for off-the-shelf network            现成网络
                    ['L2'] for L2 distance in Lab colorspace     实验室色彩空间，接近人类视觉
                    ['SSIM'] for ssim in RGB colorspace
            net - ['squeeze','alex','vgg']
            model_path - if None, will look in weights/[NET_NAME].pth
            colorspace - ['Lab','RGB'] colorspace to use for L2 and SSIM
            use_gpu - bool - whether or not to use a GPU
            printNet - bool - whether or not to print network architecture out
            spatial - bool - whether to output an array containing varying distances across spatial dimensions
            spatial_shape - if given, output spatial shape. if None then spatial shape is determined automatically via spatial_factor (see below).
            spatial_factor - if given, specifies upsampling factor relative to the largest spatial extent of a convolutional layer. if None then resized to size of input images.
            spatial_order - spline order of filter for upsampling in spatial mode, by default 1 (bilinear). # 双线性插值
            is_train - bool - [True] for training mode
            lr - float - initial learning rate
            beta1 - float - initial momentum term for adam
            version - 0.1 for latest, 0.0 was original
        '''

        self.model = model
        self.net = net
        self.use_gpu = use_gpu
        self.model_name = '%s [%s]' % (model, net)
        # default net-lin
        self.net = networks.PNetLin(use_gpu=use_gpu, pnet_type=net, use_dropout=True, version=version)  # net always alex
        kw = {}
        if not use_gpu:
            kw['map_location'] = 'cpu'
        import inspect
        # model_path = './PerceptualSimilarity/weights/v%s/%s.pth'%(version,net)
        model_path = os.path.abspath(os.path.join(inspect.getfile(self.initialize), '..', '..', 'weights/v%s/%s.pth'
                                                  % (version, net)))

        self.net.load_state_dict(torch.load(model_path, **kw))
        self.parameters = list(self.net.parameters())
        self.net.eval()

        if(printNet):
            print('---------- Networks initialized -------------')
            networks.print_network(self.net)
            print('-----------------------------------------------')

    def forward_pair(self, in1, in2):
        return self.net.forward(in1, in2)  # lpips loss end here

    def forward(self, in0, in1):
        ''' Function computes the distance between image patches in0 and in1
        INPUTS
            in0, in1 - torch.Tensor object of shape Nx3xXxY - image patch scaled to [-1,1]
            retNumpy - [False] to return as torch.Tensor, [True] to return as numpy array
        OUTPUT
            computed distances between in0 and in1
        '''

        self.input_ref = in0
        self.input_p0 = in1

        if (self.use_gpu):
            self.input_ref = self.input_ref.cuda()
            self.input_p0 = self.input_p0.cuda()

        self.var_ref = Variable(self.input_ref, requires_grad=True)
        self.var_p0 = Variable(self.input_p0, requires_grad=True)

        self.d0, _ = self.forward_pair(self.var_ref, self.var_p0)
        self.loss_total = self.d0

        def convert_output(d0):
            ans = d0.cpu().data.numpy()
            ans = ans.flatten()
            return ans

        return convert_output(self.d0)
