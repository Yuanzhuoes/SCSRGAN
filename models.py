import torch.nn as nn
import torch
from torchvision.models import vgg19
import torch.nn.functional as F

# try swish?
def swish(x):
    return x * F.sigmoid(x)

class DownBlock(nn.Module):
    def __init__(self, scale):
        super().__init__()

        self.scale = scale

    def forward(self, x):
        n, c, h, w = x.size()
        x = x.view(n, c, h//self.scale, self.scale, w//self.scale, self.scale)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        x = x.view(n, c * (self.scale**2), h//self.scale, w//self.scale)
        return x


class Get_gradient(nn.Module):
    def __init__(self):
        super(Get_gradient, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def forward(self, x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=2)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=2)

        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding=2)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding=2)

        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding=2)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding=2)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        x = torch.cat([x0, x1, x2], dim=1)
        return x


class Get_gradient_nopadding(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)  # from 2 dimention to 4 dimention
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)

        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)  # fixed kernel

    def forward(self, x):
        x_list = []
        for i in range(x.shape[1]):  # channel
            x_i = x[:, i]
            x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
            x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
            x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)  # get the gradient
            x_list.append(x_i)

        x = torch.cat(x_list, dim=1)

        return x

#  没有最后一层池化层、全连接层、softmax层的VGG19，且最后一层卷积未激活，尺寸要能被16整除。与SRGAN不同，作者认为克服了两个缺点。

class FeatureExtractor(nn.Module):

    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # vgg19预训练模型
        vgg19_model = vgg19(pretrained=True)
        mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        # [0.485-1, 0.456-1, 0.406-1] if input in range [-1,1]
        std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        # [0.229*2, 0.224*2, 0.225*2] if input in range [-1,1]
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        self.vgg19_54 = nn.Sequential(*list(vgg19_model.features.children())[:35])  # 取前35层，生成列表，拆分，Sequential
        for k, v in self.vgg19_54.named_parameters():
            v.requires_grad = False

    def forward(self, img):
        img = (img - self.mean) / self.std
        output = self.vgg19_54(img)
        return output

#
class DenseResidualBlock(nn.Module):

    def __init__(self, grow_ch, res_scale=0.2):
        super(DenseResidualBlock, self).__init__()
        self.res_scale = res_scale

        def block(in_features=64, ch=32, non_linearity=True):
            layers = [nn.Conv2d(in_features, ch, 3, 1, 1, bias=True)]
            if non_linearity:
                layers += [nn.LeakyReLU(negative_slope=0.2, inplace=True)]
            return nn.Sequential(*layers)

        self.b1 = block(2 * grow_ch, grow_ch)
        self.b2 = block(3 * grow_ch, grow_ch)
        self.b3 = block(4 * grow_ch, grow_ch)
        self.b4 = block(5 * grow_ch, grow_ch)
        self.b5 = block(6 * grow_ch, 2 * grow_ch, non_linearity=False)
        self.blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]

    def forward(self, x):
        inputs = x
        outputs = x
        for block in self.blocks:
            outputs = block(inputs)
            inputs = torch.cat([inputs, outputs], 1)  # 拼接通道数，因为第0个参数是batch_size大小，所以为1
        return outputs.mul(self.res_scale) + x


class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, grow_ch, res_scale=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        self.dense_blocks = nn.Sequential(
            DenseResidualBlock(grow_ch), DenseResidualBlock(grow_ch), DenseResidualBlock(grow_ch)
        )

    def forward(self, x):
        return self.dense_blocks(x).mul(self.res_scale) + x


class Generator(nn.Module):
    def __init__(self, channels, filters=64, num_res_blocks=23, up_scale=4, grow_ch=32):
        super(Generator, self).__init__()

        # DownBlock 将图片缩小up_scale倍
        self.head = DownBlock(up_scale)
        # Gradient Extractor
        self.gradient_extractor = Get_gradient_nopadding()
        # First sr layer
        self.conv1 = nn.Conv2d(channels*up_scale**2, filters, kernel_size=3, stride=1, padding=1)
        # First gm layer
        self.conv1_0 = nn.Conv2d(channels*up_scale**2, filters, kernel_size=3, stride=1, padding=1)
        # Residual blocks
        self.res_blocks = nn.Sequential(*[ResidualInResidualDenseBlock(grow_ch=grow_ch) for _ in range(num_res_blocks)])
        # Gradient block
        self.grandient_block = nn.Sequential(
            ResidualInResidualDenseBlock(grow_ch=grow_ch * 2),
            nn.Conv2d(grow_ch * 4, grow_ch * 2, kernel_size=3, stride=1, padding=1)
        )
        # Second sr conv layer before upsampling
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        # Second gm conv layer before upsampling
        self.conv2_0 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        # SR upsampling layers
        upsample_layers = []
        for _ in range(2):
            upsample_layers += [
                nn.Conv2d(filters, filters * 4, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(),
                nn.PixelShuffle(upscale_factor=2),  # H × W × C · r**2 to rH × rW × C
            ]
        self.upsampling = nn.Sequential(*upsample_layers)
        # GM upsampling layers
        gradient_upsample_layers = []
        for _ in range(2):
            gradient_upsample_layers += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 64, 3, 1, 1)
            ]
        self.gradient_upsanmpling = nn.Sequential(*gradient_upsample_layers)
        # Final sr output block
        self.conv3 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(filters, channels, kernel_size=3, stride=1, padding=1),
        )
        # Final GM output block
        self.conv3_0 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 3, 1, 1, 0)  # no padding
        )
        # fusion block
        self.fusion_block = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),  # 64 channel
            ResidualInResidualDenseBlock(grow_ch=grow_ch),  # 64 channel
            nn.Conv2d(64, 64, 3, 1, 1)
        )
        # Final fusion output block
        self.f_sr_conv = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 3, 3, 1, 1)
        )
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)  # ???

    def forward(self, x):
        x1 = self.head(x)
        out1 = self.conv1(x1)

        out = self.res_blocks[:4](out1)
        res1 = out  # 64 channel

        out = self.res_blocks[5:9](out)
        res2 = out

        out = self.res_blocks[10:14](out)
        res3 = out

        out = self.res_blocks[14:20](out)
        res4 = out

        out = self.res_blocks[21:](out)

        out2 = self.conv2(out)
        out = torch.add(out1, out2)  # 对位相加
        out = self.upsampling(out)  # 64 * 128 * 128

        lr_grad_ref = self.gradient_extractor(x)  # 先提取梯度图
        lr_grad = self.head(lr_grad_ref)  # 再缩小
        gm = self.conv1_0(lr_grad)  # Gradient Map 64 channel
        gm1 = self.grandient_block(torch.cat([gm, res1], 1))
        gm2 = self.grandient_block(torch.cat([gm1, res2], 1))
        gm3 = self.grandient_block(torch.cat([gm2, res3], 1))
        gm4 = self.grandient_block(torch.cat([gm3, res4], 1))
        gm4 = self.conv2_0(gm4)
        gm4 = torch.add(gm4, gm)  # 64 channel
        gm4 = self.gradient_upsanmpling(gm4)  # 64 * 128 * 128
        gm_out_d = gm4  # 64 * 128 * 128  gm_out_d的更新会传给gm4吗
        gm_out = self.conv3_0(gm4)  # 3 * 128 * 128 gm branch

        # 在上采样之后直接fusion 然后卷积生成SR
        input_fusion = torch.cat([out, gm_out_d], 1)  # 128 channel
        output_fusion = self.fusion_block(input_fusion)  # 64 channel
        output_fusion = self.conv3(output_fusion)  # 3 channel conv3

        return output_fusion, gm_out, lr_grad_ref

class Generator_pre(nn.Module):
    def __init__(self, channels, filters=64, num_res_blocks=23, up_scale=4, grow_ch=32):
        super(Generator_pre, self).__init__()

        # DownBlock 将图片缩小up_scale倍
        self.head = DownBlock(up_scale)
        # First sr layer
        self.conv1 = nn.Conv2d(channels*up_scale**2, filters, kernel_size=3, stride=1, padding=1)
        # Residual blocks
        self.res_blocks = nn.Sequential(*[ResidualInResidualDenseBlock(grow_ch=grow_ch) for _ in range(num_res_blocks)])
        # Second sr conv layer before upsampling
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        # SR upsampling layers
        upsample_layers = []
        for _ in range(2):
            upsample_layers += [
                nn.Conv2d(filters, filters * 4, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(),
                nn.PixelShuffle(upscale_factor=2),  # H × W × C · r**2 to rH × rW × C
            ]
        self.upsampling = nn.Sequential(*upsample_layers)
        # Final sr output block
        self.conv3 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(filters, channels, kernel_size=3, stride=1, padding=1),
        )
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)  # ???

    def forward(self, x):
        x1 = self.head(x)
        out1 = self.conv1(x1)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)  # 对位相加
        out = self.upsampling(out)  # 64 * 128 * 128
        output = self.conv3(out)  # 3 channel conv3

        return output
# batch norm or spectrum?
class Discriminator(nn.Module):
    def __init__(self, in_channels, out_filters, mode='CNA'):
        super(Discriminator, self).__init__()

        self.conv0_0 = nn.Conv2d(in_channels, out_filters, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(out_filters, out_filters, 4, 2, 1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(out_filters, affine=True)

        self.conv1_0 = nn.Conv2d(out_filters, out_filters * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.BatchNorm2d(out_filters * 2, affine=True)
        self.conv1_1 = nn.Conv2d(
            out_filters * 2, out_filters * 2, 4, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(out_filters * 2, affine=True)

        self.conv2_0 = nn.Conv2d(
            out_filters * 2, out_filters * 4, 3, 1, 1, bias=False)
        self.bn2_0 = nn.BatchNorm2d(out_filters * 4, affine=True)
        self.conv2_1 = nn.Conv2d(
            out_filters * 4, out_filters * 4, 4, 2, 1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(out_filters * 4, affine=True)

        self.conv3_0 = nn.Conv2d(
            out_filters * 4, out_filters * 8, 3, 1, 1, bias=False)
        self.bn3_0 = nn.BatchNorm2d(out_filters * 8, affine=True)
        self.conv3_1 = nn.Conv2d(
            out_filters * 8, out_filters * 8, 4, 2, 1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(out_filters * 8, affine=True)

        self.conv4_0 = nn.Conv2d(
            out_filters * 8, out_filters * 8, 3, 1, 1, bias=False)
        self.bn4_0 = nn.BatchNorm2d(out_filters * 8, affine=True)
        self.conv4_1 = nn.Conv2d(
            out_filters * 8, out_filters * 8, 4, 2, 1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(out_filters * 8, affine=True)

        self.linear1 = nn.Linear(out_filters * 8 * 4 * 4, 100)
        self.linear2 = nn.Linear(100, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, img):

        feat = self.lrelu(self.conv0_0(img))
        feat = self.lrelu(self.bn0_1(
            self.conv0_1(feat)))  # output spatial size: (64, 64)

        feat = self.lrelu(self.bn1_0(self.conv1_0(feat)))
        feat = self.lrelu(self.bn1_1(
            self.conv1_1(feat)))  # output spatial size: (32, 32)

        feat = self.lrelu(self.bn2_0(self.conv2_0(feat)))
        feat = self.lrelu(self.bn2_1(
            self.conv2_1(feat)))  # output spatial size: (16, 16)

        feat = self.lrelu(self.bn3_0(self.conv3_0(feat)))
        feat = self.lrelu(self.bn3_1(
            self.conv3_1(feat)))  # output spatial size: (8, 8)

        feat = self.lrelu(self.bn4_0(self.conv4_0(feat)))
        feat = self.lrelu(self.bn4_1(
            self.conv4_1(feat)))  # output spatial size: (4, 4)

        feat = feat.view(feat.size(0), -1)  # 拉伸为二维，batch_size * (c * h * w)
        feat = self.lrelu(self.linear1(feat))  # 全连接层 512 * 4 * 4 与输入矩阵的列数相同，转换为batch_size * 100二维矩阵
        out = self.linear2(feat)  # 输出Batch_size * 1个数
        return out


### U-Net Discriminator ###
# Residual block for the discriminator
class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, which_conv=nn.Conv2d, which_bn=nn.BatchNorm2d, wide=True,
                 preactivation=True, activation=nn.LeakyReLU(0.1, inplace=False), downsample=nn.AvgPool2d(2, stride=2)):
        super(DBlock, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        # If using wide D (as in SA-GAN and BigGAN), change the channel pattern
        self.hidden_channels = self.out_channels if wide else self.in_channels
        self.which_conv, self.which_bn = which_conv, which_bn
        self.preactivation = preactivation
        self.activation = activation
        self.downsample = downsample

        # Conv layers
        self.conv1 = self.which_conv(self.in_channels, self.hidden_channels, kernel_size=3, padding=1)
        self.conv2 = self.which_conv(self.hidden_channels, self.out_channels, kernel_size=3, padding=1)
        self.learnable_sc = True if (in_channels != out_channels) or downsample else False
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels,
                                           kernel_size=1, padding=0)

        self.bn1 = self.which_bn(self.hidden_channels)
        self.bn2 = self.which_bn(out_channels)

    # def shortcut(self, x):
    #     if self.preactivation:
    #         if self.learnable_sc:
    #             x = self.conv_sc(x)
    #         if self.downsample:
    #             x = self.downsample(x)
    #     else:
    #         if self.downsample:
    #             x = self.downsample(x)
    #         if self.learnable_sc:
    #             x = self.conv_sc(x)
    #     return x

    def forward(self, x):
        if self.preactivation:
            # h = self.activation(x) # NOT TODAY SATAN
            # Andy's note: This line *must* be an out-of-place ReLU or it
            #              will negatively affect the shortcut connection.
            h = self.activation(x)
        else:
            h = x
        h = self.bn1(self.conv1(h))
        # h = self.conv2(self.activation(h))
        if self.downsample:
            h = self.downsample(h)

        return h  # + self.shortcut(x)


class GBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 which_conv=nn.Conv2d, which_bn=nn.BatchNorm2d, activation=nn.LeakyReLU(0.1, inplace=False),
                 upsample=nn.Upsample(scale_factor=2, mode='nearest')):
        super(GBlock, self).__init__()

        self.in_channels, self.out_channels = in_channels, out_channels
        self.which_conv, self.which_bn = which_conv, which_bn
        self.activation = activation
        self.upsample = upsample
        # Conv layers
        self.conv1 = self.which_conv(self.in_channels, self.out_channels, kernel_size=3, padding=1)
        self.conv2 = self.which_conv(self.out_channels, self.out_channels, kernel_size=3, padding=1)
        self.learnable_sc = in_channels != out_channels or upsample
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels,
                                           kernel_size=1, padding=0)
        # Batchnorm layers
        self.bn1 = self.which_bn(out_channels)
        self.bn2 = self.which_bn(out_channels)
        # upsample layers
        self.upsample = upsample

    def forward(self, x):
        h = self.activation(x)
        if self.upsample:
            h = self.upsample(h)
            # x = self.upsample(x)
        h = self.bn1(self.conv1(h))
        # h = self.activation(self.bn2(h))
        # h = self.conv2(h)
        # if self.learnable_sc:
        #     x = self.conv_sc(x)
        return h  # + x


class UnetD(torch.nn.Module):
    def __init__(self):
        super(UnetD, self).__init__()

        self.enc_b1 = DBlock(3, 64, preactivation=False)
        self.enc_b2 = DBlock(64, 128)
        self.enc_b3 = DBlock(128, 192)
        self.enc_b4 = DBlock(192, 256)
        self.enc_b5 = DBlock(256, 320)
        self.enc_b6 = DBlock(320, 384)

        self.enc_out = nn.Conv2d(384, 1, kernel_size=1, padding=0)

        self.dec_b1 = GBlock(384, 320)
        self.dec_b2 = GBlock(320 * 2, 256)
        self.dec_b3 = GBlock(256 * 2, 192)
        self.dec_b4 = GBlock(192 * 2, 128)
        self.dec_b5 = GBlock(128 * 2, 64)
        self.dec_b6 = GBlock(64 * 2, 32)

        self.dec_out = nn.Conv2d(32, 1, kernel_size=1, padding=0)

        # Init weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.lower().find('conv') != -1:
                # print(classname)
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif classname.find('bn') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        e1 = self.enc_b1(x)  # 64 * 64 * 64
        e2 = self.enc_b2(e1)  # 128 * 32 * 32
        e3 = self.enc_b3(e2)  # 192 * 16 * 16
        e4 = self.enc_b4(e3)  # 256 * 8 * 8
        e5 = self.enc_b5(e4)  # 320 * 4 * 4
        e6 = self.enc_b6(e5)  # 384 * 2 * 2

        e_out = self.enc_out(F.leaky_relu(e6, 0.1))
        # print(e1.size())
        # print(e2.size())
        # print(e3.size())
        # print(e4.size())
        # print(e5.size())
        # print(e6.size())

        d1 = self.dec_b1(e6)
        d2 = self.dec_b2(torch.cat([d1, e5], 1))
        d3 = self.dec_b3(torch.cat([d2, e4], 1))
        d4 = self.dec_b4(torch.cat([d3, e3], 1))
        d5 = self.dec_b5(torch.cat([d4, e2], 1))
        d6 = self.dec_b6(torch.cat([d5, e1], 1))

        d_out = self.dec_out(F.leaky_relu(d6, 0.1))

        return e_out, d_out, [e1, e2, e3, e4, e5, e6], [d1, d2, d3, d4, d5, d6]


### My U-Net Discriminator ###
# Residual block for the discriminator
class My_DBlock(nn.Module):  # dont use BN?
    def __init__(self, in_channels, out_filters, not_first_block=True, active=True):
        super(My_DBlock, self).__init__()

        self.in_channels, self.out_channels = in_channels, out_filters
        self.not_first_block = not_first_block
        self.active = active
        self.conv0 = nn.Conv2d(in_channels, out_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels, out_filters, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(out_filters, out_filters, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_filters, affine=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=False)  # must be false, 0.1 or 0.2?

    def forward(self, x):
        if self.active:
            h = self.lrelu(x)
            h = self.conv0(h)
        else:
            h = x
            h = self.conv1(h)
        if self.not_first_block:
            h = self.bn1(h)
        h = self.lrelu(h)
        h = self.conv2(h)
        h = self.bn1(h)
        return h


class My_GBlock(nn.Module):  # dont use BN?
    def __init__(self, in_channels, out_filters):
        super(My_GBlock, self).__init__()

        self.in_channels, self.out_channels = in_channels, out_filters
        self.conv1 = nn.Conv2d(in_channels, out_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_filters, affine=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        upsample_layers = []
        upsample_layers += [
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.LeakyReLU(0.1, inplace=True),
        ]
        self.upsampling = nn.Sequential(*upsample_layers)

    def forward(self, x):
        h = self.lrelu(x)
        h = self.upsampling(h)
        h = self.bn1(self.conv1(h))
        return h

# 原文的Discriminator leaky是0.1
class My_UnetD(nn.Module):
    def __init__(self):
        super(My_UnetD, self).__init__()

        self.enc_b1 = My_DBlock(3, 64, not_first_block=False, active=False) # 64
        self.enc_b2 = My_DBlock(64, 128)  # 32
        self.enc_b3 = My_DBlock(128, 192)  # 16
        self.enc_b4 = My_DBlock(192, 256)  # 8
        self.enc_b5 = My_DBlock(256, 320)  # 4
        self.enc_b6 = My_DBlock(320, 384)  # 2

        self.enc_out = nn.Conv2d(384, 1, kernel_size=1, padding=0)

        self.dec_b1 = My_GBlock(384, 320)
        self.dec_b2 = My_GBlock(320 * 2, 256)
        self.dec_b3 = My_GBlock(256 * 2, 192)
        self.dec_b4 = My_GBlock(192 * 2, 128)
        self.dec_b5 = My_GBlock(128 * 2, 64)
        self.dec_b6 = My_GBlock(64 * 2, 32)

        self.dec_out = nn.Conv2d(32, 1, kernel_size=1, padding=0)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=False)

    def forward(self, img):
        assert img.size(2) == 128 and img.size(3) == 128, (
            f'Input spatial size must be 128x128, '
            f'but received {img.size()}.')

        e1 = self.enc_b1(img)  # 64 * 64 * 64
        e2 = self.enc_b2(e1)   # 128 * 32 * 32
        e3 = self.enc_b3(e2)  # 192 * 16 * 16
        e4 = self.enc_b4(e3)  # 256 * 8 * 8
        e5 = self.enc_b5(e4)  # 320 * 4 * 4
        e6 = self.enc_b6(e5)  # 384 * 2 * 2

        e_out = self.enc_out(self.lrelu(e6))  # 1 * 2 * 2

        d1 = self.dec_b1(e6)  # 384 * 4 * 4
        d2 = self.dec_b2(torch.cat([d1, e5], 1))  # 256 * 8 * 8
        d3 = self.dec_b3(torch.cat([d2, e4], 1))  # 192 * 16 * 16
        d4 = self.dec_b4(torch.cat([d3, e3], 1))  # 124 * 32 * 32
        d5 = self.dec_b5(torch.cat([d4, e2], 1))  # 64 * 64 * 64
        d6 = self.dec_b6(torch.cat([d5, e1], 1))  # 32 * 128 * 128

        d_out = self.dec_out(self.lrelu(d6))  # 1 * 128 * 128
        return e_out, d_out, [e1, e2, e3, e4, e5, e6], [d1, d2, d3, d4, d5, d6]
