import torch
import torch.nn as nn
from PIL import Image
from os import listdir
from os.path import join
import numpy as np
from torchvision.transforms import Compose, ToTensor, ToPILImage, CenterCrop, Resize, RandomCrop
from torch.utils.data.dataset import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
        ToTensor()
    ])


def im2tensor(image, cent=1., factor=255./2.):
    return torch.Tensor((image / factor - cent)
                        [:, :, :, np.newaxis].transpose((3, 2, 0, 1)))


# 将裁剪的高清图像(Tensor)转化为PIL,然后缩小upscale_factor倍并双三次插值为低清图像，再返回为Tensor
def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])


def display_transform():
    return Compose([
        ToPILImage(),
        CenterCrop(800),
        ToTensor()
    ])

def DownSample2DMatlab(tensor, scale, method='cubic', antialiasing=True, cuda=True):
    '''
    This gives same result as MATLAB downsampling
    tensor: 4D tensor [Batch, Channel, Height, Width],
            height and width must be divided by the denominator of scale factor
    scale: Even integer denominator scale factor only (e.g. 1/2,1/4,1/8,...)
           Or list [1/2, 1/4] : [V scale, H scale]
    method: 'cubic' as default, currently cubic supported
    antialiasing: True as default
    '''

    # For cubic interpolation,
    # Cubic Convolution Interpolation for Digital Image Processing, ASSP, 1981
    def cubic(x):
        absx = np.abs(x)
        absx2 = np.multiply(absx, absx)
        absx3 = np.multiply(absx2, absx)

        f = np.multiply((1.5*absx3 - 2.5*absx2 + 1), np.less_equal(absx, 1)) + \
            np.multiply((-0.5*absx3 + 2.5*absx2 - 4*absx + 2),
            np.logical_and(np.less(1, absx), np.less_equal(absx, 2)))

        return f

    # Generate resize kernel (resize weight computation)
    def contributions(scale, kernel, kernel_width, antialiasing):
        if scale < 1 and antialiasing:
          kernel_width = kernel_width / scale

        x = np.ones((1, 1))

        u = x/scale + 0.5 * (1 - 1/scale)

        left = np.floor(u - kernel_width/2)

        P = int(np.ceil(kernel_width) + 2)

        indices = np.tile(left, (1, P)) + np.expand_dims(np.arange(0, P), 0)

        if scale < 1 and antialiasing:
          weights = scale * kernel(scale * (np.tile(u, (1, P)) - indices))
        else:
          weights = kernel(np.tile(u, (1, P)) - indices)

        weights = weights / np.expand_dims(np.sum(weights, 1), 1)

        save = np.where(np.any(weights, 0))
        weights = weights[:, save[0]]

        return weights

    # Resize along a specified dimension
    def resizeAlongDim(tensor, scale_v, scale_h, kernel_width, weights):#, indices):
        if scale_v < 1 and antialiasing:
           kernel_width_v = kernel_width / scale_v
        else:
           kernel_width_v = kernel_width
        if scale_h < 1 and antialiasing:
           kernel_width_h = kernel_width / scale_h
        else:
           kernel_width_h = kernel_width

        # Generate filter
        f_height = np.transpose(weights[0][0:1, :])
        f_width = weights[1][0:1, :]
        f = np.dot(f_height, f_width)
        f = f[np.newaxis, np.newaxis, :, :]
        F = torch.from_numpy(f.astype('float32'))

        # Reflect padding
        i_scale_v = int(1/scale_v)
        i_scale_h = int(1/scale_h)
        pad_top = int((kernel_width_v - i_scale_v) / 2)
        if i_scale_v == 1:
            pad_top = 0
        pad_bottom = int((kernel_width_h - i_scale_h) / 2)
        if i_scale_h == 1:
            pad_bottom = 0
        pad_array = ([pad_bottom, pad_bottom, pad_top, pad_top])
        kernel_width_v = int(kernel_width_v)
        kernel_width_h = int(kernel_width_h)

        #
        tensor_shape = tensor.size()
        num_channel = tensor_shape[1]
        FT = nn.Conv2d(1, 1, (kernel_width_v, kernel_width_h), (i_scale_v, i_scale_h), bias=False)
        FT.weight.data = F
        FT.to(device)
        FT.requires_grad = False

        # actually, we want 'symmetric' padding, not 'reflect'
        outs = []
        for c in range(num_channel):
            padded = nn.functional.pad(tensor[:,c:c+1,:,:], pad_array, 'reflect')
            outs.append(FT(padded))
        out = torch.cat(outs, 1)

        return out

    if method == 'cubic':
        kernel = cubic

    kernel_width = 4

    if type(scale) is list:
        scale_v = float(scale[0])
        scale_h = float(scale[1])

        weights = []
        for i in range(2):
            W = contributions(float(scale[i]), kernel, kernel_width, antialiasing)
            weights.append(W)
    else:
        scale = float(scale)

        scale_v = scale
        scale_h = scale

        weights = []
        for i in range(2):
            W = contributions(scale, kernel, kernel_width, antialiasing)
            weights.append(W)

    # np.save('bic_x4_downsample_h.npy', weights[0])

    tensor = resizeAlongDim(tensor, scale_v, scale_h, kernel_width, weights)

    return tensor


def Huber(input, target, delta=0.01, reduce=True):
    abs_error = torch.abs(input - target)
    # abs_error < delta, abs_error
    # abs_error > delta, delta
    quadratic = torch.clamp(abs_error, max=delta)

    # The following expression is the same in value as
    # tf.maximum(abs_error - delta, 0), but importantly the gradient for the
    # expression when abs_error == delta is 0 (for tf.maximum it would be 1).
    # This is necessary to avoid doubling the gradient, since there is already a
    # nonzero contribution to the gradient from the quadratic term.
    linear = (abs_error - quadratic)
    losses = 0.5 * torch.pow(quadratic, 2) + delta * linear

    if reduce:
        return torch.mean(losses)
    else:
        return losses


class TrainDatasetFromFolder(Dataset):
    #  形参为文件路径，裁剪区域，上采样参数
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        # 让TrainDatasetFromFolder包含Dataset所有属性
        super(TrainDatasetFromFolder, self).__init__()
        # 图像文件名为文件夹路径+图像格式后缀(已调用is_image_file()函数检查是否为图像)，所有文件名合为一个列表
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        # 计算有效裁剪区域(能被upscale_factor整除)
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        # 裁剪高清图像
        self.hr_transform = Compose([RandomCrop(crop_size), ToTensor()])


    #  实例对象（假定为p），可以像这样p[key]取值
    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))

        return hr_image

    # 这个类是一个图片列表，返回列表元素个数
    def __len__(self):
        return len(self.image_filenames)


class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(ValDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        # w, h = hr_image.size
        # crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)

        return ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_filenames)

class PairDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(PairDatasetFromFolder, self).__init__()
        self.sr_path = dataset_dir + '/SR/'
        self.hr_path = dataset_dir + '/HR/'
        self.upscale_factor = upscale_factor
        self.sr_filenames = [join(self.sr_path, x) for x in listdir(self.sr_path) if is_image_file(x)]
        self.hr_filenames = [join(self.hr_path, x) for x in listdir(self.hr_path) if is_image_file(x)]

    def __getitem__(self, index):
        image_name = self.sr_filenames[index].split('/')[-1]

        sr_image = Image.open(self.sr_filenames[index])
        hr_image = Image.open(self.hr_filenames[index])

        return image_name,ToTensor()(sr_image), ToTensor()(hr_image)

    def __len__(self):
        return len(self.sr_filenames)