import argparse
import os
from math import log10
from PIL import Image
import numpy as np
import pandas as pd

import torch.nn.functional as F
import torch
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tq
from torchvision.transforms import ToTensordm import tqdm

import pytorch_ssim
from utils import TestDatasetFromFolder
import metrics.LPIPS.models.dist_model as dm
from metrics.psnr_ssim import calculate_psnr, calculate_ssim
from metrics.niqe import calculate_niqe
from models import Generator
from option import parse_args

if __name__ == '__main__':
    opt = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {'Set5': {'psnr': [], 'ssim': [], 'lpips': []},
               'Set14': {'psnr': [], 'ssim': [], 'lpips': []},
               'BSD100': {'psnr': [], 'ssim': [], 'lpips': []},
               'Urban100': {'psnr': [], 'ssim': [], 'lpips': []},
               'SunHays80': {'psnr': [], 'ssim': [], 'lpips': []}}

    generator = Generator(3, filters=64, num_res_blocks=opt.residual_blocks, up_scale=opt.upSampling).eval().to(device)
    checkpoint = torch.load('./checkpoints/model_G_i0085'
                            '00_best.pth', map_location=torch.device('cpu')) # GPU to CPU
    generator.load_state_dict(checkpoint['model_G'])
    test_set = TestDatasetFromFolder('data/test', upscale_factor=opt.upSampling)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
    test_bar = tqdm(test_loader, desc='[testing benchmark datasets]')
    model_LPIPS = dm.DistModel()
    model_LPIPS.initialize(model='net-lin', net='alex', use_gpu=False)
    generator.eval()
    out_path = 'benchmark_results/SRF_' + str(opt.upSampling) + '/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for image_name, lr_image, hr_restore_img, hr_image in test_bar:
        image_name = image_name[0]

        lr_image = F.interpolate(lr_image, scale_factor=opt.upSampling, mode="nearest")  # Tensor [0, 1]
        lr_image = lr_image.data.float().cpu().clamp_(0, 1)
        sr_image, _, _ = generator(lr_image)
        sr_image = sr_image.data.float().cpu().clamp_(0, 1)
        # sr_image = torch.clamp(sr_image, 0, 1)  # for correct display
        hr_image = hr_image.data.float().cpu().clamp_(0, 1)
        lpips = batch_lpips = model_LPIPS.forward(sr_image, hr_image)

        # batch size dimention deduction for calculating psnr and ssim
        sr_image = sr_image.squeeze(0)
        hr_image = hr_image.squeeze(0)

        psnr = calculate_psnr(sr_image, hr_image, test_y_channel=True)
        ssim = calculate_ssim(sr_image, hr_image, test_y_channel=True)

        # test_images = torch.stack(
            # [hr_restore_img.squeeze(0), hr_image.data.cpu(), sr_image.data.cpu()])
        # image = utils.make_grid(test_images, nrow=3, padding=1)
        # utils.save_image(image, out_path + image_name.split('.')[0] + '_psnr_%.4f_ssim_%.4f_lpips_%.4f.' %
                         # (psnr, ssim, lpips) + image_name.split('.')[-1], padding=1)

        # save psnr\ssim
        results[image_name.split('_')[0]]['psnr'].append(psnr)  # 得到第一个_前的内容
        results[image_name.split('_')[0]]['ssim'].append(ssim)
        results[image_name.split('_')[0]]['lpips'].append(lpips)

    out_path = 'statistics/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    saved_results = {'psnr': [], 'ssim': [], 'lpips': []}
    for item in results.values():
        psnr = np.array(item['psnr'])
        ssim = np.array(item['ssim'])
        niqe = np.array(item['lpips'])
        if (len(psnr) == 0) or (len(ssim) == 0) or (len(niqe) == 0):
            psnr = 'No data'
            ssim = 'No data'
            niqe = 'No data'
        else:
            psnr = psnr.mean()
            ssim = ssim.mean()
            niqe = niqe.mean()
        saved_results['psnr'].append(psnr)
        saved_results['ssim'].append(ssim)
        saved_results['lpips'].append(niqe)

    data_frame = pd.DataFrame(saved_results, results.keys())
    data_frame.to_csv(out_path + 'srf_' + str(opt.upSampling) + '_benchmark_test_results.csv', index_label='DataSet')
