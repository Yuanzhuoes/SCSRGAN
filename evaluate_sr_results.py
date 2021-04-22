import os

import numpy as np
import pandas as pd


import torch
import torchvision.utils as utils
from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader
from tqdm import tqdm


from utils import PairDatasetFromFolder
import metrics.LPIPS.models.dist_model as dm
from metrics.psnr_ssim import calculate_psnr, calculate_ssim
from models import Generator
from option import parse_args

if __name__ == '__main__':
    opt = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {'Set5': {'psnr': [], 'ssim': [], 'lpips': []},
               'Set14': {'psnr': [], 'ssim': [], 'lpips': []},
               'BSD100': {'psnr': [], 'ssim': [], 'lpips': []},
               'Urban100': {'psnr': [], 'ssim': [], 'lpips': []},
               'im': {'psnr': [], 'ssim': [], 'lpips': []}}

    test_set = PairDatasetFromFolder('data/benchmark_dataset/pair_dataset', upscale_factor=opt.upSampling)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
    test_bar = tqdm(test_loader, desc='[testing benchmark datasets]')
    model_LPIPS = dm.DistModel()
    model_LPIPS.initialize(model='net-lin', net='alex', use_gpu=False)
    out_path = 'data/benchmark_pair_results/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for image_name, SR, HR in test_bar:
        image_name = image_name[0]

        SR_lpips = ToPILImage()(SR.squeeze(0))
        HR_lpips = ToPILImage()(HR.squeeze(0))

        SR_lpips = np.array(SR_lpips)
        HR_lpips = np.array(HR_lpips)
        SR_lpips = torch.Tensor((SR_lpips / 127.5 - 1)[:, :, :, np.newaxis].transpose((3, 2, 0, 1)))
        HR_lpips = torch.Tensor((HR_lpips / 127.5 - 1)[:, :, :, np.newaxis].transpose((3, 2, 0, 1)))

        SR = SR.data.float().to(device).clamp_(0, 1)
        HR = HR.data.float().to(device).clamp_(0, 1)

        lpips = batch_lpips = model_LPIPS.forward(SR_lpips, HR_lpips)

        # batch size dimention deduction for calculating psnr and ssim

        SR = SR.squeeze(0)
        HR = HR.squeeze(0)

        psnr = calculate_psnr(SR, HR, test_y_channel=True)
        ssim = calculate_ssim(SR, HR, test_y_channel=True)

        test_images = torch.stack([HR.data.to(device), SR.data.to(device)])
        image = utils.make_grid(test_images, nrow=2, padding=1)
        utils.save_image(image, out_path + image_name.split('.')[0] + '_psnr_%.4f_ssim_%.4f_lpips_%.4f.' %
        (psnr, ssim, lpips) + image_name.split('.')[-1], padding=1)

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
