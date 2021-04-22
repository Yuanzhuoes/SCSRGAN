import os
import sys
import pandas as pd
from os import mkdir
from os.path import isdir

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F

from tensorboard_logger import configure, log_value

from option import parse_args
from models import Generator
from utils import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform, DownSample2DMatlab
from augments import apply_augment
from metrics.psnr_ssim import calculate_psnr, calculate_ssim

from torch.optim.lr_scheduler import CosineAnnealingLR


if __name__ == '__main__':
    opt = parse_args()

    print(opt)
    try:
        os.makedirs(opt.out)
    except OSError:
        pass

    START_ITER = 2
    EXP_NAME = "PSNR"
    I_SAVE = 2
    I_VALIDATION = 2
    BEST_PSNR = 1.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = TrainDatasetFromFolder('data/DIV2K_train_HR/Train_HR', crop_size=opt.crop_size,
                                           upscale_factor=opt.upSampling)
    val_dataset = ValDatasetFromFolder('data/DIV2K_valid_HR/Val_HR', upscale_factor=opt.upSampling)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=opt.batchSize, shuffle=True,
                                  num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=4)

    generator = Generator(3, filters=64, num_res_blocks=opt.residual_blocks, up_scale=opt.upSampling).to(device)
    optim_generator_pre = optim.Adam(generator.parameters(), lr=opt.generatorLR_pre)
    scheduler = CosineAnnealingLR(optim_generator_pre, T_max=3, eta_min=1e-7)

    criterion_pixel = torch.nn.L1Loss().to(device)

    configure(
        'logs/' + opt.train_dataroot + '-' + str(64) + '-' + str(opt.generatorLR_pre), flush_secs=5)

    saved_results = {'epoch': [], 'psnr': []}

    def SaveCheckpoint(i, best=False):
        str_best = ''
        if best:
            str_best = '_best'

        if not isdir('./checkpoints/{}'.format(EXP_NAME)):
            mkdir('./checkpoints/{}'.format(EXP_NAME))

        generator_pre_state = {'model_G_pre': generator.state_dict(),
                               'model_G_optimizer_pre': optim_generator_pre.state_dict(),
                               'scheduler': scheduler.state_dict(),
                               'epoch': epoch}

        torch.save(generator_pre_state, './checkpoints/{}/model_G_i{:06d}{}.pth'.format(EXP_NAME, i, str_best))
    # Pre-train generator using L1 Loss, PSNR oriented model
    print('Generator pre-training')

    if START_ITER > 0:
        checkpoint = torch.load('./checkpoints/{}/model_G_i{:06d}.pth'.format(EXP_NAME, START_ITER))
        generator.load_state_dict(checkpoint['model_G_pre'])
        optim_generator_pre.load_state_dict(checkpoint['model_G_optimizer_pre'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
        print('Load Generator epoch {} successfully!'.format(start_epoch))
    else:
        start_epoch = 0
        print('start training generator from scratch!')

    for epoch in range(start_epoch + 1, opt.nEpochs_pre + 1):
        mean_generator_pixel_loss = 0.0
        # data-lr target-hr

        for i, target in enumerate(train_dataloader):  # 16700个样本，一次epoch迭代1045次
            data = DownSample2DMatlab(target, 1/(float(opt.upSampling)))
            data = torch.clamp(data, 0, 1)
            # MoA数据增强
            if target.size() != data.size():
                scale = target.size(2) // data.size(2)
                data = F.interpolate(data, scale_factor=scale, mode="nearest")  # 从第三维reshape,前两维是batchsize和channel

            HR, LR, mask, aug = apply_augment(
                target, data,
                opt.augs, opt.prob, opt.alpha,
                opt.aux_alpha, opt.aux_alpha, opt.mix_p
            )
            #        Train generator         #
            generator.zero_grad()

            # Generate real and fake inputs
            HR = Variable(HR.to(device))
            SR = generator(Variable(LR).to(device))

            if aug == "cutout":
                SR, HR = SR * mask, HR * mask

            generator_pixel_loss = criterion_pixel(SR, HR).to(device)
            mean_generator_pixel_loss += generator_pixel_loss.item()  # item()得到元素张量的值

            generator_pixel_loss.backward()
            optim_generator_pre.step()

            #       Status and display       #
            sys.stdout.write('\r[%d/%d][%d/%d] G pixel loss: %.4f' % (
                epoch, opt.nEpochs_pre, i + 1, len(train_dataloader), generator_pixel_loss.item()))

        sys.stdout.write('\r[%d/%d][%d/%d] G pixel loss: %.4f\n' % (
            epoch, opt.nEpochs_pre, i + 1, len(train_dataloader), mean_generator_pixel_loss / len(train_dataloader)))

        log_value('generator_pixel_loss', mean_generator_pixel_loss / len(train_dataloader), epoch)

        scheduler.step()
        # Save Model
        if epoch % I_SAVE == 0:
            SaveCheckpoint(epoch)

        if epoch % I_VALIDATION == 0:
            out_path = 'pretraining_results/SRF_' + str(opt.upSampling) + '/'
            if not os.path.exists(out_path):
                os.makedirs(out_path)

            with torch.no_grad():
                valing_results = {'psnr': 0, 'batch_sizes': 0}
                val_images = []
                for target in val_dataloader:
                    HR = target
                    batch_size = target.size(0)
                    valing_results['batch_sizes'] = valing_results['batch_sizes'] + batch_size

                    LR = DownSample2DMatlab(HR, 1/opt.upSampling)
                    LR = torch.clamp(torch.round(LR * 255) / 255.0, 0, 1)  # 下采样后的图

                    # 将图片放大upscale倍，因为网络第一层已修改
                    LR = F.interpolate(LR, scale_factor=opt.upSampling, mode="nearest").to(device)  # 近似于hr_restore
                    HR = HR.to(device)
                    SR = generator(LR)
                    SR = torch.clamp(SR, 0, 1)

                    SR = SR.squeeze(0)
                    HR = HR.squeeze(0)

                    batch_psnr = calculate_psnr(SR, HR)
                    valing_results['psnr'] = valing_results['psnr'] + batch_psnr
                    valing_results['psnr'] = valing_results['psnr'] / valing_results['batch_sizes']

                    sys.stdout.write('\r[converting LR images to SR images] PSNR: %.4f dB' % (
                        valing_results['psnr']))

                sys.stdout.write('\r[converting LR images to SR images] PSNR: %.4f dB\n' % (
                    valing_results['psnr']))

                out_path = 'statistics/'
                if not os.path.exists(out_path):
                    os.makedirs(out_path)
                saved_results['epoch'].append(epoch)
                saved_results['psnr'].append(valing_results['psnr'])
                data_frame = pd.DataFrame(saved_results)
                data_frame.to_csv(out_path + 'srf_' + str(opt.upSampling) + '_pretrain_test_results.csv', index=0)

            if epoch % I_SAVE == 0:
                if valing_results['psnr'] > BEST_PSNR:
                    BEST_PSNR = valing_results['psnr']
                    SaveCheckpoint(epoch, best=True)

    print('Generator pretrian over!')
