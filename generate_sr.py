import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import sys
from os import mkdir
from os.path import isdir

import numpy as np
import pandas as pd
from tensorboard_logger import configure, log_value

import torch
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.transforms import ToPILImage

from option import parse_args
from models import Generator_pre, My_UnetD, Discriminator, Get_gradient_nopadding, Get_gradient, FeatureExtractor
from utils import TrainDatasetFromFolder, ValDatasetFromFolder, DownSample2DMatlab, Huber
from augments import apply_augment, rand_bbox
import metrics.LPIPS.models.dist_model as dm
from metrics.psnr_ssim import calculate_psnr, calculate_ssim

from torch.optim.lr_scheduler import MultiStepLR

L_ADV = 5e-3  # Scaling params for the Adv loss
L_FM = 1  # Scaling params for the feature matching loss
L_LPIPS = 1e-3
L_SRGM = 1e-2
L_BranchGM = 5e-1
L_GM_ADV = 5e-3
L_PIX = 1e-2
START_ITER = 0
EXP_NAME = "LPIPS"
I_SAVE = 2000
I_VALIDATION = 2000
BEST_LPIPS = 0.1318

if __name__ == '__main__':
    opt = parse_args()

    print(opt)
    try:
        os.makedirs(opt.out)
    except OSError:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = TrainDatasetFromFolder('data/DIV2K_train/HR', crop_size=opt.crop_size,
                                           upscale_factor=opt.upSampling)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=opt.batchSize, shuffle=True,
                                  num_workers=16)
    val_dataset = ValDatasetFromFolder('data/Set14_valid_HR/Val_HR', upscale_factor=opt.upSampling)

    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=4)

    generator = Generator_pre(3, filters=64, num_res_blocks=opt.residual_blocks, up_scale=opt.upSampling).to(device)
    checkpoint = torch.load('./checkpoints/PSNR/model_G_i019600_best.pth')  # choice the best model
    generator.load_state_dict(checkpoint['model_G_pre'], strict=False)
    print('Load Generator pre successfully!')

    # U-net Discriminator
    discriminator = My_UnetD().to(device)
    gradient_discriminator = Discriminator(3, 64).to(device)

    # Feature Extractor
    feature_extractor = FeatureExtractor().to(device)

    # Gradient Extractor
    gm_extractor_npadding = Get_gradient_nopadding().to(device)  # need device?
    gm_extractor_padding = Get_gradient().to(device)

    MSELoss = torch.nn.MSELoss().to(device)
    L1Loss = torch.nn.L1Loss().to(device)
    BCEWithLogitsLoss = torch.nn.BCEWithLogitsLoss().to(device)
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

    model_LPIPS = dm.DistModel()
    model_LPIPS.initialize(model='net-lin', net='alex', use_gpu=True)

    configure(
        'logs/' + opt.train_dataroot + '-' + str(64) + '-' + str(opt.generatorLR) + '-' + str(opt.discriminatorLR),
        flush_secs=5)

    # ESRGAN training
    optim_generator = optim.Adam(generator.parameters(), lr=opt.generatorLR, betas=(opt.b1, opt.b2))
    optim_discriminator = optim.Adam(discriminator.parameters(), lr=opt.discriminatorLR, betas=(opt.b1, opt.b2))
    optim_gradient_discriminator = optim.Adam(gradient_discriminator.parameters(), lr=opt.discriminatorLR,
                                              betas=(opt.b1, opt.b2))

    scheduler_generator = MultiStepLR(optim_generator, milestones=[1000, 2000, 4000, 6000], gamma=0.5)
    scheduler_discriminator = MultiStepLR(optim_discriminator, milestones=[1000, 2000, 4000, 6000], gamma=0.5)
    scheduler_gradient_discriminator = MultiStepLR(optim_discriminator, milestones=[1000, 2000, 4000, 6000], gamma=0.5)

    def SaveCheckpoint(i, best=False):
        str_best = ''
        if best:
            str_best = '_best'

        if not isdir('./checkpoints/{}'.format(EXP_NAME)):
            mkdir('./checkpoints/{}'.format(EXP_NAME))

        generator_state = {'model_G': generator.state_dict(),
                           'model_G_optimizer': optim_generator.state_dict(),
                           'model_G_scheduler': scheduler_generator.state_dict(),
                           'epoch': epoch}
        discriminator_state = {'model_D': discriminator.state_dict(),
                               'model_D_optimizer': optim_discriminator.state_dict(),
                               'model_D_scheduler': scheduler_discriminator.state_dict(),
                               'epoch': epoch}
        gradient_discriminator_state = {'model_D_GM': gradient_discriminator.state_dict(),
                                        'model_D_GM_optimizer': optim_gradient_discriminator.state_dict(),
                                        'model_D_GM_scheduler': scheduler_gradient_discriminator.state_dict(),
                                        'epoch': epoch}

        torch.save(generator_state, './checkpoints/{}/model_G_i{:06d}{}.pth'.format(EXP_NAME, i, str_best))
        torch.save(discriminator_state, './checkpoints/{}/model_D_i{:06d}{}.pth'.format(EXP_NAME, i, str_best))
        torch.save(gradient_discriminator_state, './checkpoints/{}/model_D_GM_i{:06d}{}.pth'.format(EXP_NAME, i, str_best))

    saved_results = {'epoch': [], 'psnr': [], 'ssim': [], 'lpips': []}  # 从头开始会被抹掉
    print('ESRGAN training')

    if START_ITER > 0:
        checkpoint = torch.load('./checkpoints/{}/model_G_i{:06d}.pth'.format(EXP_NAME, START_ITER))
        generator.load_state_dict(checkpoint['model_G'])
        optim_generator.load_state_dict(checkpoint['model_G_optimizer'])
        scheduler_generator.load_state_dict(checkpoint['model_G_scheduler'])
        start_epoch = checkpoint['epoch']
        print('Load Generator epoch {} successfully!'.format(start_epoch))

        checkpoint = torch.load('./checkpoints/{}/model_D_i{:06d}.pth'.format(EXP_NAME, START_ITER))
        discriminator.load_state_dict(checkpoint['model_D'])
        optim_discriminator.load_state_dict(checkpoint['model_D_optimizer'])
        scheduler_discriminator.load_state_dict(checkpoint['model_D_scheduler'])
        start_epoch = checkpoint['epoch']
        print('Load Discriminator epoch {} successfully!'.format(start_epoch))

        checkpoint = torch.load('./checkpoints/{}/model_D_GM_i{:06d}.pth'.format(EXP_NAME, START_ITER))
        gradient_discriminator.load_state_dict(checkpoint['model_D_GM'])
        optim_gradient_discriminator.load_state_dict(checkpoint['model_D_GM_optimizer'])
        scheduler_gradient_discriminator.load_state_dict(checkpoint['model_D_GM_scheduler'])
        start_epoch = checkpoint['epoch']
        print('Load Gradient_Discriminator epoch {} successfully!'.format(start_epoch))
    else:
        start_epoch = 0
        print('start training generator from scratch!')

    for epoch in range(start_epoch + 1, opt.nEpochs + 1):

        mean_generator_LPIPS_loss = 0.0
        mean_generator_adversarial_loss = 0.0
        mean_generator_FM_loss = 0.0
        mean_generator_sr_gradient_loss = 0.0
        mean_generator_branch_gradient_loss = 0.0
        mean_generator_gm_adversarial_loss = 0.0
        mean_generator_pixel_loss = 0.0
        mean_generator_total_loss = 0.0
        mean_discriminator_loss = 0.0
        mean_GM_discriminator_loss = 0.0

        generator.train()
        discriminator.train()
        gradient_discriminator.train()

        for i, target in enumerate(train_dataloader):
            data = DownSample2DMatlab(target, 1 / (float(opt.upSampling)))
            data = torch.clamp(data, 0, 1)
            #  resize for augment
            if data.size() != target.size():
                scale = target.size(2) // data.size(2)
                data = F.interpolate(data, scale_factor=scale, mode="nearest")

            #  MoA
            HR, LR, mask, aug = apply_augment(
                target, data,
                opt.augs, opt.prob, opt.alpha,
                opt.aux_alpha, opt.aux_alpha, opt.mix_p
            )

            #             Train generator          #
            for p in discriminator.parameters():
                p.requires_grad = False
            for q in gradient_discriminator.parameters():
                q.requires_grad = False
            generator.zero_grad()

            # real img and fake_img
            HR = Variable(HR.to(device))
            SR = generator(Variable(LR).to(device))

            if aug == "cutout":
                SR, HR = SR * mask, HR * mask

            # Gradient map
            HR_GM = gm_extractor_padding(HR)  # HR has padding
            HR_GM_N = gm_extractor_npadding(HR)  # HR padding
            SR_GM = gm_extractor_padding(SR)  # SR has padding

            # LPIPS loss
            generator_LPIPS_loss, _ = model_LPIPS.forward_pair(HR * 2 - 1, SR * 2 - 1)
            generator_LPIPS_loss = torch.mean(generator_LPIPS_loss)

            # FM and GAN losses
            e_SR, d_SR, e_SRs, d_SRs = discriminator(SR)
            _, _, e_HRs, d_HRs = discriminator(HR.detach())

            # Gradient GAN losses
            pred_SR_GM = gradient_discriminator(SR_GM)
            pred_HR_GM = gradient_discriminator(HR_GM.detach())

            target_real = torch.empty_like(pred_HR_GM).fill_(1)
            target_fake = torch.empty_like(pred_SR_GM).fill_(0)

            loss_g_real = BCEWithLogitsLoss(pred_HR_GM - torch.mean(pred_SR_GM), target_fake)
            loss_g_fake = BCEWithLogitsLoss(pred_SR_GM - torch.mean(pred_HR_GM), target_real)
            generator_gm_adversarial_loss = (loss_g_real + loss_g_fake) / 2  # 没显示

            # Feature matching loss generator_FM_loss
            generator_FM_loss = []
            for f in range(6):
                generator_FM_loss += [L1Loss(e_SRs[f], e_HRs[f]).to(device)]
                generator_FM_loss += [L1Loss(d_SRs[f], d_HRs[f]).to(device)]
            generator_FM_loss = torch.mean(torch.stack(generator_FM_loss))

            # Gradient loss
            generator_sr_gradient_loss = MSELoss(SR_GM, HR_GM)

            # Pixel loss
            generator_pixel_loss = L1Loss(SR, HR)

            # Generator adversarial loss
            generator_adversarial_loss = []
            generator_adversarial_loss += [torch.nn.ReLU()(1.0 - e_SR).mean()]  # 目的是让SR的预测值向真趋近
            generator_adversarial_loss += [torch.nn.ReLU()(1.0 - d_SR).mean()]

            generator_adversarial_loss = torch.mean(torch.stack(generator_adversarial_loss))

            # Generator total loss
            generator_total_loss = L_LPIPS * generator_LPIPS_loss + L_ADV * generator_adversarial_loss + \
                L_FM * generator_FM_loss + L_SRGM * generator_sr_gradient_loss + \
                L_GM_ADV * generator_gm_adversarial_loss + L_PIX * generator_pixel_loss

            generator_total_loss.backward()
            optim_generator.step()

            #        Train discriminator        #
            for p in discriminator.parameters():
                p.requires_grad = True
            discriminator.zero_grad()

            e_SR, d_SR, _, _ = discriminator(SR.detach())
            e_HR, d_HR, _, _ = discriminator(HR)

            # D Loss, for encoder end and decoder end
            Encode_SR_loss = torch.nn.ReLU()(1.0 + e_SR).mean()
            Encode_HR_loss = torch.nn.ReLU()(1.0 - e_HR).mean()

            Decode_SR_loss = torch.nn.ReLU()(1.0 + d_SR).mean()
            Decode_HR_loss = torch.nn.ReLU()(1.0 - d_HR).mean()

            discriminator_loss = Encode_HR_loss + Decode_HR_loss

            SR_Cutmix = SR.clone()

            if torch.rand(1) <= 0.5:
                r_mix = torch.rand(1)  # real/fake ratio [0, 1)

                bbx1, bby1, bbx2, bby2 = rand_bbox(SR_Cutmix.size(), r_mix)
                SR_Cutmix[:, :, bbx1:bbx2, bby1:bby2] = HR[:, :, bbx1:bbx2, bby1:bby2]  # 开区间，不可能全覆盖
                # adjust lambda to exactly match pixel ratio
                r_mix = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (SR_Cutmix.size()[-1] * SR_Cutmix.size()[-2]))

                e_mix, d_mix, _, _ = discriminator(SR_Cutmix.detach())

                Encode_SR_loss = torch.nn.ReLU()(1.0 + e_mix).mean()
                Decode_SR_loss = torch.nn.ReLU()(1.0 + d_mix).mean()

                d_SR[:, :, bbx1:bbx2, bby1:bby2] = d_HR[:, :, bbx1:bbx2, bby1:bby2]

                loss_D_Cons = F.mse_loss(d_mix, d_SR)
                discriminator_loss += loss_D_Cons

            discriminator_loss += Encode_SR_loss + Decode_SR_loss

            discriminator_loss.backward()
            optim_discriminator.step()

            # Train gradient discriminator #
            for p in gradient_discriminator.parameters():
                p.requires_grad = True
            gradient_discriminator.zero_grad()

            pred_HR_GM = gradient_discriminator(HR_GM)
            pred_SR_GM = gradient_discriminator(SR_GM.detach())
            target_real = torch.empty_like(pred_HR_GM).fill_(1)
            target_fake = torch.empty_like(pred_SR_GM).fill_(0)

            real_GM_D_loss = BCEWithLogitsLoss(pred_HR_GM - torch.mean(pred_SR_GM), target_real)
            fake_GM_D_loss = BCEWithLogitsLoss(pred_SR_GM - torch.mean(pred_HR_GM), target_fake)

            GM_discriminator_loss = (real_GM_D_loss + fake_GM_D_loss) / 2

            GM_discriminator_loss.backward()

            optim_gradient_discriminator.step()

            mean_discriminator_loss += discriminator_loss.item()
            mean_GM_discriminator_loss += GM_discriminator_loss.item()
            mean_generator_LPIPS_loss += generator_LPIPS_loss.item()
            mean_generator_adversarial_loss += generator_adversarial_loss.item()
            mean_generator_FM_loss += generator_FM_loss.item()
            mean_generator_sr_gradient_loss += generator_sr_gradient_loss.item()
            mean_generator_gm_adversarial_loss += generator_gm_adversarial_loss.item()
            mean_generator_pixel_loss += generator_pixel_loss.item()
            mean_generator_total_loss += generator_total_loss.item()

            sys.stdout.write(
                '\r[%d/%d][%d/%d] Discriminator_Loss: %.4f GM_Discriminator_Loss: %.4f '
                'Generator_Loss (LPIPS/Advers/FM/SRGM/AdversGM/Pixel/Total): '
                '%.4f/%.4f/%.4f/%.4f/%.4f/%.4f/%.4f' % (
                    epoch, opt.nEpochs, i + 1, len(train_dataloader),
                    discriminator_loss.item(), GM_discriminator_loss.item(), generator_LPIPS_loss.item(),
                    generator_adversarial_loss.item(), generator_FM_loss.item(), generator_sr_gradient_loss.item(),
                    generator_gm_adversarial_loss.item(), generator_pixel_loss, generator_total_loss.item()))

        sys.stdout.write(
            '\r[%d/%d][%d/%d] Discriminator_Loss: %.4f GM_Discriminator_Loss: %.4f '
            'Generator_Loss (LPIPS/Advers/FM/SRGM/AdversGM/Pixel/Total): '
            '%.4f/%.4f/%.4f/%.4f/%.4f/%.4f/%.4f\n' % (
                epoch, opt.nEpochs, i + 1, len(train_dataloader),
                mean_discriminator_loss / len(train_dataloader), mean_GM_discriminator_loss / len(train_dataloader),
                mean_generator_LPIPS_loss / len(train_dataloader),
                mean_generator_adversarial_loss / len(train_dataloader),
                mean_generator_FM_loss / len(train_dataloader), mean_generator_sr_gradient_loss / len(train_dataloader),
                mean_generator_gm_adversarial_loss / len(train_dataloader),
                mean_generator_total_loss / len(train_dataloader),
                mean_generator_pixel_loss / len(train_dataloader)))

        log_value('discriminator_loss', mean_discriminator_loss / len(train_dataloader), epoch)
        log_value('GM_discriminator_loss', mean_GM_discriminator_loss / len(train_dataloader), epoch)
        log_value('generator_adversarial_loss', mean_generator_adversarial_loss / len(train_dataloader), epoch)
        log_value('generator_FM_loss', mean_generator_FM_loss / len(train_dataloader), epoch)
        log_value('generator_LPIPS_loss', mean_generator_LPIPS_loss / len(train_dataloader), epoch)
        log_value('generator_sr_gradient_loss', mean_generator_sr_gradient_loss / len(train_dataloader), epoch)

        log_value('generator_gm_adversarial_loss', mean_generator_gm_adversarial_loss / len(train_dataloader), epoch)
        log_value('generator_pixel_loss', mean_generator_pixel_loss / len(train_dataloader), epoch)
        log_value('generator_total_loss', mean_generator_total_loss / len(train_dataloader), epoch)

        scheduler_generator.step()
        scheduler_discriminator.step()
        scheduler_gradient_discriminator.step()

        if epoch % I_SAVE == 0:
            SaveCheckpoint(epoch)

        if epoch % I_VALIDATION == 0:
            # 验证集
            out_path = 'training_results/SRF_' + str(4) + '/'
            if not os.path.exists(out_path):
                os.makedirs(out_path)

            with torch.no_grad():
                valing_results = {'psnr': 0, 'ssim': 0, 'lpips': 0, 'batch_sizes': 0}
                val_images = []
                for image_name, target in val_dataloader:
                    image_name = image_name[0]
                    HR = target
                    batch_size = target.size(0)
                    valing_results['batch_sizes'] = valing_results['batch_sizes'] + batch_size

                    LR = DownSample2DMatlab(HR, 1 / opt.upSampling)
                    LR = torch.clamp(torch.round(LR * 255) / 255.0, 0, 1)  # 下采样后的图
                    # LR = torch.clamp(LR, 0, 1)
                    # 将图片放大upscale倍，因为网络第一层已修改
                    LR = F.interpolate(LR, scale_factor=opt.upSampling, mode="nearest").to(device)  # 近似于hr_restore
                    HR = HR.to(device)  # Tensor[0, 1]

                    SR, _, _ = generator(LR)
                    SR = torch.clamp(SR, 0, 1).to(device)

                    SR_lpips = ToPILImage()(SR.squeeze(0))
                    HR_lpips = ToPILImage()(HR.squeeze(0))

                    SR_lpips = np.array(SR_lpips)
                    HR_lpips = np.array(HR_lpips)
                    SR_lpips = torch.Tensor((SR_lpips / 127.5 - 1)[:, :, :, np.newaxis].transpose((3, 2, 0, 1)))
                    HR_lpips = torch.Tensor((HR_lpips / 127.5 - 1)[:, :, :, np.newaxis].transpose((3, 2, 0, 1)))

                    batch_lpips = model_LPIPS.forward(SR_lpips, HR_lpips)
                    valing_results['lpips'] = valing_results['lpips'] + batch_lpips

                    # batch size dimention deduction must be here, for calaculating lpips requires 4 dimentions.
                    SR = SR.data.float().to(device).clamp_(0, 1)
                    HR = HR.data.float().to(device).clamp_(0, 1)
                    SR = SR.squeeze(0)
                    HR = HR.squeeze(0)

                    batch_ssim = calculate_ssim(SR, HR, test_y_channel=True)
                    valing_results['ssim'] = valing_results['ssim'] + batch_ssim

                    batch_psnr = calculate_psnr(HR, SR, test_y_channel=True)
                    valing_results['psnr'] = valing_results['psnr'] + batch_psnr

                    test_images = torch.stack([HR.data.to(device), SR.data.to(device)])
                    image = utils.make_grid(test_images, nrow=2, padding=1)
                    utils.save_image(image, out_path + image_name, padding=1)

                    sys.stdout.write('\r[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f LPIPS: %.4f' % (
                        batch_psnr, batch_ssim, batch_lpips))

                sys.stdout.write('\r[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f LPIPS: %.4f\n' % (
                    valing_results['psnr'] / valing_results['batch_sizes'],
                    valing_results['ssim'] / valing_results['batch_sizes'],
                    valing_results['lpips'] / valing_results['batch_sizes']))

                out_path = 'statistics/'
                if not os.path.exists(out_path):
                    os.makedirs(out_path)
                saved_results['epoch'].append(epoch)
                saved_results['psnr'].append(valing_results['psnr'] / valing_results['batch_sizes'])
                saved_results['ssim'].append(valing_results['ssim'] / valing_results['batch_sizes'])
                saved_results['lpips'].append(valing_results['lpips'] / valing_results['batch_sizes'])
                data_frame = pd.DataFrame(saved_results)
                data_frame.to_csv(out_path + 'srf_' + str(opt.upSampling) + '_train_results.csv', index=0)

            if epoch % I_SAVE == 0:
                if (valing_results['lpips'] / valing_results['batch_sizes']) < BEST_LPIPS:
                    BEST_LPIPS = valing_results['lpips'] / valing_results['batch_sizes']
                    SaveCheckpoint(epoch, best=True)

    print("SRGAN Train Over.")
