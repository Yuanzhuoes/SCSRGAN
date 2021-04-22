import os
import sys
from math import log10

import pandas as pd

import pytorch_ssim
from tensorboard_logger import configure, log_value

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.utils as utils
import torch.nn.functional as F

from option import parse_args
from models import Generator, Discriminator, FeatureExtractor
from utils import display_transform, TrainDatasetFromFolder, ValDatasetFromFolder, DownSample2DMatlab
from augments import apply_augment

from torch.optim.lr_scheduler import MultiStepLR

if __name__ == '__main__':
    opt = parse_args()

    print(opt)
    try:
        os.makedirs(opt.out)
    except OSError:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = TrainDatasetFromFolder('data/DIV2K_train_HR/Train_HR', crop_size=opt.crop_size,
                                           upscale_factor=opt.upSampling)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=opt.batchSize, shuffle=True,
                                  num_workers=4)
    val_dataset = ValDatasetFromFolder('data/DIV2K_valid_HR/Val_HR', upscale_factor=opt.upSampling)

    # 使用loader，从训练集中，一次性处理一个batch的文件 （批量加载器）
    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=4)

    generator = Generator(3, filters=64, num_res_blocks=opt.residual_blocks, up_scale=opt.upSampling).to(device)
    # load pretrain model
    checkpoint = torch.load(opt.generator_pretrainWeights)
    generator.load_state_dict(checkpoint['generator_model_pre'])
    print('Load Generator pre successfully!')
    discriminator = Discriminator(in_channels=3, out_filters=64).to(device)
    feature_extractor = FeatureExtractor().to(device)

    feature_extractor.eval()

    # 内容损失和对抗损失
    criterion_pixel = torch.nn.L1Loss().to(device)  # 像素差的绝对值
    content_criterion = torch.nn.L1Loss().to(device)
    adversarial_criterion = torch.nn.BCEWithLogitsLoss().to(device)  # 交叉熵

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

    # tensorboard --logdir=logs
    configure(
        'logs/' + opt.train_dataroot + '-' + str(64) + '-' + str(opt.generatorLR) + '-' + str(opt.discriminatorLR),
        flush_secs=5)

    # ESRGAN training
    optim_generator = optim.Adam(generator.parameters(), lr=opt.generatorLR, betas=(opt.b1, opt.b2))
    optim_discriminator = optim.Adam(discriminator.parameters(), lr=opt.discriminatorLR, betas=(opt.b1, opt.b2))
    scheduler_generator = MultiStepLR(optim_generator, milestones=[3, 7], gamma=0.5)
    scheduler_discriminator = MultiStepLR(optim_discriminator, milestones=[3, 7], gamma=0.5)

    print('ESRGAN training')
    saved_results = {'epoch': [], 'psnr': [], 'ssim': []}  # 从头开始会被抹掉
    if opt.generatorWeights != '' and opt.trian_from_scratch:
        checkpoint = torch.load(opt.generatorWeights)
        generator.load_state_dict(checkpoint['generator_model'])
        optim_generator.load_state_dict(checkpoint['generator_optimizer'])
        scheduler_generator.load_state_dict(checkpoint['scheduler_generator'])
        start_epoch = checkpoint['epoch']
        print('Load Generator epoch {} successfully!'.format(start_epoch))
    else:
        start_epoch = 0
        print('start training generator from scratch!')

    # load discriminator generator model
    if opt.discriminatorWeights != '' and opt.trian_from_scratch:
        checkpoint = torch.load(opt.discriminatorWeights)
        discriminator.load_state_dict(checkpoint['discriminator_model'])
        optim_discriminator.load_state_dict(checkpoint['discriminator_optimizer'])
        scheduler_discriminator.load_state_dict(checkpoint['scheduler_discriminator'])
        start_epoch = checkpoint['epoch']
        print('Load Discriminator epoch {} successfully!'.format(start_epoch))
    else:
        start_epoch = 0
        print('start training discriminator from scratch!')

    for epoch in range(start_epoch + 1, opt.nEpochs + 1):

        mean_generator_perceptual_loss = 0.0
        mean_generator_adversarial_loss = 0.0
        mean_generator_content_loss = 0.0
        mean_generator_total_loss = 0.0
        mean_discriminator_loss = 0.0

        #  discriminator.train() D有batchnormal,启用train吗？
        for i, target in enumerate(train_dataloader):

            data = DownSample2DMatlab(target, 1 / (float(opt.upSampling)))
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

            #             Train generator          #
            for p in discriminator.parameters():
                p.requires_grad = False
            generator.zero_grad()

            # real img and fake_img
            HR = Variable(HR.to(device))
            SR = generator(Variable(LR).to(device))

            if aug == "cutout":
                SR, HR = SR * mask, HR * mask

            # Content loss
            generator_content_loss = content_criterion(SR, HR).to(device)

            # Perceptual loss
            fake_features = feature_extractor(SR)
            real_features = feature_extractor(HR)
            generator_perceptual_loss = content_criterion(fake_features, real_features)

            # Extract validity predictions from discriminator
            pred_real = discriminator(HR).detach()  # 判别为真的概率
            pred_fake = discriminator(SR)  # 判别为假的概率

            # target
            target_real = pred_real.new_ones(pred_real.size()) * 1
            target_fake = pred_fake.new_ones(pred_fake.size()) * 0

            # Generator adversarial loss
            # 论文中提到对抗损失受益于生成数据和真实数据的梯度，而SRGAN只受益于生成数据
            loss_g_real = adversarial_criterion(pred_real - torch.mean(pred_fake), target_fake)
            loss_g_fake = adversarial_criterion(pred_fake - torch.mean(pred_real), target_real)
            generator_adversarial_loss = (loss_g_real + loss_g_fake) / 2

            # Total loss
            generator_total_loss = generator_perceptual_loss + opt.lambda_adv * generator_adversarial_loss + \
                opt.lambda_pixel * generator_content_loss

            generator_total_loss.backward()
            optim_generator.step()

            #        Train discriminator        #
            for p in discriminator.parameters():
                p.requires_grad = True
            discriminator.zero_grad()

            pred_fake = discriminator(SR).detach()  # 判别为假的概率
            pred_real = discriminator(HR)  # 判别为真的概率

            # Adversarial loss for real and fake images (relativistic average GAN)
            # real
            loss_real = adversarial_criterion(pred_real - torch.mean(pred_fake), target_real) * 0.5
            loss_real.backward()

            # fake
            pred_fake = discriminator(SR.detach())
            loss_fake = adversarial_criterion(pred_fake - torch.mean(pred_real.detach()), target_fake) * 0.5
            loss_fake.backward()

            optim_discriminator.step()

            discriminator_loss = loss_real + loss_fake
            mean_discriminator_loss += discriminator_loss.item()
            mean_generator_perceptual_loss += generator_perceptual_loss.item()
            mean_generator_adversarial_loss += generator_adversarial_loss.item()
            mean_generator_content_loss += generator_content_loss.item()
            mean_generator_total_loss += generator_total_loss.item()

            sys.stdout.write(
                '\r[%d/%d][%d/%d] Discriminator_Loss: %.4f Generator_Loss (Perceptual/Advers/Content/Total): '
                '%.4f/%.4f/%.4f/%.4f' % (
                    epoch, opt.nEpochs, i + 1, len(train_dataloader),
                    discriminator_loss.item(), generator_perceptual_loss.item(), generator_adversarial_loss.item(),
                    generator_content_loss.item(),
                    generator_total_loss.item()))
        sys.stdout.write(
            '\r[%d/%d][%d/%d] Discriminator_Loss: %.4f Generator_Loss (Perceptual/Advers/Content/Total): '
            '%.4f/%.4f/%.4f/%.4f\n' % (
                epoch, opt.nEpochs, i + 1, len(train_dataloader),
                mean_discriminator_loss / len(train_dataloader), mean_generator_perceptual_loss / len(train_dataloader),
                mean_generator_adversarial_loss / len(train_dataloader), mean_generator_content_loss /
                len(train_dataloader), mean_generator_total_loss / len(train_dataloader)))

        log_value('generator_perceptual_loss', mean_generator_perceptual_loss / len(train_dataloader), epoch)
        log_value('generator_adversarial_loss', mean_generator_adversarial_loss / len(train_dataloader), epoch)
        log_value('generator_content_loss', mean_generator_content_loss / len(train_dataloader), epoch)
        log_value('generator_total_loss', mean_generator_total_loss / len(train_dataloader), epoch)
        log_value('discriminator_loss', mean_discriminator_loss / len(train_dataloader), epoch)

        scheduler_generator.step()
        scheduler_discriminator.step()

        # Do checkpointing 保存模型
        generator_state = {'generator_model': generator.state_dict(),
                           'generator_optimizer': optim_generator.state_dict(),
                           'scheduler_generator': scheduler_generator.state_dict(), 'epoch': epoch}
        discriminator_state = {'discriminator_model': discriminator.state_dict(), 'discriminator_optimizer':
                               optim_discriminator.state_dict(), 'scheduler_discriminator':
                               scheduler_discriminator.state_dict(), 'epoch': epoch}

        # save model
        torch.save(generator_state, opt.generatorWeights)
        torch.save(discriminator_state, opt.discriminatorWeights)

        if epoch % 5 == 0:
            # 验证集
            out_path = 'pretraining_results/SRF_' + str(opt.upSampling) + '/'
            if not os.path.exists(out_path):
                os.makedirs(out_path)

            with torch.no_grad():
                valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
                val_images = []
                for target in val_dataloader:
                    HR = target  # 原图
                    batch_size = target.size(0)
                    valing_results['batch_sizes'] = valing_results['batch_sizes'] + batch_size

                    LR = DownSample2DMatlab(HR, 1 / opt.upSampling)
                    LR = torch.clamp(torch.round(LR * 255) / 255.0, 0, 1)  # 下采样后的图

                    # 将图片放大upscale倍，因为网络第一层已修改
                    LR = F.interpolate(LR, scale_factor=opt.upSampling, mode="nearest").to(device)  # 近似于hr_restore
                    HR = HR.to(device)
                    SR = generator(LR)

                    SR = torch.clamp(SR, 0, 1)

                    batch_mse = ((SR - HR) ** 2).data.mean()
                    valing_results['mse'] = valing_results['mse'] + batch_mse * batch_size
                    batch_ssim = pytorch_ssim.ssim(SR, HR).item()
                    valing_results['ssims'] = valing_results['ssims'] + batch_ssim * batch_size
                    valing_results['psnr'] = 10 * log10(
                        (HR.max() ** 2) / (valing_results['mse'] / valing_results['batch_sizes']))
                    valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']

                    val_images.extend(
                        [display_transform()(LR.data.cpu().squeeze(0)), display_transform()(HR.data.cpu().squeeze(0)),
                         display_transform()(SR.data.cpu().squeeze(0))])

                sys.stdout.write('\r[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f\n' % (
                    valing_results['psnr'], valing_results['ssim']))
                val_images = torch.stack(val_images)  # 按顺序排列
                val_images = torch.chunk(val_images, val_images.size(0) // 3)  # 3张图为1个单元
                index = 1
                for image in val_images:
                    image = utils.make_grid(image, nrow=3, padding=1)  # 2行三列
                    utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
                    index = index + 1

                out_path = 'statistics/'
                if not os.path.exists(out_path):
                    os.makedirs(out_path)
                saved_results['epoch'].append(epoch)
                saved_results['psnr'].append(valing_results['psnr'])
                saved_results['ssim'].append(valing_results['ssim'])
                data_frame = pd.DataFrame(saved_results)
                data_frame.to_csv(out_path + 'srf_' + str(opt.upSampling) + '_pretrain_test_results.csv', index=0)

    print("SRGAN Train Over.")
