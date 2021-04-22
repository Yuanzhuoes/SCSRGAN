import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--train_dataroot', type=str, default='./data/DIV2K_train_HR', help='path to train dataset')
    parser.add_argument('--val_dataroot', type=str, default='./data/DIV2K_valid_HR', help='path to val_dataset')
    parser.add_argument('--crop_size', default=128, type=int, help='training images crop size')
    parser.add_argument('--upSampling', type=int, default=4, help='low to high resolution scaling factor')
    parser.add_argument('--batchSize', type=int, default=1, help='input batch size')

    # training setups
    parser.add_argument("--residual_blocks", type=int, default=23, help="number of residual blocks in the generator")
    parser.add_argument('--nEpochs', type=int, default=200, help='number of epochs to train for')
    parser.add_argument('--nEpochs_pre', type=int, default=12, help='number of epochs to pre_train for')
    parser.add_argument('--generatorLR_pre', type=float, default=0.0002, help='learning rate for generator')
    parser.add_argument('--generatorLR', type=float, default=1e-4, help='learning rate for generator')
    parser.add_argument('--discriminatorLR', type=float, default=1e-4, help='learning rate for discriminator')
    parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--lambda_adv", type=float, default=1e-3, help="adversarial loss weight")
    parser.add_argument("--lambda_pixel", type=float, default=1.0, help="pixel-wise loss weight")

    # augmentations
    parser.add_argument("--augs", nargs="*", default=["flip", "blend", "rgb", "mixup", "cutout", "cutmix", "cutmixup",
                                                      "cutblur"], help='choices of MoA')
    parser.add_argument("--prob", nargs="*", default=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], help='probs of each DA occurs')
    parser.add_argument("--mix_p", nargs="*")
    parser.add_argument("--alpha", nargs="*", default=[1.0, 0.6, 1.0, 1.2, 0.001, 0.7, 0.7, 0.7])
    parser.add_argument("--aux_prob", type=float, default=1.0)
    parser.add_argument("--aux_alpha", type=float, default=1.2)
    # pretrain augmentations
    #parser.add_argument("--augs", nargs="*", default=["flip"], help='choices of MoA')
    #parser.add_argument("--prob", nargs="*", default=[1.0], help='probs of each DA occurs')
    #parser.add_argument("--mix_p", nargs="*")
    #parser.add_argument("--alpha", nargs="*", default=[1.0])
    #parser.add_argument("--aux_prob", type=float, default=1.0)
    #parser.add_argument("--aux_alpha", type=float, default=1.2)

    # path
    parser.add_argument('--out', type=str, default='checkpoints', help='folder to output model checkpoints')

    return parser.parse_args()
