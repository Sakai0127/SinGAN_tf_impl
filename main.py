import argparse

from train import *
#from SinGAN.train import *
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode')
    parser.add_argument('input_image')
    parser.add_argument('--save_dir', default='models')

    #Network hyper parameter
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--k_size', type=int, default=3)
    parser.add_argument('--n_layers', type=int, default=5)

    #train settings
    parser.add_argument('--n_iter', type=int, default=2000)
    parser.add_argument('--lr_g', type=float, default=0.0005)
    parser.add_argument('--lr_d', type=float, default=0.0005)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--g_times', type=int, default=3)
    parser.add_argument('--d_times', type=int, default=3)
    parser.add_argument('--gp_weight', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=10.0)

    #data manipulation
    parser.add_argument('--scale_factor', type=float, default=0.75)
    parser.add_argument('--noise_weight', type=float, default=0.1)
    parser.add_argument('--min_size', type=int, default=18)

    #SR params
    parser.add_argument('--sr_factor', type=int, default=4)

    args = parser.parse_args()
    if args.mode == 'SR':
        train_SR(args)