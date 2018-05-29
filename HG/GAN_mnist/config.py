import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default='mnist', help='dataset')
parser.add_argument('--image_size', type=int, default=28, help='size of the input image')
parser.add_argument('--nchannel', type=int, default=1, help='number of input channels')
parser.add_argument('--nz', type=int, default=100, help='dimension of latent vertor Z')

parser.add_argument('--learning_rate_D', type=float, default=1e-4, help='learning rate of Discriminator')
parser.add_argument('--learning_rate_G', type=float, default=1e-4, help='learning rate of Generator')

parser.add_argument('--test_size', type=int, default=10, help='num of test images')
parser.add_argument('--print_step', type=int, default=1, help='print steps')
parser.add_argument('--niter', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--outf', default=None, help='folder to output images and model checkpoints')

def get_config():
    return parser.parse_args()
