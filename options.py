import argparse
import os

class BaseOptions():
    def __init__(self):
        self.initialized = False
    def initialize(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
        parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
        parser.add_argument("--dataset_name", type=str, default="edges2shoes", help="name of the dataset")
        parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
        parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
        parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
        parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
        parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
        parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
        parser.add_argument("--img_height", type=int, default=128, help="size of image height")
        parser.add_argument("--img_width", type=int, default=128, help="size of image width")
        parser.add_argument("--channels", type=int, default=3, help="number of image channels")
        parser.add_argument("--sample_interval", type=int, default=400, help="interval saving generator samples")
        parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between saving model checkpoints")
        parser.add_argument("--n_downsample", type=int, default=2, help="number downsampling layers in encoder")
        parser.add_argument("--n_residual", type=int, default=3, help="number of residual blocks in encoder / decoder")
        parser.add_argument("--dim", type=int, default=64, help="number of filters in first encoder layer")
        parser.add_argument("--style_dim", type=int, default=8, help="dimensionality of the style code")
        opt = parser.parse_args()
        print(opt)

        self.initialized = True

        return  parser