import argparse
import os
import numpy as np
import itertools
import time
import random

import options
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

def train():
    random.seed(1234)
    opt = options.BaseOptions()
    opt = opt.initialize().parse_args

    cuda = torch.cuda.is_available()
    pass

if __name__ == '__main__':

    train()