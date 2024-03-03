import os
import random

import numpy as np
import torch
import matplotlib as mpl

__all__ = ['init_weights', 'set_seed', 'set_mpl']

def init_weights(module):
    """ Set all weights to a small, uniform range. Set all biases to zero. """
    def _init_weights(m):
        try:
            # torch.nn.init.xavier_uniform_(m.weight)
            # torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            torch.nn.init.uniform_(m.weight, -0.01, 0.01)
        except AttributeError:
            pass
        try:
            torch.nn.init.zeros_(m.bias) #torch.nn.init.uniform_(m.bias, -0.01, 0.01)
        except AttributeError:
            pass

    module.apply(_init_weights)


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # torch.use_deterministic_algorithms(True)

def set_mpl():
    # change defaults to be less ugly for matplotlib
    mpl.rc('xtick', labelsize=14, color="#222222")
    mpl.rc('ytick', labelsize=14, color="#222222")
    mpl.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial']})
    mpl.rc('font', size=16)
    mpl.rc('xtick.major', size=6, width=1)
    mpl.rc('xtick.minor', size=3, width=1)
    mpl.rc('ytick.major', size=6, width=1)
    mpl.rc('ytick.minor', size=3, width=1)
    mpl.rc('axes', linewidth=1, edgecolor="#222222", labelcolor="#222222")
    mpl.rc('text', usetex=False, color="#222222")

