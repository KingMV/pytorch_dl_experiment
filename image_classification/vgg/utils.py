import h5py
import torch
import shutil
import numpy as np


def save_checkpoint(state, filename):
    torch.save(state, filename)
