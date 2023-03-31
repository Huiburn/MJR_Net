import numpy as np
import os
from misc import fft2c
from matplotlib import pyplot as plt
import torch


def read_data(file_name, mode, device=None):
    """
    Here we load data and change the shape of matrix to fit the Python code

    origin matlab shape
    -------------------
    b0_img@(Nx,Ny,Ncoil,1,Ncase)
    csm_data@(Nx,Ny,Ncoil,Ncase)
    full_img@(Nx,Ny,Ncoil,Nshot,Ncase)

    data transpose
    --------------------
    change the shape of matrix to fit the Python code
    b0_img@(Ncase,1,Ncoil,Nx,Ny)
    csm_data@(Ncase,Ncoil,Nx,Ny)
    full_img@(Ncase,Nshot,Ncoil,Nx,Ny)

    get the measure data
    ---------------------
    measure_data@(Ncase,Nshot,Ncoil,Nx,Ny)
    """

    # load origin data
    data = np.load(file_name + mode + '_data.npy')
    csm = np.load(file_name + mode + '_csm.npy')
    gt = np.load(file_name + mode + '_gt.npy')

    mask = np.zeros(data[0, :, :, :, :].shape)
    mask[data[0, :, :, :, :].nonzero()] = 1
    # return torch.tensor(data, device=device), torch.tensor(csm, device=device), torch.tensor(gt, device=device)


    b0 = np.load(file_name + mode + '_b0.npy')

    if device == None:
        return data, csm, gt, b0, mask
    else:
        return torch.from_numpy(data).to(device), torch.from_numpy(csm).to(device), torch.from_numpy(gt).to(
            device),torch.from_numpy(b0).to(device), torch.from_numpy(mask).to(device)
