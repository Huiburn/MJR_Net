import numpy as np
import torch
import os
from scipy import io
import misc as sf
import matplotlib.pyplot as plt
from read_data import read_data
from model import pair, pocsIce
from model import shot2shot
from model import My_dataset, Admm, MJR_Net
import os
from PIL import Image
import gc
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchviz import make_dot
from skimage.metrics import structural_similarity as ssim

CURRENT_DIR = os.path.abspath('.')
TRAIN_DIR = '../data/TrainData/'
TEST_DIR = '../data/TestData/'
VAL_DIR = '../data/ValData/'
IMG_SAVE_PATH = "../result/temp"
DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 0.001
EPOCHS = 500
BATCH_SIZE = 1

print(CURRENT_DIR)

if __name__ == '__main__':
    # --------------------load data------------------------
    # train_data@(72, 12, 4, 4, 180 , 180) = (train_num, Ndir, Nshot, Ncoil, Nx, Ny) in K-space
    # train_csm@(72, 4, 180 , 180) = (train_num, Ncoil, Nx,Ny) in image space
    # train_gt@(72, 4, 180, 180) = (train_num, Nshot, Nx, Ny) in  image sapce
    # train_mask@(12, 4, 4, 180, 180) = (Nshot, Ncoil, Nx, Ny) in K-space

    print("---------------loading data---------------")

    train_data, train_csm, train_gt, train_b0, train_mask = read_data(TRAIN_DIR, 'train')
    val_data, val_csm, val_gt, val_b0, val_mask = read_data(VAL_DIR, 'val')

    Nslice, Ndir,Nshot,Ncoil,Nx,Ny = train_data.shape
    for nslice in range(Nslice):
        for ndir in range(Ndir):
            shot1 = np.abs(train_gt[nslice, ndir, 0, :, :])
            shot1 = shot1/np.max(shot1)
            shot2 = np.abs(train_gt[nslice, ndir, 1, :, :])
            shot2 = shot2 / np.max(shot2)
            shot3 = np.abs(train_gt[nslice, ndir, 2, :, :])
            shot3 = shot3 / np.max(shot3)
            shot4 = np.abs(train_gt[nslice, ndir, 3, :, :])
            shot4 = shot4 / np.max(shot4)

            cat_image = np.concatenate((shot1, shot2, shot3, shot4), axis=1)
            image = Image.fromarray(cat_image * 255)
            image = image.convert('P')
            image.save(IMG_SAVE_PATH + '/slice' + str(nslice + 1) + '_dir' + str(ndir + 1) + '.png')

    print("---------------loading data done---------------")