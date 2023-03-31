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
import gc
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchviz import make_dot
from skimage.metrics import structural_similarity as ssim

CURRENT_DIR = os.path.abspath('.')
TRAIN_DIR = '../data/TrainData/'
TEST_DIR = '../data/TestData/'
VAL_DIR = '../data/ValData/'
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
    print("test beginning--------")
    wtri = np.load('/data/disk1/shiboxuan/PythonFile/MJR_Net/data/wtri.npy')
    pocs_model = pocsIce(iter_max=100, diff_min=1e-6)
    pocs_model.solve(np.squeeze(train_data[0,0]),np.squeeze(train_csm[0]),wtri)