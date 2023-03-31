import numpy as np
import torch
import os
from scipy import io
import misc as sf
import matplotlib.pyplot as plt
from read_data import read_data
from model import pair, pocsIce
from model import shot2shot
from model import My_dataset, Admm, Modl
import os
import gc
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchviz import make_dot
from skimage.metrics import structural_similarity as ssim

CURRENT_DIR = os.path.abspath('.')
TRAIN_DIR = '../data/TrainData/'
TEST_DIR = '../data/TestData/'
VAL_DIR = '../data/ValData/'
DEVICE = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 0.001
EPOCHS = 500
BATCH_SIZE = 1

print(CURRENT_DIR)

if __name__ == '__main__':
    # --------------------load data------------------------
    # train_data@(72, 12, 4, 4, 180 , 185) = (train_num, Ndir, Nshot, Ncoil, Nx, Ny) in K-space
    # train_csm@(72, 4, 180 , 185) = (train_num, Ncoil, Nx,Ny) in image space
    # train_gt@(72, 4, 180, 185) = (train_num, Nshot, Nx, Ny) in  image sapce
    # train_mask@(12, 4, 4, 180, 185) = (Nshot, Ncoil, Nx, Ny) in K-space

    print("---------------loading data---------------")

    train_data, train_csm, train_gt, train_mask = read_data(TRAIN_DIR, 'train', DEVICE)
    val_data, val_csm, val_gt, val_mask = read_data(VAL_DIR, 'val', DEVICE)

    train_dataset = My_dataset(train_data, train_csm, train_gt)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_dataset = My_dataset(val_data, val_csm, val_gt)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("---------------training data---------------")
    model = Modl(device=DEVICE, rho=0.01, channels=8, blocks=2, cgIter=15)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.MSELoss()
    for epoch in range(EPOCHS):
        train_loss = 0
        train_ssim = 0
        train_batches = 0
        train_loss_list = []
        train_ssim_list = []
        for batch, (train_data, train_csm, train_gt) in enumerate(train_loader):
            # train_data@(batchsize,Ndir, Nshot, Ncoil, Nx, Ny)
            # train_csm@(batchsize, Ncoil, Nx, Ny)
            # train_gt@(batchsize, Nshot, Nx, Ny)

            # model init
            batch_size, Ndir, Nshot, Ncoil, Nx, Ny = train_data.shape
            for ndir in range(Ndir):
                model.train()
                optimizer.zero_grad()
                Y, csm, mask, device=train_data[:, ndir, :, :, :, :], train_csm, train_mask[ndir, :, :, :, :], DEVICE

                batch_Size, Nshot, Ncoil, Nx, Ny = Y.shape
                # coil combination
                I_mulShot_sigCoil = sf.coil_combine_torch(Y, csm[0])  # I@(Nshot,Nx,Ny)
                mag_mulShot_sigCoil = np.mean(np.abs(I_mulShot_sigCoil.cpu().detach().numpy()),axis=0)
                K_mulShot_sigCoil = sf.torch_fft2c(I_mulShot_sigCoil)
                K_mag_mulShot_sigCoil = sf.fft2c(mag_mulShot_sigCoil)

                # params initialization
                K_init = torch.zeros([Nshot, Nx, Ny], dtype=torch.complex128, device=device)
                mag_test_K_init = np.zeros((Nx,Ny),dtype='complex')
                phase = np.angle(train_gt[0,ndir].cpu().detach().numpy())
                mag_K  = sf.mag_op(K_mulShot_sigCoil, csm[0].cpu().detach().numpy(), phase, mask[:, 0, :, :].cpu().detach().numpy(), mag_test_K_init ,0, 0.01, 1e-3, 15)

                K = sf.dc_torch(K_mulShot_sigCoil, csm[0], mask[:, 0, :, :], K_init, 0, 0.01, 1e-3, 15)
                csm_k = sf.csm_op(sf.fft2c(csm[0].cpu().detach().numpy()), train_gt[0,ndir].cpu().detach().numpy(), mask[:, 0, :, :].cpu().detach().numpy(), K_init.cpu().detach().numpy() ,0, 0.01, 1e-3, 15)
                csm_img = sf.ifft2c(csm_k)/np.max(np.abs(sf.ifft2c(csm_k)))
                I = sf.torch_ifft2c(K)