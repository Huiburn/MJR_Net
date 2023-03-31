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
MODEL_DIR = '../model/'
DEVICE = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
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

    train_data, train_csm, train_gt, train_b0, train_mask = read_data(TRAIN_DIR, 'train', DEVICE)
    val_data, val_csm, val_gt, val_b0, val_mask = read_data(VAL_DIR, 'val', DEVICE)
    wtri = torch.from_numpy(np.load('/data/disk1/shiboxuan/PythonFile/MJR_Net/data/wtri.npy')).to(DEVICE)
    train_dataset = My_dataset(train_data, train_csm, train_gt, train_b0)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_dataset = My_dataset(val_data, val_csm, val_gt, train_b0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("---------------training data---------------")
    model = MJR_Net(device=DEVICE, rho=0, channels=8,with_b0=True, blocks=2, cgIter=15)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.MSELoss()
    for epoch in range(EPOCHS):
        train_loss = 0
        train_ssim = 0
        train_mag_ssim = 0
        train_batches = 0
        train_loss_list = []
        train_ssim_list = []
        for batch, (train_data, train_csm, train_gt, train_b0) in enumerate(train_loader):
            # train_data@(batchsize,Ndir, Nshot, Ncoil, Nx, Ny)
            # train_csm@(batchsize, Ncoil, Nx, Ny)
            # train_gt@(batchsize, Ndir, Nshot, Nx, Ny)
            # train_b0@(batchsize, Nx, Ny)

            # model init
            batch_size, Ndir, Nshot, Ncoil, Nx, Ny = train_data.shape
            for ndir in range(Ndir):
                model.train()
                optimizer.zero_grad()
                mag_output, phase_output = model(train_data[:, ndir, :, :, :, :], train_csm, train_b0,
                                                        train_mask[ndir, :, :, :, :], wtri, DEVICE)

                trg_mag = torch.abs(train_gt[0, ndir]).float()
                # rec_mag = torch.mean(torch.abs(sf.torch_ifft2c(rec_k)), axis=0)
                trg_phase = sf.c2r(train_gt[0, ndir]/torch.abs(train_gt[0, ndir])).float()
                # trg_phs = torch.angle(torch.abs(train_gt[0, ndir]))
                trg_Ksp = sf.c2r(sf.torch_fft2c(train_gt[0, ndir])).float()
                # loss_normal = criterion(trg_mag,mag_output) + criterion(trg_phase, phase_output) + criterion(trg_Ksp,K_output)
                loss_normal = criterion(trg_mag, mag_output) + criterion(trg_phase, phase_output)
                loss_normal.backward()
                optimizer.step()

                recImage_numpy = np.mean(mag_output.cpu().detach().numpy(),axis=0)
                recImage_avg_norm = recImage_numpy / np.max(recImage_numpy)
                train_gt_avg = np.mean(np.abs(train_gt[0, ndir].cpu().detach().numpy()), axis=0)
                train_gt_norm = train_gt_avg / np.max(train_gt_avg)
                ssim_value = ssim(recImage_avg_norm, train_gt_norm)

                # mag_net_output = np.mean(np.abs(mag_output.cpu().detach().numpy()), axis=0)
                # mag_net_output_norm = mag_net_output/np.max(mag_net_output)
                # mag_ssim_value = ssim(mag_net_output_norm, train_gt_norm)

                train_loss += loss_normal.data.item()
                train_ssim += ssim_value
                # train_mag_ssim += mag_ssim_value
                train_batches += 1
        train_loss /= train_batches
        train_ssim /= train_batches
        train_mag_ssim /= train_batches
        train_loss_list.append(train_loss)
        train_ssim_list.append(train_ssim)
        print("train_ssim: ", train_ssim)
        print("train_mag_ssim", train_mag_ssim)
        print('Epoch: ', epoch, '| Step: ', batch, '| loss: ', train_loss)

        if epoch % 5 == 0:
            model_name = MODEL_DIR + 'epoch_' + str(
                epoch) + '.pt'
            torch.save(model.state_dict(), model_name)
        if epoch % 10 == 0:
            print("---------------validating data---------------")
            val_loss = 0
            val_ssim = 0
            val_mag_ssim = 0
            val_batches = 0
            val_loss_list = []
            val_ssim_list = []
            with torch.no_grad():
                for val_batch, (val_data, val_csm, val_gt, val_b0) in enumerate(train_loader):
                    for ndir in range(Ndir):
                        model.eval()
                        gc.collect()
                        torch.cuda.empty_cache()
                        val_mag_output, val_phase_output = model(val_data[:, ndir, :, :, :], val_csm,
                                                                               val_b0, val_mask[ndir, :, :, :, :],wtri,
                                                                               DEVICE)
                        trg_phase = sf.c2r(train_gt[0, ndir] / torch.abs(train_gt[0, ndir])).float()
                        val_trg_mag = torch.abs(val_gt[0, ndir]).float()
                        # val_rec_mag = torch.mean(torch.abs(sf.torch_ifft2c(val_output_K)), axis=0)
                        val_trg_phase = sf.c2r(val_gt[0, ndir] / torch.abs(val_gt[0, ndir])).float()
                        # val_trg_phs = torch.angle(torch.abs(val_gt[0, ndir]))
                        trg_val_Ksp = sf.c2r(sf.torch_fft2c(val_gt[0, ndir])).float()
                        # validate_loss_normal = criterion(val_mag_output, val_trg_mag) + criterion(val_phase_output, val_trg_phase)+criterion(trg_val_Ksp,K_val_output)
                        validate_loss_normal = criterion(val_mag_output, val_trg_mag) + criterion(val_phase_output, val_trg_phase)
                        val_image_numpy = np.mean(val_mag_output.cpu().detach().numpy(),axis=0)
                        val_image_avg_norm = val_image_numpy / np.max(val_image_numpy)
                        val_gt_avg = np.mean(np.abs(val_gt[0, ndir].cpu().detach().numpy()), axis=0)
                        val_gt_norm = val_gt_avg / np.max(val_gt_avg)
                        val_ssim_value = ssim(val_image_avg_norm, val_gt_norm)
                        val_loss += validate_loss_normal.data.item()
                        val_ssim += val_ssim_value
                        val_batches += 1
                val_loss /= val_batches
                val_ssim /= val_batches
                val_loss_list.append(val_loss)
                val_ssim_list.append(val_ssim)
                print("val_ssim: ", val_ssim)
                print('Epoch: ', epoch, '| Step: ', val_batch, '| val_loss: ', val_loss)
    np.save("train_loss_list.npy",train_loss_list)
    plt.figure(1)
    plt.plot(train_loss_list)
    plt.title("train_loss")
    plt.figure(2)
    plt.plot(val_loss_list)
    plt.title("val_loss")
    plt.figure(3)
    plt.plot(train_ssim_list)
    plt.title("train_ssim")
    plt.figure(4)
    plt.plot(val_ssim_list)
    plt.title("val_ssim")

    # pair_model = pair(iter_max=100)
    # pairRecImage, pairRecPhase = pair_model.solve(train_data[0, 0, :, :, :, :], train_csm[0, :, :, :])
    # pocs_model = pocsIce(iter_max=100)
    # pocsRecImage, pocsRecPhase = pocs_model.solve(train_data[0, 0, :, :, :, :], train_csm[0, :, :, :], Wtri)
