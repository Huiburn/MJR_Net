import numpy as np
import torch
import os
from PIL import Image
from scipy import io
import misc as sf
import matplotlib.pyplot as plt
from read_data import read_data
from model import pair, pocsIce
from model import shot2shot
from model import My_dataset,Admm,MJR_Net
import os
import gc
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchviz import make_dot
from skimage.metrics import structural_similarity as ssim

CURRENT_DIR = os.path.abspath('.')
TRAIN_DIR = '../data/TrainData/'
TEST_DIR = '../data/TestData/'
VAL_DIR = '../data/ValData/'
IMG_SAVE_PATH = "../result/temp"

DEVICE = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 0.001
EPOCHS = 1
BATCH_SIZE=1

print(CURRENT_DIR)

if __name__ == '__main__':
    # --------------------load data------------------------
    # train_data@(300, 4, 16, 160 , 160) = (train_num,Nshot,Ncoil,Nx,Ny) in K-space
    # train_csm@(300, 16, 160 , 160) = (train_num, Ncoil, Nx,Ny) in image space
    # train_gt@(300, 4, 160, 160) = (train_num, Ncoil, Nx, Ny) in  image sapce
    # train_mask@(4,16,160,160) = (Nshot, Ncoil, Nx, Ny) in K-space

    print("---------------loading data---------------")

    test_data, test_csm, test_gt, test_b0, test_mask = read_data(TEST_DIR, 'test', DEVICE)
    test_dataset = My_dataset(test_data, test_csm, test_gt, test_b0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    wtri = torch.from_numpy(np.load('/data/disk1/shiboxuan/PythonFile/MJR_Net/data/wtri.npy')).to(DEVICE)
    model = MJR_Net(device=DEVICE, rho=0.01, channels=8, blocks=2, cgIter=15)
    net_state = torch.load('../model/epoch_300.pt')
    model.load_state_dict(net_state)
    criterion = torch.nn.MSELoss()
    print("---------------testing data---------------")
    for epoch in range(EPOCHS):
        test_loss = 0
        test_ssim = 0
        test_mag_ssim = 0
        test_batches = 0
        test_loss_list = []
        test_ssim_list = []
        for batch, (test_data, test_csm, test_gt, test_b0) in enumerate(test_loader):
            # test_data@(batchsize, Ndir, Nshot, Ncoil, Nx, Ny)
            # train_csm@(batchsize, Ncoil, Nx, Ny)
            # train_gt@(batchsize, Ndir, Nshot,Nx, Ny)

            # model init
            batch_size, Ndir, Nshot,Ncoil,Nx,Ny = test_data.shape
            for ndir in range(Ndir):
                model.eval()
                mag_output, phase_output = model(test_data[:,ndir,:,:,:], test_csm, test_b0, test_mask[ndir,:,:,:,:],wtri, DEVICE)
                rec_mag = torch.mean(mag_output, axis=0)
                test_trg_mag = torch.mean(torch.abs(test_gt[0,ndir]),axis=0)
                loss_normal = criterion(test_trg_mag, rec_mag)

                recImage_numpy = rec_mag.cpu().detach().numpy()
                recImage_avg_norm = recImage_numpy / np.max(recImage_numpy)

                test_gt_avg = np.mean(np.abs(test_gt[0, ndir].cpu().detach().numpy()), axis=0)
                test_gt_norm = test_gt_avg / np.max(test_gt_avg)
                ssim_value = ssim(recImage_avg_norm, test_gt_norm)

                # save image
                mag_net_out = np.mean(np.abs(mag_output.cpu().detach().numpy()), axis=0)
                mag_net_out_norm = mag_net_out / np.max(mag_net_out)
                mag_ssim_value = ssim(mag_net_out_norm, test_gt_norm)

                # recKsp_avg_norm = np.mean(np.abs(sf.ifft2c(sf.r2c(K_output.cpu().detach().numpy()))), 0)
                # recKsp_avg_norm = recKsp_avg_norm/np.max(recKsp_avg_norm)
                # cat_image = np.concatenate((recKsp_avg_norm,mag_net_out_norm, test_gt_norm), axis=1)
                cat_image = np.concatenate((test_gt_norm, mag_net_out_norm), axis=1)
                image = Image.fromarray(cat_image * 255)
                image = image.convert('P')
                image.save(IMG_SAVE_PATH + 'slice' + str(batch + 1) + '_dir' + str(ndir + 1) + '.png')

                plt.imshow(np.abs(mag_net_out_norm - test_gt_norm), 'jet')
                plt.colorbar()
                plt.axis('off')
                plt.title(mag_ssim_value)
                plt.savefig(IMG_SAVE_PATH + 'slice' + str(batch + 1) + '_dir' + str(ndir + 1) + '_diff.png')
                plt.close()
                test_loss += loss_normal.data.item()
                test_ssim += ssim_value
                test_batches += 1
                test_mag_ssim+=mag_ssim_value
        test_loss /= test_batches
        test_ssim /= test_batches
        test_mag_ssim /= test_batches
        test_loss_list.append(test_loss)
        test_ssim_list.append(test_ssim)
        # sf.img2gif(IMG_SAVE_PATH, 'test_results')
        print("test_loss: ", test_loss)
        print("test_ssim: ", test_ssim)
        print("test_mag_ssim: ",test_mag_ssim)
        print('Epoch: ', epoch, '| Step: ', batch, '| loss: ', test_loss)
