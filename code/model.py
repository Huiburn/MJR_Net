import numpy as np
import torch
import os
from scipy import io
import misc as sf
import matplotlib.pyplot as plt
from read_data import read_data
from torch import nn
from torch.nn import functional as F
from unet import Unet
from torch.utils.data import Dataset
from resnet import ResNet
from unet import Unet


class My_dataset(Dataset):
    def __init__(self, data, csm, gt, b0):
        super().__init__()
        self.data = data
        self.csm = csm
        self.gt = gt
        self.b0 = b0

    def __getitem__(self, index):
        return self.data[index], self.csm[index], self.gt[index], self.b0[index]

    def __len__(self):
        return len(self.gt)


class MJR_Net(nn.Module):
    def __init__(self, device, channels, rho=0.01, with_b0=True, rho_requires_grad=False, blocks=2, cgIter=15, pocsIter = 15):
        super(MJR_Net, self).__init__()
        self.device = device
        self.blocks = blocks
        self.cgIter = cgIter
        self.rho = nn.Parameter(torch.tensor(rho).to(device), requires_grad=rho_requires_grad)
        self.with_b0=with_b0
        # self.rho=rho
        # self.resnet = ResNet(in_channels=channels).to(self.device)
        self.phaseNet = ResNet(in_channels=channels).to(self.device)
        self.Knet = Unet(input_channels=channels , output_channels=channels,
                               device=self.device).to(self.device)
        self.pocsIter = pocsIter
        if self.with_b0:
            self.magNet = Unet(input_channels=int(channels/2)+1, output_channels=int(channels/2),device = self.device).to(self.device)
        else:
            self.magNet = Unet(input_channels=int(channels / 2) , output_channels=int(channels / 2),
                               device=self.device).to(self.device)

    def preprocess(self, Y, I, csm, mask, wtri, relax_fac=1):
        """
        Input
        ------
        Y@(Nshot, Ncoil, Nx, Ny)
        csm@(Ncoil,Nx,Ny)
        Wtri@(Nx,Ny)
        """
        Nshot, Ncoil, Nx, Ny = Y.shape
        img_init = torch.mean(I, dim=0)
        I_i = I  # I_i should contain phase information, phase init with all ones here
        I_u = img_init
        diff = 1
        iter = 0

        eps = np.finfo(np.float32).eps
        # Wtri=np.hamming(Ny)*np.hamming(Nx).reshape((Nx,1))
        # mask_rep=np.expand_dims(mask, 1).repeat(4, axis=1)  # 扩展coil维
        # mask_ = np.zeros(Y.shape)
        # for i in range(Nshot):
        #     mask_[i, :, :, i:Ny:Nshot] = 1
        # mask = torch.zeros(Y.shape)
        # mask[Y.nonzero()] = 1
        csm_rep = torch.unsqueeze(csm, 0).repeat(Nshot, 1,1,1)
        # csm_rep = np.expand_dims(csm, 0).repeat(Nshot, 0)  # 扩展shot维
        csm_sos = torch.sum(torch.abs(torch.pow(csm, 2)), 0) + eps
        while iter < self.pocsIter:
            # step 1
            g = I_i * csm + 0.01*sf.torch_ifft2c(Y - mask * sf.torch_fft2c(I_i * csm_rep))
            # g = sf.torch_ifft2c(sf.dc_torch(K_mulShot_sigCoil,csm,mask[:, 0, :, :],sf.torch_fft2c(I_i),0,self.rho,1e-8,self.cgIter))
            # step 2 channel combination
            h = torch.squeeze(torch.sum((torch.conj(csm_rep) * g), -3)) / csm_sos
            # step 3 shot average
            h_hat = sf.torch_ifft2c(wtri * sf.torch_fft2c(h))
            I_avg = (torch.sum(h * (torch.conj(h_hat) / (torch.abs(h_hat + eps))), 0)) / Nshot
            # step 4 image update
            I_tmp = I_u + relax_fac * (I_avg - I_u)
            diff = torch.pow(torch.norm(I_tmp - I_u, 2), 2) / (
                    torch.pow(torch.norm(I_u, 2), 2) + eps)
            I_u = I_tmp
            # step 5 phase recovery
            phase = h_hat / torch.abs(h_hat + eps)
            phase_rep = torch.unsqueeze(phase, 1).repeat(1,Ncoil,1,1)
            I_i = I_u * phase_rep
            iter += 1
            # print('iter=%d diff=%f' % (iter, diff))
            # plt.imshow(sf.sos(I_i[0, :, :, :]), 'gray')
            # plt.show()

        return I_i[:,0,:,:]

    def forward(self, Y, csm, b0, mask, wtri, device, tol = 1e-8):
        """
        input:
        params: Y@(batch_Size, Nshot, Ncoil, Nx, Ny) in K-space
                csm@(batch_size,Ncoil,Nx,Ny)
                b0@(batch_size,Nx,Ny)
                mask@(Nshot, Ncoil, Nx, Ny)
        """
        batch_Size, Nshot, Ncoil, Nx, Ny = Y.shape
        # coil combination
        I_mulShot_sigCoil = sf.coil_combine_torch(Y, csm[0])  # I@(Nshot,Nx,Ny)
        K_mulShot_sigCoil = sf.torch_fft2c(I_mulShot_sigCoil)

        # params initialization
        K_init = torch.zeros([Nshot, Nx, Ny], dtype=torch.complex128, device=device)
        # K = sf.dc_torch(K_mulShot_sigCoil, csm[0], mask[:, 0, :, :], K_init, 0, self.rho, tol, self.cgIter)
        I = sf.cg_sense(I_mulShot_sigCoil, csm[0], mask[:, 0, :, :], K_init, 0, self.rho, tol, self.cgIter)
        # I = sf.torch_ifft2c(K)
        mag = torch.abs(I)
        P = I/(mag+0.0000001)
        mean_mag = torch.mean(mag, dim=0)
        I = mean_mag * P
        mag = torch.abs(I)
        # K = sf.dc_torch(K_mulShot_sigCoil, csm[0], mask[:, 0, :, :], sf.torch_fft2c(I), 0, 0.001, tol, self.cgIter)
        # I_mulShot_mulCoil = I[:,np.newaxis]*csm

        # I = self.preprocess(Y[0], I, csm[0], mask, wtri)
        # net_input = sf.c2r(K)
        # net_input = net_input.type(torch.FloatTensor).to(self.device)
        # net_input = torch.unsqueeze(net_input, axis=0)
        # net_output = torch.squeeze(self.resnet(net_input))
        # K_netoutput = sf.r2c(net_output)
        # K_update = K_mulShot_sigCoil + self.rho * K_netoutput
        # K = sf.dc_torch(K_update, csm[0], mask[:, 0, :, :], K, 0, self.rho, tol, self.cgIter)
        # I = sf.torch_ifft2c(K)

        # mag = torch.abs(I)
        if self.with_b0:
            cat_b0 = torch.concat([mag, b0], dim = 0).detach()
            magNetInput = cat_b0.type(torch.FloatTensor).to(self.device)
            magNetInput = torch.unsqueeze(magNetInput, axis=0)
            magNetOutput = torch.squeeze(self.magNet(magNetInput))
        else:
            mag = mag.detach()
            magNetInput = mag.type(torch.FloatTensor).to(self.device)
            magNetInput = torch.unsqueeze(magNetInput, axis=0)
            magNetOutput = torch.squeeze(self.magNet(magNetInput))
        phase = sf.c2r(I/(mag+1e-8))
        phase = phase.detach()  # detach from the graph
        phaseNetInput = phase.type(torch.FloatTensor).to(self.device)
        phaseNetInput = torch.unsqueeze(phaseNetInput, axis=0)
        phaseNetOutput = torch.squeeze(self.phaseNet(phaseNetInput))
        mean_mag_output = torch.mean(magNetOutput, 0)
        # K_net_input = sf.c2r(sf.torch_fft2c(sf.r2c(mean_mag_output * phaseNetOutput)))
        # K_net_input = K_net_input.detach()
        # K_net_input = torch.unsqueeze(K_net_input, axis=0)
        # K_net_output = torch.squeeze(self.Knet(K_net_input))
        # mag_mean = torch.mean(torch.abs(magNetOutput),dim=0)
        # I_ = mag_mean*torch.exp(1j*phaseNetOutput)
        # K_ = sf.torch_fft2c(torch.unsqueeze(I_,dim=1)*csm)
        # dc_K = Y+(1-mask)*K_
        # dc_I = sf.coil_combine_torch(dc_K, csm[0])
        return magNetOutput, phaseNetOutput


class Admm(nn.Module):
    def __init__(self, device, res_block_num, channels, blocks=5, cgIter=7, learning_rate=0.001):
        super(Admm, self).__init__()
        self.device = device
        self.blocks = blocks
        self.cgIter = cgIter
        self.rho = nn.Parameter(torch.tensor([0.001]).to(device), requires_grad=True)
        self.gamma = nn.Parameter(torch.tensor([0.1]).to(device), requires_grad=True)

        # layers = nn.ModuleList()
        self.recon_org_layer = ReconstructionOriginalLayer(0)
        self.resblocks = nn.ModuleList()
        for res_block in range(self.blocks + 1):
            self.resblocks.append(ResConvolutionLayer(block_num=res_block_num, channels=channels, device=self.device))
        self.minus_layer = MinusLayer()
        self.mul_org_layer = MultipleOriginalLayer(self.gamma)
        self.recon_update_layer = ReconstructionUpdateLayer(self.rho)
        self.add_layer = AdditionalLayer()
        self.mul_update_layer = MultipleUpdateLayer(self.gamma)

    def forward(self, Y, csm, mask, device):
        """
        input:
        params: Y@(batch_Size, Nshot, Ncoil, Nx, Ny) in K-space
                csm@(batch_size,Ncoil,Nx,Ny)
                mask@(Nshot, Ncoil, Nx, Ny)
        """
        batch_Size, Nshot, Ncoil, Nx, Ny = Y.shape
        # coil combination
        I_mulShot_sigCoil = sf.coil_combine_torch(Y, csm[0])  # I@(Nshot,Nx,Ny)
        K_mulShot_sigCoil = sf.torch_fft2c(I_mulShot_sigCoil)  # size should be (Nshot, Nx, Ny)

        # params initialization
        I_init = torch.zeros([Nshot, Nx, Ny], dtype=torch.complex128, device=device)
        I = self.recon_org_layer(K_mulShot_sigCoil, I_init, csm[0], mask[:, 0, :, :])
        m = torch.mean(torch.abs(I), axis=0)
        P = torch.angle(I)
        I = m * torch.exp(1j * P)
        net_input = sf.c2r(I)
        net_input = net_input.type(torch.FloatTensor).to(self.device)
        net_input = torch.unsqueeze(net_input, axis=0)
        net_output = self.resblocks[0](net_input)
        minus_output = self.minus_layer(net_input, net_output).to(device)
        mul_output = self.mul_org_layer(net_input, minus_output)

        for block in range(self.blocks):
            I = self.recon_update_layer(K_mulShot_sigCoil, minus_output, mul_output, I, csm[0], mask[:, 0, :, :])
            m = torch.mean(torch.abs(I), axis=0)
            P = torch.angle(I)
            I = m * torch.exp(1j * P)
            add = self.add_layer(I, mul_output)
            net_input = sf.c2r(add)
            net_input = net_input.type(torch.FloatTensor).to(self.device)
            net_input = torch.unsqueeze(net_input, axis=0)
            net_output = self.resblocks[block + 1](net_input)
            minus_output = self.minus_layer(net_input, net_output)
            mul_output = self.mul_update_layer(I, mul_output, minus_output)
        I = self.recon_update_layer(K_mulShot_sigCoil, minus_output, mul_output, I, csm[0], mask[:, 0, :, :])
        m = torch.mean(torch.abs(I), axis=0)
        return I, m


# multiple original layer
class MultipleOriginalLayer(nn.Module):
    def __init__(self, gamma):
        super(MultipleOriginalLayer, self).__init__()
        self.gamma = gamma

    def forward(self, net_input, minus_output):
        net_input = torch.squeeze(sf.r2c(net_input))
        output = torch.mul(self.gamma, torch.sub(net_input, minus_output))
        return output


class MultipleUpdateLayer(nn.Module):
    def __init__(self, gamma):
        super(MultipleUpdateLayer, self).__init__()
        self.gamma = gamma

    def forward(self, I, mul_output, minus_output):
        re_mid_output = I
        output = torch.add(mul_output, torch.mul(self.gamma, torch.sub(re_mid_output, minus_output)))
        return output


class MinusLayer(nn.Module):
    def __init__(self):
        super(MinusLayer, self).__init__()

    def forward(self, net_input, net_output):
        output = torch.sub(net_input, net_output)
        output = torch.squeeze(sf.r2c(output))
        return output


class ReconstructionOriginalLayer(nn.Module):
    def __init__(self, rho):
        super(ReconstructionOriginalLayer, self).__init__()
        self.rho = rho

    def forward(self, K_mulShot_sigCoil, I, csm, mask):
        K = sf.dc_torch(K_mulShot_sigCoil, csm, mask, I, 0, self.rho, 1e-3, 15)
        I = sf.torch_ifft2c(K)
        return I


class ReconstructionUpdateLayer(nn.Module):
    def __init__(self, rho):
        super(ReconstructionUpdateLayer, self).__init__()
        self.rho = rho

    def forward(self, K_mulShot_sigCoil, minus_output, mul_output, I, csm, mask):
        sub = torch.sub(minus_output, mul_output)
        number = sf.torch_fft2c(sub)
        rhs = K_mulShot_sigCoil + self.rho * number
        K = sf.dc_torch(rhs, csm, mask, sf.torch_fft2c(I), 0, self.rho, 1e-3, 7)
        I = sf.torch_ifft2c(K)
        return I


class ResConvolutionLayer(nn.Module):
    def __init__(self, block_num, channels, device):
        super(ResConvolutionLayer, self).__init__()
        self.block_num = block_num
        self.channels = channels
        self.resblocks = nn.ModuleList()
        for i in range(self.block_num):
            resblock = ResBlock(n_ch_in=self.channels, n_ch_out=self.channels)
            self.resblocks.append(resblock).to(device)

    def forward(self, I):
        for block in range(self.block_num):
            I = self.resblocks[block](I)
        return I


# addtional layer
class AdditionalLayer(nn.Module):
    def __init__(self):
        super(AdditionalLayer, self).__init__()

    def forward(self, I, mul_output):
        output = torch.add(I, mul_output)
        return output


class shot2shot(nn.Module):
    def __init__(self, device, blocks=10, lamK=0, lamI=0, Nshot=4, cgIter=7, learning_rate=0.001):
        super(shot2shot, self).__init__()
        self.blocks = blocks
        self.lamK = lamK
        self.lamI = lamI
        self.Nshot = Nshot
        self.cgIter = cgIter
        self.eps = np.finfo(np.float32).eps
        self.learning_rate = learning_rate
        self.resblocks = nn.ModuleList()
        for i in range(self.blocks):
            resblock = ResBlock(n_ch_in=int(Nshot)).to(device)
            self.resblocks.append(resblock)

        self.net1_input_list = [0] * blocks
        self.net2_input_list = [0] * blocks
        self.net1_output_list = [0] * blocks
        self.net2_output_list = [0] * blocks
        self.undersam_z_ksp_list = [0] * blocks

    def forward(self, mask, csm, m, K, P, I_mulShot_sigCoil, test):
        for block in range(self.blocks):
            # random choice
            net_input = torch.unsqueeze(m, dim=0)
            # input_avg = torch.mean(m_random, axis=0)
            # plt.imshow(np.abs(input_avg.cpu().detach().numpy()), 'gray')
            net1_output = self.resblocks[block](net_input)
            # plt.imshow(np.abs(m_avg.cpu().detach().numpy()), 'gray')
            m = net1_output

        # image update
        K = sf.dc_torch(sf.torch_fft2c(rhs), csm, mask[:, 0, :, :], K, 0, self.lamI, 1e-3, 7)
        I = sf.torch_ifft2c(K)
        # here shold pay attention,why input K??????????
        m = torch.abs(I)
        P = I / (torch.abs(I) + self.eps)
        # return net1_output, net2_output, undersam_z_ksp
        return self.net1_input_list, self.net2_input_list, self.net1_output_list, self.net2_output_list, self.undersam_z_ksp_list, m, P


class ContrastiveLoss(nn.Module):
    def __init__(self, loss_tv_weight=0.5, loss_rec_weight=1):
        super().__init__()
        self.loss_tv_weight = loss_tv_weight
        self.loss_rec_weight = loss_rec_weight
        self.loss_tv = 0
        self.loss_rec = 0

    def tensor_size(self, t):
        return t.size()[1] * t.size()[1] * t.size()[2]

    def forward(self, net1_input_list, net2_input_list, net1_output_list, net2_output_list, K_mulShot_sigCoil,
                undersam_z_ksp_list):
        """
        input
        -----
        :param phase_conj_estimate: (Ncase,Nshot,Nx,Ny)
        :param h: (Nshot,Nx,Ny)
        :return: loss
        """
        # compute TV loss
        # batch_size = net1_output.size()[0]
        # h_phase = net1_output.size()[2]
        # w_phase = net1_output.size()[3]
        # count_h = self.tensor_size(net1_output[:, :, 1:, :])
        # count_w = self.tensor_size(net1_output[:, :, :, 1:])
        # h1_tv = torch.pow((net1_output[:, :, 1:, :] - net1_output[:, :, :h_phase - 1, :]), 2).sum()
        # w1_tv = torch.pow((net1_output[:, :, :, 1:] - net1_output[:, :, :, :w_phase - 1]), 2).sum()
        # h2_tv = torch.pow((net2_output[:, :, 1:, :] - net2_output[:, :, :h_phase - 1, :]), 2).sum()
        # w2_tv = torch.pow((net2_output[:, :, :, 1:] - net2_output[:, :, :, :w_phase - 1]), 2).sum()
        # self.loss_tv = self.loss_tv_weight * 2 * (
        #         h1_tv / count_h + w1_tv / count_w + h2_tv / count_h + w2_tv / count_w) / batch_size

        # compute reconstruction loss
        mse_loss = torch.nn.MSELoss()
        blocks = len(net1_input_list)
        for block in range(blocks):
            self.loss_rec = mse_loss(net1_input_list[block], net2_output_list[block])
        # self.loss_rec = self.loss_rec + mse_loss(sf.c2r(K_mulShot_sigCoil), sf.c2r(undersam_z_ksp_list[blocks-1]))
        loss = self.loss_rec_weight * self.loss_rec
        return loss


class pair(object):
    """
    This function use svt to solve dwi problem
    """

    def __init__(self, iter_max=100, diff_min=1e-6, rank=20, threshold=1, winSize=5):
        self.iter_max = iter_max
        self.diff_min = diff_min
        self.threshold = threshold
        self.winSize = winSize
        self.rank = rank

    def solve(self, Y, csm):
        """
        Input
        ------
        Y@(Nshot, Ncoil, Nx, Ny): undersampled dwi data in k-space
        csm@(Ncoil,Nx,Ny): coil sensitivity map
        Wtri@(Nx,Ny)
        """
        Nshot, Ncoil, Nx, Ny = Y.shape
        mask = np.zeros(Y.shape)
        mask[Y.nonzero()] = 1
        eps = np.finfo(np.float32).eps
        csm_rep = np.expand_dims(csm, 0).repeat(Nshot, axis=0)
        CcC = np.sum(np.square(np.abs(csm)), 0)
        I = np.zeros([Nshot, Nx, Ny])
        P = np.zeros([Nshot, Nx, Ny])
        m = np.zeros([Nx, Ny])
        iter = 0
        while iter < self.iter_max:
            P = np.expand_dims(P, axis=1).repeat(Ncoil, axis=1)
            Gij = csm_rep * P * m + sf.ifft2c(Y - mask * sf.fft2c(csm_rep * P * m))
            Ij = np.sum(np.conj(csm_rep) * Gij, 1) / (CcC + eps)
            Kj = sf.fft2c(Ij)
            A = sf.im2bh(np.expand_dims(Kj, axis=0), self.winSize)
            [U, S, V] = np.linalg.svd(A[0], compute_uv=1, full_matrices=0)
            S_diag = np.diag(S)

            # hard threshold
            # A = U[:, 0:self.rank] @ S_diag[0:self.rank, 0:self.rank] @ V[:, 0:self.rank].T

            # soft threshold
            S2 = np.maximum(0, S - self.threshold)
            S3 = np.concatenate([S[0:self.rank], S2[self.rank:]], 0)
            A = U @ np.diag(S3) @ V

            I_k = sf.bh2im(np.expand_dims(A, axis=0), Nshot, Nx, Ny, self.winSize)
            I = np.squeeze(sf.ifft2c(I_k))
            P = I / (np.abs(I) + eps)

            # magnitude update
            m3 = np.sum(I * np.conj(P), axis=0) / Nshot
            m3 = m + 1.5 * (m3 - m)
            diff = np.power(np.linalg.norm(m - m3, 2), 2) / (
                    np.power(np.linalg.norm(m, 2), 2) + eps)
            m = m3
            iter = iter + 1
            print('iters: ', str(iter), 'diff: ', str(diff))

        return m, P


class pocsIce(object):
    def __init__(self, iter_max=200, diff_min=1e-6):
        self.iter_max = iter_max
        self.diff_min = diff_min

    def solve(self, Y, csm, Wtri, relax_fac=1):
        """
        Input
        ------
        Y@(Nshot, Ncoil, Nx, Ny)
        csm@(Ncoil,Nx,Ny)
        Wtri@(Nx,Ny)
        """
        Nshot, Ncoil, Nx, Ny = Y.shape
        img_init = np.zeros([Nx, Ny])
        I_i = img_init  # I_i should contain phase information, phase init with all ones here
        I_u = img_init
        diff = 1
        iter = 0
        Wtri = Wtri
        eps = np.finfo(np.float32).eps
        # Wtri=np.hamming(Ny)*np.hamming(Nx).reshape((Nx,1))
        # mask_rep=np.expand_dims(mask, 1).repeat(4, axis=1)  # 扩展coil维
        # mask_ = np.zeros(Y.shape)
        # for i in range(Nshot):
        #     mask_[i, :, :, i:Ny:Nshot] = 1
        mask = np.zeros(Y.shape)
        mask[Y.nonzero()] = 1
        csm_rep = np.expand_dims(csm, 0).repeat(Nshot, axis=0)  # 扩展shot维
        csm_sos = np.sum(np.abs(np.power(csm, 2)), 0) + eps
        while iter < self.iter_max:
            # step 1
            g = I_i * csm + sf.ifft2c(Y - mask * sf.fft2c(I_i * csm_rep))
            # step 2 channel combination
            h = np.squeeze(np.sum((np.conj(csm_rep) * g), -3)) / csm_sos
            # step 3 shot average
            h_hat = sf.ifft2c(Wtri * sf.fft2c(h))
            I_avg = (np.sum(h * (np.conj(h_hat) / (np.abs(h_hat + eps))), 0)) / Nshot
            # step 4 image update
            I_tmp = I_u + relax_fac * (I_avg - I_u)
            diff = np.power(np.linalg.norm(I_tmp - I_u, 2), 2) / (
                    np.power(np.linalg.norm(I_u, 2), 2) + eps)
            I_u = I_tmp
            # step 5 phase recovery
            phase = h_hat / np.abs(h_hat + eps)
            phase_rep = np.expand_dims(phase, 0).repeat(Ncoil, axis=0).transpose(1, 0, 2, 3)
            I_i = I_u * phase_rep
            iter += 1
            print('iter=%d diff=%f' % (iter, diff))
            plt.imshow(sf.sos(I_i[0, :, :, :]), 'gray')
            plt.show()
        rec_pocs_ice = I_tmp
        return rec_pocs_ice, phase_rep
