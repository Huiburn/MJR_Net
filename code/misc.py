"""
Created on Feb 10th, 2022

This file contains some supporting functions

@author:Huiburn
"""

import numpy as np
import torch
import torch.fft
import os
import imageio
import matplotlib.pyplot as plt


def coil_combine_torch(Y, csm):
    """
    return: single coil image
    """
    if Y.ndim == 4:
        Nshot, Ncoil, Nx, Ny = Y.shape
    if Y.ndim == 5:
        batchsize, Nshot, Ncoil, Nx, Ny = Y.shape
    if csm.ndim == 3:
        csm = torch.unsqueeze(csm, 0).repeat(Nshot, 1, 1, 1)  # 扩展shot维
    eps = np.finfo(np.float32).eps

    # coil combination
    img_mulCoil = torch_ifft2c(Y)
    img_sigCoil = torch.squeeze(torch.sum((torch.conj(csm) * img_mulCoil), -3))
    return img_sigCoil


def coil_combine(Y, csm):
    """
        return: single coil image
        """
    if Y.ndim == 4:
        Nshot, Ncoil, Nx, Ny = Y.shape
    if Y.ndim == 5:
        batchsize, Nshot, Ncoil, Nx, Ny = Y.shape
    if csm.ndim == 3:
        csm = np.expand_dims(csm, axis=0).repeat(Nshot, 0)  # 扩展shot维
    eps = np.finfo(np.float32).eps

    # coil combination
    img_mulCoil = ifft2c(Y)
    img_sigCoil = np.squeeze(np.sum((np.conj(csm) * img_mulCoil), -3))
    return img_sigCoil


def im2bh(x, w=6):
    '''
    from image to block-hankel
    --------------------------
    params: x@(batchSize, Nshot, Nx, Ny)
    return:  A@(batchSize, (Nx-w+1)*Ny-w+1, w*w, Nshot)
    '''
    [batchSize, Nshot, Nx, Ny] = x.shape
    A = np.zeros([batchSize, Nshot, (Nx - w + 1) * (Ny - w + 1), 1], dtype='complex')
    for wy in range(w):
        for wx in range(w):
            A = np.concatenate((A, np.reshape(x[:, :, wx:Nx - w + wx + 1, wy:Ny - w + wy + 1],
                                              [batchSize, Nshot, (Nx - w + 1) * (Ny - w + 1), 1])), 3)
    A = np.transpose(A[:, :, :, 1:], [0, 2, 3, 1])
    A = np.reshape(A, [batchSize, (Nx - w + 1) * (Ny - w + 1), w * w * Nshot])
    return A


def bh2im(A, Nshot, Nx, Ny, w=6):
    """
    from block-hankel to image
    --------------------------
    params: x@(batchSize, Nshot, Nx, Ny)
    return:  A@(batchSize, w*w*Nshot, (Nx-w+1)*Ny-w+1)
    """
    batchSize = A.shape[0]
    A = np.reshape(A, [batchSize, (Nx - w + 1) * (Ny - w + 1), w * w, Nshot])
    A = np.transpose(A, [0, 3, 1, 2])  # ->[batchsize, Nshot, (Nx-w+1)*(Ny-w+1), w*w]
    x_real = np.zeros([batchSize, Nshot, Nx, Ny])
    x_imag = np.zeros([batchSize, Nshot, Nx, Ny])
    overLap = np.zeros([batchSize, Nshot, Nx, Ny])
    count = 0

    for wy in range(w):
        for wx in range(w):
            x_real[:, :, wx:Nx - w + wx + 1, wy:Ny - w + wy + 1] = np.reshape(
                np.real(A[:, :, :, count]), [batchSize, Nshot, Nx - w + 1, Ny - w + 1]) + x_real[:, :,
                                                                                          wx:Nx - w + wx + 1,
                                                                                          wy:Ny - w + wy + 1]
            x_imag[:, :, wx:Nx - w + wx + 1, wy:Ny - w + wy + 1] = np.reshape(
                np.imag(A[:, :, :, count]), [batchSize, Nshot, Nx - w + 1, Ny - w + 1]) + x_imag[:, :,
                                                                                          wx:Nx - w + wx + 1,
                                                                                          wy:Ny - w + wy + 1]
            overLap[:, :, wx:Nx - w + wx + 1, wy:Ny - w + wy + 1] = 1.0 + overLap[:, :, wx:Nx - w + wx + 1,
                                                                          wy:Ny - w + wy + 1]
            count = count + 1
    x = x_real + 1j * x_imag
    x = x / overLap
    return x


def fft2c(img):
    shp = img.shape
    nimg = int(np.prod(shp[0:-2]))
    scale = 1 / np.sqrt(np.prod(shp[-2:]))
    img = np.reshape(img, (nimg, shp[-2], shp[-1]))

    tmp = np.empty_like(img, dtype=np.complex64)
    for i in range(nimg):
        tmp[i] = scale * np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img[i])))

    kspace = np.reshape(tmp, shp)
    return kspace


# def fft2c_torch(img):
#     shp = img.shape
#     nimg = int(np.prod(shp[0:-2]))
#     scale = 1 / np.sqrt(np.prod(shp[-2:]))
#     img = torch.reshape(img, (nimg, shp[-2], shp[-1]))
#
#     tmp = torch.empty_like(img, dtype=torch.complex128)
#     for i in range(nimg):
#         tmp[i] = scale * torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(img[i])))
#
#     kspace = torch.reshape(tmp, shp)
#     return kspace


def ifft2c(kspace):
    shp = kspace.shape
    scale = np.sqrt(np.prod(shp[-2:]))
    nimg = int(np.prod(shp[0:-2]))

    kspace = np.reshape(kspace, (nimg, shp[-2], shp[-1]))

    tmp = np.empty_like(kspace)
    for i in range(nimg):
        tmp[i] = scale * np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kspace[i])))

    img = np.reshape(tmp, shp)
    return img


# def ifft2c_torch(kspace):
#     shp = kspace.shape
#     scale = np.sqrt(np.prod(shp[-2:]))
#     nimg = int(np.prod(shp[0:-2]))
#
#     kspace = torch.reshape(kspace, (nimg, shp[-2], shp[-1]))
#
#     tmp = torch.empty_like(kspace, dtype=torch.complex128)
#     for i in range(nimg):
#         tmp[i] = scale * torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(kspace[i])))
#
#     img = torch.reshape(tmp, shp)
#     return img


def torch_fft2c(img):
    # kspace = myifftshift(torch.fft.fftn(myfftshift(img), dim=(-2, -1), norm='ortho'))
    # kspace = torch.fft.ifftshift(torch.fft.fft2(torch.fft.fftshift(img), dim=(-2, -1),norm='ortho'))
    shp = img.shape
    scale = 1 / np.sqrt(np.prod(shp[-2:]))
    kspace = scale * torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(img)))
    return kspace


def torch_ifft2c(kspace):
    # img = myifftshift(torch.fft.ifftn(myfftshift(kspace), dim=(-2, -1), norm='ortho'))
    # img = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fftshift(kspace), dim=(-2, -1), norm='ortho'))
    shp = kspace.shape
    scale = np.sqrt(np.prod(shp[-2:]))
    img = scale * torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(kspace)))
    return img


def sos(data, dim=-3):
    res = np.sqrt(np.sum(np.abs(data) ** 2, dim))
    return res


def epiA_np(ksp, csm, mask):
    img = ifft2c(ksp)
    coilImages = csm * img[:, np.newaxis]
    data = fft2c(coilImages)  # (shots,coils,rows,cols)
    data = mask[:, np.newaxis] * data
    return data


def epiA_csm_np(ksp, I, mask):
    csm_img = ifft2c(ksp)
    coilImages = csm_img * I[:, np.newaxis]
    data = fft2c(coilImages)  # (shots,coils,rows,cols)
    data = mask[:, np.newaxis] * data
    return data


def epiA_mag_np(ksp, csm, phase, mask):
    mag = ifft2c(ksp)
    coilImages = csm * np.exp(1j * phase[:,np.newaxis]) * mag
    data = fft2c(coilImages)  # (shots,coils,rows,cols)
    data = mask[:, np.newaxis] * data
    return data



def epiAt_np(ksp, csm, mask):
    ksp = mask[:, np.newaxis] * ksp
    gdata = ifft2c(ksp)
    data = np.conj(csm) * gdata
    img = np.sum(data, -3)
    kspace = fft2c(img)
    return kspace


def epiAt_csm_np(ksp, I, mask):
    ksp = mask[:, np.newaxis] * ksp
    gdata = ifft2c(ksp)
    data = np.conj(I[:, np.newaxis]) * gdata
    img = np.sum(data, -3)
    kspace = fft2c(img)
    return kspace


def epiAt_mag_np(ksp, csm, phase, mask):
    ksp = mask[:, np.newaxis] * ksp
    gdata = ifft2c(ksp)
    data = np.conj(csm) * gdata
    img = np.sum(data, -3)
    mag = np.exp(1j * np.conj(phase)) * img
    img = np.mean(mag, -3)
    kspace = fft2c(img)
    return kspace


def epiA_torch(ksp, csm, mask):
    img = torch_ifft2c(ksp)
    coilImages = csm * img[:, np.newaxis]
    data = torch_fft2c(coilImages)  # (shots,coils,rows,cols)
    data = mask[:, np.newaxis] * data
    return data


def epiA_csm_torch(ksp, I, mask):
    csm_img = torch_ifft2c(ksp)
    coilImages = csm_img * I[:, np.newaxis]
    data = torch_fft2c(coilImages)  # (shots,coils,rows,cols)
    data = mask[:, np.newaxis] * data
    return data


def epiAt_torch(ksp, csm, mask):
    ksp = mask[:, np.newaxis] * ksp
    gdata = torch_ifft2c(ksp)
    data = torch.conj(csm) * gdata
    img = torch.sum(data, -3)
    kspace = torch_fft2c(img)
    return kspace


def epiAt_csm_torch(ksp, I, mask):
    ksp = mask[:, np.newaxis] * ksp
    gdata = torch_ifft2c(ksp)
    data = torch.conj(I[:, np.newaxis]) * gdata
    img = torch.sum(data, -3)
    kspace = torch_fft2c(img)
    return kspace


def cg4shots(B, rhs, maxIter, cgTol, x):
    # This CG works on all N-shots simultaneously for speed
    # Conjugate-gradient on all 4-shots
    one = 1
    zero = 0
    cond = lambda i, rTr, *_: np.less(i, maxIter) and torch.sqrt(torch.min(torch.abs(rTr))) > cgTol
    fn = lambda x, y: torch.sum(torch.conj(x) * y, axis=(-1, -2), keepdims=True)

    def body(i, rTr, x, r, p):
        Ap = B(p)
        alpha = rTr / fn(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rTrNew = fn(r, r)
        beta = rTrNew / rTr
        p = r + beta * p
        return i + one, rTrNew, x, r, p

    i = zero
    r = rhs - B(x)
    p = r
    rTr = fn(r, r)
    # out = tf.while_loop(cond, body, loopVar, name='CGwhile', parallel_iterations=1)[2]
    while (cond(i, rTr, x)):
        i, rTr, x, r, p = body(i, rTr, x, r, p)
    out = x

    return out


def cg4shots_np(B, rhs, maxIter, cgTol, x):
    # This CG works on all N-shots simultaneously for speed
    # Conjugate-gradient on all 4-shots
    one = 1
    zero = 0
    cond = lambda i, rTr, *_: np.less(i, maxIter) and np.sqrt(np.min(np.abs(rTr))) > cgTol
    fn = lambda x, y: np.sum(np.conj(x) * y, axis=(-1, -2), keepdims=True)

    def body(i, rTr, x, r, p):
        Ap = B(p)
        alpha = rTr / fn(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rTrNew = fn(r, r)
        beta = rTrNew / rTr
        p = r + beta * p
        return i + one, rTrNew, x, r, p

    i = zero
    r = rhs - B(x)
    p = r
    rTr = fn(r, r)
    # out = tf.while_loop(cond, body, loopVar, name='CGwhile', parallel_iterations=1)[2]
    while (cond(i, rTr, x)):
        i, rTr, x, r, p = body(i, rTr, x, r, p)
    out = x

    return out

def cg_forward_Etranspose(img, csm, mask):
    """
    return: in kspace
    """
    coilImages = csm * img[:, np.newaxis]
    data = torch_fft2c(coilImages)  # (shots,coils,rows,cols)
    data = mask[:, np.newaxis] * data
    return data

def cg_backward_ETtranspose(ksp, csm, mask):
    # ksp = mask[:, np.newaxis] * ksp
    gdata = torch_ifft2c(ksp)
    data = torch.conj(csm) * gdata
    img = torch.sum(data, -3)
    return img

def cg_sense(rhsT, csmT, maskT, xprev, lamKT, lamIT, cgTol, cgIter):
    """
    input
    ------
    :param rhsT@(Nshot,Nx,Ny) in img-space
    :param csmT@(Ncoil,Nx,Ny)
    :param maskT@(Nshot,Nx,Ny)
    output
    ------
    :x @(Nshot, Nx, Ny) in img-space
    """
    rhs, csm, mask, xin = (rhsT, csmT, maskT, xprev)
    A = lambda x: cg_forward_Etranspose(x, csm, mask)
    At = lambda x: cg_backward_ETtranspose(x, csm, mask)
    B = lambda x: At(A(x)) + lamKT * x + lamIT * x
    x = cg4shots(B, rhs, cgIter, cgTol, xin)
    return x

def dc(rhsT, csmT, maskT, xprev, lamKT, lamIT, cgTol, cgIter):
    """
    input
    ------
    :param rhsT@(Nshot,Nx,Ny) in k-space
    :param csmT@(Ncoil,Nx,Ny)
    :param maskT@(Nshot,Nx,Ny)
    output
    ------
    :x @(Nshot, Nx, Ny) in k-space
    """
    rhs, csm, mask, xin = (rhsT, csmT, maskT, xprev)
    A = lambda x: epiA_np(x, csm, mask)
    At = lambda x: epiAt_np(x, csm, mask)
    B = lambda x: At(A(x)) + lamKT * x + lamIT * x
    x = cg4shots_np(B, rhs, cgIter, cgTol, xin)
    return x


def csm_op(rhsT, I, maskT, xprev, lamKT, lamIT, cgTol, cgIter):
    """
    :param rhsT:
    :param csmT:
    :param maskT:
    :param xprev:
    :param lamKT:
    :param lamIT:
    :param cgTol:
    :param cgIter:
    :return:
    """
    rhs, I, mask, xin = (rhsT, I, maskT, xprev)
    A = lambda x: epiA_csm_np(x, I, mask)
    At = lambda x: epiAt_csm_np(x, I, mask)
    B = lambda x: At(A(x)) + lamKT * x + lamIT * x
    x = cg4shots_np(B, rhs, cgIter, cgTol, xin)
    return x


def mag_op(rhsT, csmT, phase, maskT, xprev, lamKT, lamIT, cgTol, cgIter):
    rhs, csm, phase, mask, xin = (rhsT, csmT, phase, maskT, xprev)
    A = lambda x: epiA_mag_np(x, csm, phase, mask)
    At = lambda x: epiAt_mag_np(x, csm, phase, mask)
    B = lambda x: At(A(x)) + lamKT * x + lamIT * x
    x = cg4shots_np(B, rhs, cgIter, cgTol, xin)
    return x


def csm_op_torch(rhsT, I, maskT, xprev, lamKT, lamIT, cgTol, cgIter):
    """
    :param rhsT:
    :param csmT:
    :param maskT:
    :param xprev:
    :param lamKT:
    :param lamIT:
    :param cgTol:
    :param cgIter:
    :return:
    """
    rhs, I, mask, xin = (rhsT, I, maskT, xprev)
    A = lambda x: epiA_csm_torch(x, I, mask)
    At = lambda x: epiAt_csm_torch(x, I, mask)
    B = lambda x: At(A(x)) + lamKT * x + lamIT * x
    x = cg4shots(B, rhs, cgIter, cgTol, xin)
    return x


def dc_torch(rhsT, csmT, maskT, xprev, lamKT, lamIT, cgTol, cgIter):
    """
    input
    ------
    :param rhsT@(Nshot,Nx,Ny) in k-space
    :param csmT@(Ncoil,Nx,Ny)
    :param maskT@(Nshot,Nx,Ny)
    output
    ------
    :x @(Nshot, Nx, Ny) in k-space
    """
    rhs, csm, mask, xin = (rhsT, csmT, maskT, xprev)
    A = lambda x: epiA_torch(x, csm, mask)
    At = lambda x: epiAt_torch(x, csm, mask)
    B = lambda x: At(A(x)) + lamKT * x + lamIT * x
    x = cg4shots(B, rhs, cgIter, cgTol, xin)
    return x


# %% split(or combine) the real and imaginary parts of the complex number(pytorch version) into different channels
def r2c(inp):
    idx = inp.shape[-3] // 2
    out = inp[..., 0:idx, :, :] + 1j * inp[..., idx:, :, :]
    return out


def c2r(inp):
    if torch.is_tensor(inp):
        out = torch.cat([torch.real(inp), torch.imag(inp)], dim=-3)
    else:
        out = np.concatenate([np.real(inp), np.imag(inp)], axis=-3)
    return out


def complex_matmul(x, y):
    """
    input
    :param x: (Ncase, Nshot, Nx, Ny)
    :param y: (Ncase, Nshot, Nx, Ny)
    :return: z
    """
    if torch.is_tensor(x):
        Ncase, Nshot, Nx, Ny = x.size()
        Nshot = int(Nshot / 2)
        z = torch.zeros(x.size())
    else:
        Ncase, Nshot, Nx, Ny = x.shape
        Nshot = int(Nshot / 2)
        z = np.zeros(x.size)
    for i in range(0, Ncase):
        for nshot in range(0, Nshot):
            # x = a+bj
            a = x[i, nshot, :, :]
            b = x[i, nshot + Nshot, :, :]
            # y = c+dj
            c = y[i, nshot, :, :]
            d = y[i, nshot + Nshot, :, :]

            # real part
            z[i, nshot, :, :] = a * c - b * d
            # imag part
            z[i, nshot + Nshot, :, :] = a * d + b * c

    return z


def myfftshift(x):
    shp = x.shape[-2:]
    dim = [s // 2 for s in shp]
    y = torch.roll(x, dim, (-2, -1))
    return y


def myifftshift(x):
    shp = x.shape[-2:]
    dim = [(s + 1) // 2 for s in shp]
    y = torch.roll(x, dim, (-2, -1))
    return y


def img2gif(img_path, gif_name):
    pic_lst = os.listdir(img_path)
    gif_images = []
    for name in pic_lst:
        filename = os.path.join(img_path, name)
        gif_images.append(imageio.imread(filename))

    imageio.mimsave(gif_name, gif_images, 'GIF', duration=0.5)
