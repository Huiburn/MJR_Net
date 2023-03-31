import numpy as np
import matplotlib.pyplot as plt
import h5py
import misc as sf
from scipy.io import loadmat

if __name__ == '__main__':

    # Train_1 = h5py.File('../data/MatData/Train_1.mat')
    # Train_2 = h5py.File('../data/MatData/Train_2.mat')
    # Train_3 = h5py.File('../data/MatData/Train_3.mat')
    # Train_4 = h5py.File('../data/MatData/Train_4.mat')
    # Train_5 = h5py.File('../data/MatData/Train_5.mat')
    # Train_6 = h5py.File('../data/MatData/Train_6.mat')
    # Train_7 = h5py.File('../data/MatData/Train_7.mat')
    # Train_8 = h5py.File('../data/MatData/Train_8.mat')
    # #
    # train_data=np.transpose(Train_8['Y'])
    # train_data.dtype = 'complex'
    # train_csm=np.transpose(Train_8['C'])
    # train_csm.dtype = 'complex'
    # train_gt = np.transpose(Train_8['GT'])
    # train_gt.dtype = 'complex'
    # # #
    # train_data=np.transpose(train_data,(0,1,5,4,2,3))  # train_data@(Nslice,Ndir,Nshot,Ncoil,Nx,Ny)
    # train_gt = np.transpose(train_gt,(0,1,4,2,3))  # train_gt@(Nslice,Ndir,Nshot,Nx,Ny)
    # train_csm = np.transpose(train_csm,(0,3,1,2))  # train_csm@(Nslice,Ncoil,Nx,Ny)
    # #
    # np.save('train_data_8.npy',train_data)
    # np.save('train_csm_8.npy',train_csm)
    # np.save('train_gt_8.npy',train_gt)


    # ---------transform from mat------------
    # Train = loadmat('/data/disk1/shiboxuan/Data for boxuan/data/Philphs_PAIR_180/Train/Train_8.mat')
    # # Train_2 = loadmat('/data/disk1/shiboxuan/Data for boxuan/data/180_2th_shotPhase/Train/Train_2.mat')
    # # Train_3 = loadmat('/data/disk1/shiboxuan/Data for boxuan/data/180_2th_shotPhase/Train/Train_3.mat')
    # # Train_4 = loadmat('/data/disk1/shiboxuan/Data for boxuan/data/180_2th_shotPhase/Train/Train_4.mat')
    # # Train_5 = loadmat('/data/disk1/shiboxuan/Data for boxuan/data/180_2th_shotPhase/Train/Train_5.mat')
    # # Train_6 = loadmat('/data/disk1/shiboxuan/Data for boxuan/data/180_2th_shotPhase/Train/Train_6.mat')
    # # Train_7 = loadmat('/data/disk1/shiboxuan/Data for boxuan/data/180_2th_shotPhase/Train/Train_7.mat')
    #
    # train_data = Train['Y']
    # train_data = np.transpose(train_data, [0, 1, 5, 4, 2, 3])  # train_data@(Nslice,Ndir,Nshot,Ncoil,Nx,Ny)
    # train_gt = Train['GT']  # train_gt@(Nslice,Ndir,Nshot,Nx,Ny)
    # train_gt = np.transpose(train_gt, [0, 1, 4, 2, 3])
    # train_csm = Train['C']  # train_csm@(Nslice,Ncoil,Nx,Ny)
    # train_csm = np.transpose(train_csm, [0, 3, 1, 2])
    # del Train
    #
    # np.save('train_data_8.npy', train_data)
    # np.save('train_csm_8.npy', train_csm)
    # np.save('train_gt_8.npy', train_gt)

    # ----------preprocess------------
    # train_data_1=np.load('../data/TrainData/train_data_1.npy')
    # train_data_2=np.load('../data/TrainData/train_data_2.npy')
    # train_data_3=np.load('../data/TrainData/train_data_3.npy')
    # train_data_4 = np.load('../data/TrainData/train_data_4.npy')
    # train_data_5 = np.load('../data/TrainData/train_data_5.npy')
    # train_data_6 = np.load('../data/TrainData/train_data_6.npy')
    # train_data = np.concatenate((train_data_1, train_data_2, train_data_3, train_data_4, train_data_5, train_data_6),
    #                             axis=0)
    # np.save('train_data.npy', train_data)
    #
    # train_csm_1 = np.load('../data/TrainData/train_csm_1.npy')
    # train_csm_2 = np.load('../data/TrainData/train_csm_2.npy')
    # train_csm_3 = np.load('../data/TrainData/train_csm_3.npy')
    # train_csm_4 = np.load('../data/TrainData/train_csm_4.npy')
    # train_csm_5 = np.load('../data/TrainData/train_csm_5.npy')
    # train_csm_6 = np.load('../data/TrainData/train_csm_6.npy')
    # train_csm = np.concatenate((train_csm_1, train_csm_2, train_csm_3, train_csm_4, train_csm_5, train_csm_6),
    #                             axis=0)
    # np.save('train_csm.npy', train_csm)
    #
    train_gt_1 = np.load('../data/TrainData/train_gt_1.npy')
    train_gt_2 = np.load('../data/TrainData/train_gt_2.npy')
    train_gt_3 = np.load('../data/TrainData/train_gt_3.npy')
    train_gt_4 = np.load('../data/TrainData/train_gt_4.npy')
    train_gt_5 = np.load('../data/TrainData/train_gt_5.npy')
    train_gt_6 = np.load('../data/TrainData/train_gt_6.npy')
    train_gt = np.concatenate((train_gt_1, train_gt_2, train_gt_3, train_gt_4, train_gt_5, train_gt_6),
                               axis=0)
    np.save('train_gt.npy', train_gt)
# %%
#     b0_1 = h5py.File('/data/disk1/shiboxuan/PythonFile/AlternateUpdate_v3.0/data/MatData/train_b0_1.mat')
#     b0_2 = h5py.File('/data/disk1/shiboxuan/PythonFile/AlternateUpdate_v3.0/data/MatData/train_b0_2.mat')
#     b0_3 = h5py.File('/data/disk1/shiboxuan/PythonFile/AlternateUpdate_v3.0/data/MatData/train_b0_3.mat')
#     b0_4 = h5py.File('/data/disk1/shiboxuan/PythonFile/AlternateUpdate_v3.0/data/MatData/train_b0_4.mat')
#     b0_5 = h5py.File('/data/disk1/shiboxuan/PythonFile/AlternateUpdate_v3.0/data/MatData/train_b0_5.mat')
#     b0_6 = h5py.File('/data/disk1/shiboxuan/PythonFile/AlternateUpdate_v3.0/data/MatData/train_b0_6.mat')
#     b0_7 = h5py.File('/data/disk1/shiboxuan/PythonFile/AlternateUpdate_v3.0/data/MatData/train_b0_7.mat')
#     b0_8 = h5py.File('/data/disk1/shiboxuan/PythonFile/AlternateUpdate_v3.0/data/MatData/train_b0_8.mat')
#
#     train_b0_1 = np.transpose(np.transpose(b0_1['b0img']),[2,0,1])
#     train_b0_2 = np.transpose(np.transpose(b0_2['b0img']),[2,0,1])
#     train_b0_3 = np.transpose(np.transpose(b0_3['b0img']),[2,0,1])
#     train_b0_4 = np.transpose(np.transpose(b0_4['b0img']),[2,0,1])
#     train_b0_5 = np.transpose(np.transpose(b0_5['b0img']),[2,0,1])
#     train_b0_6 = np.transpose(np.transpose(b0_6['b0img']),[2,0,1])
#     train_b0_7 = np.transpose(np.transpose(b0_7['b0img']),[2,0,1])
#     train_b0_8 = np.transpose(np.transpose(b0_8['b0img']),[2,0,1])
#
#     train_b0 = np.concatenate((train_b0_1,train_b0_2,train_b0_3,train_b0_4,train_b0_5,train_b0_6),axis=0)
#     test_b0 = train_b0_7
#     val_b0 = train_b0_8
#     np.save("train_b0.npy", train_b0)
#     np.save("test_b0.npy", test_b0)
#     np.save("val_b0.npy", val_b0)
