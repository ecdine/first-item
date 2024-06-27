import os
from scipy import io
import numpy as np
import torch
from torch.utils.data import Dataset as dataset
import pywt
from collections import OrderedDict
import scipy.fft as fft
import h5py
from torch.utils.data import DataLoader
import torch.nn as nn
class WifiPoseDataset(dataset):
    CLASSES = ('person', )
    def __init__(self, dataset_root, mode, **kwargs):
        
        self.data_root = dataset_root
        self.filename_list = self.load_file_name_list(os.path.join(self.data_root, mode + '_data_list.txt'))
        self._set_group_flag()
        
    def pre_pipeline(self, results):
        results['seg_fields'] = []
        results['img_prefix'] = self.img_dir

    def get_item_single_frame(self,index): 
        data_name = self.filename_list[index]
        csi_path = os.path.join(self.data_root,'csi',(str(data_name)+'.mat'))
        keypoint_path = os.path.join(self.data_root,'keypoint',(str(data_name)+'.npy'))
        with h5py.File(csi_path, 'r') as file:
            grouped_data = file['grouped_data']
            real_part = torch.from_numpy(grouped_data['real'][()])  # 将数据转换为 PyTorch 张量
            imag_part = torch.from_numpy(grouped_data['imag'][()])  # 将数据转换为 PyTorch 张量
            csi = torch.complex(real_part, imag_part).to(torch.complex128)
            csi = csi.unsqueeze(0).repeat(3, 1, 1, 1)  # 这行代码替代了 np.tile
            csi = csi.permute(0, 3, 2, 1)
        csi_amp = self.dwt_amp(csi)
        csi_ph = self.phase_deno(csi)
        csi_ph = np.angle(csi_ph)
        csi = np.concatenate((csi_amp, csi_ph), axis=2)#3*9*60*50
        #csi = torch.FloatTensor(csi)
        
        '''csi_amp = self.dwt_amp(csi)
        csi = torch.FloatTensor(csi_amp)'''
        
        #csi = torch.cat((csi_amp, csi_ph), 2)
        csi = np.reshape(csi, (27, 60, 50))#27*60*50
        csi = torch.FloatTensor(csi).permute(1,0,2)#60channel*27个数*50数据包

        keypoint = np.array(np.load(keypoint_path))
        #keypoint = self.keypoint_process(keypoint)
        keypoint = torch.FloatTensor(keypoint) # keypoint tensor: (N*14*3)(应该是17个点N*17*3)

        numOfPerson = keypoint.shape[0]
        gt_labels = np.zeros(numOfPerson, dtype=np.int64) #label (N,)

        gt_bboxes = torch.zeros(numOfPerson,4)
        result = dict(img=csi, gt_keypoints=keypoint, gt_labels = gt_labels, gt_bboxes = gt_bboxes,  img_name = data_name)
        return result
    
    def __len__(self):
        return len(self.filename_list)
    
    def __getitem__(self, index):
        result = self.get_item_single_frame(index)
        return result

    def load_file_name_list(self, file_path):
        file_name_list = []
        with open(file_path, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline().strip()  
                if not lines:
                    break
                file_name_list.append(lines.split()[0])
        return file_name_list

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def CSI_sanitization(self, csi_rx):
        one_csi = csi_rx[0,:,:]
        two_csi = csi_rx[1,:,:]
        three_csi = csi_rx[2,:,:]
        four_csi = csi_rx[3,:,:]
        five_csi = csi_rx[4,:,:]
        six_csi = csi_rx[5,:,:]
        seven_csi = csi_rx[6,:,:]
        eight_csi = csi_rx[7,:,:]
        nine_csi = csi_rx[8,:,:]
        pi = np.pi
        M = 9  # 天线数量3
        N = 30  # 子载波数目30
        T = 50  # 总包数
        fi = 312.5 * 2  # 子载波间隔312.5 * 2
        csi_phase = np.zeros((M, N, T))
        for t in range(T):  # 遍历时间戳上的CSI包，每根天线上都有30个子载波
            csi_phase[0, :, t] = np.unwrap(np.angle(one_csi[:, t]))
            csi_phase[1, :, t] = np.unwrap(csi_phase[0, :, t] + np.angle(two_csi[:, t] * np.conj(one_csi[:, t])))
            csi_phase[2, :, t] = np.unwrap(csi_phase[1, :, t] + np.angle(three_csi[:, t] * np.conj(two_csi[:, t])))
            csi_phase[3, :, t] = np.unwrap(csi_phase[2, :, t] + np.angle(four_csi[:, t] * np.conj(three_csi[:, t])))
            csi_phase[4, :, t] = np.unwrap(csi_phase[3, :, t] + np.angle(five_csi[:, t] * np.conj(four_csi[:, t])))
            csi_phase[5, :, t] = np.unwrap(csi_phase[4, :, t] + np.angle(six_csi[:, t] * np.conj(five_csi[:, t])))
            csi_phase[6, :, t] = np.unwrap(csi_phase[5, :, t] + np.angle(seven_csi[:, t] * np.conj(six_csi[:, t])))
            csi_phase[7, :, t] = np.unwrap(csi_phase[6, :, t] + np.angle(eight_csi[:, t] * np.conj(seven_csi[:, t])))
            csi_phase[8, :, t] = np.unwrap(csi_phase[7, :, t] + np.angle(nine_csi[:, t] * np.conj(eight_csi[:, t])))
            ai = np.tile(2 * pi * fi * np.array(range(N)), M)
            bi = np.ones(M * N)
            ci = np.concatenate((csi_phase[0, :, t], csi_phase[1, :, t], csi_phase[2, :, t],csi_phase[3, :, t],csi_phase[4, :, t],csi_phase[5, :, t],csi_phase[6, :, t],csi_phase[7, :, t],csi_phase[8, :, t]))
            A = np.dot(ai, ai)
            B = np.dot(ai, bi)
            C = np.dot(bi, bi)
            D = np.dot(ai, ci)
            E = np.dot(bi, ci)
            rho_opt = (B * E - C * D) / (A * C - B ** 2)
            beta_opt = (B * D - A * E) / (A * C - B ** 2)
            temp = np.tile(np.array(range(N)), M).reshape(M, N)
            csi_phase[:, :, t] = csi_phase[:, :, t] + 2 * pi * fi * temp * rho_opt + beta_opt
        antennaPair_One = abs(one_csi) * np.exp(1j * csi_phase[0, :, :])
        antennaPair_Two = abs(two_csi) * np.exp(1j * csi_phase[1, :, :])
        antennaPair_Three = abs(three_csi) * np.exp(1j * csi_phase[2, :, :])
        antennaPair_Four = abs(four_csi) * np.exp(1j * csi_phase[3, :, :])
        antennaPair_Five = abs(five_csi) * np.exp(1j * csi_phase[4, :, :])
        antennaPair_Six = abs(six_csi) * np.exp(1j * csi_phase[5, :, :])
        antennaPair_Seven = abs(seven_csi) * np.exp(1j * csi_phase[6, :, :])
        antennaPair_Eight = abs(eight_csi) * np.exp(1j * csi_phase[7, :, :])
        antennaPair_Nine = abs(nine_csi) * np.exp(1j * csi_phase[8, :, :])
        antennaPair = np.concatenate((np.expand_dims(antennaPair_One,axis=0), 
                                      np.expand_dims(antennaPair_Two,axis=0), 
                                      np.expand_dims(antennaPair_Three,axis=0),
                                      np.expand_dims(antennaPair_Four,axis=0),
                                      np.expand_dims(antennaPair_Five,axis=0),
                                      np.expand_dims(antennaPair_Six,axis=0),
                                      np.expand_dims(antennaPair_Seven,axis=0),
                                      np.expand_dims(antennaPair_Eight,axis=0),
                                      np.expand_dims(antennaPair_Nine,axis=0),))
        return antennaPair


    def phase_deno(self, csi):
        #input csi shape (3*3*30*20)        
        ph_rx1 = self.CSI_sanitization(csi[0,:,:,:])
        ph_rx2 = self.CSI_sanitization(csi[1,:,:,:])
        ph_rx3 = self.CSI_sanitization(csi[2,:,:,:])
        csi_phde = np.concatenate((np.expand_dims(ph_rx1,axis=0), 
                                   np.expand_dims(ph_rx2,axis=0), 
                                   np.expand_dims(ph_rx3,axis=0),))
        #csi_phde = csi_phde.transpose(0,1,3,2)
        return csi_phde
    
    def dwt_amp(self, csi):
        w = pywt.Wavelet('dB11')
        list = pywt.wavedec(abs(csi), w,'sym')
        csi_amp = pywt.waverec(list, w)
        return csi_amp
dataset = WifiPoseDataset(dataset_root='/megadisk/fanghengyu/XRF55/opera/datasets/train',mode = 'train')
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
batch_norm = nn.BatchNorm2d(60)
for batch in dataloader:
    img = batch['img']
    batch_norm = nn.BatchNorm2d(60)
    img_normalized = batch_norm(img)
    batch['img'] = img_normalized 
    print(batch['img'][0][0][0])
