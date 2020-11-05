import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import math
from GGP.models.cond_refinenet_dilated import CondRefineNetDilated
from torchvision import transforms
from torch.utils.data import DataLoader
from scipy.io import loadmat,savemat
import matplotlib.pyplot as plt
from skimage.measure import compare_psnr,compare_ssim
import glob
import h5py
import time
from scipy.misc import imread,imsave
from scipy.linalg import norm,orth

#import imutils
__all__ = ['GGP_EXE']
class GGP_EXE():
    def __init__(self, args, config):
        self.args = args
        self.config = config
        if not os.path.exists(args.image_folder):
            os.mkdir(args.image_folder)
    def write_images(self,x,image_save_path):
        x = np.array(x,dtype=np.uint8)
        cv2.imwrite(image_save_path, x)
    def test(self):

        # Load the score network
        states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'), map_location=self.config.device)
        scorenet = CondRefineNetDilated(self.config).to(self.config.device)
        scorenet = torch.nn.DataParallel(scorenet, device_ids=[0])
        scorenet.load_state_dict(states[0])
        scorenet.eval()
    
        test_img = imread('./flowers_128.png') # test image

        test_img = np.array(test_img,dtype=np.float32)/255.0 
        print('Creating Phi and measurments')
        
        N = 128#64,32
        # m = 10
        mr = 0.10  				## set sample rate
        m = int(mr*N**2)
        print('Creating Phi and measurments')
        np.random.seed(4)
        Phi = np.zeros((m,N**2,3))
        Phi = loadmat('./phi/phi_10/Phi_10.mat')['Phi']
        test_img_r = np.cast[np.float64](test_img)  ## (0-1)
        y = np.einsum('mnr,ndr->mdr',Phi,test_img_r.reshape(-1,1,3))
        x0 = nn.Parameter(torch.Tensor(1,3,128,128).uniform_(-1,1)).cuda()#128
        x01 = x0

        step_lr=0.00001 

        # Noise amounts
        sigmas = np.array([1., 0.59948425, 0.35938137, 0.21544347, 0.12915497,
                           0.07742637, 0.04641589, 0.02782559, 0.01668101, 0.01])
        n_steps_each = 80
        max_psnr = 0
        max_ssim = 0
        min_hfen = 100
        start_start = time.time()
        for idx, sigma in enumerate(sigmas):
            start_out = time.time()
            print(idx)
            lambda_recon = 1./sigma**2
            labels = torch.ones(1, device=x0.device) * idx
            labels = labels.long()

            step_size = step_lr * (sigma / sigmas[-1]) ** 2
            
            print('sigma = {}'.format(sigma))
            for step in range(n_steps_each):
                start_in = time.time()
                noise1 = torch.rand_like(x0)* np.sqrt(step_size * 2)
                grad1 = scorenet(x01, labels).detach()
                x0 = x0 + step_size * grad1
                x01 = x0 + noise1
                x0=np.array(x0.cpu().detach(),dtype = np.float32)

                x_rec_r = x0.squeeze()[0,:,:]
                x_rec_g = x0.squeeze()[1,:,:]
                x_rec_b = x0.squeeze()[2,:,:]
                
                x_rec = np.stack([x_rec_r,x_rec_g,x_rec_b],axis=2)

                l = 0
                res = y - np.einsum('mnr,ndr->mdr',Phi,x_rec.reshape(-1,1,3))
                l += norm(res)
                x_rec += np.einsum('mnr,ndr->mdr',Phi.transpose([1,0,2]),res).reshape(128,128,3)
                l = l/3
                x_rec   = np.clip(x_rec,0,1)      
                imsave(os.path.join(self.args.image_folder,'x_rec.png'),(255.0*x_rec).astype(np.uint8))
                
                end_in = time.time()
                print("inner run time :%.2f s"%(end_in-start_in))
                psnr = compare_psnr(255*abs(x_rec),255*abs(test_img),data_range=255)
                ssim = compare_ssim(abs(x_rec),abs(test_img),data_range=1,multichannel=True)

                print("current {} step".format(step),'PSNR :', psnr,'SSIM :', ssim)
                
                x_mid = np.zeros([1,3,128,128],dtype=np.float32)

                x_rec = np.transpose(x_rec,[2,0,1])
                
                x0 = torch.tensor(x_rec,dtype=torch.float32).cuda()
            end_out = time.time()
            print("outer run time:%.2f s"%(end_out-start_out))
        
        end_end = time.time()
        print("image reconstruction cost time:%.2f s"%(end_end-start_start))

        
