import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
from glob import glob
from imread_from_url import imread_from_url
import tqdm
import time

from nets import Model
import matplotlib.pyplot as plt

device = 'cuda'
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

#Ref: https://github.com/megvii-research/CREStereo/blob/master/test.py
def inference(left, right, model, n_iter=20):

        #print("Model Forwarding...")
        imgL = left.transpose(2, 0, 1)
        imgR = right.transpose(2, 0, 1)
        imgL = np.ascontiguousarray(imgL[None, :, :, :])
        imgR = np.ascontiguousarray(imgR[None, :, :, :])

        imgL = torch.tensor(imgL.astype("float32")).to(device)
        imgR = torch.tensor(imgR.astype("float32")).to(device)
        #print("imgL:",imgL.dtype,imgL.shape)
        imgL_dw2 = F.interpolate(
                imgL,
                size=(imgL.shape[2] // 2, imgL.shape[3] // 2),
                mode="bilinear",
                align_corners=True,
        )
        imgR_dw2 = F.interpolate(
                imgR,
                size=(imgL.shape[2] // 2, imgL.shape[3] // 2),
                mode="bilinear",
                align_corners=True,
        )
        # print(imgR_dw2.shape)
        with torch.inference_mode():
                pred_flow_dw2 = model(imgL_dw2, imgR_dw2, iters=n_iter, flow_init=None)
                pred_flow = model(imgL, imgR, iters=n_iter, flow_init=pred_flow_dw2)
        pred_disp = torch.squeeze(pred_flow[:, 0, :, :]).cpu().detach().numpy()

        return pred_disp

if __name__ == '__main__':
        dirbase = '/nas/dataset_wavue_camera/wavue_orbbec/W001B/waibao/20240111/matched/2/'
        # dirbase = '/nas/dataset_wavue_camera/wavue_orbbec/W001B/HAND/ORIGIN/20240220/matched/0/'
        # dirbase = '/nas/dataset_wavue_camera/wavue_orbbec/W002A/20231219_1/matched/2/'
        #dirbase = '20231212_test/'
        savebase = dirbase
        model_path = "models/crestereo_eth3d.pth"

        model = Model(max_disp=256, mixed_precision=False, test_mode=True)
        model.load_state_dict(torch.load(model_path), strict=True)
        model.to(device)
        model.eval()
        if not os.path.exists(savebase):
            os.makedirs(savebase)
        rpaths = glob(os.path.join(dirbase,'**/*right_2.png'),recursive=True)
        paths = [path.replace('right_2','2') for path in rpaths]
        # rpaths = glob(os.path.join(dirbase,'**/*right_1.png'),recursive=True)
        # paths = [path.replace('right_1','1') for path in rpaths]
        # rpaths = glob(os.path.join(dirbase,'**/*right_0.png'),recursive=True)
        # paths = [path.replace('right_0','0') for path in rpaths]
        print(paths[:3])
        #rpaths = glob(dirbase+'**/right*.jpg',recursive=True)
        #paths = [path.replace('right','left') for path in rpaths]
        for i,path in tqdm.tqdm(enumerate(paths),total=len(paths)):
            #start = time.time()
            stamp = path.split('/')[-1].split('_')[1]#.split('.')[0]
            # print(stamp)
            left_img = cv2.imread(path)[...,::-1]
            right_img = cv2.imread(rpaths[i])[...,::-1]

            in_h, in_w = left_img.shape[:2]

            # Resize image in case the GPU memory overflows
            eval_h, eval_w = (in_h,in_w)
            assert eval_h%8 == 0, "input height should be divisible by 8"
            assert eval_w%8 == 0, "input width should be divisible by 8"
            
            imgL = cv2.resize(left_img, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)
            imgR = cv2.resize(right_img, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)
            pred = inference(imgL, imgR, model, n_iter=20)
          
            t = float(in_w) / float(eval_w)
            disp = cv2.resize(pred, (in_w, in_h), interpolation=cv2.INTER_LINEAR) * t
            depth = 117*533/disp
            #depth = 117*1063/disp
            #print(depth.shape)
            #np.savetxt(f'{savebase}Raw_Depth_{stamp}_2.txt',depth.astype('uint16'))
            #np.savetxt(f'{savebase}{stamp}.txt',depth)
            depth.astype('uint16').tofile(f'{savebase}Raw_Depth_{stamp}_2.bin')
        #     depth.astype('uint16').tofile(f'{savebase}Raw_Depth_{stamp}_1.bin')
        #     depth.astype('uint16').tofile(f'{savebase}Raw_Depth_{stamp}_0.bin')
            plt.imsave(f"{savebase}Raw_Depth_{stamp}.png", depth.squeeze(), cmap='jet',vmin=0,vmax=2000)    
            #end = time.time()
            #print(f"time elapsed:{end-start}")
            #plt.imshow(depth)
            #plt.show()


