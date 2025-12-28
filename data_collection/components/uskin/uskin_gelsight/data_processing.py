import sys
import math
import yaml
import os
import cv2
import pandas as pd
import numpy as np
import nidaqmx
import atiiaftt
import datetime
from time import sleep
from tqdm import tqdm


# save ft to a csv file
class FT_Sensor():
    def __init__(self,task,dir_ft,baseline_count=50):
        # FTsensor Configuration
        self.instance = atiiaftt.FTSensor()
        self.instance.createCalibration(CalFilePath='components/ur5e/Nano17/FT31439.cal', index=1)
        self.instance.setToolTransform([0, 0, 0, 0, 0, 0], atiiaftt.FTUnit.DIST_MM, atiiaftt.FTUnit.ANGLE_DEG)
        self.task = task
        self.baseline_count = baseline_count
        self.dir_ft = dir_ft
        self.baseline = self.baselineF()

    def ft_readout_generator(self):
        ee = (np.array(self.task.read()) - np.array(self.baseline)).tolist()
        self.instance.convertToFt(ee)
        Ft_readout = self.instance.ft_vector
        return Ft_readout    

    def baselineF(self):
        base_sum = np.zeros(6)
        for _ in range(self.baseline_count):
            base_sum = base_sum + np.array(self.task.read())
        base_line = (base_sum / self.baseline_count).tolist()
        return base_line
    
    def save(self,c_dir):

        ft_path = os.path.join(c_dir,'ft.csv')
        ft_readout = self.ft_readout_generator()
        data = {
                'Fx(N)': [ft_readout[0]],
                'Fy(N)': [ft_readout[1]],
                'Fz(N)': [ft_readout[2]],
                'Tx(Nmm)': [ft_readout[3]],
                'Ty(Nmm)': [ft_readout[4]],
                'Tz(Nmm)': [ft_readout[5]]
            }
        if os.path.exists(ft_path):
            existing_df = pd.read_csv(ft_path)
        else:
            existing_df = pd.DataFrame()
        additional_df = pd.DataFrame(data)
        updated_df = pd.concat([existing_df, additional_df], ignore_index=True)
        updated_df.to_csv(ft_path, index=False)

class GelSight():

    def __init__(self, port, dir_imgs):
        self.video = cv2.VideoCapture(port,cv2.CAP_DSHOW)
        self.dir_imgs = dir_imgs

    def c_ref(self):
        ret, frame = self.video.read()
        if not ret:
            raise RuntimeError("Camera does not run normally.")
        cv2.imwrite(os.path.join(self.dir_imgs,'ref.jpg'), frame)
    
    def save(self,img_path):
        os.makedirs(img_path,exist_ok=True)
        ret, frame = self.video.read()
        if not ret:
            raise RuntimeError("Camera does not run normally.")
        count = len(os.listdir(img_path))
        cv2.imwrite(os.path.join(img_path,f'{count:04d}.jpg'), frame)
    
    def imgShow(self):
        while True:
            ret, frame = self.video.read()
            if not ret:
                raise RuntimeError('Camera open fail!')
            cv2.imshow('Tactile Image',frame)
            cv2.waitKey(1)

def img_save(video,img_path):
    os.makedirs(img_path,exist_ok=True)
    ret, frame = video.read()
    if not ret:
        raise RuntimeError("Camera does not run normally.")
    count = len(os.listdir(img_path))
    cv2.imshow('tactile image',frame)
    cv2.waitKey(1)
    cv2.imwrite(os.path.join(img_path,f'{count:04d}.jpg'), frame)






