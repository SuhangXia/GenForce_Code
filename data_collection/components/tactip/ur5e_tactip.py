import sys
import urx
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


def relativePath(inital_P,stepX,stepY,stepZ,maxZ,angleN,radius,**kwargs):

    def constructPixels():

        pixels = []
        # five points 
        ori = inital_P.copy()
        pix_leftTop= [ori[0]-stepX,ori[1]+stepY,ori[2],ori[3],ori[4],ori[5]]
        pix_rightTop= [ori[0]-stepX,ori[1]-stepY,ori[2],ori[3],ori[4],ori[5]]
        pix_rightBottom= [ori[0]+stepX,ori[1]-stepY,ori[2],ori[3],ori[4],ori[5]]
        pix_leftBottom= [ori[0]+stepX,ori[1]+stepY,ori[2],ori[3],ori[4],ori[5]]
        ###
        pixels.append(ori)
        ##modulus comment from here
        pixels.append(pix_leftTop)
        pixels.append(pix_rightTop)
        pixels.append(pix_rightBottom)
        pixels.append(pix_leftBottom)
        return pixels
    
    def genCirclePoints(center,angleN,radius):

        cirPoints=[]
        for i in range(angleN):
            p = center.copy()
            theta = 2.0 * math.pi * i / angleN
            x = radius * math.cos(theta)
            y = radius * math.sin(theta)
            p[0] += x
            p[1] += y
            cirPoints.append(p)
        return cirPoints
    
    # pixel with initial points
    pixels = constructPixels()
    allPoints = []
    pixelPoints = []
    # add all pixels in depth
    for pixel in pixels:
        #add initial point
        if pixel == pixels[0]:
            pixelPoints.append(pixel)
        else:
            pixel[2] -= 0.0015
            pixelPoints.append(pixel)
        #add points with circle
        for j in np.arange(stepZ,maxZ+stepZ,stepZ):
            center = pixel.copy()
            center[2] -= j
            cirPoints = genCirclePoints(center,angleN,radius)
            pixelPoints.extend(cirPoints)
        allPoints.append(pixelPoints)
        pixelPoints = []
    allPoints = np.array(allPoints)
    print(f"Total Point: {allPoints.shape[0]*allPoints.shape[1]}")
    return allPoints

# save ft to a csv file
class FT_Sensor():
    def __init__(self,task,dir_ft,baseline_count=50):
        # FTsensor Configuration
        self.instance = atiiaftt.FTSensor()
        self.instance.createCalibration(CalFilePath='components/nano17/FT31439.cal', index=1)
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
        self.video.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 1 = manual mode for DirectShow  
        self.video.set(cv2.CAP_PROP_EXPOSURE,-7) 
        self.video.set(cv2.CAP_PROP_BRIGHTNESS, 100) 
        self.video.set(cv2.CAP_PROP_CONTRAST, 64) 
        self.video.set(cv2.CAP_PROP_SHARPNESS, 2) 
        # current_exposure = video.get(cv2.CAP_PROP_EXPOSURE)  
        # current_brightness = video.get(cv2.CAP_PROP_BRIGHTNESS)  
        # current_contrast = video.get(cv2.CAP_PROP_CONTRAST)  
        # current_sharpness = video.get(cv2.CAP_PROP_SHARPNESS)
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
            cropped = frame[50:380,180:510]  # [y1:y2, x1:x2]  
            img_crop = cv2.resize(cropped, (640, 480))
            if not ret:
                raise RuntimeError('Camera open fail!')
            cv2.imshow('Tactile Image',img_crop)
            cv2.waitKey(1)

def img_save(video,img_path):
    os.makedirs(img_path,exist_ok=True)
    ret, frame = video.read()
    cropped = frame[50:380,180:510]  # [y1:y2, x1:x2] 
    img_crop = cv2.resize(cropped, (640, 480))
    if not ret:
        raise RuntimeError("Camera does not run normally.")
    count = len(os.listdir(img_path))
    cv2.imshow('tactile image',img_crop)
    cv2.waitKey(1)
    cv2.imwrite(os.path.join(img_path,f'{count:04d}.jpg'), img_crop)

def pose_save(rob, pos_path):
    pose = rob.getl()
    pose_path = os.path.join(pos_path,'pose.csv')
    data = {
            'X(m)': [pose[0]],
            'Y(m)': [pose[1]],
            'Z(m)': [pose[2]],
            'RX(rad)': [pose[3]],
            'RY(rad)': [pose[4]],
            'RZ(rad)': [pose[5]]
        }
    if os.path.exists(pose_path):
        existing_df = pd.read_csv(pose_path)
    else:
        existing_df = pd.DataFrame()
    additional_df = pd.DataFrame(data)
    updated_df = pd.concat([existing_df, additional_df], ignore_index=True)
    updated_df.to_csv(pose_path, index=False)


def record(port,ft,rob,img_path,ft_path, pos_path, ban_img=False):
    pose_save(rob, pos_path)
    ft.save(ft_path)
    if not ban_img:
        img_save(port,img_path)

def init_robo(ip):
    rob = urx.Robot(ip)
    rob.set_tcp((0, 0, 0.1, 0, 0, 0))
    rob.set_payload(2, (0, 0, 0.1))
    sleep(0.2)
    return rob

def creat_dir(ft_root,img_root,pos_root,name):
    ft_dir = os.path.join(ft_root,name)
    os.makedirs(ft_dir,exist_ok=True)
    img_dir = os.path.join(img_root,name)
    os.makedirs(ft_dir,exist_ok=True)
    pos_dir = os.path.join(pos_root,name)
    os.makedirs(pos_dir,exist_ok=True)
    return ft_dir, img_dir, pos_dir

def video_set(video):

    video.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 1 = manual mode for DirectShow  
    video.set(cv2.CAP_PROP_EXPOSURE,-7) 
    video.set(cv2.CAP_PROP_BRIGHTNESS, 100) 
    video.set(cv2.CAP_PROP_CONTRAST, 64) 
    video.set(cv2.CAP_PROP_SHARPNESS, 2) 

    for i in range(5):
        ret, frame = video.read()
            

def robo_run(port,ft,rob,allPoints,dir_ft,dir_imgs,dir_pos,a=0.001,v=0.01,frequency=60):
    
    count = 1
    for idx_pixel,pixel in enumerate(allPoints):
        # if idx_pixel!=4:continue
        for i in tqdm(range(len(pixel))):
            if i==0:
                ori = pixel[i].copy()
                # move to pixel origin
                rob.movel(ori, 0.01, 0.5,wait=True)
                print('\nmoved to p0 ',rob.getl()) 
                continue
            # if i<29:continue
            video = cv2.VideoCapture(port,cv2.CAP_DSHOW)
            video_set(video)
            
            ft_root = os.path.join(dir_ft,f'{count}')
            os.makedirs(ft_root,exist_ok=True)
            img_root = os.path.join(dir_imgs,f'{count}')
            os.makedirs(img_root,exist_ok=True)
            pos_root = os.path.join(dir_pos,f'{count}')
            os.makedirs(pos_root,exist_ok=True)
            print(f'runing pixel {idx_pixel+1} - Point {i+1}')

            # move to center (normal force)
            print("Normal +++++")   
            p1 = pixel[i].copy()
            p1[0], p1[1] = ori[0],ori[1]
            ft_normal_inc, img_normal_inc, pos_shear_inc = creat_dir(ft_root,img_root,pos_root,'normal_inc')
            rob.movel(p1, a, v,wait=False)
            while(abs(rob.getl()[2]-p1[2])>3e-5):
                sleep(1/frequency)
                record(video,ft,rob, img_normal_inc,ft_normal_inc, pos_shear_inc)
            while rob.is_program_running():
                continue

            # move to target (shear force increases)
            print("Shear +++++")  
            p2 = pixel[i].copy()
            ft_shear_inc, img_shear_inc, pos_shear_inc = creat_dir(ft_root,img_root,pos_root,'shear_inc')
            rob.movel(p2, a, v,wait=False)
            while(abs(rob.getl()[0]-p2[0])/2+abs(rob.getl()[1]-p2[1])/2>3e-5):
                sleep(1/frequency)
                record(video,ft,rob,img_shear_inc,ft_shear_inc, pos_shear_inc)
            while rob.is_program_running():
                continue

            # move to center (shear force decreases) 
            # p3 = p2.copy()
            # p3[2] = ori[2]
            # rob.movel(p3, a, v,wait=True)
            print("Shear -----")  
            ft_shear_dec, img_shear_dec, pos_shear_dec = creat_dir(ft_root,img_root,pos_root,'shear_dec')
            rob.movel(p1, a, v,wait=False)
            while(abs(rob.getl()[0]-p1[0])/2+abs(rob.getl()[1]-p1[1])/2>3e-5):
                sleep(1/frequency)
                record(video,ft,rob, img_shear_dec,ft_shear_dec, pos_shear_dec)
            while rob.is_program_running():
                continue

            # move to ori (normal force decreases) 
            print("Normal -----")  
            ft_normal_dec, img_normal_dec, pos_normal_dec = creat_dir(ft_root,img_root,pos_root,'normal_dec')
            rob.movel(ori, a, v,wait=False)
            while(abs(rob.getl()[2]-ori[2])>3e-5):
                sleep(1/frequency)
                record(video,ft,rob, img_normal_dec,ft_normal_dec, pos_normal_dec)
            while rob.is_program_running():
                continue
            # move to origin
            # rob.movel(ori, a, v,wait=True)
            # print('moved to p0 ',rob.getl()) 
            count += 1
            video.release()
        print(f"finish Pixel {count}")
    print(f"Done!")
    cv2.destroyAllWindows()

def robo_run_modulus(port,ft,rob,depth,dir_ft,dir_imgs,dir_pose,a=0.001,v=0.01,frequency=60,ban_img=False):
    
    video = cv2.VideoCapture(port,cv2.CAP_DSHOW)
    ft_down = os.path.join(dir_ft,'down')
    os.makedirs(ft_down,exist_ok=True)
    pose_down = os.path.join(dir_pose,'down')
    os.makedirs(pose_down,exist_ok=True)
    img_down = os.path.join(dir_imgs,'down')
    if not ban_img:
        os.makedirs(img_down,exist_ok=True)

    print("down++++++")
    ori = rob.getl()
    target = list(ori)
    target[2] -= depth
    rob.movel(target, a, v,wait=False)
    while(abs(rob.getl()[2]-target[2])>3e-5):
        sleep(1/frequency)
        record(video,ft,rob,img_down,ft_down,pose_down, ban_img=ban_img)
    while rob.is_program_running():
        continue

    # move to ori
    print("up++++++")
    ft_up = os.path.join(dir_ft,'up')
    os.makedirs(ft_up,exist_ok=True)
    img_up = os.path.join(dir_imgs,'up')
    if not ban_img:
        os.makedirs(img_up,exist_ok=True)
    pose_up = os.path.join(dir_pose,'up')
    os.makedirs(pose_up,exist_ok=True)
    rob.movel(ori, a, v,wait=False)
    while(abs(rob.getl()[2]-ori[2])>3e-5):
        sleep(1/frequency)
        record(video,ft,rob,img_up,ft_up,pose_up, ban_img=ban_img)

    while rob.is_program_running():
        continue
    video.release()
    print(f"Done!")
    cv2.destroyAllWindows()
    rob.close()
    sys.exit()

if __name__ == "__main__":
    pass
    # with open("components/ur5e/config/config.yaml",'r') as fp:
    #     config = yaml.safe_load(fp)
    # ctime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # dir_ft = f"data/FT_data/{ctime}"
    # dir_imgs = f"data/Images/{ctime}"
    # os.makedirs(dir_ft,exist_ok=True)
    # os.makedirs(dir_imgs,exist_ok=True)

    # rob = init_robo(config['ip'])
    # allPoints = np.array(relativePath(rob.getl(),**config['path']))
    # np.savetxt(f'data/Path/keyPoints_{ctime}.csv',allPoints.reshape(-1,6),delimiter=',',fmt="%f")

    # port = config['GelSight']['port']
    # gelsight = GelSight(port,dir_imgs)
    # gelsight.c_ref()

    # with nidaqmx.Task() as task:

    #     task.ai_channels.add_ai_voltage_chan("Dev1/ai0")
    #     task.ai_channels.add_ai_voltage_chan("Dev1/ai1")
    #     task.ai_channels.add_ai_voltage_chan("Dev1/ai2")
    #     task.ai_channels.add_ai_voltage_chan("Dev1/ai3")
    #     task.ai_channels.add_ai_voltage_chan("Dev1/ai4")
    #     task.ai_channels.add_ai_voltage_chan("Dev1/ai5")
    #     ft = FT_Sensor(task,dir_ft)
    #     record = Record(gelsight,ft,port)
    #     robo_run(record,rob,allPoints,dir_ft,dir_imgs,**config['robo'])






