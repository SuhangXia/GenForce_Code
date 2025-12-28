#!/usr/bin/env python3  
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
import threading  
import websocket  
import json  
import matplotlib.pyplot as plt  
import queue  
import time  
from matplotlib.animation import FuncAnimation  
from matplotlib import use  
from time import sleep  
from tqdm import tqdm  

use('TkAgg')  

# Global Configuration  
ip = "10.70.151.133"  
port = 5000  
lastmessage = {"message": "No message"}  
data_queue = queue.Queue(maxsize=1)  
first_frame = None  
running = True   

# Pre-calculate grid coordinates  
GRID_X = np.array([0, 1, 2, 3] * 4)  
GRID_Y = np.repeat([0, 1, 2, 3], 4)  

# Screen dimensions  
SCREEN_WIDTH = 1920  
SCREEN_HEIGHT = 1080  

def relativePath(inital_P,stepX,stepY,stepZ,maxZ,angleN,radius,onlycenter=0,**kwargs):

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
        if onlycenter==1:
            if pixel != pixels[0]:continue
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

class FT_Sensor():  
    def __init__(self, task, dir_ft, baseline_count=50):  
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
    
    def save(self, c_dir, ft_readout):  
        ft_path = os.path.join(c_dir, 'ft.csv')   
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

class USKIN:  
    def __init__(self, dir_imgs, dir_raw):  
        plt.style.use('dark_background')
        plt.ion()    
        self.window_width = 640  
        self.window_height = 480  
        self.dir_imgs = dir_imgs  
        self.dir_raw = dir_raw
        # Modified figure size calculation  
        self.dpi = 100  
        self.figwidth = self.window_width / self.dpi  
        self.figheight = self.window_height / self.dpi  
        
        self.fig = plt.figure(figsize=(self.figwidth, self.figheight),  
                            dpi=self.dpi, facecolor='black')  
        self.ax = self.fig.add_subplot(111)  
        self.scatter = self.ax.scatter(GRID_X, GRID_Y, s=100,  
                                     color='white', alpha=1.0)  
        # Set figure size explicitly  
        self.fig.set_size_inches(self.window_width/self.dpi, self.window_height/self.dpi)  
        
        self.setup_plot()   
        self.position_window()  
        # Start WebSocket connection  
        websocket.setdefaulttimeout(1)  
        self.wsapp = websocket.WebSocketApp(f"ws://{ip}:{port}",  
                                          on_message=self.on_message)  
        
        # Start threads  
        self.ws_thread = threading.Thread(target=self.wsapp.run_forever,  
                                        daemon=True)  
        # self.mes_thread = threading.Thread(target=self.mesreader,  
        #                                  daemon=True)  
        self.ws_thread.start()  
        # self.mes_thread.start()  
        self.get_first_frame()

        self.fig.show()  

    def position_window(self):  
        x_position = max(0, 3 * SCREEN_WIDTH // 4 - self.window_width // 2)  
        y_position = max(0, SCREEN_HEIGHT // 2)  
        
        mng = plt.get_current_fig_manager()  
        try:  
            mng.window.wm_geometry(f"+{x_position}+{y_position}")  
        except:  
            try:  
                mng.window.setGeometry(x_position, y_position,  
                                     self.window_width, self.window_height)  
            except:  
                print("Could not position matplotlib window")  

    def setup_plot(self):  
        self.ax.set_xlim(-1, 4)  
        self.ax.set_ylim(-1, 4)  
        self.ax.set_xticks([])  
        self.ax.set_yticks([])  
        self.ax.set_xticklabels([])  
        self.ax.set_yticklabels([])  
        self.ax.set_facecolor('black')  
        
        # Remove padding  
        self.ax.set_position([0, 0, 1, 1])  
        
        for spine in self.ax.spines.values():  
            spine.set_visible(False)  

    def on_message(self, wsapp, message):  
        global lastmessage  
        try:  
            lastmessage = json.loads(message)  
        except Exception as e:  
            print(f"Error in on_message: {e}")  
    
    def get_first_frame(self):

        global lastmessage, first_frame 
        flag = True
        while flag:
            if lastmessage.get("message") != "No message" and '1' in lastmessage:  
                data = [int(d, 16) for d in lastmessage['1']['data'].split(",")]  
                points = np.array(data).reshape(16, 3)  

                if first_frame is None:  
                    first_frame = points.copy()  
                    print("First frame captured") 
                    np.save(os.path.join(self.dir_raw,"ref.npy"),first_frame) 
                    flag = False


    # def mesreader(self):  
    #     global first_frame, running  
    #     while running:  
    #         try:  
    #             if lastmessage.get("message") != "No message" and '1' in lastmessage:  
    #                 data = [int(d, 16) for d in lastmessage['1']['data'].split(",")]  
    #                 points = np.array(data).reshape(16, 3)  

    #                 if first_frame is None:  
    #                     first_frame = points.copy()  
    #                     print("First frame captured")  
    #                 else:  
    #                     try:  
    #                         data_queue.put_nowait(points - first_frame)  
    #                     except queue.Full:  
    #                         try:  
    #                             data_queue.get_nowait()  
    #                             data_queue.put_nowait(points - first_frame)  
    #                         except queue.Empty:  
    #                             pass  
    #             time.sleep(0.001)  
    #         except Exception as e:  
    #             print(f"Error in mesreader: {e}")  

    def get_cur_uskin_data(self):
        global first_frame
        try:
            if lastmessage.get("message") != "No message" and '1' in lastmessage:  
                data = [int(d, 16) for d in lastmessage['1']['data'].split(",")]  
                points = np.array(data).reshape(16, 3)  
                points = points - first_frame
                time.sleep(0.001)
        except Exception as e:
            print(f"Error in mesreader: {e}") 
        return points


    def update(self,points):  
        delta_x = np.clip(points[:, 0]/2000, -0.8, 0.8)  
        delta_y = np.clip(points[:, 1]/2000, -0.8, 0.8)  
        plot_x = GRID_X + delta_x  
        plot_y = GRID_Y + delta_y  
        sizes = 100 + points[:, 2]    #4
        sizes = np.clip(sizes, 100, 6000)  #4000
        plot_x = np.clip(plot_x, -0.6, 3.6)  
        plot_y = np.clip(plot_y, -0.6, 3.6)  
        
        self.scatter.set_offsets(np.c_[plot_x, plot_y])  
        self.scatter.set_sizes(sizes)  
        
        self.fig.canvas.draw()
        # self.fig.canvas.draw_idle()  
        self.fig.canvas.flush_events()  
        return self.scatter,  

    def save(self, raw_path, img_path, uskin_points):  
        os.makedirs(img_path, exist_ok=True) 
        count = len(os.listdir(img_path)) 
        np.save(os.path.join(raw_path,f'{count:04d}.npy'),uskin_points) 
        self.update(uskin_points) 
        self.fig.set_size_inches(self.window_width/self.dpi, self.window_height/self.dpi) 
        plt.savefig(os.path.join(img_path, f'{count:04d}.jpg'),  
                   facecolor='black',  
                   edgecolor='none',  
                   bbox_inches=None,  # Changed from 'tight' to None  
                   pad_inches=0,  
                   format='jpg',  
                   dpi=self.dpi)  
        

def pose_save(pos_path,pose):  
    pose_path = os.path.join(pos_path, 'pose.csv')  
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

def record(Uskin, ft, rob, img_path, ft_path, pos_path, raw_path):  
    # time_start = time.time()
    pose = rob.getl() 
    ft_value = ft.ft_readout_generator()  
    # uskin_points = data_queue.get_nowait()  
    uskin_points =  Uskin.get_cur_uskin_data()
    pose_save(pos_path,pose) 
    ft.save(ft_path,ft_value)  
    Uskin.save(raw_path,img_path,uskin_points)  
    # time_end = time.time()
    # print(f"time cost saving:{time_end-time_start}")

def init_robo(ip):  
    rob = urx.Robot(ip)  
    rob.set_tcp((0, 0, 0.1, 0, 0, 0))  
    rob.set_payload(2, (0, 0, 0.1))  
    sleep(0.2)  
    return rob  

def creat_dir(ft_root, img_root, pos_root, uskin_raw_root, name):  
    ft_dir = os.path.join(ft_root, name)  
    os.makedirs(ft_dir, exist_ok=True)  
    img_dir = os.path.join(img_root, name)  
    os.makedirs(img_dir, exist_ok=True)  
    pos_dir = os.path.join(pos_root, name)  
    os.makedirs(pos_dir, exist_ok=True)  
    uskin_raw_dir = os.path.join(uskin_raw_root, name)  
    os.makedirs(uskin_raw_dir, exist_ok=True)  
    return ft_dir, img_dir, pos_dir, uskin_raw_dir  

def cleanup(rob, Uskin):  
    global running  
    running = False  
    plt.close('all')  
    rob.close()  
    sys.exit()  

def sleep_ms(sleep_time=25):
    time_s = time.time()
    time_e = time.time()
    while ((time_e-time_s)*1000<sleep_time):
        time_e = time.time() 

def robo_run(Uskin, ft, rob, allPoints, dir_ft, dir_imgs, dir_pos, dir_uskin_raw, a=0.001, v=0.01, frequency=60):  
    count = 1  
    sleep_time = (1/frequency)*1000
    for idx_pixel, pixel in enumerate(allPoints):  
        for i in tqdm(range(len(pixel))):  
            if i == 0:  
                ori = pixel[i].copy()  
                rob.movel(ori, 0.01, 0.5, wait=True)  
                print('\nmoved to p0 ', rob.getl())  
                continue  
            ft_root = os.path.join(dir_ft, f'{count}')  
            os.makedirs(ft_root, exist_ok=True)  
            img_root = os.path.join(dir_imgs, f'{count}')  
            os.makedirs(img_root, exist_ok=True)  
            pos_root = os.path.join(dir_pos, f'{count}')  
            os.makedirs(pos_root, exist_ok=True)  
            uskin_raw_root = os.path.join(dir_uskin_raw, f'{count}') 
            os.makedirs(uskin_raw_root, exist_ok=True)  
            print(f'runing pixel {idx_pixel+1} - Point {i+1}') 
            
            print("Normal +++++")   
            p1 = pixel[i].copy()  
            p1[0], p1[1] = ori[0], ori[1]  
            ft_normal_inc, img_normal_inc, pos_normal_inc, raw_normal_inc = creat_dir(ft_root, img_root, pos_root, uskin_raw_root, 'normal_inc')  
            rob.movel(p1, a, v, wait=False)  
            while(abs(rob.getl()[2]-p1[2]) > 3e-5):  
                record(Uskin, ft, rob, img_normal_inc, ft_normal_inc, pos_normal_inc, raw_normal_inc) 
                sleep_ms(sleep_time) 
            while rob.is_program_running():  
                continue  
            
            print("Shear +++++")  
            p2 = pixel[i].copy()  
            ft_shear_inc, img_shear_inc, pos_shear_inc, raw_shear_inc = creat_dir(ft_root, img_root, pos_root, uskin_raw_root, 'shear_inc')  
            rob.movel(p2, a, v, wait=False)  
            while(abs(rob.getl()[0]-p2[0])/2+abs(rob.getl()[1]-p2[1])/2>3e-5): 
                record(Uskin, ft, rob, img_shear_inc, ft_shear_inc, pos_shear_inc,raw_shear_inc)  
                sleep_ms(sleep_time)
            while rob.is_program_running():  
                continue  
            
            print("Shear -----")  
            ft_shear_dec, img_shear_dec, pos_shear_dec, raw_shear_dec = creat_dir(ft_root, img_root, pos_root, uskin_raw_root, 'shear_dec')  
            rob.movel(p1, a, v, wait=False)  
            while(abs(rob.getl()[0]-p1[0])/2+abs(rob.getl()[1]-p1[1])/2>3e-5):  
                record(Uskin, ft, rob, img_shear_dec, ft_shear_dec, pos_shear_dec, raw_shear_dec)  
                sleep_ms(sleep_time)
            while rob.is_program_running():  
                continue  

            print("Normal -----")  
            ft_normal_dec, img_normal_dec, pos_normal_dec, raw_normal_dec = creat_dir(ft_root, img_root, pos_root, uskin_raw_root, 'normal_dec')  
            rob.movel(ori, a, v, wait=False)  
            while(abs(rob.getl()[2]-ori[2]) > 3e-5):  
                record(Uskin, ft, rob, img_normal_dec, ft_normal_dec, pos_normal_dec, raw_normal_dec)  
                sleep_ms(sleep_time) 
            print('moved to ori ', rob.getl())  
            while rob.is_program_running():  
                continue  

            sleep_ms(sleep_time*5) 
            count += 1  

    print(f"Done!")  
    cleanup(rob, Uskin)

def robo_run_modulus(Uskin, ft, rob, depth, dir_ft, dir_imgs, dir_pos, ban_img=True, a=0.001, v=0.01, frequency=60):  
    ori = rob.getl()  
    target = ori.copy()  
    target[2] -= depth  
    sleep_time = (1/frequency)*1000

    # Move down (normal force) 
    print("Normal +++++")  
    ft_normal_inc, img_normal_inc, pos_normal_inc = creat_dir(dir_ft, dir_imgs, dir_pos, 'normal_inc')  
    rob.movel(target, a, v, wait=False)  
    while(abs(rob.getl()[2]-target[2]) > 3e-5):  
        sleep_ms(sleep_time) 
        record(Uskin, ft, rob, img_normal_inc, ft_normal_inc, pos_normal_inc)  
    # print('moved to target ', rob.getl())  
    while rob.is_program_running():  
        continue  

    # Move up (normal force)  
    print("Normal -----")  
    ft_normal_dec, img_normal_dec, pos_normal_dec = creat_dir(dir_ft, dir_imgs, dir_pos, 'normal_dec')  
    rob.movel(ori, a, v, wait=False)  
    while(abs(rob.getl()[2]-ori[2]) > 3e-5):  
        sleep_ms(sleep_time) 
        record(Uskin, ft, rob, img_normal_dec, ft_normal_dec, pos_normal_dec)  
    # print('moved to ori ', rob.getl())  
    while rob.is_program_running():  
        continue  

    print(f"Done!")  
    cleanup(rob, Uskin)  

if __name__ == '__main__':  
    sensor = "hetero/uskin"  
    with open(f"components/uskin/config/uskin.yaml",'r') as fp:  
        config = yaml.safe_load(fp)  
    ctime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  
    indenter = config["Indenter"]  
    dir_ft = f"data/{sensor}/force/{indenter}-{ctime}"  
    dir_imgs = f"data/{sensor}/image/{indenter}-{ctime}"  
    dir_pos = f"data/{sensor}/pose/{indenter}-{ctime}"  
    dir_path = f'data/{sensor}/path/{indenter}-{ctime}'  
    dir_uskin_raw = f'data/{sensor}/uskin_raw/{indenter}-{ctime}'  
    os.makedirs(dir_ft, exist_ok=True)  
    os.makedirs(dir_imgs, exist_ok=True)  
    os.makedirs(dir_pos, exist_ok=True)  
    os.makedirs(dir_path, exist_ok=True)  
    os.makedirs(dir_uskin_raw, exist_ok=True)  

    rob = init_robo(config['ip'])  
    allPoints = relativePath(rob.getl(), **config['path'])  
    np.savetxt(f'{dir_path}/path.csv', allPoints.reshape(-1,6), delimiter=',', fmt="%f")  

    # Initialize Uskin  
    Uskin = USKIN(dir_imgs,dir_uskin_raw)  
    # ani = FuncAnimation(Uskin.fig, Uskin.update,  
    #                    interval=20,  
    #                    cache_frame_data=False)  

    collect_calib = True  
    collect_modulus = False

    with nidaqmx.Task() as task:  
        task.ai_channels.add_ai_voltage_chan("Dev1/ai0")  
        task.ai_channels.add_ai_voltage_chan("Dev1/ai1")  
        task.ai_channels.add_ai_voltage_chan("Dev1/ai2")  
        task.ai_channels.add_ai_voltage_chan("Dev1/ai3")  
        task.ai_channels.add_ai_voltage_chan("Dev1/ai4")  
        task.ai_channels.add_ai_voltage_chan("Dev1/ai5")  
        ft = FT_Sensor(task, dir_ft)  

        t_start = time.time()  
        try:  
            if collect_calib:   
                robo_run(Uskin, ft, rob, allPoints, dir_ft, dir_imgs, dir_pos, dir_uskin_raw, **config['robo'])  
            elif collect_modulus:  
                robo_run_modulus(Uskin, ft, rob, config["modulus"]["depth"],   
                               dir_ft, dir_imgs, dir_pos, ban_img=True, **config['robo'])  
        finally:  
            t_end = time.time()  
            t_cost = abs(t_start-t_end)/60  
            print(f"finished: {t_cost} mins")  
            running = False   
            Uskin.wsapp.close()  
            plt.close('all')