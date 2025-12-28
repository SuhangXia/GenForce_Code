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

os.chdir(os.path.dirname(__file__))  

# save ft to a csv file  
class FT_Sensor():  
    def __init__(self,task,dir_ft,baseline_count=50):  
        # FTsensor Configuration  
        self.instance = atiiaftt.FTSensor()  
        self.instance.createCalibration(CalFilePath='config/FT31439.cal', index=1)  
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
    
    def save(self):  
        ft_path = os.path.join(self.dir_ft,'ft.csv')  
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
    
    def save(self):  
        os.makedirs(self.dir_imgs,exist_ok=True)  
        ret, frame = self.video.read()  
        if not ret:  
            raise RuntimeError("Camera does not run normally.")  
        count = len(os.listdir(self.dir_imgs))  
        cv2.imwrite(os.path.join(self.dir_imgs,f'{count:04d}.jpg'), frame)  
        return count
    
    def imgShow(self):  
        while True:  
            ret, frame = self.video.read()  
            if not ret:  
                raise RuntimeError('Camera open fail!')  
            cv2.imshow('Tactile Image',frame)  
            cv2.waitKey(1)  

def record(ft, gelsight):  
    
    while True:  
        # Show live camera feed  
        ret, frame = gelsight.video.read()  
        if not ret:  
            raise RuntimeError("Camera does not run normally.")  
        cv2.imshow('Tactile Image', frame)  
        key = cv2.waitKey(1) & 0xFF  
        if key == ord('s'):  
            ft.save()  
            count=gelsight.save()  
            print(f"recorded - {count}")  
        elif key == ord('q'):  
            break  
    gelsight.video.release()  
    cv2.destroyAllWindows()  

if __name__ == '__main__':  
    sensor = "gelsight"  
    with open("config/hetero_img_force.yaml", 'r') as fp:  
        config = yaml.safe_load(fp)  
    # Create timestamp and directories  
    ctime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  
    indenter = config["Indenter"]  
    dir_ft = f"data/img_force/{sensor}/force/{indenter}-{ctime}"  
    dir_imgs = f"data/img_force/{sensor}/image/{indenter}-{ctime}"  
    os.makedirs(dir_ft, exist_ok=True)  
    os.makedirs(dir_imgs, exist_ok=True)  

    # Initialize GelSight camera  
    port = config['GelSight']['port']  
    gelsight = GelSight(port, dir_imgs)  

    # Initialize F/T sensor  
    with nidaqmx.Task() as task:  
        # Configure DAQ channels  
        task.ai_channels.add_ai_voltage_chan("Dev1/ai0")  
        task.ai_channels.add_ai_voltage_chan("Dev1/ai1")  
        task.ai_channels.add_ai_voltage_chan("Dev1/ai2")  
        task.ai_channels.add_ai_voltage_chan("Dev1/ai3")  
        task.ai_channels.add_ai_voltage_chan("Dev1/ai4")  
        task.ai_channels.add_ai_voltage_chan("Dev1/ai5")  
        
        # Initialize F/T sensor  
        ft = FT_Sensor(task, dir_ft)  
        
        try:  
            # Start recording  
            record(ft, gelsight) 
        except Exception as e:  
            print(f"Error occurred: {e}")  
        finally:  
            # Ensure proper cleanup  
            gelsight.video.release()  
            cv2.destroyAllWindows()