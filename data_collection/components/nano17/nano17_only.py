import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import nidaqmx
import atiiaftt
import numpy as np
import pandas as pd
class RealTimeVisual():
    def __init__(self, num_channels, x_timeMax, frames, interval, baseline_count,task):

        # Parameters
        self.num_channels = num_channels
        self.x_timeMax = x_timeMax
        self.frames = frames
        self.interval = interval
        self.baseline_count = baseline_count
        
        # Raw Figure Setting
        self.x_data = np.linspace(0, self.x_timeMax, int(self.x_timeMax*1000/self.interval))
        self.y_data = [np.zeros_like(self.x_data) for _ in range(self.num_channels)]
        self.fig, self.axes = plt.subplots(self.num_channels, 1, sharex=True, figsize=(8, 10))
        self.lines = [self.axes[i].plot([], [])[0] for i in range(self.num_channels)]
                
        # FTsensor Configuration
        self.instance = atiiaftt.FTSensor()
        self.instance.createCalibration(CalFilePath='components/nano17/FT31439.cal', index=1)
        self.instance.setToolTransform([0, 0, 0, 0, 0, 0], atiiaftt.FTUnit.DIST_MM, atiiaftt.FTUnit.ANGLE_DEG)

        self.task = task
        self.baseline = self.baselineF()
        self.figure_set()

    # Set each ax
    def figure_set(self):

        #channel x
        self.axes[0].set_ylim(-6, 6)
        self.axes[0].set_ylabel('Fx(N)')

        #channel y
        self.axes[1].set_ylim(-6, 6)
        self.axes[1].set_ylabel('Fy(N)')

        #channel z
        self.axes[2].set_ylim(-15, 5)
        self.axes[2].set_ylabel('Fz(N)')

        #channel Tx
        self.axes[3].set_ylim(-100, 100)
        self.axes[3].set_ylabel('Tx(N*mm)')

        #channel Ty
        self.axes[4].set_ylim(-100, 100)
        self.axes[4].set_ylabel('Ty(N*mm)')

        #channel Tz
        self.axes[5].set_ylim(-100, 100)
        self.axes[5].set_ylabel('Tz(N*mm)')

        self.axes[0].set_title('Nano17 Real-time Readout')
        self.axes[-1].set_xlabel('Time(s)')

    # Generator function to yield updated Ft_readout values
    def ft_readout_generator(self):
        ee = (np.array(self.task.read()) - np.array(self.baseline)).tolist()
        # print("Raw Data : {}".format(ee))
        # print("-"*100)
        self.instance.convertToFt(ee)
        # print("Transformed Data : {}".format(self.instance.ft_vector))
        # print("-"*100)
        Ft_readout =  self.instance.ft_vector
        return Ft_readout

    # Move window for y
    def y_moveWindow(self,i,new_y):

        self.y_data[i][:-1] = self.y_data[i][1:]
        self.y_data[i][-1] = new_y
        
    # Update function for animation
    def update(self,frame):

        # start_time = time.time()  # Record the start time
        ft_readout = self.ft_readout_generator()
        x_new = self.x_data + frame * self.interval/1000
        for i in range(self.num_channels):
            self.y_moveWindow(i, ft_readout[i])
            self.lines[i].set_data(x_new, self.y_data[i])
            self.axes[i].set_xlim(x_new[0], x_new[-1])
        print(ft_readout,sep=' ',end='\n', flush=True)
        # sys.stdout.flush()
    # baseline
    def baselineF(self):
        base_sum = np.zeros(6)
        for _ in range(self.baseline_count):
            base_sum = base_sum + np.array(self.task.read())
        base_line = (base_sum / self.baseline_count).tolist()
        return base_line

    def data_save(self,ft_readout,num_saved_images):
    
        data = {
                'Fx(N)': [ft_readout[0]],
                'Fy(N)': [ft_readout[1]],
                'Fz(N)': [ft_readout[2]],
                'Tx(Nmm)': [ft_readout[3]],
                'Ty(Nmm)': [ft_readout[4]],
                'Tz(Nmm)': [ft_readout[5]],
                'Num': [num_saved_images]
            }
        
        # Writing to CSV file
        if os.path.exists(self.save_file_name):
            # Reading the existing CSV file into a DataFrame
            existing_df = pd.read_csv(self.save_file_name)
        else:
            # If the file doesn't exist, create a new DataFrame with the additional data
            existing_df = pd.DataFrame()
        # Creating a DataFrame from the additional data
        additional_df = pd.DataFrame(data)
        updated_df =  pd.concat([existing_df, additional_df], ignore_index=True)
        # Writing the updated DataFrame to the CSV file
        updated_df.to_csv(self.save_file_name, index=False)
        print(f'Data has been updated and saved to {self.save_file_name}')



if __name__ == '__main__':
    
     # Number of channels
    num_channels = 6
    # Time range in X-axis 
    x_timeMax = 5
    # Frames to Display
    frames = 10**6
    # Interval
    interval= 50
    # Baseline_count
    baseline_count = 50
    
    with nidaqmx.Task() as task:

        task.ai_channels.add_ai_voltage_chan("Dev1/ai0")
        task.ai_channels.add_ai_voltage_chan("Dev1/ai1")
        task.ai_channels.add_ai_voltage_chan("Dev1/ai2")
        task.ai_channels.add_ai_voltage_chan("Dev1/ai3")
        task.ai_channels.add_ai_voltage_chan("Dev1/ai4")
        task.ai_channels.add_ai_voltage_chan("Dev1/ai5")

        rtv = RealTimeVisual(num_channels, x_timeMax, frames, interval, baseline_count,task)
        # calculate baseline to calibrate
        print(f"baseline:{rtv.baseline}")
        # real-time display update
        ani = FuncAnimation(rtv.fig, rtv.update, frames=range(rtv.frames), interval=rtv.interval, repeat=False)
        # Show the plot
        plt.tight_layout()
        plt.show()



