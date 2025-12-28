#!/usr/bin/env python3  
import matplotlib  
matplotlib.use('TkAgg', force=True)  
import cv2  
import yaml  
from yaml.loader import SafeLoader  
import os  
import datetime  
import argparse  
import threading  
import websocket  
import json  
import matplotlib.pyplot as plt  
import numpy as np  
import queue  
from matplotlib.animation import FuncAnimation  
import nidaqmx  
import atiiaftt  
import pandas as pd  
import time  
from pynput import keyboard  

plt.ion()  # Enable interactive mode  
os.chdir(os.path.dirname(__file__))  

# Global Configuration  
ip = "10.70.151.133"  
port = 5000  
lastmessage = {"message": "No message"}  
data_queue = queue.Queue(maxsize=1)  
save_queue = queue.Queue()  
first_frame = None  
running = True  
save_event = threading.Event()  

# Pre-calculate grid coordinates  
GRID_X = np.array([0, 1, 2, 3] * 4)  
GRID_Y = np.repeat([0, 1, 2, 3], 4)  

# Screen dimensions  
SCREEN_WIDTH = 1920  
SCREEN_HEIGHT = 1080  

# Global variable for USKIN_DIR  
USKIN_DIR = None  

class FT_Sensor():  
    def __init__(self, task, dir_ft, baseline_count=50):  
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
    
    def save(self, c_dir):  
        try:  
            ft_path = os.path.join(c_dir, 'ft.csv')  
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
                additional_df = pd.DataFrame(data)  
                updated_df = pd.concat([existing_df, additional_df], ignore_index=True)  
            else:  
                updated_df = pd.DataFrame(data)  
                
            updated_df.to_csv(ft_path, index=False)  
        except Exception as e:  
            print(f"Error saving FT data: {e}")  

class KeyboardListener:  
    def __init__(self, visualizer):  
        self.visualizer = visualizer  
        self.listener = keyboard.Listener(on_press=self.on_press)  
        self.last_press_time = 0  
        self.listener.start()  

    def on_press(self, key):  
        try:  
            current_time = time.time()  
            if hasattr(key, 'char'):  
                if key.char == 'a':  
                    if current_time - self.last_press_time > 1.0:  # 1 second debounce  
                        save_queue.put('save')  
                        self.last_press_time = current_time  
                elif key.char == 'q':  
                    global running  
                    running = False  
                    return False  
        except AttributeError:  
            pass  

class Visualizer:  
    def __init__(self, ft_sensor, dir_ft):  
        plt.style.use('dark_background')  
        self.window_width = 640  
        self.window_height = 480  
        self.ft_sensor = ft_sensor  
        self.dir_ft = dir_ft  
        self.save_lock = threading.Lock()  
        self.is_saving = False  
        
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
        self.setup_keyboard_handler()  
        self.position_window()  
        
        self.save_event = save_event  
        self.num_saved_images = 0  
        
        self.keyboard_listener = KeyboardListener(self)  
        self.fig.canvas.draw()  

    def position_window(self):  
        try:  
            x_position = max(0, 3 * SCREEN_WIDTH // 4 - self.window_width // 2)  
            y_position = max(0, SCREEN_HEIGHT // 2 - self.window_height // 2)  
            
            mng = plt.get_current_fig_manager()  
            if mng is not None and hasattr(mng, 'window'):  
                try:  
                    mng.window.wm_geometry(f"+{x_position}+{y_position}")  
                except:  
                    print("Could not set window position")  
        except Exception as e:  
            print(f"Error positioning window: {e}")  

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

    def save_data(self):  
        if self.is_saving:  
            return  
        
        try:  
            with self.save_lock:  
                self.is_saving = True  
                global USKIN_DIR  
                if USKIN_DIR is None:  
                    print("Error: USKIN_DIR is not set")  
                    return  

                filename = os.path.join(USKIN_DIR, f'{self.num_saved_images:04d}.jpg')  
                os.makedirs(os.path.dirname(filename), exist_ok=True)  
                
                # Modified save parameters  
                self.fig.set_size_inches(self.window_width/self.dpi, self.window_height/self.dpi)  
                self.fig.savefig(filename,  
                                facecolor='black',  
                                edgecolor='none',  
                                bbox_inches=None,  
                                pad_inches=0,  
                                format='jpg',  
                                dpi=self.dpi)  
                
                print(f"Saved uSkin plot: {filename}")  
                
                # Save force data  
                self.ft_sensor.save(self.dir_ft)  
                print(f"Saved force data")  
                
                self.num_saved_images += 1  
                self.save_event.set()  
                
        except Exception as e:  
            print(f"Error in save_data: {e}")  
        finally:  
            self.is_saving = False  
            # Clear the save queue  
            while not save_queue.empty():  
                try:  
                    save_queue.get_nowait()  
                except queue.Empty:  
                    break  

    def setup_keyboard_handler(self):  
        def on_key(event):  
            if event.key == 'a' and not self.is_saving:  
                save_queue.put('save')  
        
        self.fig.canvas.mpl_connect('key_press_event', on_key)  

    def update(self, frame):  
        try:  
            # Process at most one save command per update  
            if not self.is_saving:  
                try:  
                    cmd = save_queue.get_nowait()  
                    if cmd == 'save':  
                        self.save_data()  
                except queue.Empty:  
                    pass  

            if not data_queue.empty():  
                points = data_queue.get_nowait()  
                delta_x = np.clip(points[:, 0]/1500, -0.8, 0.8)  
                delta_y = np.clip(points[:, 1]/1500, -0.8, 0.8)  
                plot_x = GRID_X + delta_x  
                plot_y = GRID_Y + delta_y  
                sizes = 100 + points[:, 2]*2  
                sizes = np.clip(sizes, 100, 9000)  
                plot_x = np.clip(plot_x, -0.6, 3.6)  
                plot_y = np.clip(plot_y, -0.6, 3.6)  
                
                self.scatter.set_offsets(np.c_[plot_x, plot_y])  
                self.scatter.set_sizes(sizes)  
                
                self.fig.canvas.draw_idle()  
                self.fig.canvas.flush_events()  
                
        except Exception as e:  
            if str(e):  
                print(f"Error in update: {e}")  
        return self.scatter,  

def on_message(wsapp, message):  
    global lastmessage  
    try:  
        lastmessage = json.loads(message)  
    except Exception as e:  
        print(f"Error in on_message: {e}")  

def mesreader():  
    global first_frame, running  
    print("Message reader started")  
    
    while running:  
        try:  
            if lastmessage.get("message") != "No message" and '1' in lastmessage:  
                data = [int(d, 16) for d in lastmessage['1']['data'].split(",")]  
                points = np.array(data).reshape(16, 3)  

                if first_frame is None:  
                    first_frame = points.copy()  
                    print("First frame captured")  
                else:  
                    try:  
                        data_queue.put_nowait(points - first_frame)  
                    except queue.Full:  
                        try:  
                            data_queue.get_nowait()  
                            data_queue.put_nowait(points - first_frame)  
                        except queue.Empty:  
                            pass  
            time.sleep(0.001)  
        except Exception as e:  
            print(f"Error in mesreader: {e}")  

def main():  
    global running, USKIN_DIR  
    
    # Create directories  
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  
    indenter = "prism"  
    SAVE_DIR = f"data/modulus/uskin"  
    USKIN_DIR = os.path.join(SAVE_DIR, "image", indenter+"-"+current_time)  
    dir_ft = os.path.join(SAVE_DIR, "force", indenter+"-"+current_time)  
    os.makedirs(USKIN_DIR, exist_ok=True)  
    os.makedirs(dir_ft, exist_ok=True)  

    # Initialize F/T sensor  
    with nidaqmx.Task() as task:  
        task.ai_channels.add_ai_voltage_chan("Dev1/ai0")  
        task.ai_channels.add_ai_voltage_chan("Dev1/ai1")  
        task.ai_channels.add_ai_voltage_chan("Dev1/ai2")  
        task.ai_channels.add_ai_voltage_chan("Dev1/ai3")  
        task.ai_channels.add_ai_voltage_chan("Dev1/ai4")  
        task.ai_channels.add_ai_voltage_chan("Dev1/ai5")  
        
        ft = FT_Sensor(task, dir_ft)  
        
        # Start WebSocket connection  
        websocket.setdefaulttimeout(1)  
        wsapp = websocket.WebSocketApp(f"ws://{ip}:{port}",  
                                     on_message=on_message)  
        
        # Start threads  
        ws_thread = threading.Thread(target=wsapp.run_forever,  
                                   daemon=True)  
        mes_thread = threading.Thread(target=mesreader,  
                                    daemon=True)  
        
        ws_thread.start()  
        mes_thread.start()  

        # Initialize visualizer with F/T sensor  
        vis = Visualizer(ft, dir_ft)  
        ani = FuncAnimation(vis.fig, vis.update,  
                           interval=20,  
                           cache_frame_data=False)  
        
        try:  
            while running:  
                try:  
                    plt.pause(0.1)  
                except Exception as e:  
                    print(f"Warning: Display update error: {e}")  
                    time.sleep(0.1)  
            
        except KeyboardInterrupt:  
            running = False  
        finally:  
            running = False  
            save_event.clear()  
            wsapp.close()  
            plt.close('all')  

if __name__ == "__main__":  
    main()