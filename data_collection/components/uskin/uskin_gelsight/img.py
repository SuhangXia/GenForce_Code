#!/usr/bin/env python3  
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
from matplotlib import use  
import time  
use('TkAgg')  

os.chdir(os.path.dirname(__file__))  

# Global Configuration  
ip = "10.70.151.133"  
port = 5000  
lastmessage = {"message": "No message"}  
data_queue = queue.Queue(maxsize=1)  
first_frame = None  
running = True  
save_event = threading.Event()  

# Pre-calculate grid coordinates  
GRID_X = np.array([0, 1, 2, 3] * 4)  
GRID_Y = np.repeat([0, 1, 2, 3], 4)  

# Create save directories  
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
indenter = "prism"  
SAVE_DIR = f"Data_Save/Images/{indenter}/{current_time}"  
USKIN_DIR = os.path.join(SAVE_DIR, "uskin")  
GELSIGHT_DIR = os.path.join(SAVE_DIR, "gelsight")  
os.makedirs(SAVE_DIR, exist_ok=True)  
os.makedirs(USKIN_DIR, exist_ok=True)  
os.makedirs(GELSIGHT_DIR, exist_ok=True)  

# Screen dimensions  
SCREEN_WIDTH = 1920  
SCREEN_HEIGHT = 1080  

class ImageSee():  
    def __init__(self, video):  
        self.args = self.options()  
        self.config = self.configF()  
        self.stream = self.config["stream"]  
        self.resize_ratio = self.stream["resize_ratio"]  
        
        self.video = video  
        self.H = 0  
        self.W = 0  
        self.running = True  
        self.num_saved_images = 0  
        self.save_event = save_event  
        
        self.window_name = "GelSight Camera"  
        self.window_width = 640  
        self.window_height = 480  

    def options(self):  
        save_ref_dir = os.path.join(GELSIGHT_DIR, "ref_gelsight.jpg")  
        parser = argparse.ArgumentParser()  
        parser.add_argument("--config", type=str, default="config/gelsight_config.yaml")  
        parser.add_argument("--save_dir", type=str, default=GELSIGHT_DIR)  
        parser.add_argument("--save_fmt", type=str, default="%04d.jpg")  
        parser.add_argument("--ref_img", type=str, default=save_ref_dir)  
        return parser.parse_args()  

    def configF(self):  
        with open(self.args.config, 'r') as f:  
            return yaml.load(f, Loader=SafeLoader)  

    def ref_capture(self):  
        res, frame = self.video.read()  
        if not res:  
            raise RuntimeError("Camera does not run normally.")  
        self.H, self.W = frame.shape[:2]  

    def save_images(self):  
        # Position window on the left side  
        x_position = max(0, SCREEN_WIDTH // 4 - self.window_width // 2)  
        y_position = max(0, SCREEN_HEIGHT // 2 - self.window_height // 2)  
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)  
        cv2.resizeWindow(self.window_name, self.window_width, self.window_height)  
        cv2.moveWindow(self.window_name, x_position, y_position)  

        while self.running:  
            res, frame = self.video.read()  
            if not res:  
                break  
            
            frame = cv2.resize(frame, (self.window_width, self.window_height))  
            cv2.imshow(self.window_name, frame)  

            key = cv2.waitKey(1)  
            if key == ord('a') or self.save_event.is_set():  
                saved_name = os.path.join(self.args.save_dir,  
                                        self.args.save_fmt % self.num_saved_images)  
                cv2.imwrite(saved_name, frame)  
                print(f"Saved GelSight image: {saved_name}")  
                self.num_saved_images += 1  
                self.save_event.clear()  
            elif key == 27:  # ESC  
                self.running = False  
                break  

class Visualizer:  
    def __init__(self):  
        plt.style.use('dark_background')  
        self.window_width = 640  
        self.window_height = 480  
        dpi = 100  
        figwidth = self.window_width / dpi  
        figheight = self.window_height / dpi  
        
        self.fig = plt.figure(figsize=(figwidth, figheight),   
                            dpi=dpi, facecolor='black')  
        self.ax = self.fig.add_subplot(111)  
        self.scatter = self.ax.scatter(GRID_X, GRID_Y, s=100,   
                                     color='white', alpha=1.0)  
        
        self.setup_plot()  
        self.setup_keyboard_handler()  
        self.position_window()  
        
        self.save_event = save_event  
        self.num_saved_images = 0  

    def position_window(self):  
        # Position window on the right side  
        x_position = max(0, 3 * SCREEN_WIDTH // 4 - self.window_width // 2)  
        y_position = max(0, SCREEN_HEIGHT // 2 - self.window_height // 2)  
        
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
        
        for spine in self.ax.spines.values():  
            spine.set_visible(False)  

    def setup_keyboard_handler(self):  
        def on_key(event):  
            if event.key == 'a':  
                filename = os.path.join(USKIN_DIR, f'{self.num_saved_images:04d}.jpg')  
                
                plt.savefig(filename,  
                           facecolor='black',  
                           edgecolor='none',  
                           bbox_inches='tight',  
                           pad_inches=0,  
                           format='jpg',  
                           dpi=100)  
                print(f"Saved uSkin plot: {filename}")  
                self.num_saved_images += 1  
                self.save_event.set()  # Trigger GelSight save  
        
        self.fig.canvas.mpl_connect('key_press_event', on_key)  

    def update(self, frame):  
        try:  
            if not data_queue.empty():  
                points = data_queue.get_nowait()  
                delta_x = np.clip(points[:, 0]/20000,-0.8,0.8)
                delta_y = np.clip(points[:, 1]/20000,-0.8,0.8)
                plot_x = GRID_X + delta_x 
                plot_y = GRID_Y + delta_y 
                # plot_x = GRID_X + points[:, 0]/2000 
                # plot_y = GRID_Y + points[:, 1]/2000 
                sizes = 50 + points[:, 2]/4
                # print(np.max(sizes))
                sizes = np.clip(sizes, 50, 4000)  
                plot_x = np.clip(plot_x,-0.6,3.6)
                plot_y = np.clip(plot_y,-0.6,3.6)
                # plot_y = np.clip(plot_y,-1,4)
                
                self.scatter.set_offsets(np.c_[plot_x, plot_y])  
                self.scatter.set_sizes(sizes)  
                
                self.fig.canvas.draw_idle()  
                self.fig.canvas.flush_events()  
                
        except Exception as e:  
            if str(e):  # Only print if there's an actual error message  
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
    global running  
    
    # Initialize camera  
    video = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
    if not video.isOpened():  
        print("Error: Could not open camera")  
        return  

    # Initialize camera handling  
    cam_handler = ImageSee(video)  
    cam_handler.ref_capture()  
    
    # Start WebSocket connection  
    websocket.setdefaulttimeout(1)  
    wsapp = websocket.WebSocketApp(f"ws://{ip}:{port}",   
                                 on_message=on_message)  
    
    # Start threads  
    ws_thread = threading.Thread(target=wsapp.run_forever,   
                               daemon=True)  
    mes_thread = threading.Thread(target=mesreader,   
                                daemon=True)  
    cam_thread = threading.Thread(target=cam_handler.save_images)  
    
    ws_thread.start()  
    mes_thread.start()  
    cam_thread.start()  

    # Initialize visualizer  
    vis = Visualizer()  
    ani = FuncAnimation(vis.fig, vis.update,   
                       interval=20,   
                       cache_frame_data=False)  
    
    # Show plot and keep main thread alive  
    try:  
        plt.show(block=True)  
    except KeyboardInterrupt:  
        running = False  
    finally:  
        running = False  
        save_event.clear()  
        video.release()  
        cv2.destroyAllWindows()  
        wsapp.close()  
        plt.close('all')  

if __name__ == "__main__":  
    main()