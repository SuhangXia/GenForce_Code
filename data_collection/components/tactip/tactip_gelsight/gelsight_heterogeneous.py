from functools import partial  
import cv2  
import yaml  
from yaml.loader import SafeLoader  
import os  
import datetime  
import argparse  
import multiprocessing  
import keyboard  
import time  

# Ensure the script runs in the directory where it resides  
os.chdir(os.path.dirname(__file__))  

class ImageSee():  

    def __init__(self, video, camera_id, save_event, sync_event, saved_names):  
        self.current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  
        self.current_date = datetime.datetime.now().strftime("%Y%m%d")  
        self.saved_names = saved_names  
        self.args = self.options(camera_id)  
        self.config = self.configF()  
        self.stream = self.config["stream"]  
        self.resize_ratio = self.stream["resize_ratio"]  

        self.video = video  
        self.camera_id = camera_id  
        self.save_event = save_event  # Shared event for saving images  
        self.sync_event = sync_event  # Sync event for camera coordination  
        self.H = 0  
        self.W = 0  
        self.num_saved_images = len(os.listdir(self.args.save_dir))  

    def options(self, camera_id):  
        save_name = self.saved_names[camera_id]  
        default_save_dir = f"../Data_Save/Images/downstream/{self.current_date}/camera_{save_name}"  
        parser = argparse.ArgumentParser()  
        parser.add_argument("--config", type=str, default="config/gelsight_config.yaml")  
        parser.add_argument("--save_dir", type=str, default=default_save_dir)  
        parser.add_argument("--save_fmt", type=str, default="%04d.jpg")  
        parser.add_argument("--ref_img", type=str, default=f"../Data_Save/Ref_Images/{self.current_date}/ref_{self.current_time}_camera_{camera_id}.jpg")  
        parser.add_argument("--_continue", type=bool, default=True)  
        os.makedirs(default_save_dir, exist_ok=True)  
        return parser.parse_args()  

    def configF(self):  
        with open(self.args.config, 'r') as f:  
            config = yaml.load(f, Loader=SafeLoader)  
            return config  

    def ref_capture(self):  
        res, frame = self.video.read()  
        if not res:  
            raise RuntimeError(f"Camera {self.camera_id} does not run normally.")  
        self.H, self.W = frame.shape[:2]  
        resize_func = self.resizeF()  
        img_crop = resize_func(frame)  
        cv2.imwrite(self.args.ref_img, img_crop)  

    def resizeF(self):  
        return partial(cv2.resize, dsize=(int(self.W * self.resize_ratio), int(self.H * self.resize_ratio)), interpolation=cv2.INTER_AREA)  

    def save_images(self):  
        os.makedirs(self.args.save_dir, exist_ok=True)  
        if self.args._continue:  
            num_saved_images = len(os.listdir(self.args.save_dir))  

        while True:  
            saved_name = os.path.join(self.args.save_dir, self.args.save_fmt % num_saved_images)  
            res, frame = self.video.read()  
            if not res:  
                raise RuntimeError(f"Camera {self.camera_id} open error.")  
            curr_window_name = f"Camera {self.camera_id}"  # Unique window name for each camera  
            frame = cv2.resize(frame, (640, 480))  
            cv2.imshow(curr_window_name, frame)  
            resize_func = self.resizeF()  
            img_crop = resize_func(frame)  

            # Check if the save event is triggered  
            if self.save_event.is_set():  
                # Wait for sync signal  
                if self.camera_id == 1:
                    counterpart = 0
                else:
                    counterpart = 1
                self.sync_event[self.camera_id].set()  
                self.sync_event[counterpart].wait()  
                cv2.imwrite(saved_name, img_crop)  
                print(f"Saved image from Camera {self.camera_id}: {saved_name}")  
                num_saved_images += 1  
                self.save_event.clear()  
                self.sync_event[self.camera_id].clear()  

                # Align the windows side by side after saving  
                if self.camera_id == 0:  
                    cv2.moveWindow(curr_window_name, 100, 100)  # Position for Camera 0  
                elif self.camera_id == 1:  
                    cv2.moveWindow(curr_window_name, 800, 100)  # Position for Camera 1  

            key = cv2.waitKey(1)  
            if key == 27:  # esc key  
                break  

def camera_process(camera_id, save_event, sync_event, save_names):  
    """  
    Function to handle a single camera process.  
    """  
    video = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)  
    if not video.isOpened():  
        print(f"Error: Could not open camera {camera_id}")  
        return  
    
    # Create an instance of ImageSee for the current camera  
    ImSv = ImageSee(video, camera_id, save_event, sync_event, save_names)  

    # Capture reference image  
    ImSv.ref_capture()  

    # Start saving images  
    ImSv.save_images()  

    # Release the video and close windows  
    video.release()  
    cv2.destroyAllWindows()  

if __name__ == "__main__":  
    # Define camera IDs (change IDs as per your setup)  
    camera_ids = [0, 1]  

    # Create shared events for saving images and synchronization  
    save_event = multiprocessing.Event()  
    sync_event_c1 = multiprocessing.Event()  
    sync_event_c2 = multiprocessing.Event()  
    sync_event =[sync_event_c1, sync_event_c2] 
    save_names = ["gelsight_diamond", "tactip"]  


    # Create a process for each camera  
    processes = []  
    for camera_id in camera_ids:  
        p = multiprocessing.Process(target=camera_process, args=(camera_id, save_event, sync_event, save_names))  
        processes.append(p)  
        p.start()  

    # Variables for key debouncing  
    key_pressed = False  
    last_press_time = 0  
    debounce_delay = 0.3  # 300ms debounce delay  

    while True:  
        current_time = time.time()  
        
        if keyboard.is_pressed('s'):  
            if not key_pressed and (current_time - last_press_time) > debounce_delay:  
                print("Main process: 's' key pressed!")  
                save_event.set()  # Trigger the save event  
                key_pressed = True  
                last_press_time = current_time  
        else:  
            key_pressed = False  # Reset key state when released  

        if keyboard.is_pressed('esc'):  # Detect 'Esc' key press  
            print("Main process: 'Esc' key pressed. Exiting.")  
            break  

        time.sleep(0.01)  # Small sleep to prevent CPU overuse  

    # Wait for all processes to finish  
    for p in processes:  
        p.join()