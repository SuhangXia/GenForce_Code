from functools import partial
import cv2
import yaml
from yaml.loader import SafeLoader
import os
import datetime
import argparse

os.chdir(os.path.dirname(__file__))

class ImageSee():

    def __init__(self,video):

        self.current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_date = datetime.datetime.now().strftime("%Y%m%d")
        self.args = self.options()
        self.config = self.configF()
        self.current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_date = datetime.datetime.now().strftime("%Y%m%d")
        self.stream = self.config["stream"]
        self.resize_ratio = self.stream["resize_ratio"]
        
        self.video = video
        self.H = 0
        self.W = 0
        self.num_saved_images = len(os.listdir(self.args.save_dir))
        self.num_saved_images = len(os.listdir(self.args.save_dir))

    def options(self):
        defult_save_dir = f"../Data_Save/Images/downstream/{self.current_date}"
        parser = argparse.ArgumentParser()
        parser.add_argument("--config",type=str,default="config/gelsight.yaml")
        parser.add_argument("--save_dir",type=str,default= defult_save_dir)
        parser.add_argument("--save_fmt",type=str,default="%04d.jpg")
        parser.add_argument("--ref_img",type=str,default=f"../Data_Save/Ref_Images/{self.current_date}/ref_{self.current_time}.jpg")
        parser.add_argument("--_continue",type=bool,default=True)
        os.makedirs(defult_save_dir,exist_ok=True)
        return parser.parse_args()

    def configF(self):
        with open(self.args.config,'r') as f:
            config = yaml.load(f, Loader=SafeLoader)
            return config

    def ref_capture(self):
        res, frame = self.video.read()
        if not res:
            raise RuntimeError("Camera does not run normally.")
        self.H, self.W = frame.shape[:2]
        resize_func = self.resizeF()
        img_crop = resize_func(frame)
        cv2.imwrite(self.args.ref_img, img_crop)

    def resizeF(self):
       return partial(cv2.resize, dsize=(int(self.W*self.resize_ratio), int(self.H*self.resize_ratio)),interpolation=cv2.INTER_AREA)

    def save_images(self):

        os.makedirs(self.args.save_dir,exist_ok=True)
        if self.args._continue:
            num_saved_images = len(os.listdir(self.args.save_dir))
        while True:
            saved_name = os.path.join(self.args.save_dir,self.args.save_fmt%num_saved_images)
            # used for GelSight
            res, frame = self.video.read()
            if not res:
                raise RuntimeError("Camera open error.")
            # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)  
            last_window_name = "Img %d"%(num_saved_images-1)
            curr_window_name = "Img %d"%num_saved_images
            try:
                cv2.destroyWindow(last_window_name)
            except Exception as e:
                pass
            frame = cv2.resize(frame,(640,480))
            cv2.imshow(curr_window_name, frame)
            resize_func = self.resizeF()
            img_crop = resize_func(frame)
            key = cv2.waitKey(1)
            if key == 115:
                cv2.imwrite(saved_name, img_crop)
                print("Saved images:{}".format(num_saved_images))
                num_saved_images += 1
            if key == 27: # esc
                break

if __name__ == "__main__":

    video = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    ImSv = ImageSee(video)
    ImSv.ref_capture()
    ImSv.save_images()
    cv2.destroyAllWindows()
    ImSv.video.release()

