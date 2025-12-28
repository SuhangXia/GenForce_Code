#!/usr/bin/env python3
import os
import sys
import rospy
import datetime
import numpy as np
import pandas as pd
import cv2
from threading import Lock, Thread
from genforce.msg import StampedFloat64MultiArray

# ======= User-configurable acquisition parameters =======
DATA_ROOT = '/home/zhuo/catkin_ws/src/genforce/data'
USKIN_DIM = 48          # Actual uskin data size per frame
ROBOT_DIM = 21          # Actual robot state size per frame
MARKER_DIM = 7          # [x y z ox oy oz ow]
GELSIGHT_DIM = 224*224*3   # 640*480*3 for RGB image
H, W, C = 224, 224, 3   # Gelsight image shape
# =======================================================

class DataCollector:
    def __init__(self):
        rospy.init_node('data_collection', anonymous=True)
        print('\n===== Incremental Synchronized /syn_data Saving Started =====\n')

        # Prepare data output directories and file paths
        self.timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.save_dir = os.path.join(DATA_ROOT, f"data_{self.timestamp}")
        self.gelsight_img_dir = os.path.join(self.save_dir, "gelsight_img")
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.gelsight_img_dir, exist_ok=True)
        print(f"Data will be saved to: {self.save_dir}")
        print(f"Gelsight images will be saved in: {self.gelsight_img_dir}")
        print("Press 's' to START, 't' to PAUSE. Ctrl+C to exit.")

        # Prepare CSV paths
        self.uskin_file = os.path.join(self.save_dir, f'{self.timestamp}_uskin.csv')
        self.robot_file = os.path.join(self.save_dir, f'{self.timestamp}_robot_state.csv')
        self.marker_file = os.path.join(self.save_dir, f'{self.timestamp}_marker.csv')

        self._init_csv_files = False

        self.lock = Lock()
        self.collecting = False
        self.frame_counter = 0  # Gelsight image file naming

        self.key_thread = Thread(target=self.keyboard_listener, daemon=True)
        self.key_thread.start()

        rospy.Subscriber('/syn_data', StampedFloat64MultiArray, self.cb_syn)

    def keyboard_listener(self):
        """Listen for 's' (start) and 't' (pause) keyboard commands."""
        import termios, tty, select
        print("\nPress 's'+Enter to start, 't'+Enter to pause collection.\n")
        while True:
            if select.select([sys.stdin], [], [], 0.1)[0]:
                key = sys.stdin.readline().strip()
                if key == 's' and not self.collecting:
                    self.collecting = True
                    print("\n[INFO] Data collection started.\n")
                elif key == 't' and self.collecting:
                    self.collecting = False
                    print("\n[INFO] Data collection paused.\n")

    def _write_csv_row(self, file, row, cols, first_row=False):
        """Append a single data row to a CSV file."""
        mode = 'a' if os.path.exists(file) else 'w'
        header = not os.path.exists(file) or first_row
        pd.DataFrame([row], columns=cols).to_csv(
            file, mode='a', index=False, header=header, float_format='%.8f'
        )

    def cb_syn(self, msg):
        """Main callback saving data for every received synced frame."""
        if not self.collecting:
            return

        with self.lock:
            stamp = msg.header.stamp.to_sec()
            arr = np.array(msg.data, dtype=np.float32)
            expected_len = USKIN_DIM + ROBOT_DIM + MARKER_DIM + GELSIGHT_DIM
            if arr.size != expected_len:
                print(f"[ERROR] Received frame size {arr.size}, expected {expected_len}. Dropping frame.")
                return

            # Data slicing by index
            uskin    = arr[:USKIN_DIM]
            robot    = arr[USKIN_DIM:USKIN_DIM+ROBOT_DIM]
            marker   = arr[USKIN_DIM+ROBOT_DIM:USKIN_DIM+ROBOT_DIM+MARKER_DIM]
            gelsight = arr[USKIN_DIM+ROBOT_DIM+MARKER_DIM:USKIN_DIM+ROBOT_DIM+MARKER_DIM+GELSIGHT_DIM]

            # CSV file headers (only on the very first frame)
            if not self._init_csv_files:
                self.uskin_cols = ['stamp'] + [f'uskin_{i}' for i in range(USKIN_DIM)]
                self.robot_cols = ['stamp'] + [f'robot_{i}' for i in range(ROBOT_DIM)]
                self.marker_cols = ['stamp'] + [f'marker0_{s}' for s in ['x','y','z','ox','oy','oz','ow']]
                self._write_csv_row(self.uskin_file, [stamp]+uskin.tolist(), self.uskin_cols, first_row=True)
                self._write_csv_row(self.robot_file, [stamp]+robot.tolist(), self.robot_cols, first_row=True)
                self._write_csv_row(self.marker_file, [stamp]+marker.tolist(), self.marker_cols, first_row=True)
                self._init_csv_files = True
            else:
                self._write_csv_row(self.uskin_file, [stamp]+uskin.tolist(), self.uskin_cols)
                self._write_csv_row(self.robot_file, [stamp]+robot.tolist(), self.robot_cols)
                self._write_csv_row(self.marker_file, [stamp]+marker.tolist(), self.marker_cols)

            # Save gelsight RGB image as JPG
            try:
                vec = np.array(gelsight, dtype=np.float32)
                # If data is [0,1] float, convert to [0,255] and uint8
                img = (vec.reshape((H,W,C)) * 255).clip(0,255).astype(np.uint8)
                # Use "img" for RGB or "img[..., ::-1]" for BGR, depending on your pipeline
                fname = os.path.join(self.gelsight_img_dir, f'gelsight_{self.frame_counter:06d}_{stamp:.3f}.jpg')
                cv2.imwrite(fname, img)  # Switch to img if colors are wrong
                self.frame_counter += 1
                print(f"[INFO] Frame {self.frame_counter} saved.")
            except Exception as e:
                print(f"[ERROR] Failed to save gelsight image for frame {self.frame_counter}: {e}")

    def spin(self):
        """Main ROS spin loop."""
        try:
            while not rospy.is_shutdown():
                rospy.sleep(0.2)
        except KeyboardInterrupt:
            print("\n[INFO] Data collection stopped by user (Ctrl+C).\n")
            print(f"[INFO] Total frames saved: {self.frame_counter}\n")

if __name__ == "__main__":
    import tty, termios
    stdin_fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(stdin_fd)
    try:
        tty.setcbreak(stdin_fd)
        collector = DataCollector()
        collector.spin()
    finally:
        termios.tcsetattr(stdin_fd, termios.TCSADRAIN, old_settings)