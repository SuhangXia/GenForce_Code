#!/usr/bin/env python3
import yaml
from yaml.loader import SafeLoader
import os
import datetime
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

# ROS imports
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

os.chdir(os.path.dirname(__file__))

# Global Configuration
ip = "10.70.102.29"
port = 5000
lastmessage = {"message": "No message"}
data_queue = queue.Queue(maxsize=1)
first_frame = None
running = True

# Pre-calculate grid coordinates
GRID_X = np.array([0, 1, 2, 3] * 4)
GRID_Y = np.repeat([0, 1, 2, 3], 4)

# Screen dimensions for position
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

class Visualizer:
    def __init__(self, ros_pub, bridge):
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

        self.ros_pub = ros_pub
        self.bridge = bridge

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
            pass
        self.fig.canvas.mpl_connect('key_press_event', on_key)

    def update(self, frame):
        try:
            if not data_queue.empty():
                points = data_queue.get_nowait()
                delta_x = np.clip(points[:, 0]/500, -1, 1)
                delta_y = np.clip(points[:, 1]/500, -1, 1)
                plot_x = GRID_X + delta_x
                plot_y = GRID_Y + delta_y
                sizes = 50 + points[:, 2]
                sizes = np.clip(sizes, 50, 4000)
                plot_x = np.clip(plot_x, -0.6, 3.6)
                plot_y = np.clip(plot_y, -0.6, 3.6)

                self.scatter.set_offsets(np.c_[plot_x, plot_y])
                self.scatter.set_sizes(sizes)

                self.fig.canvas.draw_idle()
                self.fig.canvas.flush_events()

                # --- ROS IMAGE PUBLISHING ---
                # Capture the current matplotlib figure as an RGB image
                self.fig.canvas.draw()
                img_np = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
                img_np = img_np.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
                # (Optional) Convert RGB to BGR for OpenCV compatibility
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                # Publish as ROS Image
                img_msg = self.bridge.cv2_to_imgmsg(img_bgr, encoding="bgr8")
                self.ros_pub.publish(img_msg)
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

    # --- ROS Node and publisher setup ---
    rospy.init_node('uskin_publisher', anonymous=True)
    ros_pub = rospy.Publisher('/uskin/marker_image', Image, queue_size=1)
    bridge = CvBridge()

    # Start WebSocket connection
    websocket.setdefaulttimeout(1)
    wsapp = websocket.WebSocketApp(f"ws://{ip}:{port}",
                                   on_message=on_message)

    # Start threads
    ws_thread = threading.Thread(target=wsapp.run_forever, daemon=True)
    mes_thread = threading.Thread(target=mesreader, daemon=True)

    ws_thread.start()
    mes_thread.start()

    # Initialize visualizer
    vis = Visualizer(ros_pub, bridge)
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
        wsapp.close()
        plt.close('all')

if __name__ == "__main__":
    main()