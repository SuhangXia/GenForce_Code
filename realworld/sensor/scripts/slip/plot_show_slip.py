#!/usr/bin/env python3
import time
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication
import threading
import roslibpy
import argparse
import sys
from PyQt5.QtCore import QTimer
import numpy as np
import cv2
import os
from datetime import datetime

WINDOW_LENGTH = 6

pg.setConfigOption('background', 'w')  # Set global background to white
pg.setConfigOption('foreground', 'k')  # Set axes/scale color to black (optional, for visibility)

class DataStore:
    def __init__(self):
        self.lock = threading.Lock()
        self.force1_x, self.force1_x_t = [], []
        self.force1_y, self.force1_y_t = [], []
        self.force1_z, self.force1_z_t = [], []
        self.force2_x, self.force2_x_t = [], []
        self.force2_y, self.force2_y_t = [], []
        self.force2_z, self.force2_z_t = [], []
        self.start_time = None

store = DataStore()
exit_event = threading.Event()

def ros_subscribe_thread(client, sensor1, sensor2):
    def make_cb(data_list, t_list):
        def cb(msg):
            t_now = time.time()
            with store.lock:
                if store.start_time is None:
                    store.start_time = t_now
                rel_t = t_now - store.start_time
                t_list.append(rel_t)
                data_list.append(msg['data'])

                while t_list and (rel_t - t_list[0]) > WINDOW_LENGTH:
                    t_list.pop(0)
                    data_list.pop(0)
        return cb

    topics = {
        f'/force/{sensor1}/x':    (store.force1_x, store.force1_x_t),
        f'/force/{sensor1}/y':    (store.force1_y, store.force1_y_t),
        f'/force/{sensor1}/z':    (store.force1_z, store.force1_z_t),
        f'/force/{sensor2}/x':    (store.force2_x, store.force2_x_t),
        f'/force/{sensor2}/y':    (store.force2_y, store.force2_y_t),
        f'/force/{sensor2}/z':    (store.force2_z, store.force2_z_t),
    }

    for topic_name, (data_list, t_list) in topics.items():
        topic = roslibpy.Topic(client, topic_name, 'std_msgs/Float32')
        topic.subscribe(make_cb(data_list, t_list))
        print(f"subscribed: {topic_name}")

    try:
        while client.is_connected:
            if exit_event.is_set():
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    exit_event.set()
    client.terminate()


def plot_window_main(sensor1, sensor2, client, save_video, video_filename, video_fps, video_width, video_height):
    app = QApplication([])
    win = pg.GraphicsLayoutWidget(title="Force Real-time Curves")
    win.resize(video_width, video_height)

    # ================= 布局：三行两列，每行一个量（Fx, Fy, Fz），两列两个传感器 =================
    # 第 1 行：Fx（左：sensor1，右：sensor2） Y 轴固定 [-1, 1]
    pw_fx_1 = win.addPlot(row=0, col=0, title=f'{sensor1} Fx')
    pw_fx_1.setLabels(left='N', bottom='t(s)')
    pw_fx_1.setYRange(-1.5, 1.5, padding=0)
    curve1x = pw_fx_1.plot(pen=pg.mkPen((200, 0, 0), width=3), name=f'{sensor1} Fx')

    pw_fx_2 = win.addPlot(row=0, col=1, title=f'{sensor2} Fx')
    pw_fx_2.setLabels(left='N', bottom='t(s)')
    pw_fx_2.setYRange(-1.5, 1.5, padding=0)
    curve2x = pw_fx_2.plot(pen=pg.mkPen((200, 0, 0), width=3), name=f'{sensor2} Fx')

    # 第 2 行：Fy（左：sensor1，右：sensor2） Y 轴固定 [-1, 1]
    pw_fy_1 = win.addPlot(row=1, col=0, title=f'{sensor1} Fy')
    pw_fy_1.setLabels(left='N', bottom='t(s)')
    pw_fy_1.setYRange(-1.5, 1.5, padding=0)
    curve1y = pw_fy_1.plot(pen=pg.mkPen((0, 150, 0), width=3), name=f'{sensor1} Fy')

    pw_fy_2 = win.addPlot(row=1, col=1, title=f'{sensor2} Fy')
    pw_fy_2.setLabels(left='N', bottom='t(s)')
    pw_fy_2.setYRange(-1.5, 1.5, padding=0)
    curve2y = pw_fy_2.plot(pen=pg.mkPen((0, 150, 0), width=3), name=f'{sensor2} Fy')

    # 第 3 行：Fz（左：sensor1，右：sensor2） Y 轴固定 [-4, 0]
    pw_fz_1 = win.addPlot(row=2, col=0, title=f'{sensor1} Fz')
    pw_fz_1.setLabels(left='N', bottom='t(s)')
    pw_fz_1.setYRange(-5.0, 1.0, padding=0)
    curve1z = pw_fz_1.plot(pen=pg.mkPen((0, 0, 180), width=3), name=f'{sensor1} Fz')

    pw_fz_2 = win.addPlot(row=2, col=1, title=f'{sensor2} Fz')
    pw_fz_2.setLabels(left='N', bottom='t(s)')
    pw_fz_2.setYRange(-5.0, 1.0, padding=0)
    curve2z = pw_fz_2.plot(pen=pg.mkPen((0, 0, 180), width=3), name=f'{sensor2} Fz')

    win.show()

    # --------- Video Writer + timestamp log --------------
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_filename, fourcc, video_fps, (video_width, video_height))
        ts_log = open(video_filename + '.frame_ts.csv', 'w')
        ts_log.write('tick_epoch_sec,frame_index\n')
        print(f"[INFO] Saving video as: {video_filename}")
    else:
        video_writer = None
        ts_log = None

    # ====== 时间节拍控制（真实时间驱动写帧）======
    frame_interval = 1.0 / video_fps
    t0 = time.time()
    next_t = t0
    frame_index = 0
    last_frame = np.zeros((video_height, video_width, 3), dtype=np.uint8)

    def grab_current_frame():
        qimg = win.grab().toImage().convertToFormat(4)  # QImage.Format_RGBA8888
        width, height = qimg.width(), qimg.height()
        ptr = qimg.bits()
        ptr.setsize(qimg.byteCount())
        arr = np.array(ptr).reshape(height, width, 4)
        frame_bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        if (width, height) != (video_width, video_height):
            frame_bgr = cv2.resize(frame_bgr, (video_width, video_height))
        return frame_bgr

    def update():
        nonlocal next_t, frame_index, last_frame, t0

        # 刷新曲线数据
        with store.lock:
            curve1x.setData(store.force1_x_t, store.force1_x)
            curve1y.setData(store.force1_y_t, store.force1_y)
            curve1z.setData(store.force1_z_t, store.force1_z)
            curve2x.setData(store.force2_x_t, store.force2_x)
            curve2y.setData(store.force2_y_t, store.force2_y)
            curve2z.setData(store.force2_z_t, store.force2_z)

        # 抓取并按“时间节拍”写帧
        if save_video and video_writer is not None:
            last_frame = grab_current_frame()
            now = time.time()
            while now >= next_t:
                video_writer.write(last_frame)  # 需要时重复写以补齐节拍
                if ts_log is not None:
                    ts_log.write('%.6f,%d\n' % (next_t, frame_index))  # 记录 tick 时间
                frame_index += 1
                next_t += frame_interval

        # 退出逻辑
        if (not client.is_connected) or exit_event.is_set():
            print("[INFO] ROS client disconnected or exit requested, closing window.")
            timer.stop()
            win.close()
            app.quit()

    timer = QTimer()
    timer.timeout.connect(update)
    timer.start(max(5, int(1000 / video_fps)))  # 提高刷新频率更利于追帧

    try:
        app.exec_()
    except KeyboardInterrupt:
        pass
    finally:
        if video_writer is not None:
            video_writer.release()
        if ts_log is not None:
            ts_log.close()
        win.close()

def main():
    parser = argparse.ArgumentParser(description="Force Real-time Plot Script")
    parser.add_argument('--sensor1', required=True, help='Name of the first sensor (e.g. AII)')
    parser.add_argument('--sensor2', required=True, help='Name of the second sensor (e.g. uskin)')
    parser.add_argument('--save-video', action='store_true', help='Save video of plotting window')
    parser.add_argument('--video_name', default='orange', help='Output video file base name (.mp4)')
    parser.add_argument('--fps', type=float, default=30, help='Video framerate')
    parser.add_argument('--width', type=int, default=1500, help='Video width (pixels)')
    parser.add_argument('--height', type=int, default=1000, help='Video height (pixels)')
    args = parser.parse_args()

    sensor1 = args.sensor1
    sensor2 = args.sensor2

    # --------- Video filename with timestamp -------------
    if args.save_video:
        video_name = f'video/force/{args.video_name}.mp4'
        base, ext = os.path.splitext(video_name)
        # Use local time for timestamp
        now = datetime.now()
        timestamp_str = now.strftime('%Y%m%d_%H%M%S')
        video_filename = f"{base}_{timestamp_str}{ext}"
        print(f"[INFO] Saving video as: {video_filename}")
    else:
        video_filename = f'video/force/{args.video_name}.mp4'

    client = roslibpy.Ros(host='localhost', port=9090)
    client.run()
    print("[roslibpy] Connected to rosbridge websocket.")
    t_ros = threading.Thread(target=ros_subscribe_thread, args=(client, sensor1, sensor2), daemon=True)
    t_ros.start()
    try:
        plot_window_main(sensor1, sensor2, client, args.save_video, video_filename, args.fps, args.width, args.height)
    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt detected, shutting down...")
        exit_event.set()
    finally:
        client.terminate()
        print("[INFO] Clean exit.")

if __name__ == "__main__":
    main()
