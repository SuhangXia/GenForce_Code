#!/usr/bin/env python3
import numpy as np
import cv2
import threading
import roslibpy
import base64
import argparse
import os
from datetime import datetime
import time

class DataStore(object):
    def __init__(self):
        self.lock = threading.Lock()
        self.img1 = np.zeros((240, 320, 3), np.uint8)  # ROS topic1
        self.img2 = np.zeros((240, 320, 3), np.uint8)  # ROS topic2
        self.img3 = np.zeros((240, 320, 3), np.uint8)  # Camera (cv2)
        self.img4 = np.zeros((240, 320, 3), np.uint8)  # ROS topic4

store = DataStore()

def decode_ros_image_message(msg):
    if 'data' in msg:
        data = msg['data']
        if isinstance(data, str):
            img_bytes = np.frombuffer(base64.b64decode(data), dtype=np.uint8)
        else:
            img_bytes = np.frombuffer(bytearray(data), dtype=np.uint8)
        h, w = msg['height'], msg['width']
        encoding = msg['encoding'].lower()
        if encoding in ['bgr8']:
            img = img_bytes.reshape((h, w, 3))
            return cv2.resize(img, (320, 240))
        elif encoding in ['rgb8']:
            img = img_bytes.reshape((h, w, 3))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return cv2.resize(img, (320, 240))
        elif encoding in ['mono8']:
            img = img_bytes.reshape((h, w))
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return cv2.resize(img, (320, 240))
        else:
            img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
            if img is not None:
                return cv2.resize(img, (320, 240))
    return np.zeros((240, 320, 3), np.uint8)

def safe_img(img, size=(320, 240)):
    try:
        if img is None or img.shape != (size[1], size[0], 3):
            return np.zeros((size[1], size[0], 3), np.uint8)
        return cv2.resize(img, size)
    except:
        return np.zeros((size[1], size[0], 3), np.uint8)

def cam_thread_function(port):
    cam = cv2.VideoCapture(port)
    if not cam.isOpened():
        print(f"[ERROR] Cannot open camera port {port}")
        return
    orig_w = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
    orig_h = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)

    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    while True:
        ret, frame = cam.read()
        if not ret:
            time.sleep(0.01)
            continue
        img = safe_img(frame)
        with store.lock:
            store.img3 = img
        if getattr(store, "cam_exit", False):
            break

    cam.set(cv2.CAP_PROP_FRAME_WIDTH, orig_w)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, orig_h)
    cam.release()
    print("[INFO] Camera thread terminated and cam property reset.")

def ros_image_thread(client, sensor1, sensor2, save_video=False, video_base='out_video.mp4', fps=30):
    topic_image1 = f'/{sensor1}/marker'
    topic_image2 = f'/{sensor2}/marker'
    topic_image4 = f'/{sensor1}/raw'

    def get_imgcb(attr):
        def cb(msg):
            img = decode_ros_image_message(msg)
            with store.lock:
                setattr(store, attr, img)
        return cb

    t1 = roslibpy.Topic(client, topic_image1, 'sensor_msgs/Image')
    t2 = roslibpy.Topic(client, topic_image2, 'sensor_msgs/Image')
    t4 = roslibpy.Topic(client, topic_image4, 'sensor_msgs/Image')
    t1.subscribe(get_imgcb('img1'))
    t2.subscribe(get_imgcb('img2'))
    t4.subscribe(get_imgcb('img4'))

    out_width, out_height = 640, 480
    
    # 初始化记录用变量
    start_time = None
    frame_count = 0

    if save_video:
        base, ext = os.path.splitext(video_base)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = f"{base}_{timestamp}{ext}"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_path, fourcc, fps, (out_width, out_height))
        print('[INFO]', "Save video to", video_path)
        ts_log = open(video_path + '.frame_ts.txt', 'w')
        start_time = time.time()
    else:
        writer = None
        ts_log = None

    # 初始化时钟节拍控制
    frame_interval = 1.0 / fps
    next_t = time.time()
    last_frame = np.zeros((out_height, out_width, 3), np.uint8)  # 用于写重复帧

    print("[INFO] Main loop running...")
    try:
        while client.is_connected:
            # 拼接当前最新的四路图像
            with store.lock:
                img1 = safe_img(store.img1)
                img2 = safe_img(store.img2)
                img3 = safe_img(store.img3)
                img4 = safe_img(store.img4)

            top = np.hstack([img1, img2])
            bot = np.hstack([img3, img4])
            main_img = np.vstack([top, bot])
            
            # 更新最后一帧用于重复写入
            last_frame = main_img.copy()

            # 添加时间戳文本
            now = time.time()
            time_text = datetime.fromtimestamp(now).strftime('%H:%M:%S.%f')[:-3]
            
            # 实时显示
            cv2.imshow('4-View', main_img)
            
            # 按时钟节拍写帧
            while now >= next_t and writer is not None:
                writer.write(last_frame)
                if ts_log is not None:
                    ts_log.write('%.6f\n' % next_t)  # 使用tick时间点而非now
                frame_count += 1
                next_t += frame_interval
            
            # 检查退出条件
            k = cv2.waitKey(1) & 0xFF
            if k == 27 or (cv2.getWindowProperty('4-View', cv2.WND_PROP_VISIBLE) < 1):
                break
            
    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt detected, shutting down...")
    finally:
        # 取消订阅和清理
        t1.unsubscribe()
        t2.unsubscribe()
        t4.unsubscribe()
        
        # 打印录制统计信息
        if writer is not None:
            end_time = time.time()
            real_dur = end_time - start_time
            expected_frames = real_dur * fps
            print(f"[INFO] Recording statistics:")
            print(f"  - Real duration: {real_dur:.3f}s")
            print(f"  - Target FPS: {fps}")
            print(f"  - Frames written: {frame_count}")
            print(f"  - Expected frames: ~{expected_frames:.0f}")
            print(f"  - Video duration: {frame_count/fps:.3f}s")
            writer.release()
        
        if ts_log is not None:
            ts_log.close()
            
        client.terminate()
        cv2.destroyAllWindows()
        store.cam_exit = True
        print("[INFO] Shut down completed.")

def main():
    parser = argparse.ArgumentParser(description="ROS+Camera 4-View Video Recorder")
    parser.add_argument('--sensor1', required=True, help='Name of the first sensor (e.g. AII)')
    parser.add_argument('--sensor2', required=True, help='Name of the second sensor (e.g. uskin)')
    parser.add_argument('--cam-port', type=int, default=4, help='Camera port integer (default 10 for /dev/video10)')
    parser.add_argument('--save-video', action='store_true', help='Save the 4-view video')
    parser.add_argument('--video_name', default='orange', help='Output video base name (.mp4)')
    parser.add_argument('--fps', type=float, default=60, help='Video FPS')
    args = parser.parse_args()
    sensor1 = args.sensor1
    sensor2 = args.sensor2
    cam_port = args.cam_port
    video_name = f'video/img/{args.video_name}.mp4'
    setattr(args,"video",video_name)
    store.cam_exit = False
    th_cam = threading.Thread(target=cam_thread_function, args=(cam_port,), daemon=True)
    th_cam.start()

    client = roslibpy.Ros(host='localhost', port=9090)
    client.run()
    ros_image_thread(
        client, sensor1, sensor2,
        save_video=args.save_video, video_base=args.video, fps=args.fps
    )
    th_cam.join()

if __name__ == "__main__":
    main()