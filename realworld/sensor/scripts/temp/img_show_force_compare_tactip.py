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
        self.marker_img = np.zeros((240, 320, 3), np.uint8)  # ROS topic marker
        self.raw_img = np.zeros((240, 320, 3), np.uint8)     # ROS topic raw

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

def ros_image_thread(client, sensor_name, save_video=False, video_base='out_video.mp4', fps=30):
    # 只订阅sensor1的两个话题
    topic_marker = f'/{sensor_name}/marker'
    topic_raw = f'/{sensor_name}/raw'

    def get_imgcb(attr):
        def cb(msg):
            img = decode_ros_image_message(msg)
            with store.lock:
                setattr(store, attr, img)
        return cb

    # 订阅两个话题
    t_marker = roslibpy.Topic(client, topic_marker, 'sensor_msgs/Image')
    t_raw = roslibpy.Topic(client, topic_raw, 'sensor_msgs/Image')
    t_marker.subscribe(get_imgcb('marker_img'))
    t_raw.subscribe(get_imgcb('raw_img'))

    # 输出视频设置
    out_width, out_height = 640, 240  # 只显示两张图像，宽度保持不变，高度减半
    
    # 初始化记录用变量
    start_time = None
    frame_count = 0

    if save_video:
        # 创建保存目录（如果不存在）
        os.makedirs(os.path.dirname(video_base), exist_ok=True)
        
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

    print(f"[INFO] Receiving images from {sensor_name}...")
    try:
        while client.is_connected:
            # 获取当前最新的两张图像
            with store.lock:
                marker_img = safe_img(store.marker_img)
                raw_img = safe_img(store.raw_img)

            # 水平拼接两张图像
            main_img = np.hstack([marker_img, raw_img])
            
            # 更新最后一帧用于重复写入
            last_frame = main_img.copy()

            # # 添加传感器名称和时间戳
            now = time.time()
            # time_text = datetime.fromtimestamp(now).strftime('%H:%M:%S.%f')[:-3]
            # cv2.putText(main_img, f"{sensor_name} - {time_text}", (10, 20), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # 实时显示
            cv2.imshow(f'{sensor_name} Images', main_img)
            
            # 按时钟节拍写帧
            while now >= next_t and writer is not None:
                writer.write(last_frame)
                if ts_log is not None:
                    ts_log.write('%.6f\n' % next_t)  # 使用tick时间点而非now
                frame_count += 1
                next_t += frame_interval
            
            # 检查退出条件
            k = cv2.waitKey(1) & 0xFF
            if k == 27 or (cv2.getWindowProperty(f'{sensor_name} Images', cv2.WND_PROP_VISIBLE) < 1):
                break
            
    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt detected, shutting down...")
    finally:
        # 取消订阅和清理
        t_marker.unsubscribe()
        
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
        print("[INFO] Shut down completed.")

def main():
    parser = argparse.ArgumentParser(description="ROS Image Viewer for Single Sensor")
    parser.add_argument('--sensor', required=True, help='Name of the sensor (e.g. gelsight)')
    parser.add_argument('--save-video', action='store_true', help='Save the video')
    parser.add_argument('--video_name', default='sensor_view', help='Output video base name (.mp4)')
    parser.add_argument('--fps', type=float, default=60, help='Video FPS')
    args = parser.parse_args()
    
    sensor_name = args.sensor
    video_name = f'video/img/{args.video_name}.mp4'

    client = roslibpy.Ros(host='localhost', port=9090)
    client.run()
    print(f"[INFO] Connected to ROS bridge, subscribing to {sensor_name} topics...")
    
    ros_image_thread(
        client, sensor_name,
        save_video=args.save_video, video_base=video_name, fps=args.fps
    )

if __name__ == "__main__":
    main()