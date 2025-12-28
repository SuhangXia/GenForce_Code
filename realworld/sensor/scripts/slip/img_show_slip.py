
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
from collections import deque

cv2.setNumThreads(1)

class DataStore(object):
    def __init__(self):
        self.lock = threading.Lock()
        self.img3 = np.zeros((240, 320, 3), np.uint8)
        self.cam_exit = False
        # 添加最新图像缓存
        self.latest_img1 = np.zeros((240, 320, 3), np.uint8)
        self.latest_img2 = np.zeros((240, 320, 3), np.uint8)
        self.latest_img4 = np.zeros((240, 320, 3), np.uint8)
        self.img1_timestamp = 0
        self.img2_timestamp = 0
        self.img4_timestamp = 0

store = DataStore()

def safe_img(img, size=(320, 240)):
    try:
        if img is None:
            return np.zeros((size[1], size[0], 3), np.uint8)
        if img.shape != (size[1], size[0], 3):
            return cv2.resize(img, size)
        return img
    except Exception:
        return np.zeros((size[1], size[0], 3), np.uint8)

def decode_ros_image_message(msg):
    img = np.zeros((240, 320, 3), np.uint8)
    if 'data' in msg:
        try:
            data = msg['data']
            # 优化：直接检查数据类型，减少不必要的转换
            if isinstance(data, str):
                img_bytes = np.frombuffer(base64.b64decode(data), dtype=np.uint8)
            else:
                img_bytes = np.frombuffer(bytearray(data), dtype=np.uint8)
                
            h, w = int(msg.get('height', 0)), int(msg.get('width', 0))
            encoding = str(msg.get('encoding', '')).lower()

            if h > 0 and w > 0 and img_bytes.size >= h * w:
                if encoding == 'bgr8' and img_bytes.size >= h * w * 3:
                    img = img_bytes[:h*w*3].reshape((h, w, 3))
                elif encoding == 'rgb8' and img_bytes.size >= h * w * 3:
                    img = cv2.cvtColor(img_bytes[:h*w*3].reshape((h, w, 3)), cv2.COLOR_RGB2BGR)
                elif encoding == 'mono8' and img_bytes.size >= h * w:
                    img = cv2.cvtColor(img_bytes[:h*w].reshape((h, w)), cv2.COLOR_GRAY2BGR)
                else:
                    tmp = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
                    if tmp is not None:
                        img = tmp
        except Exception as e:
            print(f"[WARN] Decode failed: {e}")

    img = safe_img(img, (320, 240))

    stamp = None
    try:
        sec = msg['header']['stamp']['secs']
        nsec = msg['header']['stamp']['nsecs']
        stamp = float(sec) + float(nsec) * 1e-9
    except Exception:
        stamp = time.time()  # 如果没有时间戳，使用当前时间

    return img, stamp

def cam_thread_function(port):
    cam = cv2.VideoCapture(port)
    if not cam.isOpened():
        print(f"[ERROR] Cannot open camera port {port}")
        return
        
    # 优化相机设置
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 最小缓冲
    cam.set(cv2.CAP_PROP_FPS, 30)  # 设置帧率

    frame_skip = 0
    max_skip = 2  # 最多跳过2帧以保持实时性

    try:
        while not getattr(store, "cam_exit", False):
            ret, frame = cam.read()
            if not ret:
                time.sleep(0.005)  # 减少等待时间
                continue
                
            # 跳帧策略：如果处理不过来就跳过一些帧
            frame_skip += 1
            if frame_skip <= max_skip:
                continue
            frame_skip = 0
            
            img = safe_img(frame)
            with store.lock:
                store.img3 = img
    finally:
        cam.release()
        print("[INFO] Camera thread terminated.")

def ros_image_thread(client, sensor1, sensor2, save_video=False, video_base='out_video.mp4', fps=30.0):
    topic_image1 = f'/{sensor1}/marker'
    topic_image2 = f'/{sensor2}/marker'  
    topic_image4 = f'/{sensor1}/raw'

    # 直接更新store，不使用缓冲区
    def create_callback(img_attr, ts_attr, name):
        def cb(msg):
            img, ts = decode_ros_image_message(msg)
            recv_time = time.time()
            
            with store.lock:
                setattr(store, img_attr, img)
                setattr(store, ts_attr, ts if ts else recv_time)
            
            # 减少打印频率
            if hasattr(cb, 'counter'):
                cb.counter += 1
            else:
                cb.counter = 1
                
            if cb.counter % 50 == 1:  # 每50帧打印一次
                delay = recv_time - ts if ts != recv_time else 0
                print(f"[{name}] delay: {delay*1000:.1f}ms")
                
        return cb

    # 移除节流，使用原始帧率
    t1 = roslibpy.Topic(client, topic_image1, 'sensor_msgs/Image', queue_length=1)
    t2 = roslibpy.Topic(client, topic_image2, 'sensor_msgs/Image', queue_length=1)
    t4 = roslibpy.Topic(client, topic_image4, 'sensor_msgs/Image', queue_length=1)
                        
    t1.subscribe(create_callback('latest_img1', 'img1_timestamp', 's1_marker'))
    t2.subscribe(create_callback('latest_img2', 'img2_timestamp', 's2_marker'))
    t4.subscribe(create_callback('latest_img4', 'img4_timestamp', 's1_raw'))

    # 录制初始化
    if save_video:
        base, ext = os.path.splitext(video_base)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = f"{base}_{timestamp}{ext}"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_path, fourcc, fps, (640, 480))
        print('[INFO]', "Save video to", video_path)
        ts_log = open(video_path + '.frame_ts.txt', 'w')
        rec_start = time.time()
    else:
        writer = None
        ts_log = None

    frame_count = 0
    frame_interval = 1.0 / fps
    next_t = time.time()
    
    print("[INFO] Main loop running (Low-latency mode)...")
    
    # 帧率控制
    target_interval = 1.0 / 30  # 30 FPS显示
    last_display = time.time()
    
    try:
        while client.is_connected:
            now = time.time()

            # 控制显示帧率，减少CPU负载
            if now - last_display < target_interval:
                time.sleep(0.001)
                continue
            last_display = now

            # 直接从store获取最新图像，加锁时间最短
            with store.lock:
                img1 = store.latest_img1.copy()
                img2 = store.latest_img2.copy()
                img4 = store.latest_img4.copy()
                img3 = store.img3.copy()
                ts1, ts2, ts4 = store.img1_timestamp, store.img2_timestamp, store.img4_timestamp
            
            # 快速组合图像
            top = np.hstack([img1, img2])
            bot = np.hstack([img3, img4])
            main_img = np.vstack([top, bot])
            
            # 显示延迟信息
            current_time = time.time()
            delays = [
                current_time - ts1 if ts1 > 0 else 0,
                current_time - ts2 if ts2 > 0 else 0, 
                current_time - ts4 if ts4 > 0 else 0
            ]
            
            # status_text = f"Delays(ms): s1_m={delays[0]*1000:.0f}, s2_m={delays[1]*1000:.0f}, s1_r={delays[2]*1000:.0f}"
            # cv2.putText(main_img, status_text, (6, 18), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
            # cv2.putText(main_img, "Mode: Low-latency direct", (6, 40),
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            cv2.imshow('4-View', main_img)

            # 录制：固定帧率
            while writer is not None and now >= next_t:
                writer.write(main_img)
                if ts_log is not None:
                    ts_log.write('%.6f\n' % next_t)
                frame_count += 1
                next_t += frame_interval

            k = cv2.waitKey(1) & 0xFF
            if k == 27 or (cv2.getWindowProperty('4-View', cv2.WND_PROP_VISIBLE) < 1):
                break
                
    except KeyboardInterrupt:
        print("[INFO] KeyboardInterrupt detected, shutting down...")
    finally:
        try:
            t1.unsubscribe()
            t2.unsubscribe() 
            t4.unsubscribe()
        except Exception:
            pass

        if writer is not None:
            writer.release()
            end_time = time.time()
            real_dur = end_time - rec_start
            print(f"[INFO] Recording: {frame_count} frames in {real_dur:.3f}s")
        if ts_log is not None:
            ts_log.close()

        client.terminate()
        cv2.destroyAllWindows()
        store.cam_exit = True
        print("[INFO] Shut down completed.")

def main():
    parser = argparse.ArgumentParser(description="ROS+Camera 4-View (Low-latency Mode)")
    parser.add_argument('--sensor1', required=True, help='Name of the first sensor')
    parser.add_argument('--sensor2', required=True, help='Name of the second sensor')
    parser.add_argument('--cam-port', type=int, default=4, help='Camera port')
    parser.add_argument('--save-video', action='store_true', help='Save video')
    parser.add_argument('--video_name', default='orange', help='Video name')
    parser.add_argument('--fps', type=float, default=30.0, help='FPS')
    args = parser.parse_args()

    video_name = f'video/img/{args.video_name}.mp4'
    setattr(args, "video", video_name)

    # 创建目录
    os.makedirs(os.path.dirname(video_name), exist_ok=True)

    store.cam_exit = False
    th_cam = threading.Thread(target=cam_thread_function, args=(args.cam_port,), daemon=True)
    th_cam.start()

    client = roslibpy.Ros(host='localhost', port=9090)
    client.run()

    ros_image_thread(
        client, args.sensor1, args.sensor2,
        save_video=args.save_video, video_base=args.video, fps=args.fps
    )
    th_cam.join()

if __name__ == "__main__":
    main()
