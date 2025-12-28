#!/usr/bin/env python3
import time
import sys
import os
import argparse
import threading
from collections import deque
import numpy as np
import cv2
import zmq
import roslibpy
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QMainWindow, QGridLayout, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QComboBox, QCheckBox, QDoubleSpinBox
from PyQt5.QtCore import Qt, QTimer, QPointF
from PyQt5.QtGui import QColor, QPen, QPainter
from PyQt5.QtChart import QChart, QChartView, QLineSeries, QValueAxis
import socket

# Constants
WINDOW_LENGTH = 6  # seconds

class DataStore:
    def __init__(self, max_points=2000):
        self.lock = threading.Lock()

        # ROS data
        self.ros_force_x, self.ros_force_x_t = [], []
        self.ros_force_y, self.ros_force_y_t = [], []
        self.ros_force_z, self.ros_force_z_t = [], []

        # Nano17 data (用于绘图，已经做窗口裁剪/下采样)
        self.nano17_force_x, self.nano17_force_x_t = [], []
        self.nano17_force_y, self.nano17_force_y_t = [], []
        self.nano17_force_z, self.nano17_force_z_t = [], []

        # 高频采集缓冲队列（采集线程 -> GUI 消费）
        self.nano17_queue = deque(maxlen=10000)
        self.max_points = max_points

        self.start_time = None


class ForceVisualizer(QMainWindow):
    def __init__(self, ros_sensor_type=None, video_width=1200, video_height=800):
        super().__init__()

        # Parameters
        self.zmq_host = "192.168.1.118"
        self.zmq_port = 5555
        self.video_width = video_width
        self.video_height = video_height
        self.update_interval = 50  # ms
        self.x_timeMax = WINDOW_LENGTH
        self.ros_sensor_type = ros_sensor_type

        # Video recording parameters
        self.save_video = False
        self.video_fps = 30.0
        self.video_writer = None
        self.ts_log = None

        # 录制时钟与节拍
        self.recording_start_time = None  # 使用 time.time() 与参考保持一致
        self.frame_interval = 1.0 / self.video_fps
        self.next_tick_time = None
        self.frame_count = 0
        self.last_frame_bgr = None  # 重复帧缓存

        # 视频异步写入
        self.video_queue = deque(maxlen=600)  # 略大，避免短暂阻塞导致丢最新帧
        self.video_thread = None
        self.video_thread_stop = threading.Event()

        # Data store
        self.store = DataStore()

        # Exit event
        self.exit_event = threading.Event()

        # ZMQ setup
        self.socket = None
        self.context = None
        self.connected = False
        self.connection_retries = 0
        self.max_retries = 10

        # ZMQ reader thread
        self.zmq_thread = None
        self.zmq_thread_stop = threading.Event()

        # ROS setup
        self.ros_client = None
        self.ros_connected = False

        # Setup UI
        self.init_ui()

        # Start update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_data)

        # Status update timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.check_connection)

        # Auto connect ROS
        self.auto_connect_ros()

        # Chart axis update throttle
        self._axis_update_counter = 0

    def auto_connect_ros(self):
        QTimer.singleShot(500, self._delayed_ros_connect)

    def _delayed_ros_connect(self):
        self.status_label.setText("Auto-connecting to ROS...")
        self.status_label.setStyleSheet("color: orange; font-weight: bold;")

        self.setup_ros_connection()

        if self.ros_connected:
            self.timer.start(self.update_interval)
            self.status_timer.start(5000)

    def get_local_ip(self):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(('10.255.255.255', 1))
            local_ip = s.getsockname()[0]
            s.close()
            return local_ip
        except Exception:
            return "127.0.0.1"

    def init_ui(self):
        self.setWindowTitle("Force Sensor Comparison with Video Recording")
        self.setGeometry(100, 100, self.video_width, self.video_height)

        main_widget = QWidget()
        main_layout = QVBoxLayout()

        # Connection panel
        control_panel = QWidget()
        control_layout = QHBoxLayout()

        control_layout.addWidget(QLabel("Nano17 Server IP:"))
        self.ip_input = QLineEdit("192.168.1.118")
        control_layout.addWidget(self.ip_input)

        control_layout.addWidget(QLabel("Port:"))
        self.port_input = QLineEdit("5555")
        control_layout.addWidget(self.port_input)

        control_layout.addWidget(QLabel("ROS Sensor:"))
        self.sensor_combo = QComboBox()
        self.sensor_combo.addItems(["AII", "uskin"])
        if self.ros_sensor_type:
            index = self.sensor_combo.findText(self.ros_sensor_type)
            if index >= 0:
                self.sensor_combo.setCurrentIndex(index)
        control_layout.addWidget(self.sensor_combo)

        self.connect_button = QPushButton("Connect ZMQ")
        self.connect_button.clicked.connect(self.connect_clicked)
        control_layout.addWidget(self.connect_button)

        control_panel.setLayout(control_layout)
        main_layout.addWidget(control_panel)

        # Video panel
        video_panel = QWidget()
        video_layout = QHBoxLayout()

        self.record_checkbox = QCheckBox("Record Video")
        self.record_checkbox.stateChanged.connect(self.on_record_checkbox_changed)
        video_layout.addWidget(self.record_checkbox)

        video_layout.addWidget(QLabel("Video Name:"))
        self.video_name_input = QLineEdit("force_comparison")
        video_layout.addWidget(self.video_name_input)

        video_layout.addWidget(QLabel("FPS:"))
        self.fps_spinbox = QDoubleSpinBox()
        self.fps_spinbox.setRange(1.0, 120.0)
        self.fps_spinbox.setValue(30.0)
        self.fps_spinbox.setSingleStep(1.0)
        video_layout.addWidget(self.fps_spinbox)

        self.record_button = QPushButton("Start Recording")
        self.record_button.clicked.connect(self.toggle_recording)
        self.record_button.setEnabled(False)
        video_layout.addWidget(self.record_button)

        self.record_status_label = QLabel("Not Recording")
        self.record_status_label.setStyleSheet("color: gray; font-weight: bold;")
        video_layout.addWidget(self.record_status_label)

        video_panel.setLayout(video_layout)
        main_layout.addWidget(video_panel)

        # Status label
        self.status_label = QLabel("Starting up...")
        self.status_label.setStyleSheet("font-weight: bold;")
        main_layout.addWidget(self.status_label)

        # Charts
        chart_widget = QWidget()
        chart_layout = QGridLayout()
        chart_widget.setLayout(chart_layout)
        main_layout.addWidget(chart_widget)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        self.charts = []
        self.x_axis = []
        self.y_axis = []

        self.setup_force_charts(chart_layout)

    def on_record_checkbox_changed(self, state):
        self.record_button.setEnabled(state == Qt.Checked)
        if state != Qt.Checked and self.save_video:
            self.stop_recording()

    def toggle_recording(self):
        if not self.save_video:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        if self.save_video:
            return

        # Params
        self.video_fps = self.fps_spinbox.value()
        self.frame_interval = 1.0 / self.video_fps
        video_name = self.video_name_input.text().strip() or "force_comparison"

        video_dir = "video/force"
        os.makedirs(video_dir, exist_ok=True)

        now = datetime.now()
        timestamp_str = now.strftime('%Y%m%d_%H%M%S')
        video_filename = os.path.join(video_dir, f"{video_name}_{timestamp_str}.mp4")

        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(video_filename, fourcc, self.video_fps, (self.video_width, self.video_height))

            self.ts_log = open(video_filename + '.frame_ts.txt', 'w')

            # 用墙钟作为时间轴（与参考一致）
            self.recording_start_time = time.time()
            self.next_tick_time = self.recording_start_time
            self.frame_count = 0
            self.last_frame_bgr = None
            self.save_video = True

            # start video writer thread
            self.video_thread_stop.clear()
            self.video_thread = threading.Thread(target=self._video_writer_loop, daemon=True)
            self.video_thread.start()

            self.record_button.setText("Stop Recording")
            self.record_button.setStyleSheet("background-color: red; color: white; font-weight: bold;")
            self.record_status_label.setText(f"Recording to: {os.path.basename(video_filename)}")
            self.record_status_label.setStyleSheet("color: red; font-weight: bold;")

            print(f"[INFO] Started recording video: {video_filename}")
            print(f"[INFO] Recording parameters: FPS={self.video_fps}, Size={self.video_width}x{self.video_height}")

        except Exception as e:
            print(f"[ERROR] Failed to start recording: {e}")
            self.record_status_label.setText(f"Recording failed: {str(e)}")
            self.record_status_label.setStyleSheet("color: red; font-weight: bold;")

    def stop_recording(self):
        if not self.save_video:
            return

        self.save_video = False

        # stop writer thread
        self.video_thread_stop.set()
        if self.video_thread is not None:
            self.video_thread.join(timeout=2.0)
            self.video_thread = None

        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        if self.ts_log is not None:
            self.ts_log.close()
            self.ts_log = None

        if self.recording_start_time is not None:
            end_time = time.time()
            real_duration = end_time - self.recording_start_time
            expected_frames = real_duration * self.video_fps

            print(f"[INFO] Recording stopped. Statistics:")
            print(f"  - Real duration: {real_duration:.3f}s")
            print(f"  - Target FPS: {self.video_fps}")
            print(f"  - Frames written: {self.frame_count}")
            print(f"  - Expected frames: ~{expected_frames:.0f}")
            print(f"  - Video duration: {self.frame_count/self.video_fps:.3f}s")

        self.record_button.setText("Start Recording")
        self.record_button.setStyleSheet("")
        self.record_status_label.setText("Recording stopped")
        self.record_status_label.setStyleSheet("color: gray; font-weight: bold;")

        self.recording_start_time = None
        self.next_tick_time = None
        self.frame_count = 0
        self.last_frame_bgr = None

    def _video_writer_loop(self):
        while not self.video_thread_stop.is_set() or self.video_queue:
            if self.video_queue:
                frame_bgr, ts = self.video_queue.popleft()
                try:
                    if self.video_writer is not None:
                        self.video_writer.write(frame_bgr)
                    if self.ts_log is not None:
                        self.ts_log.write(f'{ts:.6f}\n')
                except Exception as e:
                    print(f"[VideoWriter] write error: {e}")
            else:
                time.sleep(0.001)

    def _grab_current_frame(self):
        # 抓取窗口 -> QImage -> numpy，不落盘
        pixmap = self.grab()
        qimg = pixmap.toImage().convertToFormat(4)  # QImage.Format_RGBA8888 = 4
        width = qimg.width()
        height = qimg.height()
        ptr = qimg.bits()
        ptr.setsize(qimg.byteCount())
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
        frame_bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        if (width, height) != (self.video_width, self.video_height):
            frame_bgr = cv2.resize(frame_bgr, (self.video_width, self.video_height), interpolation=cv2.INTER_AREA)
        return frame_bgr

    def capture_frame_for_video(self):
        if not self.save_video or self.video_writer is None:
            return

        now = time.time()

        # 到达或超过节拍时，按 tick 写入帧；必要时重复上一帧，保持时间轴不缩短
        wrote_any = False
        while now >= self.next_tick_time:
            # 若没有缓存帧，则抓一次作为本轮所有重复帧的内容
            if self.last_frame_bgr is None:
                try:
                    self.last_frame_bgr = self._grab_current_frame()
                except Exception as e:
                    print(f"[Grab] error: {e}")
                    break  # 本次无法抓取，退出循环避免死循环

            # 入队写线程，时间戳用 tick 时间
            if len(self.video_queue) >= self.video_queue.maxlen:
                # 丢弃最旧，保持实时
                try:
                    self.video_queue.popleft()
                except IndexError:
                    pass
            self.video_queue.append((self.last_frame_bgr, self.next_tick_time))
            self.frame_count += 1
            self.next_tick_time += self.frame_interval
            wrote_any = True

        # 一轮 while 完成后，重置缓存，下一次再抓最新画面
        if wrote_any:
            self.last_frame_bgr = None

        # 更新录制状态显示
        if self.recording_start_time is not None:
            elapsed = now - self.recording_start_time
            self.record_status_label.setText(f"Recording... {elapsed:.1f}s ({self.frame_count} frames)")

    def setup_force_charts(self, layout):
        # X
        chart_fx = QChart()
        chart_fx.setTitle("Fx Comparison (N)")
        series_nano17_fx = QLineSeries()
        series_nano17_fx.setName("Nano17 Fx")
        series_nano17_fx.setColor(QColor("blue"))
        series_ros_fx = QLineSeries()
        series_ros_fx.setName("ROS Sensor Fx")
        series_ros_fx.setColor(QColor("darkblue"))
        dash_pen = QPen(QColor("darkblue"))
        dash_pen.setStyle(Qt.DashLine)
        dash_pen.setWidth(1)
        series_ros_fx.setPen(dash_pen)
        chart_fx.addSeries(series_nano17_fx)
        chart_fx.addSeries(series_ros_fx)

        # Y
        chart_fy = QChart()
        chart_fy.setTitle("Fy Comparison (N)")
        series_nano17_fy = QLineSeries()
        series_nano17_fy.setName("Nano17 Fy")
        series_nano17_fy.setColor(QColor("green"))
        series_ros_fy = QLineSeries()
        series_ros_fy.setName("ROS Sensor Fy")
        series_ros_fy.setColor(QColor("darkgreen"))
        dash_pen_green = QPen(QColor("darkgreen"))
        dash_pen_green.setStyle(Qt.DashLine)
        dash_pen_green.setWidth(1)
        series_ros_fy.setPen(dash_pen_green)
        chart_fy.addSeries(series_nano17_fy)
        chart_fy.addSeries(series_ros_fy)

        # Z
        chart_fz = QChart()
        chart_fz.setTitle("Fz Comparison (N)")
        series_nano17_fz = QLineSeries()
        series_nano17_fz.setName("Nano17 Fz")
        series_nano17_fz.setColor(QColor("red"))
        series_ros_fz = QLineSeries()
        series_ros_fz.setName("ROS Sensor Fz")
        series_ros_fz.setColor(QColor("darkred"))
        dash_pen_red = QPen(QColor("darkred"))
        dash_pen_red.setStyle(Qt.DashLine)
        dash_pen_red.setWidth(1)
        series_ros_fz.setPen(dash_pen_red)
        chart_fz.addSeries(series_nano17_fz)
        chart_fz.addSeries(series_ros_fz)

        for i, chart in enumerate([chart_fx, chart_fy, chart_fz]):
            axis_x = QValueAxis()
            axis_x.setRange(0, self.x_timeMax)
            axis_x.setLabelFormat("%0.1f")
            axis_x.setTitleText("Time (s)")
            chart.addAxis(axis_x, Qt.AlignBottom)

            axis_y = QValueAxis()
            axis_y.setRange(-1, 1)
            axis_y.setTitleText("Force (N)")
            chart.addAxis(axis_y, Qt.AlignLeft)

            for s in chart.series():
                s.attachAxis(axis_x)
                s.attachAxis(axis_y)

            chart.legend().hide()  # 可选，减少绘制

            chart_view = QChartView(chart)
            chart_view.setRenderHint(QPainter.Antialiasing)
            layout.addWidget(chart_view, 0, i)

            self.charts.append(chart)
            self.x_axis.append(axis_x)
            self.y_axis.append(axis_y)

        self.series_nano17_fx = series_nano17_fx
        self.series_nano17_fy = series_nano17_fy
        self.series_nano17_fz = series_nano17_fz
        self.series_ros_fx = series_ros_fx
        self.series_ros_fy = series_ros_fy
        self.series_ros_fz = series_ros_fz

        # 可选：启用 OpenGL（部分平台/Qt 版本可能不稳定）
        try:
            self.series_nano17_fx.setUseOpenGL(True)
            self.series_ros_fx.setUseOpenGL(True)
            self.series_nano17_fy.setUseOpenGL(True)
            self.series_ros_fy.setUseOpenGL(True)
            self.series_nano17_fz.setUseOpenGL(True)
            self.series_ros_fz.setUseOpenGL(True)
        except Exception:
            pass

    def setup_zmq_connection(self):
        try:
            if hasattr(self, 'socket') and self.socket:
                self.socket.close()
            if hasattr(self, 'context') and self.context:
                self.context.term()

            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.SUB)
            self.socket.setsockopt(zmq.LINGER, 0)
            self.socket.setsockopt(zmq.RCVTIMEO, 0)  # 非阻塞
            connection_string = f"tcp://{self.zmq_host}:{self.zmq_port}"
            self.socket.connect(connection_string)
            self.socket.setsockopt_string(zmq.SUBSCRIBE, "")

            self.connected = True
            self.connection_retries = 0
            self.update_status_label()
            print(f"Connected to ZMQ publisher at {self.zmq_host}:{self.zmq_port}")

            # 启动采集线程
            if self.zmq_thread is None or not self.zmq_thread.is_alive():
                self.zmq_thread_stop.clear()
                self.zmq_thread = threading.Thread(target=self.zmq_reader_loop, daemon=True)
                self.zmq_thread.start()

        except Exception as e:
            self.connected = False
            self.connection_retries += 1
            self.status_label.setText(f"ZMQ connection error: {str(e)}")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
            print(f"Error connecting to ZeroMQ server: {e}")

    def zmq_reader_loop(self):
        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)
        while not self.zmq_thread_stop.is_set():
            try:
                socks = dict(poller.poll(5))  # 5 ms
                if self.socket in socks and socks[self.socket] == zmq.POLLIN:
                    msg = self.socket.recv_json(flags=zmq.NOBLOCK)
                    ft_values = msg.get("force_torque")
                    if not ft_values or len(ft_values) < 3:
                        continue
                    t_now = time.time()
                    with self.store.lock:
                        if self.store.start_time is None:
                            self.store.start_time = t_now
                        rel_t = t_now - self.store.start_time
                        self.store.nano17_queue.append((
                            rel_t,
                            float(ft_values[0]),
                            float(ft_values[1]),
                            float(ft_values[2])
                        ))
            except zmq.Again:
                continue
            except Exception as e:
                print(f"[ZMQ reader] error: {e}")
                time.sleep(0.001)

    def setup_ros_connection(self):
        try:
            self.ros_client = roslibpy.Ros(host='localhost', port=9090)
            self.ros_client.run()

            if self.ros_client.is_connected:
                self.ros_connected = True
                self.ros_sensor_type = self.sensor_combo.currentText()
                print(f"[roslibpy] Connected to rosbridge websocket. Sensor type: {self.ros_sensor_type}")
                self.subscribe_to_ros_topics()
                self.update_status_label()
            else:
                self.ros_connected = False
                self.status_label.setText("Failed to connect to ROS bridge")
                self.status_label.setStyleSheet("color: red; font-weight: bold;")

        except Exception as e:
            self.ros_connected = False
            self.status_label.setText(f"ROS connection error: {str(e)}")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
            print(f"Error connecting to ROS bridge: {e}")

    def update_status_label(self):
        zmq_status = "Connected" if self.connected else "Disconnected"
        ros_status = f"Connected to {self.ros_sensor_type}" if self.ros_connected else "Disconnected"

        self.status_label.setText(f"ZMQ: {zmq_status} | ROS: {ros_status}")

        if self.connected and self.ros_connected:
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
        elif self.connected or self.ros_connected:
            self.status_label.setStyleSheet("color: orange; font-weight: bold;")
        else:
            self.status_label.setStyleSheet("color: red; font-weight: bold;")

    def subscribe_to_ros_topics(self):
        sensor_name = self.ros_sensor_type

        def make_cb(data_list, t_list):
            def cb(msg):
                t_now = time.time()
                with self.store.lock:
                    if self.store.start_time is None:
                        self.store.start_time = t_now
                    rel_t = t_now - self.store.start_time
                    t_list.append(rel_t)
                    data_list.append(msg['data'])
                    # Window trim
                    while t_list and (rel_t - t_list[0]) > WINDOW_LENGTH:
                        t_list.pop(0)
                        data_list.pop(0)
            return cb

        topics = {
            f'/force/{sensor_name}/x': (self.store.ros_force_x, self.store.ros_force_x_t),
            f'/force/{sensor_name}/y': (self.store.ros_force_y, self.store.ros_force_y_t),
            f'/force/{sensor_name}/z': (self.store.ros_force_z, self.store.ros_force_z_t),
        }

        for topic_name, (data_list, t_list) in topics.items():
            topic = roslibpy.Topic(self.ros_client, topic_name, 'std_msgs/Float32')
            topic.subscribe(make_cb(data_list, t_list))
            print(f"Subscribed to: {topic_name}")

    def check_connection(self):
        if not self.connected and self.connection_retries < self.max_retries:
            print(f"Attempting to reconnect ZMQ... (attempt {self.connection_retries + 1}/{self.max_retries})")
            self.setup_zmq_connection()
        elif not self.connected:
            self.status_label.setText(f"Failed to connect to ZMQ after {self.max_retries} attempts")

        if self.ros_client and not self.ros_client.is_connected:
            self.ros_connected = False
            print("ROS connection lost, attempting to reconnect...")
            self.setup_ros_connection()

        self.update_status_label()

    def connect_clicked(self):
        self.zmq_host = self.ip_input.text().strip()
        try:
            self.zmq_port = int(self.port_input.text().strip())
        except ValueError:
            self.status_label.setText("Invalid port number")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
            return

        self.connected = False
        self.connection_retries = 0

        self.status_label.setText(f"Connecting to ZMQ: {self.zmq_host}:{self.zmq_port}...")
        self.status_label.setStyleSheet("color: orange; font-weight: bold;")

        self.setup_zmq_connection()

        if not self.timer.isActive():
            self.timer.start(self.update_interval)
            self.status_timer.start(5000)

    def update_data(self):
        now = time.time()
        window_len = WINDOW_LENGTH

        # 消费高频队列，窗口裁剪，下采样
        with self.store.lock:
            while self.store.nano17_queue:
                t, fx, fy, fz = self.store.nano17_queue.popleft()
                self.store.nano17_force_x_t.append(t)
                self.store.nano17_force_x.append(fx)
                self.store.nano17_force_y_t.append(t)
                self.store.nano17_force_y.append(fy)
                self.store.nano17_force_z_t.append(t)
                self.store.nano17_force_z.append(fz)

            def trim_window(ts, xs):
                while ts and (ts[-1] - ts[0]) > window_len:
                    ts.pop(0); xs.pop(0)

            trim_window(self.store.nano17_force_x_t, self.store.nano17_force_x)
            trim_window(self.store.nano17_force_y_t, self.store.nano17_force_y)
            trim_window(self.store.nano17_force_z_t, self.store.nano17_force_z)

            trim_window(self.store.ros_force_x_t, self.store.ros_force_x)
            trim_window(self.store.ros_force_y_t, self.store.ros_force_y)
            trim_window(self.store.ros_force_z_t, self.store.ros_force_z)

            # 下采样（仅对 Nano17）
            target_points = 200  # 可调 200~600
            def downsample(ts, xs, n=target_points):
                if len(ts) <= n or n <= 0:
                    return ts[:], xs[:]
                step = len(ts) / n
                idxs = [int(i * step) for i in range(n)]
                return [ts[i] for i in idxs], [xs[i] for i in idxs]

            nx_t, nx = downsample(self.store.nano17_force_x_t, self.store.nano17_force_x)
            ny_t, ny = downsample(self.store.nano17_force_y_t, self.store.nano17_force_y)
            nz_t, nz = downsample(self.store.nano17_force_z_t, self.store.nano17_force_z)

            # 缓存给绘图函数，避免反复持锁
            self._plot_cache = {
                'nx': (nx_t, nx),
                'ny': (ny_t, ny),
                'nz': (nz_t, nz),
                'rx': (self.store.ros_force_x_t[:], self.store.ros_force_x[:]),
                'ry': (self.store.ros_force_y_t[:], self.store.ros_force_y[:]),
                'rz': (self.store.ros_force_z_t[:], self.store.ros_force_z[:]),
            }

        self.update_charts_from_cache(now)
        self.capture_frame_for_video()

    def update_charts_from_cache(self, current_time):
        c = getattr(self, '_plot_cache', None)
        if not c:
            return

        def to_points(ts, xs):
            return [QPointF(t, x) for t, x in zip(ts, xs)]

        self.series_nano17_fx.replace(to_points(*c['nx']))
        self.series_ros_fx.replace(to_points(*c['rx']))

        self.series_nano17_fy.replace(to_points(*c['ny']))
        self.series_ros_fy.replace(to_points(*c['ry']))

        self.series_nano17_fz.replace(to_points(*c['nz']))
        self.series_ros_fz.replace(to_points(*c['rz']))

        # 滚动 X 轴
        with self.store.lock:
            if self.store.start_time:
                current_rel_time = current_time - self.store.start_time
                if current_rel_time > WINDOW_LENGTH:
                    start_time = current_rel_time - WINDOW_LENGTH
                    end_time = current_rel_time
                else:
                    start_time = 0
                    end_time = WINDOW_LENGTH
            else:
                start_time = 0
                end_time = WINDOW_LENGTH

        for axis_x in self.x_axis:
            axis_x.setRange(start_time, end_time)

        # 降低 Y 轴的动态更新频率（每 5 次）
        self._axis_update_counter += 1
        if self._axis_update_counter % 5 == 0:
            for i, chart in enumerate(self.charts):
                all_y_values = []
                for series in chart.series():
                    pts = series.pointsVector()
                    if pts:
                        all_y_values.extend([p.y() for p in pts])
                if all_y_values:
                    mn, mx = min(all_y_values), max(all_y_values)
                    pad = max(0.1, (mx - mn) * 0.1)
                    self.y_axis[i].setRange(mn - pad, mx + pad)

    def closeEvent(self, event):
        # Stop recording if active
        if self.save_video:
            self.stop_recording()

        self.timer.stop()
        self.status_timer.stop()

        # 停止后台 ZMQ 线程
        if self.zmq_thread is not None:
            self.zmq_thread_stop.set()
            self.zmq_thread.join(timeout=1.0)
            self.zmq_thread = None

        # ZMQ 资源
        if hasattr(self, 'socket') and self.socket:
            try:
                self.socket.close()
            except Exception:
                pass
        if hasattr(self, 'context') and self.context:
            try:
                self.context.term()
            except Exception:
                pass

        # ROS 资源
        if hasattr(self, 'ros_client') and self.ros_client:
            try:
                if self.ros_client.is_connected:
                    self.ros_client.terminate()
            except Exception:
                pass

        self.exit_event.set()
        event.accept()


def main():
    parser = argparse.ArgumentParser(description="Force Comparison Visualizer with Video Recording")
    parser.add_argument('--sensor', default='AII', help='ROS sensor type (AII or uskin)')
    parser.add_argument('--width', type=int, default=3200, help='Window width (pixels)')
    parser.add_argument('--height', type=int, default=1000, help='Window height (pixels)')
    args = parser.parse_args()

    app = QApplication(sys.argv)
    visualizer = ForceVisualizer(ros_sensor_type=args.sensor, video_width=args.width, video_height=args.height)
    visualizer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()