#!/usr/bin/env python3  
import time  
import sys  
import os  
import argparse  
import threading  
import numpy as np  
import cv2  
import zmq  
import roslibpy  
from datetime import datetime  
from PyQt5.QtWidgets import QApplication, QMainWindow, QGridLayout, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QComboBox  
from PyQt5.QtCore import Qt, QTimer, QPointF  
from PyQt5.QtGui import QColor, QPen , QPainter  # 添加QPen导入  
from PyQt5.QtChart import QChart, QChartView, QLineSeries, QValueAxis  
import socket  

# Constants  
WINDOW_LENGTH = 6  

class DataStore:  
    def __init__(self):  
        self.lock = threading.Lock()  
        
        # ROS data  
        self.ros_force_x, self.ros_force_x_t = [], []  
        self.ros_force_y, self.ros_force_y_t = [], []  
        self.ros_force_z, self.ros_force_z_t = [], []  
        
        # ZMQ data (Nano17)  
        self.nano17_force_x, self.nano17_force_x_t = [], []  
        self.nano17_force_y, self.nano17_force_y_t = [], []  
        self.nano17_force_z, self.nano17_force_z_t = [], []  
        
        self.start_time = None  

class ForceVisualizer(QMainWindow):  
    def __init__(self, ros_sensor_type=None, video_width=1200, video_height=800):  
        super().__init__()  
        
        # Parameters  
        self.zmq_host = "192.168.1.118"  # 修改默认IP  
        self.zmq_port = 5555  
        self.video_width = video_width  
        self.video_height = video_height  
        self.update_interval = 50  # ms  
        self.x_timeMax = WINDOW_LENGTH  
        self.ros_sensor_type = ros_sensor_type  # "gelsight" or "uskin"  
        
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
        
        # 自动连接ROS  
        self.auto_connect_ros()  
        
    def auto_connect_ros(self):  
        """自动连接ROS并开始数据更新"""  
        # 延迟一点时间让UI完全初始化  
        QTimer.singleShot(500, self._delayed_ros_connect)  
        
    def _delayed_ros_connect(self):  
        """延迟执行的ROS连接"""  
        self.status_label.setText("Auto-connecting to ROS...")  
        self.status_label.setStyleSheet("color: orange; font-weight: bold;")  
        
        # 连接ROS  
        self.setup_ros_connection()  
        
        # 开始更新定时器  
        if self.ros_connected:  
            self.timer.start(self.update_interval)  
            self.status_timer.start(5000)  
        
    def get_local_ip(self):  
        """Get the local IP address of this machine."""  
        try:  
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  
            s.connect(('10.255.255.255', 1))  
            local_ip = s.getsockname()[0]  
            s.close()  
            return local_ip  
        except Exception:  
            return "127.0.0.1"  
        
    def init_ui(self):  
        """Initialize the user interface with charts."""  
        self.setWindowTitle("Force Sensor Comparison")  
        self.setGeometry(100, 100, self.video_width, self.video_height)  
        
        # Create main widget and layout  
        main_widget = QWidget()  
        main_layout = QVBoxLayout()  
        
        # Create connection control panel  
        control_panel = QWidget()  
        control_layout = QHBoxLayout()  
        
        # IP address input - 修改默认IP为192.168.1.118  
        control_layout.addWidget(QLabel("Nano17 Server IP:"))  
        self.ip_input = QLineEdit("192.168.1.118")  
        control_layout.addWidget(self.ip_input)  
        
        # Port input  
        control_layout.addWidget(QLabel("Port:"))  
        self.port_input = QLineEdit("5555")  
        control_layout.addWidget(self.port_input)  
        
        # Sensor type selection  
        control_layout.addWidget(QLabel("ROS Sensor:"))  
        self.sensor_combo = QComboBox()  
        self.sensor_combo.addItems(["DI", "uskin"])  
        if self.ros_sensor_type:  
            index = self.sensor_combo.findText(self.ros_sensor_type)  
            if index >= 0:  
                self.sensor_combo.setCurrentIndex(index)  
        control_layout.addWidget(self.sensor_combo)  
        
        # Connect button  
        self.connect_button = QPushButton("Connect ZMQ")  # 修改按钮文字  
        self.connect_button.clicked.connect(self.connect_clicked)  
        control_layout.addWidget(self.connect_button)  
        
        control_panel.setLayout(control_layout)  
        main_layout.addWidget(control_panel)  
        
        # Add status label  
        self.status_label = QLabel("Starting up...")  
        self.status_label.setStyleSheet("font-weight: bold;")  
        main_layout.addWidget(self.status_label)  
        
        # Create chart widget and layout  
        chart_widget = QWidget()  
        chart_layout = QGridLayout()  
        chart_widget.setLayout(chart_layout)  
        main_layout.addWidget(chart_widget)  
        
        main_widget.setLayout(main_layout)  
        self.setCentralWidget(main_widget)  
        
        self.charts = []  
        self.series = []  
        self.x_axis = []  
        self.y_axis = []  
        
        # Create charts for Nano17 (Fx, Fy, Fz)  
        self.setup_force_charts(chart_layout)  
        
    def setup_force_charts(self, layout):  
        """Setup force comparison charts."""  
        # X Force comparison (Nano17 vs ROS sensor)  
        chart_fx = QChart()  
        chart_fx.setTitle("Fx Comparison (N)")  
        
        # Series for Nano17 Fx and ROS Sensor Fx  
        series_nano17_fx = QLineSeries()  
        series_nano17_fx.setName("Nano17 Fx")  
        series_nano17_fx.setColor(QColor("blue"))  
        
        series_ros_fx = QLineSeries()  
        series_ros_fx.setName("ROS Sensor Fx")  
        series_ros_fx.setColor(QColor("darkblue"))  
        
        # 创建一个虚线QPen对象  
        dash_pen = QPen(QColor("darkblue"))  
        dash_pen.setStyle(Qt.DashLine)  
        dash_pen.setWidth(1)  
        series_ros_fx.setPen(dash_pen)  
        
        chart_fx.addSeries(series_nano17_fx)  
        chart_fx.addSeries(series_ros_fx)  
        
        # Y Force comparison (Nano17 vs ROS sensor)  
        chart_fy = QChart()  
        chart_fy.setTitle("Fy Comparison (N)")  
        
        # Series for Nano17 Fy and ROS Sensor Fy  
        series_nano17_fy = QLineSeries()  
        series_nano17_fy.setName("Nano17 Fy")  
        series_nano17_fy.setColor(QColor("green"))  
        
        series_ros_fy = QLineSeries()  
        series_ros_fy.setName("ROS Sensor Fy")  
        series_ros_fy.setColor(QColor("darkgreen"))  
        
        # 创建一个虚线QPen对象  
        dash_pen_green = QPen(QColor("darkgreen"))  
        dash_pen_green.setStyle(Qt.DashLine)  
        dash_pen_green.setWidth(1)  
        series_ros_fy.setPen(dash_pen_green)  
        
        chart_fy.addSeries(series_nano17_fy)  
        chart_fy.addSeries(series_ros_fy)  
        
        # Z Force comparison (Nano17 vs ROS sensor)  
        chart_fz = QChart()  
        chart_fz.setTitle("Fz Comparison (N)")  
        
        # Series for Nano17 Fz and ROS Sensor Fz  
        series_nano17_fz = QLineSeries()  
        series_nano17_fz.setName("Nano17 Fz")  
        series_nano17_fz.setColor(QColor("red"))  
        
        series_ros_fz = QLineSeries()  
        series_ros_fz.setName("ROS Sensor Fz")  
        series_ros_fz.setColor(QColor("darkred"))  
        
        # 创建一个虚线QPen对象  
        dash_pen_red = QPen(QColor("darkred"))  
        dash_pen_red.setStyle(Qt.DashLine)  
        dash_pen_red.setWidth(1)  
        series_ros_fz.setPen(dash_pen_red)  
        
        chart_fz.addSeries(series_nano17_fz)  
        chart_fz.addSeries(series_ros_fz)  
        
        # Setup axes for all charts  
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
            
            # Attach axes to all series in the chart  
            for series in chart.series():  
                series.attachAxis(axis_x)  
                series.attachAxis(axis_y)  
            
            # Create chart view and add to layout  
            chart_view = QChartView(chart)  
            chart_view.setRenderHint(QPainter.Antialiasing)  
            
            layout.addWidget(chart_view, 0, i)  
            
            # Store references  
            self.charts.append(chart)  
            self.x_axis.append(axis_x)  
            self.y_axis.append(axis_y)  
        
        # Store series for later updates  
        self.series_nano17_fx = series_nano17_fx  
        self.series_nano17_fy = series_nano17_fy  
        self.series_nano17_fz = series_nano17_fz  
        self.series_ros_fx = series_ros_fx  
        self.series_ros_fy = series_ros_fy  
        self.series_ros_fz = series_ros_fz  
    
    def setup_zmq_connection(self):  
        """Setup ZeroMQ connection with proper error handling."""  
        try:  
            # Clean up existing connection if any  
            if hasattr(self, 'socket') and self.socket:  
                self.socket.close()  
            if hasattr(self, 'context') and self.context:  
                self.context.term()  
                
            # Create new connection  
            self.context = zmq.Context()  
            self.socket = self.context.socket(zmq.SUB)  
            
            # Set connection timeout to 1 second for faster feedback  
            self.socket.setsockopt(zmq.LINGER, 0)  
            self.socket.setsockopt(zmq.RCVTIMEO, 1000)  
            
            # Connect to the server  
            connection_string = f"tcp://{self.zmq_host}:{self.zmq_port}"  
            self.socket.connect(connection_string)  
            self.socket.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all messages  
            
            self.connected = True  
            self.connection_retries = 0  
            self.update_status_label()  
            print(f"Connected to ZMQ publisher at {self.zmq_host}:{self.zmq_port}")  
            
        except Exception as e:  
            self.connected = False  
            self.connection_retries += 1  
            self.status_label.setText(f"ZMQ connection error: {str(e)}")  
            self.status_label.setStyleSheet("color: red; font-weight: bold;")  
            print(f"Error connecting to ZeroMQ server: {e}")  
    
    def setup_ros_connection(self):  
        """Setup ROS connection with proper error handling."""  
        try:  
            self.ros_client = roslibpy.Ros(host='localhost', port=9090)  
            self.ros_client.run()  
            
            if self.ros_client.is_connected:  
                self.ros_connected = True  
                self.ros_sensor_type = self.sensor_combo.currentText()  
                print(f"[roslibpy] Connected to rosbridge websocket. Sensor type: {self.ros_sensor_type}")  
                
                # Subscribe to ROS topics  
                self.subscribe_to_ros_topics()  
                
                # Update status  
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
        """更新状态标签"""
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
        """Subscribe to ROS force topics for the selected sensor."""  
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
                    
                    # Keep only data within the time window  
                    while t_list and (rel_t - t_list[0]) > WINDOW_LENGTH:  
                        t_list.pop(0)  
                        data_list.pop(0)  
            return cb  
        
        # Subscribe to X, Y, Z force topics  
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
        """Check if connections are active and reconnect if necessary."""
        # Check ZMQ connection
        if not self.connected and self.connection_retries < self.max_retries:
            print(f"Attempting to reconnect ZMQ... (attempt {self.connection_retries + 1}/{self.max_retries})")
            self.setup_zmq_connection()
        elif not self.connected:
            self.status_label.setText(f"Failed to connect to ZMQ after {self.max_retries} attempts")
        
        # Check ROS connection
        if self.ros_client and not self.ros_client.is_connected:
            self.ros_connected = False
            print("ROS connection lost, attempting to reconnect...")
            self.setup_ros_connection()
            
        # Update status
        self.update_status_label()
    
    def connect_clicked(self):
        """Handle connect button click."""
        # Get IP and port for ZMQ connection
        self.zmq_host = self.ip_input.text().strip()
        try:
            self.zmq_port = int(self.port_input.text().strip())
        except ValueError:
            self.status_label.setText("Invalid port number")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
            return
        
        # Reset connection states
        self.connected = False
        self.connection_retries = 0
        
        # Attempt to connect to ZMQ
        self.status_label.setText(f"Connecting to ZMQ: {self.zmq_host}:{self.zmq_port}...")
        self.status_label.setStyleSheet("color: orange; font-weight: bold;")
        
        # Setup ZMQ connection
        self.setup_zmq_connection()
        
        # Start timer if not already running
        if not self.timer.isActive():
            self.timer.start(self.update_interval)
            self.status_timer.start(5000)  # Check connection every 5 seconds
    
    def update_data(self):
        """Update data from both ZMQ and ROS sources."""
        current_time = time.time()
        
        # Update ZMQ data if connected
        if self.connected:
            try:
                # Try to receive data from Nano17
                message = self.socket.recv_json()
                ft_values = message["force_torque"]
                
                # Extract force components (first three values: Fx, Fy, Fz)
                with self.store.lock:
                    if self.store.start_time is None:
                        self.store.start_time = current_time
                    rel_t = current_time - self.store.start_time
                    
                    # Update Nano17 force data
                    self.store.nano17_force_x.append(ft_values[0])
                    self.store.nano17_force_x_t.append(rel_t)
                    
                    self.store.nano17_force_y.append(ft_values[1])
                    self.store.nano17_force_y_t.append(rel_t)
                    
                    self.store.nano17_force_z.append(ft_values[2])
                    self.store.nano17_force_z_t.append(rel_t)
                    
                    # Keep only data within the time window
                    while self.store.nano17_force_x_t and (rel_t - self.store.nano17_force_x_t[0]) > WINDOW_LENGTH:
                        self.store.nano17_force_x.pop(0)
                        self.store.nano17_force_x_t.pop(0)
                        
                    while self.store.nano17_force_y_t and (rel_t - self.store.nano17_force_y_t[0]) > WINDOW_LENGTH:
                        self.store.nano17_force_y.pop(0)
                        self.store.nano17_force_y_t.pop(0)
                        
                    while self.store.nano17_force_z_t and (rel_t - self.store.nano17_force_z_t[0]) > WINDOW_LENGTH:
                        self.store.nano17_force_z.pop(0)
                        self.store.nano17_force_z_t.pop(0)
                    
            except zmq.Again:
                # No data received within timeout, do nothing
                pass
            except zmq.ZMQError as e:
                print(f"ZMQ error: {e}")
                self.connected = False
                self.update_status_label()
            except Exception as e:
                print(f"Error receiving ZMQ data: {e}")
        
        # Update the charts
        self.update_charts()
    
    def update_charts(self):
        """Update all charts with the latest data."""
        current_time = time.time()
        
        with self.store.lock:
            # Update Fx chart series
            self.series_nano17_fx.replace([QPointF(t, x) for t, x in zip(self.store.nano17_force_x_t, self.store.nano17_force_x)])
            self.series_ros_fx.replace([QPointF(t, x) for t, x in zip(self.store.ros_force_x_t, self.store.ros_force_x)])
            
            # Update Fy chart series
            self.series_nano17_fy.replace([QPointF(t, y) for t, y in zip(self.store.nano17_force_y_t, self.store.nano17_force_y)])
            self.series_ros_fy.replace([QPointF(t, y) for t, y in zip(self.store.ros_force_y_t, self.store.ros_force_y)])
            
            # Update Fz chart series
            self.series_nano17_fz.replace([QPointF(t, z) for t, z in zip(self.store.nano17_force_z_t, self.store.nano17_force_z)])
            self.series_ros_fz.replace([QPointF(t, z) for t, z in zip(self.store.ros_force_z_t, self.store.ros_force_z)])
            
            # 更新X轴范围以实现滚动效果
            if self.store.start_time:
                current_rel_time = current_time - self.store.start_time
                
                # 如果当前时间超过了窗口长度，开始滚动
                if current_rel_time > WINDOW_LENGTH:
                    start_time = current_rel_time - WINDOW_LENGTH
                    end_time = current_rel_time
                else:
                    start_time = 0
                    end_time = WINDOW_LENGTH
                
                # 更新所有图表的X轴范围
                for axis_x in self.x_axis:
                    axis_x.setRange(start_time, end_time)
            
            # Dynamically update Y-axis ranges
            for i, chart in enumerate(self.charts):
                # Get all data points from all series in this chart
                all_y_values = []
                for series in chart.series():
                    points = series.pointsVector()
                    all_y_values.extend([point.y() for point in points])
                
                if all_y_values:
                    min_value = min(all_y_values)
                    max_value = max(all_y_values)
                    
                    # Add padding to the axis range
                    range_padding = (max_value - min_value) * 0.1
                    if range_padding < 0.1:
                        range_padding = 0.1
                        
                    self.y_axis[i].setRange(min_value - range_padding, max_value + range_padding)
    
    def closeEvent(self, event):
        """Clean up resources when the window is closed."""
        self.timer.stop()
        self.status_timer.stop()
        
        # Clean up ZMQ resources
        if hasattr(self, 'socket') and self.socket:
            self.socket.close()
        if hasattr(self, 'context') and self.context:
            self.context.term()
        
        # Clean up ROS resources
        if hasattr(self, 'ros_client') and self.ros_client:
            if self.ros_client.is_connected:
                self.ros_client.terminate()
        
        self.exit_event.set()
        event.accept()


def main():
    parser = argparse.ArgumentParser(description="Force Comparison Visualizer")
    parser.add_argument('--sensor', default='gelsight', help='ROS sensor type (gelsight or uskin)')
    parser.add_argument('--width', type=int, default=3200, help='Window width (pixels)')
    parser.add_argument('--height', type=int, default=1000, help='Window height (pixels)')
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    visualizer = ForceVisualizer(ros_sensor_type=args.sensor, video_width=args.width, video_height=args.height)
    visualizer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()