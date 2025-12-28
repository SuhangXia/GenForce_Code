import sys
import os
import time
import numpy as np
import zmq
import socket
from PyQt5.QtCore import Qt, QTimer, QPointF
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (QApplication, QMainWindow, QGridLayout, QWidget, QLabel, 
                            QVBoxLayout, QPushButton, QHBoxLayout, QLineEdit)
from PyQt5.QtChart import QChart, QChartView, QLineSeries, QValueAxis


class ForceClient(QMainWindow):
    def __init__(self, x_timeMax=5, update_interval=50):
        super().__init__()

        # Parameters
        self.zmq_host = "localhost"  # Default, will be updated by user
        self.zmq_port = 5555         # Default, will be updated by user
        self.num_channels = 6        # 6 channels: Fx, Fy, Fz, Tx, Ty, Tz
        self.x_timeMax = x_timeMax   # Time window (seconds)
        self.update_interval = update_interval  # Update interval (ms)
        self.connection_retries = 0
        self.max_retries = 10
        self.connected = False
        self.socket = None
        self.context = None

        # Initialize data structures
        self.x_data = np.linspace(0, self.x_timeMax, int(self.x_timeMax * 1000 / self.update_interval))
        self.y_data = [np.zeros_like(self.x_data) for _ in range(self.num_channels)]
        
        # Setup UI
        self.init_ui()
        
        # Start the update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_data)
        
        # Status update timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.check_connection)

    def get_local_ip(self):
        """Get the local IP address of this machine."""
        try:
            # Create a temporary socket to get the local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # Doesn't need to be reachable
            s.connect(('10.255.255.255', 1))
            local_ip = s.getsockname()[0]
            s.close()
            return local_ip
        except Exception:
            return "127.0.0.1"  # Fallback to localhost

    def setup_zmq_connection(self):
        """Setup ZeroMQ connection with proper error handling."""
        try:
            # Clean up existing connection if any
            if self.socket:
                self.socket.close()
            if self.context:
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
            self.status_label.setText(f"Connected to {self.zmq_host}:{self.zmq_port}")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
            print(f"Connected to publisher at {self.zmq_host}:{self.zmq_port}")
            
            # Start timers
            self.timer.start(self.update_interval)
            self.status_timer.start(5000)  # Check connection every 5 seconds
            
        except Exception as e:
            self.connected = False
            self.connection_retries += 1
            self.status_label.setText(f"Connection error: {str(e)}")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
            print(f"Error connecting to ZeroMQ server: {e}")

    def check_connection(self):
        """Check if connection is active and reconnect if necessary."""
        if not self.connected and self.connection_retries < self.max_retries:
            print(f"Attempting to reconnect... (attempt {self.connection_retries + 1}/{self.max_retries})")
            self.setup_zmq_connection()
        elif not self.connected:
            self.status_label.setText(f"Failed to connect after {self.max_retries} attempts")

    def connect_clicked(self):
        """Handle connect button click - attempt to connect to the server."""
        # Get IP and port from input fields
        self.zmq_host = self.ip_input.text().strip()
        try:
            self.zmq_port = int(self.port_input.text().strip())
        except ValueError:
            self.status_label.setText("Invalid port number")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
            return
            
        # Reset connection state
        self.connected = False
        self.connection_retries = 0
        
        # Attempt to connect
        self.status_label.setText(f"Connecting to {self.zmq_host}:{self.zmq_port}...")
        self.status_label.setStyleSheet("color: orange; font-weight: bold;")
        self.setup_zmq_connection()

    def init_ui(self):
        """Initialize the user interface with charts."""
        self.setWindowTitle("Nano17 Force-Torque Client")
        self.setGeometry(100, 100, 1200, 800)  # Set a reasonable window size

        # Create main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # Create connection control panel
        control_panel = QWidget()
        control_layout = QHBoxLayout()
        
        # IP address input
        control_layout.addWidget(QLabel("Server IP:"))
        self.ip_input = QLineEdit(self.get_local_ip())
        control_layout.addWidget(self.ip_input)
        
        # Port input
        control_layout.addWidget(QLabel("Port:"))
        self.port_input = QLineEdit("5555")
        control_layout.addWidget(self.port_input)
        
        # Connect button
        connect_button = QPushButton("Connect")
        connect_button.clicked.connect(self.connect_clicked)
        control_layout.addWidget(connect_button)
        
        control_panel.setLayout(control_layout)
        main_layout.addWidget(control_panel)
        
        # Add status label
        self.status_label = QLabel("Click 'Connect' to start")
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

        # Set up Force (Fx, Fy, Fz) and Torque (Tx, Ty, Tz) in a 3-row, 2-column layout
        titles = ["Fx(N)", "Fy(N)", "Fz(N)", "Tx(Nmm)", "Ty(Nmm)", "Tz(Nmm)"]
        colors = ["blue", "green", "red", "blue", "green", "red"]  # BGR colors for XYZ

        for i in range(self.num_channels):
            chart = QChart()
            series = QLineSeries()
            series.setColor(QColor(colors[i]))
            chart.addSeries(series)

            # X-axis (Time)
            axis_x = QValueAxis()
            axis_x.setRange(0, self.x_timeMax)
            axis_x.setLabelFormat("%0.1f")
            axis_x.setTitleText("Time (s)")
            chart.addAxis(axis_x, Qt.AlignBottom)
            series.attachAxis(axis_x)

            # Y-axis (Force/Torque values) - Initial range will be updated dynamically
            axis_y = QValueAxis()
            axis_y.setTitleText(titles[i])
            axis_y.setRange(-1, 1)  # Initial range
            chart.addAxis(axis_y, Qt.AlignLeft)
            series.attachAxis(axis_y)

            chart.setTitle(titles[i])
            chart_view = QChartView(chart)

            # Add Force charts to the first column, Torque charts to the second column
            if i < 3:  # Force (Fx, Fy, Fz)
                chart_layout.addWidget(chart_view, i, 0)
            else:  # Torque (Tx, Ty, Tz)
                chart_layout.addWidget(chart_view, i - 3, 1)

            # Store references
            self.charts.append(chart)
            self.series.append(series)
            self.x_axis.append(axis_x)
            self.y_axis.append(axis_y)

    def update_data(self):
        """Receive data from ZeroMQ and update charts."""
        if not self.connected or not self.socket:
            return
            
        try:
            # Try to receive data from the server with a short timeout
            message = self.socket.recv_json()
            ft_values = message["force_torque"]

            # Update the data for each channel
            for i in range(self.num_channels):
                self.y_data[i][:-1] = self.y_data[i][1:]
                self.y_data[i][-1] = ft_values[i]

                # Update the series data
                self.series[i].replace(
                    [QPointF(self.x_data[j], self.y_data[i][j]) for j in range(len(self.x_data))]
                )

                # Dynamically update the y-axis range
                min_value = min(self.y_data[i])
                max_value = max(self.y_data[i])
                
                # Add padding to the axis range to prevent it from changing too frequently
                range_padding = (max_value - min_value) * 0.1
                if range_padding < 0.1:  # If range is very small, use a minimum padding
                    range_padding = 0.1
                    
                self.y_axis[i].setRange(min_value - range_padding, max_value + range_padding)

        except zmq.Again:
            # No data received within timeout, do nothing
            pass
        except zmq.ZMQError as e:
            print(f"ZMQ error: {e}")
            self.connected = False
            self.status_label.setText(f"Connection lost: {str(e)}")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
            self.timer.stop()  # Stop the update timer when connection is lost
        except Exception as e:
            print(f"Error receiving data: {e}")
            self.status_label.setText(f"Data error: {str(e)}")
            self.status_label.setStyleSheet("color: orange; font-weight: bold;")

    def closeEvent(self, event):
        """Clean up ZeroMQ resources when the window is closed."""
        self.timer.stop()
        self.status_timer.stop()
        
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()
        event.accept()


if __name__ == "__main__":
    # Launch the application
    app = QApplication(sys.argv)
    client = ForceClient()
    client.show()
    sys.exit(app.exec_())