
import sys
import os
import time
import numpy as np
import zmq
import socket
import csv
from datetime import datetime
from PyQt5.QtCore import Qt, QTimer, QPointF
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (QApplication, QMainWindow, QGridLayout, QWidget, QLabel, 
                            QVBoxLayout, QPushButton, QHBoxLayout, QLineEdit, QCheckBox,
                            QFileDialog, QMessageBox)
from PyQt5.QtChart import QChart, QChartView, QLineSeries, QValueAxis


class ForceClient(QMainWindow):
    def __init__(self, x_timeMax=5, update_interval=50):
        super().__init__()

        # Parameters
        self.zmq_host = "192.168.1.118"  # Updated default IP address
        self.zmq_port = 5555         # Default, will be updated by user
        self.num_channels = 6        # 6 channels: Fx, Fy, Fz, Tx, Ty, Tz
        self.x_timeMax = x_timeMax   # Time window (seconds)
        self.update_interval = update_interval  # Update interval (ms)
        self.connection_retries = 0
        self.max_retries = 10
        self.connected = False
        self.socket = None
        self.context = None

        # CSV logging parameters
        self.logging_enabled = False
        self.csv_file = None
        self.csv_writer = None
        self.csv_filename = ""
        self.data_count = 0  # Counter for logged data points

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

    def browse_csv_file(self):
        """Browse for CSV file location and name."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"force_data_{timestamp}.csv"
        
        filename, _ = QFileDialog.getSaveFileName(
            self, 
            "Save CSV File", 
            default_filename, 
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if filename:
            self.csv_filename_input.setText(filename)

    def start_logging(self):
        """Start CSV logging."""
        if self.logging_enabled:
            return
            
        filename = self.csv_filename_input.text().strip()
        if not filename:
            # If no filename specified, create one with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"force_data_{timestamp}.csv"
            self.csv_filename_input.setText(filename)
        
        try:
            # Create directory if it doesn't exist
            directory = os.path.dirname(filename)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
            
            # Open CSV file for writing
            self.csv_file = open(filename, 'w', newline='', encoding='utf-8')
            self.csv_writer = csv.writer(self.csv_file)
            
            # Write header
            header = ['Timestamp', 'Time_Relative(s)', 'Fx(N)', 'Fy(N)', 'Fz(N)', 'Tx(Nmm)', 'Ty(Nmm)', 'Tz(Nmm)']
            self.csv_writer.writerow(header)
            self.csv_file.flush()
            
            # Set logging state
            self.logging_enabled = True
            self.data_count = 0
            self.start_time = time.time()
            
            # Update UI
            self.start_logging_button.setEnabled(False)
            self.stop_logging_button.setEnabled(True)
            self.csv_filename_input.setEnabled(False)
            self.browse_button.setEnabled(False)
            self.logging_status_label.setText(f"Recording to: {os.path.basename(filename)}")
            self.logging_status_label.setStyleSheet("color: green; font-weight: bold;")
            
            print(f"Started logging to: {filename}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start logging:\n{str(e)}")
            print(f"Error starting logging: {e}")

    def stop_logging(self):
        """Stop CSV logging."""
        if not self.logging_enabled:
            return
            
        try:
            # Close CSV file
            if self.csv_file:
                self.csv_file.close()
                self.csv_file = None
                self.csv_writer = None
            
            # Update logging state
            self.logging_enabled = False
            
            # Update UI
            self.start_logging_button.setEnabled(True)
            self.stop_logging_button.setEnabled(False)
            self.csv_filename_input.setEnabled(True)
            self.browse_button.setEnabled(True)
            self.logging_status_label.setText(f"Stopped. Saved {self.data_count} data points.")
            self.logging_status_label.setStyleSheet("color: blue; font-weight: bold;")
            
            print(f"Stopped logging. Total data points: {self.data_count}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to stop logging:\n{str(e)}")
            print(f"Error stopping logging: {e}")

    def log_data_to_csv(self, ft_values):
        """Log current data to CSV file."""
        if not self.logging_enabled or not self.csv_writer:
            return
            
        try:
            current_time = time.time()
            timestamp = datetime.fromtimestamp(current_time).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            relative_time = current_time - self.start_time
            
            # Create row with timestamp, relative time, and force-torque values
            row = [timestamp, f"{relative_time:.3f}"] + [f"{val:.6f}" for val in ft_values]
            
            # Write to CSV
            self.csv_writer.writerow(row)
            self.csv_file.flush()  # Ensure data is written immediately
            
            self.data_count += 1
            
            # Update logging status every 100 data points
            if self.data_count % 100 == 0:
                self.logging_status_label.setText(f"Recording: {self.data_count} points to {os.path.basename(self.csv_filename_input.text())}")
            
        except Exception as e:
            print(f"Error logging data: {e}")
            self.stop_logging()

    def init_ui(self):
        """Initialize the user interface with charts."""
        self.setWindowTitle("Nano17 Force-Torque Client with CSV Logging")
        self.setGeometry(100, 100, 1200, 900)  # Increased height for logging controls

        # Create main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # Create connection control panel
        control_panel = QWidget()
        control_layout = QHBoxLayout()
        
        # IP address input - now defaults to 192.168.1.118
        control_layout.addWidget(QLabel("Server IP:"))
        self.ip_input = QLineEdit("192.168.1.118")  # Updated default IP
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
        
        # Create CSV logging control panel
        logging_panel = QWidget()
        logging_layout = QVBoxLayout()
        
        # File selection row
        file_row = QHBoxLayout()
        file_row.addWidget(QLabel("CSV File:"))
        self.csv_filename_input = QLineEdit()
        # Set default filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_filename_input.setText(f"force_data_{timestamp}.csv")
        file_row.addWidget(self.csv_filename_input)
        
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self.browse_csv_file)
        file_row.addWidget(self.browse_button)
        
        # Logging control buttons
        button_row = QHBoxLayout()
        self.start_logging_button = QPushButton("Start Recording")
        self.start_logging_button.clicked.connect(self.start_logging)
        button_row.addWidget(self.start_logging_button)
        
        self.stop_logging_button = QPushButton("Stop Recording")
        self.stop_logging_button.clicked.connect(self.stop_logging)
        self.stop_logging_button.setEnabled(False)
        button_row.addWidget(self.stop_logging_button)
        
        # Logging status label
        self.logging_status_label = QLabel("Ready to record")
        self.logging_status_label.setStyleSheet("color: gray; font-weight: bold;")
        
        # Add to logging layout
        logging_layout.addLayout(file_row)
        logging_layout.addLayout(button_row)
        logging_layout.addWidget(self.logging_status_label)
        
        logging_panel.setLayout(logging_layout)
        logging_panel.setStyleSheet("QWidget { background-color: #f0f0f0; padding: 10px; border: 1px solid #ccc; }")
        main_layout.addWidget(logging_panel)
        
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

            # Log data to CSV if logging is enabled
            if self.logging_enabled:
                self.log_data_to_csv(ft_values)

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
        
        # Stop logging if it's active
        if self.logging_enabled:
            self.stop_logging()
        
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
