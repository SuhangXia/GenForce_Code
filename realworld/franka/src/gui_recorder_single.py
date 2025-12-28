#!/usr/bin/env python3  

import time  
import pyqtgraph as pg  
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel  
from PyQt5.QtCore import QTimer, Qt  
from PyQt5.QtGui import QFont  
import threading  
import numpy as np  
import cv2  
import os  
import csv  
from datetime import datetime  

# Set global PyQtGraph configuration
pg.setConfigOption('background', 'w')  # white background
pg.setConfigOption('foreground', 'k')  # black foreground

target_fz = -1.5

class DataSaver:
    """Data saver - saves experiment data to CSV files"""
    def __init__(self):  
        self.data_buffer = []  
        self.lock = threading.Lock()  
        self.filename = None  
        self.start_time = None  
        self.is_saving = False  
        
    def start_saving(self, filename_base="data/experiment_data"):
        """Start saving data"""
        try:  
            # Create a timestamped filename
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')  
            self.filename = f"{filename_base}_{timestamp_str}.csv"  
            
            # Create directory
            os.makedirs(os.path.dirname(self.filename), exist_ok=True)  
            
            # Clear buffer
            with self.lock:  
                self.data_buffer = []  
                self.start_time = time.time()  
                self.is_saving = True  
            
            print(f"[INFO] Started data saving: {self.filename}")  
            return True  
            
        except Exception as e:  
            print(f"[ERROR] Failed to start data saving: {e}")  
            return False  
    
    def add_data_point(self, fx, fy, fz, gripper_width, slip_detected, control_active, phase, corrections):
        """Add a data point"""
        if not self.is_saving:  
            return  
            
        current_time = time.time()  
        
        with self.lock:  
            if self.start_time is None:  
                self.start_time = current_time  
                
            relative_time = current_time - self.start_time  
            
            data_point = {  
                'timestamp': current_time,  
                'relative_time': relative_time,  
                'force_x': fx,  
                'force_y': fy,  
                'force_z': fz,  
                'gripper_width': gripper_width,  
                'slip_detected': int(slip_detected),  
                'control_active': int(control_active),  
                'phase': phase,  
                'slip_corrections': corrections  
            }  
            
            self.data_buffer.append(data_point)  
    
    def stop_saving(self):
        """Stop saving and write to file"""
        if not self.is_saving:  
            return  
            
        try:  
            with self.lock:  
                self.is_saving = False  
                
                if self.data_buffer and self.filename:  
                    # Write CSV file
                    with open(self.filename, 'w', newline='') as csvfile:  
                        fieldnames = ['timestamp', 'relative_time', 'force_x', 'force_y', 'force_z',   
                                    'gripper_width', 'slip_detected', 'control_active', 'phase', 'slip_corrections']  
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)  
                        
                        # Write header row
                        writer.writeheader()  
                        
                        # Write all data points
                        for data_point in self.data_buffer:  
                            writer.writerow(data_point)  
                    
                    print(f"[INFO] Data saved: {self.filename}")  
                    print(f"[INFO] Total data points: {len(self.data_buffer)}")  
                    
                    if len(self.data_buffer) > 1:  
                        duration = self.data_buffer[-1]['relative_time']  
                        avg_rate = len(self.data_buffer) / duration if duration > 0 else 0  
                        print(f"[INFO] Data collection - Duration: {duration:.2f}s, Avg Rate: {avg_rate:.2f} Hz")  
                    
                    # Also save a summary report
                    self._save_summary()  
                    
        except Exception as e:  
            print(f"[ERROR] Failed to save data: {e}")  
    
    def _save_summary(self):
        """Save experiment data summary"""
        if not self.data_buffer:  
            return  
            
        try:  
            summary_filename = self.filename.replace('.csv', '_summary.txt')  
            
            # Compute statistics
            forces_x = [d['force_x'] for d in self.data_buffer]  
            forces_y = [d['force_y'] for d in self.data_buffer]  
            forces_z = [d['force_z'] for d in self.data_buffer]  
            widths = [d['gripper_width'] for d in self.data_buffer]  
            
            total_slip_events = sum(d['slip_detected'] for d in self.data_buffer)  
            max_corrections = max(d['slip_corrections'] for d in self.data_buffer) if self.data_buffer else 0  
            
            phases = list(set(d['phase'] for d in self.data_buffer))  
            duration = self.data_buffer[-1]['relative_time'] if self.data_buffer else 0  
            
            with open(summary_filename, 'w') as f:  
                f.write("=== SLIP DETECTION EXPERIMENT SUMMARY ===\n\n")  
                f.write(f"Experiment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")  
                f.write(f"Duration: {duration:.2f} seconds\n")  
                f.write(f"Total Data Points: {len(self.data_buffer)}\n")  
                f.write(f"Data Rate: {len(self.data_buffer)/duration:.2f} Hz\n\n")  
                
                f.write("=== FORCE STATISTICS ===\n")  
                f.write(f"Force X: Min={min(forces_x):.3f}, Max={max(forces_x):.3f}, Avg={np.mean(forces_x):.3f}, Std={np.std(forces_x):.3f}\n")  
                f.write(f"Force Y: Min={min(forces_y):.3f}, Max={max(forces_y):.3f}, Avg={np.mean(forces_y):.3f}, Std={np.std(forces_y):.3f}\n")  
                f.write(f"Force Z: Min={min(forces_z):.3f}, Max={max(forces_z):.3f}, Avg={np.mean(forces_z):.3f}, Std={np.std(forces_z):.3f}\n\n")  
                
                f.write("=== GRIPPER STATISTICS ===\n")  
                f.write(f"Gripper Width: Min={min(widths):.4f}, Max={max(widths):.4f}, Avg={np.mean(widths):.4f}, Std={np.std(widths):.4f}\n\n")  
                
                f.write("=== SLIP DETECTION RESULTS ===\n")  
                f.write(f"Total Slip Events Detected: {total_slip_events}\n")  
                f.write(f"Total Slip Corrections Applied: {max_corrections}\n")  
                f.write(f"Slip Event Rate: {total_slip_events/duration:.2f} events/second\n\n")  
                
                f.write("=== EXPERIMENT PHASES ===\n")  
                f.write(f"Phases: {', '.join(phases)}\n\n")  
                
                # Phase timing analysis  
                f.write("=== PHASE TIMING ===\n")  
                current_phase = None  
                phase_start_time = 0  
                for data_point in self.data_buffer:  
                    if data_point['phase'] != current_phase:  
                        if current_phase is not None:  
                            f.write(f"{current_phase}: {phase_start_time:.2f}s - {data_point['relative_time']:.2f}s (Duration: {data_point['relative_time'] - phase_start_time:.2f}s)\n")  
                        current_phase = data_point['phase']  
                        phase_start_time = data_point['relative_time']  
                
                # Final phase
                if current_phase and self.data_buffer:  
                    final_time = self.data_buffer[-1]['relative_time']  
                    f.write(f"{current_phase}: {phase_start_time:.2f}s - {final_time:.2f}s (Duration: {final_time - phase_start_time:.2f}s)\n")  
            
            print(f"[INFO] Summary saved: {summary_filename}")  
            
        except Exception as e:  
            print(f"[ERROR] Failed to save summary: {e}")  

class DataStore:  
    def __init__(self, window_length=10):  
        self.lock = threading.Lock()  
        self.window_length = window_length  
        
        # Force data  
        self.force_x, self.force_x_t = [], []  
        self.force_y, self.force_y_t = [], []  
        self.force_z, self.force_z_t = [], []  
        
        # Gripper width data  
        self.gripper_width, self.gripper_width_t = [], []  
        
        # Slip detection status data  
        self.slip_detected, self.slip_detected_t = [], []  
        self.control_active, self.control_active_t = [], []  
        
        # System status  
        self.current_phase = "Initializing"  
        self.slip_corrections = 0  
        
        self.start_time = None  
    
    def add_force_data(self, fx, fy, fz):  
        t_now = time.time()  
        with self.lock:  
            if self.start_time is None:  
                self.start_time = t_now  
            rel_t = t_now - self.start_time  
            
            self.force_x_t.append(rel_t)  
            self.force_x.append(fx)  
            self.force_y_t.append(rel_t)  
            self.force_y.append(fy)  
            self.force_z_t.append(rel_t)  
            self.force_z.append(fz)  
            
            self._trim_data(self.force_x, self.force_x_t, rel_t)  
            self._trim_data(self.force_y, self.force_y_t, rel_t)  
            self._trim_data(self.force_z, self.force_z_t, rel_t)  
    
    def add_gripper_data(self, width):  
        t_now = time.time()  
        with self.lock:  
            if self.start_time is None:  
                self.start_time = t_now  
            rel_t = t_now - self.start_time  
            
            self.gripper_width_t.append(rel_t)  
            self.gripper_width.append(width)  
            
            self._trim_data(self.gripper_width, self.gripper_width_t, rel_t)  
    
    def add_status_data(self, slip_detected, control_active):  
        t_now = time.time()  
        with self.lock:  
            if self.start_time is None:  
                self.start_time = t_now  
            rel_t = t_now - self.start_time  
            
            self.slip_detected_t.append(rel_t)  
            self.slip_detected.append(1.0 if slip_detected else 0.0)  
            self.control_active_t.append(rel_t)  
            self.control_active.append(2.0 if control_active else 0.0)  
            
            self._trim_data(self.slip_detected, self.slip_detected_t, rel_t)  
            self._trim_data(self.control_active, self.control_active_t, rel_t)  
    
    def update_system_status(self, phase, corrections):  
        with self.lock:  
            self.current_phase = phase  
            self.slip_corrections = corrections  
    
    def _trim_data(self, data_list, time_list, current_time):  
        while time_list and (current_time - time_list[0]) > self.window_length:  
            time_list.pop(0)  
            data_list.pop(0)  

class VideoRecorder:  
    def __init__(self, fps=30):  
        self.fps = fps  
        self.recording = False  
        self.video_writer = None  
        self.filename = None  
        self.frame_timestamps = []  
        self.start_real_time = None  
    
    def start(self, filename):  
        try:  
            self.recording = True  
            self.start_real_time = time.time()  
            
            base, ext = os.path.splitext(filename)  
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')  
            self.filename = f"{base}_{timestamp_str}{ext}"  
            
            os.makedirs(os.path.dirname(self.filename), exist_ok=True)  
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
            self.video_writer = cv2.VideoWriter(self.filename, fourcc, self.fps, (1500, 1000))  
            
            self.frame_timestamps = []  
            print(f"[INFO] Started video recording: {self.filename}")  
            return True  
            
        except Exception as e:  
            print(f"[ERROR] Failed to start video recording: {e}")  
            return False  
    
    def write_frame(self, frame):  
        if self.recording and self.video_writer is not None:  
            try:  
                current_time = time.time()  
                frame_timestamp = current_time - self.start_real_time  
                
                frame_resized = cv2.resize(frame, (1500, 1000))  
                self.video_writer.write(frame_resized)  
                
                self.frame_timestamps.append({  
                    'frame_number': len(self.frame_timestamps),  
                    'real_timestamp': frame_timestamp,  
                    'wall_clock_time': current_time  
                })  
                
            except Exception as e:  
                print(f"[WARNING] Frame write error: {e}")  
    
    def stop(self):  
        if self.recording:  
            self.recording = False  
            
            if self.video_writer:  
                self.video_writer.release()  
                print(f"[INFO] Video saved: {self.filename}")  
            
            if self.frame_timestamps:  
                timestamp_filename = self.filename + '.timestamps.csv'  
                try:  
                    with open(timestamp_filename, 'w') as f:  
                        f.write('frame_number,real_timestamp_sec,wall_clock_time,fps_actual\n')  
                        for i, ts in enumerate(self.frame_timestamps):  
                            if i > 0:  
                                time_diff = ts['real_timestamp'] - self.frame_timestamps[i-1]['real_timestamp']  
                                fps_actual = 1.0 / time_diff if time_diff > 0 else 0  
                            else:  
                                fps_actual = self.fps  
                            
                            f.write(f"{ts['frame_number']},{ts['real_timestamp']:.6f},"  
                                   f"{ts['wall_clock_time']:.6f},{fps_actual:.2f}\n")  
                    
                    print(f"[INFO] Timestamps saved: {timestamp_filename}")  
                    
                    if len(self.frame_timestamps) > 1:  
                        total_duration = self.frame_timestamps[-1]['real_timestamp']  
                        avg_fps = len(self.frame_timestamps) / total_duration  
                        print(f"[INFO] Recording stats - Duration: {total_duration:.2f}s, Avg FPS: {avg_fps:.2f}")  
                        
                except Exception as e:  
                    print(f"[ERROR] Failed to save timestamps: {e}")  

class SlipDetectionGUI:  
    def __init__(self, window_length=10, video_fps=30, video_width=1500, video_height=1000):  
        self.window_length = window_length  
        self.video_fps = video_fps  
        self.video_width = video_width  
        self.video_height = video_height  
        
        # Data storage
        self.data_store = DataStore(window_length)  
        
        # Video recorder
        self.video_recorder = VideoRecorder(video_fps)  
        
        # Data saver
        self.data_saver = DataSaver()  
        
        # GUI components
        self.app = None  
        self.main_widget = None  
        self.timer = None  
        self.exit_event = threading.Event()  
        
        # Plot widgets and curve objects
        self.plot_widgets = {}  
        self.curves = {}  
        self.info_label = None  
        
        self.frame_interval = 1.0 / video_fps  
        self.next_frame_time = 0  
        self.frame_index = 0  
    
    def setup_window(self):  
        """Set up the GUI window"""
        self.app = QApplication([])  
        
        # Create main window
        self.main_widget = QWidget()  
        self.main_widget.setWindowTitle("Slip Detection - Real-time Monitoring")  
        self.main_widget.resize(self.video_width, self.video_height)  
        
        main_layout = QVBoxLayout(self.main_widget)  
        
        # Status label layout
        status_layout = QHBoxLayout()  
        
        # Phase label
        self.phase_label = QLabel("Phase: Initializing")  
        self.phase_label.setFont(QFont("Arial", 14, QFont.Bold))  
        self.phase_label.setStyleSheet("""  
            color: blue;   
            padding: 10px;   
            border: 2px solid blue;   
            border-radius: 5px;  
            margin: 5px;  
        """)  
        
        self.status_label = QLabel("System Ready")  
        self.status_label.setFont(QFont("Arial", 12, QFont.Bold))  
        self.status_label.setStyleSheet("""  
            color: green;   
            padding: 10px;   
            border: 2px solid green;   
            border-radius: 5px;  
            margin: 5px;  
        """)  
        
        # Recording label
        self.recording_label = QLabel("🔴 Recording")  
        self.recording_label.setFont(QFont("Arial", 12, QFont.Bold))  
        self.recording_label.setStyleSheet("""  
            color: red;   
            padding: 10px;   
            border: 2px solid red;   
            border-radius: 5px;  
            margin: 5px;  
        """)  
        
        # Data saving label
        self.data_saving_label = QLabel("💾 Data Saving")  
        self.data_saving_label.setFont(QFont("Arial", 12, QFont.Bold))  
        self.data_saving_label.setStyleSheet("""  
            color: purple;  
            padding: 10px;  
            border: 2px solid purple;  
            border-radius: 5px;  
            margin: 5px;  
        """)  
        
        status_layout.addWidget(self.phase_label)  
        status_layout.addWidget(self.status_label)  
        status_layout.addWidget(self.recording_label)  
        status_layout.addWidget(self.data_saving_label)  
        status_layout.addStretch()  
        
        main_layout.addLayout(status_layout)  
        
        # Create plot area layout
        plots_layout = QVBoxLayout()  
        
        # First row: Fx and Fy
        row1_layout = QHBoxLayout()  
        
        # Force X plot
        self.plot_widgets['fx'] = pg.PlotWidget(title='Force X (Fx)')  
        self.plot_widgets['fx'].setLabels(left='Force (N)', bottom='Time (s)')  
        self.plot_widgets['fx'].setYRange(-2.0, 2.0, padding=0)  
        self.curves['fx'] = self.plot_widgets['fx'].plot(pen=pg.mkPen((200, 0, 0), width=3), name='Fx')  
        # Add threshold lines (optional)
        # self.plot_widgets['fx'].addLine(y=0.5, pen=pg.mkPen((200, 0, 0, 100), style=pg.QtCore.Qt.DashLine))  
        # self.plot_widgets['fx'].addLine(y=-0.5, pen=pg.mkPen((200, 0, 0, 100), style=pg.QtCore.Qt.DashLine))  
        
        # Force Y   
        self.plot_widgets['fy'] = pg.PlotWidget(title='Force Y (Fy)')  
        self.plot_widgets['fy'].setLabels(left='Force (N)', bottom='Time (s)')  
        self.plot_widgets['fy'].setYRange(-2.0, 2.0, padding=0)  
        self.curves['fy'] = self.plot_widgets['fy'].plot(pen=pg.mkPen((0, 150, 0), width=3), name='Fy')  
        # self.plot_widgets['fy'].addLine(y=0.5, pen=pg.mkPen((0, 150, 0, 100), style=pg.QtCore.Qt.DashLine))  
        # self.plot_widgets['fy'].addLine(y=-0.5, pen=pg.mkPen((0, 150, 0, 100), style=pg.QtCore.Qt.DashLine))  
        
        row1_layout.addWidget(self.plot_widgets['fx'])  
        row1_layout.addWidget(self.plot_widgets['fy'])  
        plots_layout.addLayout(row1_layout)  
        
        # Second row: Fz and Gripper Width
        row2_layout = QHBoxLayout()  
        
        # Force Z   
        self.plot_widgets['fz'] = pg.PlotWidget(title='Force Z (Fz)')  
        self.plot_widgets['fz'].setLabels(left='Force (N)', bottom='Time (s)')  
        self.plot_widgets['fz'].setYRange(-5.0, 1.0, padding=0)  
        self.curves['fz'] = self.plot_widgets['fz'].plot(pen=pg.mkPen((0, 0, 180), width=3), name='Fz')  
        # self.plot_widgets['fz'].addLine(y=target_fz, pen=pg.mkPen((0, 0, 180, 100), style=pg.QtCore.Qt.DashLine))  
        
        # Gripper Width 图  
        self.plot_widgets['width'] = pg.PlotWidget(title='Gripper Width')  
        self.plot_widgets['width'].setLabels(left='Width (m)', bottom='Time (s)')  
        self.plot_widgets['width'].setYRange(0.0, 0.08, padding=0)  
        self.curves['width'] = self.plot_widgets['width'].plot(pen=pg.mkPen((128, 0, 128), width=3), name='Width')  
        
        row2_layout.addWidget(self.plot_widgets['fz'])  
        row2_layout.addWidget(self.plot_widgets['width'])  
        plots_layout.addLayout(row2_layout)  
        
        # Third row: Status plots
        row3_layout = QHBoxLayout()  
        
        # Slip Detection Status  
        self.plot_widgets['slip'] = pg.PlotWidget(title='Slip Detection Status')  
        self.plot_widgets['slip'].setLabels(left='Status', bottom='Time (s)')  
        self.plot_widgets['slip'].setYRange(-0.1, 1.1, padding=0)  
        self.curves['slip'] = self.plot_widgets['slip'].plot(pen=pg.mkPen((255, 0, 0), width=3),  
                                                            fillLevel=0, brush=(255, 0, 0, 100), name='Slip Detected')  
        
        # Control Active Status  
        self.plot_widgets['control'] = pg.PlotWidget(title='Control Active Status')  
        self.plot_widgets['control'].setLabels(left='Status', bottom='Time (s)')  
        self.plot_widgets['control'].setYRange(-0.1, 2.1, padding=0)  
        self.curves['control'] = self.plot_widgets['control'].plot(pen=pg.mkPen((0, 0, 255), width=3),  
                                                                  fillLevel=0, brush=(0, 0, 255, 100), name='Control Active')  
        
        row3_layout.addWidget(self.plot_widgets['slip'])  
        row3_layout.addWidget(self.plot_widgets['control'])  
        plots_layout.addLayout(row3_layout)  
        
        main_layout.addLayout(plots_layout)  
        
        # System information label
        self.info_label = QLabel("System Information")  
        self.info_label.setFont(QFont("Monospace", 10))  
        self.info_label.setStyleSheet("""  
            color: black;  
            background-color: #f0f0f0;  
            padding: 10px;  
            border: 1px solid gray;  
            border-radius: 5px;  
        """)  
        self.info_label.setMinimumHeight(120)  
        self.info_label.setAlignment(Qt.AlignTop)  
        main_layout.addWidget(self.info_label)  
        
        self.main_widget.show()  
    
    def grab_current_frame(self):
        """Grab the current frame"""
        try:  
            qimg = self.main_widget.grab().toImage().convertToFormat(4)  # QImage.Format_RGBA8888  
            width, height = qimg.width(), qimg.height()  
            ptr = qimg.bits()  
            ptr.setsize(qimg.byteCount())  
            arr = np.array(ptr).reshape(height, width, 4)  
            frame_bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)  
            return frame_bgr  
        except Exception as e:  
            print(f"[WARNING] Frame capture error: {e}")  
            # Return a black frame as fallback
            return np.zeros((self.video_height, self.video_width, 3), dtype=np.uint8)  
    
    def update_display(self):
        """Update display"""
        try:  
            with self.data_store.lock:  
                # Update force sensor curves
                if self.data_store.force_x_t:  
                    self.curves['fx'].setData(self.data_store.force_x_t, self.data_store.force_x)  
                if self.data_store.force_y_t:  
                    self.curves['fy'].setData(self.data_store.force_y_t, self.data_store.force_y)  
                if self.data_store.force_z_t:  
                    self.curves['fz'].setData(self.data_store.force_z_t, self.data_store.force_z)  
                
                # Update gripper width curve
                if self.data_store.gripper_width_t:  
                    self.curves['width'].setData(self.data_store.gripper_width_t, self.data_store.gripper_width)  
                
                # Update status curves
                if self.data_store.slip_detected_t:  
                    self.curves['slip'].setData(self.data_store.slip_detected_t, self.data_store.slip_detected)  
                if self.data_store.control_active_t:  
                    # Normalize control_active values to 0-1 for display
                    control_normalized = [val/2.0 for val in self.data_store.control_active]  
                    self.curves['control'].setData(self.data_store.control_active_t, control_normalized)  
                
                # Update labels
                self.phase_label.setText(f"Phase: {self.data_store.current_phase}")  
                
                # Update system information
                current_time = time.time() - self.data_store.start_time if self.data_store.start_time else 0  
                current_fx = self.data_store.force_x[-1] if self.data_store.force_x else 0
                current_fy = self.data_store.force_y[-1] if self.data_store.force_y else 0
                current_fz = self.data_store.force_z[-1] if self.data_store.force_z else 0
                current_width = self.data_store.gripper_width[-1] if self.data_store.gripper_width else 0
                
                # Determine status
                slip_detected = (self.data_store.slip_detected and 
                               self.data_store.slip_detected[-1] > 0.5) if self.data_store.slip_detected else False
                control_active = (self.data_store.control_active and 
                                self.data_store.control_active[-1] > 1.0) if self.data_store.control_active else False
                
                # Save data point to data saver
                if self.data_saver.is_saving:
                    self.data_saver.add_data_point(
                        fx=current_fx,
                        fy=current_fy,
                        fz=current_fz,
                        gripper_width=current_width,
                        slip_detected=slip_detected,
                        control_active=control_active,
                        phase=self.data_store.current_phase,
                        corrections=self.data_store.slip_corrections
                    )
                
                if slip_detected:
                    self.status_label.setText("SLIP DETECTED!")
                    self.status_label.setStyleSheet("""
                        color: white; 
                        background-color: red; 
                        padding: 10px; 
                        border: 2px solid red; 
                        border-radius: 5px; 
                        margin: 5px;
                        font-weight: bold;
                    """)
                elif control_active:
                    self.status_label.setText("CONTROL ACTIVE")
                    self.status_label.setStyleSheet("""
                        color: white; 
                        background-color: blue; 
                        padding: 10px; 
                        border: 2px solid blue; 
                        border-radius: 5px; 
                        margin: 5px;
                        font-weight: bold;
                    """)
                else:
                    self.status_label.setText("System Running")
                    self.status_label.setStyleSheet("""
                        color: green; 
                        padding: 10px; 
                        border: 2px solid green; 
                        border-radius: 5px;
                        margin: 5px;
                    """)
                
                info_text = f"""Phase: {self.data_store.current_phase}
Time: {current_time:.1f}s

Current Values:
Fx: {current_fx:.3f} N      Fy: {current_fy:.3f} N
Fz: {current_fz:.3f} N      Width: {current_width:.4f} m

Status:
Slip Corrections: {self.data_store.slip_corrections}
Recording: {'ON' if self.video_recorder.recording else 'OFF'}
Data Saving: {'ON' if self.data_saver.is_saving else 'OFF'}

Thresholds:
Fx/Fy: ±0.5 N      Fz Target: -1.0 N"""
                
                self.info_label.setText(info_text)
            
            # Video recording - based on timing
            if self.video_recorder.recording:
                current_frame = self.grab_current_frame()
                now = time.time()
                
                # Write frames at fixed intervals
                while now >= self.next_frame_time:
                    self.video_recorder.write_frame(current_frame)
                    self.frame_index += 1
                    self.next_frame_time += self.frame_interval
            
            # Check exit condition
            if self.exit_event.is_set():
                print("[INFO] Exit event detected, closing window.")
                self.timer.stop()
                self.main_widget.close()
                self.app.quit()
                
        except Exception as e:
            print(f"[ERROR] Display update error: {e}")
            import traceback
            traceback.print_exc()
    
    def start_recording(self, filename="video/slip_detection.mp4"):
        """Start recording video"""
        success = self.video_recorder.start(filename)
        if success:
            self.next_frame_time = time.time()
            self.frame_index = 0
            self.recording_label.setVisible(True)
        return success
    
    def stop_recording(self):
        """Stop recording video"""
        self.video_recorder.stop()
        if hasattr(self, 'recording_label'):
            self.recording_label.setVisible(False)
    
    def start_data_saving(self, filename_base="data/experiment_data"):
        """Start saving data"""
        success = self.data_saver.start_saving(filename_base)
        if success:
            self.data_saving_label.setVisible(True)
            print("[INFO] Data saving started successfully!")
        return success
    
    def stop_data_saving(self):
        """Stop saving data"""
        if self.data_saver.is_saving:
            self.data_saver.stop_saving()
            if hasattr(self, 'data_saving_label'):
                self.data_saving_label.setVisible(False)
            print("[INFO] Data saving stopped and files written!")
    
    def update_force_data(self, fx, fy, fz):
        """Update force sensor data"""
        self.data_store.add_force_data(fx, fy, fz)
    
    def update_gripper_data(self, width):
        """Update gripper data"""
        self.data_store.add_gripper_data(width)
    
    def update_status_data(self, slip_detected, control_active):
        """Update status data"""
        self.data_store.add_status_data(slip_detected, control_active)
    
    def update_system_status(self, phase, corrections):
        """Update system status"""
        self.data_store.update_system_status(phase, corrections)
    
    def run(self):
        """Run the GUI main loop"""
        try:
            # Set up update timer
            self.timer = QTimer()
            self.timer.timeout.connect(self.update_display)
            # Use a higher refresh rate to improve video recording accuracy
            timer_interval = max(16, int(1000 / (self.video_fps * 2)))  # at least ~60fps refresh
            self.timer.start(timer_interval)
            
            print(f"[INFO] GUI started with timer interval: {timer_interval}ms")
            self.app.exec_()
            
        except KeyboardInterrupt:
            print("[INFO] KeyboardInterrupt detected in GUI")
        finally:
            if self.video_recorder.recording:
                self.stop_recording()
            if self.data_saver.is_saving:
                self.stop_data_saving()
            if self.main_widget:
                self.main_widget.close()
            print("[INFO] GUI cleanup completed")
    
    def stop(self):
        """停止GUI"""
        self.exit_event.set()

# Test function
def test_gui():
    """Test GUI functionality"""
    import math
    
    gui = SlipDetectionGUI(window_length=10, video_fps=30)
    gui.setup_window()
    
    gui.start_recording("test_slip_detection.mp4")
    gui.start_data_saving("test_experiment_data")
    
    print("=" * 80)
    print("🚀 TEST MODE - Slip Detection GUI with Data Recording 🚀")
    print("=" * 80)
    print("📹 Video Recording: ACTIVE")
    print("💾 Data Saving: ACTIVE")
    print("📊 Real-time Visualization: ACTIVE")
    print("=" * 80)
    print("Watch the GUI for:")
    print("- Force curves with simulated slip events")
    print("- Gripper width adjustments")
    print("- Phase transitions")
    print("- Status changes")
    print("=" * 80)
    
    # Simulated data update thread
    def data_simulation():
        t = 0
        phase_names = ["Initial", "Moving Down", "Grasping", "Lifting", "Slip Detection", "Completed"]
        phase_idx = 0
        slip_corrections = 0
        slip_event_counter = 0
        
        while not gui.exit_event.is_set():
            try:
                # Simulate force sensor data
                base_fx = 0.3 * math.sin(0.1 * t)
                base_fy = 0.2 * math.cos(0.15 * t)
                base_fz = -1.0 + 0.5 * math.sin(0.05 * t)
                
                # Add noise
                fx = base_fx + 0.1 * np.random.randn()
                fy = base_fy + 0.1 * np.random.randn()
                fz = base_fz + 0.1 * np.random.randn()
                
                # Simulate slip events - a pronounced slip every 30 seconds
                if int(t / 30) > slip_event_counter:
                    slip_event_counter = int(t / 30)
                    # Create a slip event
                    if t % 30 < 5: 
                        fx += 0.8 * math.sin(2 * t)  # strong lateral force
                        fy += 0.6 * math.cos(3 * t)  # strong sideways force
                
                # Simulate gripper width changes
                if t < 50:
                    width = 0.055 - 0.001 * t 
                else:
                   
                    base_width = 0.005
                    if 50 < t < 150:  
                        if abs(fx) > 0.4 or abs(fy) > 0.4:  # tighten when slip detected
                            base_width -= 0.001
                        width = max(0.0, base_width + 0.001 * math.sin(0.02 * t))
                    else:
                        width = base_width + 0.002 * math.sin(0.02 * t)
                
                width = max(0.0, min(0.08, width))
                
                # Simulate slip detection
                slip_detected = abs(fx) > 0.5 or abs(fy) > 0.5
                control_active = 20 < t < 150 
                
                if slip_detected and control_active:
                    slip_corrections += 1
                
                # Simulate phase changes
                new_phase_idx = min(int(t / 25), len(phase_names) - 1)
                if new_phase_idx != phase_idx:
                    phase_idx = new_phase_idx
                    current_phase = phase_names[phase_idx]
                    print(f"[SIMULATION] Phase change: {current_phase} at t={t:.1f}s")
                else:
                    current_phase = phase_names[phase_idx]
                
                gui.update_force_data(fx, fy, fz)
                gui.update_gripper_data(width)
                gui.update_status_data(slip_detected, control_active)
                gui.update_system_status(current_phase, slip_corrections)
                
                if int(t) % 10 == 0 and t != int(t):
                    print(f"[SIMULATION] t={t:.1f}s | Phase: {current_phase} | "
                          f"Fx: {fx:.3f}N, Fy: {fy:.3f}N, Fz: {fz:.3f}N | "
                          f"Width: {width:.4f}m | Slip: {slip_detected} | "
                          f"Corrections: {slip_corrections}")
                
                t += 0.1
                time.sleep(0.033)  
                
            except Exception as e:
                print(f"[ERROR] Data simulation error: {e}")
                break
    
    # Start simulated data thread
    data_thread = threading.Thread(target=data_simulation, daemon=True)
    data_thread.start()
    

    try:
        gui.run()
    finally:
        print("[INFO] Test completed!")
        print("Files generated:")
        print("- Video file (.mp4)")
        print("- Video timestamps (.mp4.timestamps.csv)")
        print("- Experiment data (.csv)")
        print("- Experiment summary (.txt)")

if __name__ == "__main__":
    test_gui()