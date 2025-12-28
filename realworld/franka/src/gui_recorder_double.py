#!/usr/bin/env python3

import os
import sys


os.environ['QT_X11_NO_MITSHM'] = '1'
os.environ['QT_OPENGL'] = 'software'

import time
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont, QImage
import threading
import numpy as np
import cv2
import csv
from datetime import datetime

# Set global PyQtGraph configuration
pg.setConfigOption('background', 'w')  # white background
pg.setConfigOption('foreground', 'k')  # black foreground
pg.setConfigOption('antialias', True)  # enable antialiasing

class DataSaver:
    """Data saver - adapted for dual sensors"""
    def __init__(self):
        self.data_buffer = []
        self.lock = threading.Lock()
        self.filename = None
        self.start_time = None
        self.is_saving = False
        
    def start_saving(self, filename_base="data/experiment_data"):
        try:
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.filename = f"{filename_base}_{timestamp_str}.csv"
            os.makedirs(os.path.dirname(self.filename), exist_ok=True)
            
            with self.lock:
                self.data_buffer = []
                self.start_time = time.time()
                self.is_saving = True
            print(f"[INFO] Started data saving: {self.filename}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to start data saving: {e}")
            return False
    
    def add_data_point(self, t_fx, t_fy, t_fz, u_fx, u_fy, u_fz, m_fz, 
                       gripper_width, slip_detected, control_active, phase, corrections):
        if not self.is_saving: return
        
        current_time = time.time()
        with self.lock:
            if self.start_time is None: self.start_time = current_time
            relative_time = current_time - self.start_time
            
            data_point = {
                'timestamp': current_time,
                'relative_time': relative_time,
                'tactip_fx': t_fx, 'tactip_fy': t_fy, 'tactip_fz': t_fz,
                'uskin_fx': u_fx, 'uskin_fy': u_fy, 'uskin_fz': u_fz,
                'mean_fz': m_fz,
                'gripper_width': gripper_width,
                'slip_detected': int(slip_detected),
                'control_active': int(control_active),
                'phase': phase,
                'slip_corrections': corrections
            }
            self.data_buffer.append(data_point)
    
    def stop_saving(self):
        if not self.is_saving: return
        try:
            with self.lock:
                self.is_saving = False
                if self.data_buffer and self.filename:
                    with open(self.filename, 'w', newline='') as csvfile:
                        fieldnames = ['timestamp', 'relative_time', 
                                    'tactip_fx', 'tactip_fy', 'tactip_fz',
                                    'uskin_fx', 'uskin_fy', 'uskin_fz',
                                    'mean_fz', 'gripper_width', 
                                    'slip_detected', 'control_active', 'phase', 'slip_corrections']
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        for dp in self.data_buffer:
                            writer.writerow(dp)
                    print(f"[INFO] Data saved: {self.filename} ({len(self.data_buffer)} points)")
        except Exception as e:
            print(f"[ERROR] Failed to save data: {e}")

class DataStore:
    """Data storage - adds lists only, logic unchanged"""
    def __init__(self, window_length=10):
        self.lock = threading.Lock()
        self.window_length = window_length
        self.start_time = None
        
        # Tactip Data (Red/Orange)
        self.t_fx, self.t_fx_t = [], []
        self.t_fy, self.t_fy_t = [], []
        self.t_fz, self.t_fz_t = [], []
        
        # Uskin Data (Green/Lime)
        self.u_fx, self.u_fx_t = [], []
        self.u_fy, self.u_fy_t = [], []
        self.u_fz, self.u_fz_t = [], []
        
        # Mean Fz (Blue)
        self.m_fz, self.m_fz_t = [], []
        
        # Others
        self.gripper_width, self.gripper_width_t = [], []
        self.slip_detected, self.slip_detected_t = [], []
        self.control_active, self.control_active_t = [], []
        
        self.current_phase = "Initializing"
        self.slip_corrections = 0
    
    def add_force_data(self, fx, fy, fz, sensor_name="tactip"):
        t_now = time.time()
        with self.lock:
            if self.start_time is None: self.start_time = t_now
            rel_t = t_now - self.start_time
            
            # Dispatch data by sensor
            if sensor_name == "tactip":
                self._append(self.t_fx, self.t_fx_t, fx, rel_t)
                self._append(self.t_fy, self.t_fy_t, fy, rel_t)
                self._append(self.t_fz, self.t_fz_t, fz, rel_t)
            elif sensor_name == "uskin":
                self._append(self.u_fx, self.u_fx_t, fx, rel_t)
                self._append(self.u_fy, self.u_fy_t, fy, rel_t)
                self._append(self.u_fz, self.u_fz_t, fz, rel_t)
            
            # Compute mean (if both sensors have data)
            if self.t_fz and self.u_fz:
                mean_val = (self.t_fz[-1] + self.u_fz[-1]) / 2.0
                self._append(self.m_fz, self.m_fz_t, mean_val, rel_t)

    def add_gripper_data(self, width):
        t_now = time.time()
        with self.lock:
            if self.start_time is None: self.start_time = t_now
            self._append(self.gripper_width, self.gripper_width_t, width, t_now - self.start_time)

    def add_status_data(self, slip, control):
        t_now = time.time()
        with self.lock:
            if self.start_time is None: self.start_time = t_now
            rel_t = t_now - self.start_time
            self._append(self.slip_detected, self.slip_detected_t, 1.0 if slip else 0.0, rel_t)
            self._append(self.control_active, self.control_active_t, 2.0 if control else 0.0, rel_t)
            
    def update_system_status(self, phase, corrections):
        with self.lock:
            self.current_phase = phase
            self.slip_corrections = corrections

    def _append(self, data_list, time_list, val, t):
        time_list.append(t)
        data_list.append(val)
        while time_list and (t - time_list[0]) > self.window_length:
            time_list.pop(0)
            data_list.pop(0)

class VideoRecorder:
    def __init__(self, fps=30):
        self.fps = fps
        self.recording = False
        self.video_writer = None
        self.filename = None
        self.start_real_time = None
    
    def start(self, filename):
        try:
            self.recording = True
            self.start_real_time = time.time()
            
            # Simple fix: ensure the filename has an extension
            base, ext = os.path.splitext(filename)
            if not ext: ext = ".mp4"
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.filename = f"{base}_{timestamp_str}{ext}"
            
            os.makedirs(os.path.dirname(self.filename), exist_ok=True)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(self.filename, fourcc, self.fps, (1500, 1000))
            return True
        except Exception as e:
            print(f"[ERROR] Start video failed: {e}")
            self.recording = False
            return False
    
    def write_frame(self, frame):
        if self.recording and self.video_writer:
            try:
                # Force-resize to the initialized size to avoid errors
                frame_resized = cv2.resize(frame, (1500, 1000))
                self.video_writer.write(frame_resized)
            except: pass
    
    def stop(self):
        if self.recording:
            self.recording = False
            if self.video_writer:
                self.video_writer.release()
                print(f"[INFO] Video saved: {self.filename}")

class SlipDetectionGUI:
    def __init__(self, window_length=10, video_fps=30, video_width=1500, video_height=1000):
        self.window_length = window_length
        self.video_fps = video_fps
        self.video_width = video_width
        self.video_height = video_height
        
        self.data_store = DataStore(window_length)
        self.video_recorder = VideoRecorder(video_fps)
        self.data_saver = DataSaver()
        
        self.app = None
        self.main_widget = None
        self.timer = None
        self.exit_event = threading.Event()
        
        self.plot_widgets = {}
        self.curves = {}
        self.info_label = None
        
        self.frame_interval = 1.0 / video_fps
        self.next_frame_time = 0

    def setup_window(self):
        # Must pass in sys.argv
        self.app = QApplication(sys.argv)
        
        self.main_widget = QWidget()
        self.main_widget.setWindowTitle("Dual Sensor Slip Detection (Compact)")
        self.main_widget.resize(self.video_width, self.video_height)
        
        main_layout = QVBoxLayout(self.main_widget)
        
        # --- Status bar (unchanged) ---
        status_layout = QHBoxLayout()
        self.phase_label = self._create_label("Phase: Init", "blue")
        self.status_label = self._create_label("System Ready", "green")
        self.recording_label = self._create_label("🔴 Recording", "red")
        self.recording_label.setVisible(False)
        self.data_saving_label = self._create_label("💾 Data Saving", "purple")
        self.data_saving_label.setVisible(False)
        
        status_layout.addWidget(self.phase_label)
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.recording_label)
        status_layout.addWidget(self.data_saving_label)
        status_layout.addStretch()
        main_layout.addLayout(status_layout)
        
        # --- Plot area (structure unchanged; only curves added) ---
        plots_layout = QVBoxLayout()
        
        # First row: Fx and Fy
        r1 = QHBoxLayout()
        
        # Fx window: Tactip (red) + Uskin (green)
        self.plot_widgets['fx'] = self._create_plot("Force X (TacTip=Red, uSkin=Green)", "N", [-2, 2])
        self.curves['t_fx'] = self.plot_widgets['fx'].plot(pen=pg.mkPen((200,0,0), width=2), name='TacTip')
        self.curves['u_fx'] = self.plot_widgets['fx'].plot(pen=pg.mkPen((0,180,0), width=2), name='uSkin')
        
        # Fy window: Tactip (orange) + Uskin (lime)
        self.plot_widgets['fy'] = self._create_plot("Force Y (TacTip=Orange, uSkin=Lime)", "N", [-2, 2])
        self.curves['t_fy'] = self.plot_widgets['fy'].plot(pen=pg.mkPen((255,100,0), width=2), name='TacTip')
        self.curves['u_fy'] = self.plot_widgets['fy'].plot(pen=pg.mkPen((100,255,100), width=2), name='uSkin')
        
        r1.addWidget(self.plot_widgets['fx'])
        r1.addWidget(self.plot_widgets['fy'])
        plots_layout.addLayout(r1)
        
        # Second row: Fz and Gripper
        r2 = QHBoxLayout()
        
        # Fz window: Tactip (red) + Uskin (green) + Mean (blue, bold)
        self.plot_widgets['fz'] = self._create_plot("Force Z (TacTip=Red, uSkin=Green, Mean=Blue)", "N", [-5, 1])
        self.curves['t_fz'] = self.plot_widgets['fz'].plot(pen=pg.mkPen((200,0,0), width=1))
        self.curves['u_fz'] = self.plot_widgets['fz'].plot(pen=pg.mkPen((0,180,0), width=1))
        self.curves['m_fz'] = self.plot_widgets['fz'].plot(pen=pg.mkPen((0,0,255), width=3)) # highlight mean
        
        self.plot_widgets['width'] = self._create_plot("Gripper Width", "m", [0, 0.08])
        self.curves['width'] = self.plot_widgets['width'].plot(pen=pg.mkPen((128,0,128), width=3))
        
        r2.addWidget(self.plot_widgets['fz'])
        r2.addWidget(self.plot_widgets['width'])
        plots_layout.addLayout(r2)
        
        # Third row: Status (unchanged)
        r3 = QHBoxLayout()
        self.plot_widgets['slip'] = self._create_plot("Slip Status", "", [-0.1, 1.1])
        self.curves['slip'] = self.plot_widgets['slip'].plot(pen=pg.mkPen('r', width=2), fillLevel=0, brush=(255,0,0,100))
        
        self.plot_widgets['control'] = self._create_plot("Control Active", "", [-0.1, 2.1])
        self.curves['control'] = self.plot_widgets['control'].plot(pen=pg.mkPen('b', width=2), fillLevel=0, brush=(0,0,255,100))
        
        r3.addWidget(self.plot_widgets['slip'])
        r3.addWidget(self.plot_widgets['control'])
        plots_layout.addLayout(r3)
        
        main_layout.addLayout(plots_layout)
        
        # Bottom information
        self.info_label = QLabel("Initializing...")
        self.info_label.setFont(QFont("Monospace", 10))
        self.info_label.setStyleSheet("background: #f0f0f0; border: 1px solid gray; padding: 5px;")
        self.info_label.setMinimumHeight(100)
        self.info_label.setAlignment(Qt.AlignTop)
        main_layout.addWidget(self.info_label)
        
        self.main_widget.show()

    def _create_label(self, text, color):
        lbl = QLabel(text)
        lbl.setFont(QFont("Arial", 12, QFont.Bold))
        lbl.setStyleSheet(f"color: {color}; border: 2px solid {color}; border-radius: 4px; padding: 5px;")
        return lbl

    def _create_plot(self, title, units, y_range):
        p = pg.PlotWidget(title=title)
        p.setLabels(left=units, bottom='s')
        p.setYRange(y_range[0], y_range[1])
        return p

    def grab_current_frame(self):
        try:
            qimg = self.main_widget.grab().toImage().convertToFormat(QImage.Format_RGB32)
            w, h = qimg.width(), qimg.height()
            ptr = qimg.bits()
            ptr.setsize(h * w * 4)
            arr = np.frombuffer(ptr, np.uint8).reshape((h, w, 4))
            return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        except:
            return np.zeros((self.video_height, self.video_width, 3), dtype=np.uint8)

    def update_display(self):
        try:
            with self.data_store.lock:
                ds = self.data_store
                
                # 1. Update all curves
                # Tactip
                if ds.t_fx_t: self.curves['t_fx'].setData(ds.t_fx_t, ds.t_fx)
                if ds.t_fy_t: self.curves['t_fy'].setData(ds.t_fy_t, ds.t_fy)
                if ds.t_fz_t: self.curves['t_fz'].setData(ds.t_fz_t, ds.t_fz)
                
                # Uskin
                if ds.u_fx_t: self.curves['u_fx'].setData(ds.u_fx_t, ds.u_fx)
                if ds.u_fy_t: self.curves['u_fy'].setData(ds.u_fy_t, ds.u_fy)
                if ds.u_fz_t: self.curves['u_fz'].setData(ds.u_fz_t, ds.u_fz)
                
                # Mean Fz
                if ds.m_fz_t: self.curves['m_fz'].setData(ds.m_fz_t, ds.m_fz)
                
                # Others
                if ds.gripper_width_t: self.curves['width'].setData(ds.gripper_width_t, ds.gripper_width)
                if ds.slip_detected_t: self.curves['slip'].setData(ds.slip_detected_t, ds.slip_detected)
                if ds.control_active_t: self.curves['control'].setData(ds.control_active_t, ds.control_active)

                # 2. Save data
                if self.data_saver.is_saving:
                    self.data_saver.add_data_point(
                        ds.t_fx[-1] if ds.t_fx else 0, ds.t_fy[-1] if ds.t_fy else 0, ds.t_fz[-1] if ds.t_fz else 0,
                        ds.u_fx[-1] if ds.u_fx else 0, ds.u_fy[-1] if ds.u_fy else 0, ds.u_fz[-1] if ds.u_fz else 0,
                        ds.m_fz[-1] if ds.m_fz else 0,
                        ds.gripper_width[-1] if ds.gripper_width else 0,
                        ds.slip_detected[-1] > 0.5 if ds.slip_detected else False,
                        ds.control_active[-1] > 1.0 if ds.control_active else False,
                        ds.current_phase, ds.slip_corrections
                    )

                # 3. Update text
                self.phase_label.setText(f"Phase: {ds.current_phase}")
                
                # Status
                is_slip = ds.slip_detected[-1] > 0.5 if ds.slip_detected else False
                is_control = ds.control_active[-1] > 1.0 if ds.control_active else False
                
                if is_slip:
                    self.status_label.setText("SLIP DETECTED!")
                    self.status_label.setStyleSheet("color: white; background: red; padding: 5px;")
                elif is_control:
                    self.status_label.setText("CONTROL ACTIVE")
                    self.status_label.setStyleSheet("color: white; background: blue; padding: 5px;")
                else:
                    self.status_label.setText("System Running")
                    self.status_label.setStyleSheet("color: green; border: 2px solid green; padding: 5px;")
                
                # Info Label
                t_fx = ds.t_fx[-1] if ds.t_fx else 0
                u_fx = ds.u_fx[-1] if ds.u_fx else 0
                m_fz = ds.m_fz[-1] if ds.m_fz else 0
                
                info = (f"Phase: {ds.current_phase} | Corrections: {ds.slip_corrections}\n"
                        f"Tactip Fx/Fy: {t_fx:.2f} / {ds.t_fy[-1] if ds.t_fy else 0:.2f}\n"
                        f"Uskin  Fx/Fy: {u_fx:.2f} / {ds.u_fy[-1] if ds.u_fy else 0:.2f}\n"
                        f"Mean Fz: {m_fz:.3f} N | Width: {ds.gripper_width[-1] if ds.gripper_width else 0:.4f}")
                self.info_label.setText(info)

            # 4. Video recording
            if self.video_recorder.recording:
                now = time.time()
                while now >= self.next_frame_time:
                    frame = self.grab_current_frame()
                    self.video_recorder.write_frame(frame)
                    self.next_frame_time += self.frame_interval

            if self.exit_event.is_set():
                self.main_widget.close()
                self.app.quit()
        except Exception as e:
            print(f"Update error: {e}")

    def start_recording(self, filename="video/rec.mp4"):
        if self.video_recorder.start(filename):
            self.recording_label.setVisible(True)
            self.next_frame_time = time.time()
            return True
        return False

    def stop_recording(self):
        self.video_recorder.stop()
        self.recording_label.setVisible(False)

    def start_data_saving(self, filename="data/exp"):
        if self.data_saver.start_saving(filename):
            self.data_saving_label.setVisible(True)
            return True
        return False

    def stop_data_saving(self):
        self.data_saver.stop_saving()
        self.data_saving_label.setVisible(False)

    def update_force_data(self, fx, fy, fz, sensor_name="tactip"):
        """Key adaptation: dispatch data according to sensor_name"""
        self.data_store.add_force_data(fx, fy, fz, sensor_name)

    def update_gripper_data(self, width):
        self.data_store.add_gripper_data(width)
        
    def update_status_data(self, slip, control):
        self.data_store.add_status_data(slip, control)
        
    def update_system_status(self, phase, corrections):
        self.data_store.update_system_status(phase, corrections)

    def run(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_display)
        self.timer.start(33) # 30fps
        print("[INFO] GUI Running...")
        self.app.exec_()

    def stop(self):
        self.exit_event.set()

if __name__ == "__main__":
    gui = SlipDetectionGUI()
    gui.setup_window()
    gui.start_recording("test_dual.mp4")
    
    # Simple test
    def sim():
        import math
        t = 0
        while not gui.exit_event.is_set():
            v = math.sin(t)
            gui.update_force_data(v, v, -1+v*0.1, "tactip")
            gui.update_force_data(v*0.8, v*0.8, -1+v*0.2, "uskin")
            gui.update_gripper_data(0.05)
            time.sleep(0.03)
            t += 0.1
    t = threading.Thread(target=sim, daemon=True)
    t.start()
    
    gui.run()