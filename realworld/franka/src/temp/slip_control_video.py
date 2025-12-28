#!/usr/bin/env python  

import rospy  
from moveit_commander import MoveGroupCommander, roscpp_initialize, roscpp_shutdown  
from franka_gripper.msg import GraspAction, GraspGoal, MoveAction, MoveGoal, GraspEpsilon  
import actionlib  
import numpy as np  
from std_msgs.msg import Float32  
from sensor_msgs.msg import JointState  
import threading  
import time  
from collections import deque  
import cv2  
import pyautogui  
import datetime  

# Matplotlib for plotting  
import matplotlib.pyplot as plt  
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas  
from matplotlib.figure import Figure  

# PyQt5 imports  
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel  
from PyQt5.QtCore import QTimer, Qt  
from PyQt5.QtGui import QFont  
import sys  

# --- Force control parameters ---  
TARGET_FORCE = 1  
KP = 0.0004  
WIDTH_MIN = 0.0  
WIDTH_MAX = 0.08  
START_WIDTH = 0.055  
FORCE_TOL = 0.1  
CONTROL_HZ = 10  
SPEED = 0.005  
MOVE_DOWN = 0.03  

# --- Slip detection parameters for top position only ---  
SLIP_THRESHOLD_FX = 0.5  
SLIP_THRESHOLD_FY = 0.5  
SLIP_WIDTH_DECREASE = 0.002  
SLIP_DETECTION_WINDOW = 0.3  
TOP_HOLD_TIME = 10.0  
TOP_CONTROL_HZ = 20  

# --- GUI parameters ---  
PLOT_WINDOW_SIZE = 200  
UPDATE_INTERVAL = 100  

# --- Video recording parameters ---  
VIDEO_FPS = 60  

# Global variables  
last_gripper_width = 0.0  
last_force_z = 0.0  
last_force_x = 0.0  
last_force_y = 0.0  
force_history_x = []  
force_history_y = []  
force_timestamps = []  
lock = threading.Lock()  

# Control flags  
slip_detection_active = False  
top_position_slip_corrections = 0  
control_active = False  
current_phase = "Initializing"  

class ScreenRecorder:  
    def __init__(self, fps=60):  
        self.fps = fps  
        self.recording = False  
        self.video_writer = None  
        self.thread = None  
        self.filename = None  

    def start(self):  
        """Start screen recording"""  
        try:  
            self.recording = True  
            # Create filename with real world timestamp  
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  
            self.filename = f"slip_detection_recording_{timestamp}.avi"  
            
            # Get screen resolution  
            screen_size = pyautogui.size()  
            rospy.loginfo(f"Recording screen at {screen_size[0]}x{screen_size[1]} resolution")  
            
            # Define codec and create VideoWriter object  
            fourcc = cv2.VideoWriter_fourcc(*"XVID")  
            self.video_writer = cv2.VideoWriter(self.filename, fourcc, self.fps, screen_size)  
            
            # Start recording thread  
            self.thread = threading.Thread(target=self._record)  
            self.thread.daemon = True  
            self.thread.start()  
            
            rospy.loginfo(f"Started video recording: {self.filename}")  
            return True  
            
        except Exception as e:  
            rospy.logerr(f"Failed to start video recording: {e}")  
            return False  

    def _record(self):  
        """Recording loop running in separate thread"""  
        frame_duration = 1.0 / self.fps  
        
        while self.recording:  
            try:  
                start_time = time.time()  
                
                # Capture screen  
                img = pyautogui.screenshot()  
                # Convert PIL Image to numpy array  
                frame = np.array(img)  
                # Convert RGB to BGR for OpenCV  
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  
                
                # Write frame  
                if self.video_writer:  
                    self.video_writer.write(frame)  
                
                # Maintain frame rate  
                elapsed = time.time() - start_time  
                sleep_time = max(0, frame_duration - elapsed)  
                if sleep_time > 0:  
                    time.sleep(sleep_time)  
                    
            except Exception as e:  
                rospy.logwarn(f"Frame capture error: {e}")  
                time.sleep(0.1)  

    def stop(self):  
        """Stop recording and save file"""  
        if self.recording:  
            self.recording = False  
            
            if self.thread and self.thread.is_alive():  
                self.thread.join(timeout=5.0)  
                
            if self.video_writer:  
                self.video_writer.release()  
                rospy.loginfo(f"Video recording saved: {self.filename}")  
            
            rospy.loginfo("Screen recording stopped")  

class MatplotlibCanvas(FigureCanvas):  
    def __init__(self, parent=None):  
        # Create larger figure with better spacing  
        self.fig = Figure(figsize=(18, 10))  
        super().__init__(self.fig)  
        self.setParent(parent)  
        
        # Data storage  
        self.time_data = deque(maxlen=PLOT_WINDOW_SIZE)  
        self.force_x_data = deque(maxlen=PLOT_WINDOW_SIZE)  
        self.force_y_data = deque(maxlen=PLOT_WINDOW_SIZE)  
        self.force_z_data = deque(maxlen=PLOT_WINDOW_SIZE)  
        self.width_data = deque(maxlen=PLOT_WINDOW_SIZE)  
        self.slip_data = deque(maxlen=PLOT_WINDOW_SIZE)  
        self.control_data = deque(maxlen=PLOT_WINDOW_SIZE)  
        
        self.start_time = time.time()  
        
        # Create subplots  
        self.setup_plots()  
        
    def setup_plots(self):  
        # Clear any existing plots  
        self.fig.clear()  
        
        # Create 2x3 subplot grid with proper spacing  
        self.fig.subplots_adjust(hspace=0.4, wspace=0.25, top=0.93, bottom=0.08, left=0.06, right=0.98)  
        
        self.ax1 = self.fig.add_subplot(2, 3, 1)  # Force X  
        self.ax2 = self.fig.add_subplot(2, 3, 2)  # Force Y  
        self.ax3 = self.fig.add_subplot(2, 3, 3)  # Force Z  
        self.ax4 = self.fig.add_subplot(2, 3, 4)  # Gripper Width  
        self.ax5 = self.fig.add_subplot(2, 3, 5)  # Status  
        self.ax6 = self.fig.add_subplot(2, 3, 6)  # Phase info  
        
        # Setup each plot  
        self.setup_force_plot(self.ax1, "Force X (Fx)", "red", SLIP_THRESHOLD_FX)  
        self.setup_force_plot(self.ax2, "Force Y (Fy)", "green", SLIP_THRESHOLD_FY)  
        self.setup_force_plot(self.ax3, "Force Z (Fz)", "blue", TARGET_FORCE)  
        self.setup_width_plot()  
        self.setup_status_plot()  
        self.setup_info_plot()  
        
    def setup_force_plot(self, ax, title, color, threshold):  
        ax.set_title(title, fontsize=11, fontweight='bold', pad=20)  
        ax.set_xlabel('Time (s)', fontsize=9)  
        ax.set_ylabel('Force (N)', fontsize=9)  
        ax.grid(True, alpha=0.3)  
        ax.axhline(y=threshold, color=color, linestyle='--', alpha=0.7, label=f'Threshold: +/-{threshold}')  
        ax.axhline(y=-threshold, color=color, linestyle='--', alpha=0.7)  
        ax.legend(fontsize=8)  
        
    def setup_width_plot(self):  
        self.ax4.set_title("Gripper Width", fontsize=11, fontweight='bold', pad=20)  
        self.ax4.set_xlabel('Time (s)', fontsize=9)  
        self.ax4.set_ylabel('Width (m)', fontsize=9)  
        self.ax4.grid(True, alpha=0.3)  
        
    def setup_status_plot(self):  
        self.ax5.set_title("System Status", fontsize=11, fontweight='bold', pad=20)  
        self.ax5.set_xlabel('Time (s)', fontsize=9)  
        self.ax5.set_ylabel('Status', fontsize=9)  
        self.ax5.grid(True, alpha=0.3)  
        self.ax5.set_ylim(-0.1, 2.1)  
        
    def setup_info_plot(self):  
        self.ax6.axis('off')  
        self.ax6.set_title("System Information", fontsize=11, fontweight='bold', pad=20)  
        
    def update_plots(self):  
        # Update data  
        current_time = time.time() - self.start_time  
        self.time_data.append(current_time)  
        self.force_x_data.append(last_force_x)  
        self.force_y_data.append(last_force_y)  
        self.force_z_data.append(last_force_z)  
        self.width_data.append(last_gripper_width)  
        
        # Status data  
        slip_detected = detect_slip_at_top() and slip_detection_active  
        self.slip_data.append(1.0 if slip_detected else 0.0)  
        self.control_data.append(2.0 if control_active else 0.0)  
        
        if len(self.time_data) < 2:  
            return  
            
        times = list(self.time_data)  
        
        # Clear and plot Force X  
        self.ax1.clear()  
        self.setup_force_plot(self.ax1, "Force X (Fx)", "red", SLIP_THRESHOLD_FX)  
        self.ax1.plot(times, list(self.force_x_data), 'r-', linewidth=2)  
        
        # Clear and plot Force Y  
        self.ax2.clear()  
        self.setup_force_plot(self.ax2, "Force Y (Fy)", "green", SLIP_THRESHOLD_FY)  
        self.ax2.plot(times, list(self.force_y_data), 'g-', linewidth=2)  
        
        # Clear and plot Force Z  
        self.ax3.clear()  
        self.setup_force_plot(self.ax3, "Force Z (Fz)", "blue", TARGET_FORCE)  
        self.ax3.plot(times, list(self.force_z_data), 'b-', linewidth=2)  
        
        # Clear and plot Width  
        self.ax4.clear()  
        self.setup_width_plot()  
        self.ax4.plot(times, list(self.width_data), 'purple', linewidth=3)  
        
        # Clear and plot Status  
        self.ax5.clear()  
        self.setup_status_plot()  
        self.ax5.fill_between(times, list(self.slip_data), alpha=0.7, color='red', label='Slip Detected')  
        self.ax5.fill_between(times, list(self.control_data), alpha=0.5, color='blue', label='Control Active')  
        self.ax5.legend(fontsize=8)  
        
        # Update info panel  
        self.ax6.clear()  
        self.ax6.axis('off')  
        self.ax6.set_title("System Information", fontsize=11, fontweight='bold', pad=20)  
        
        info_text = f"""  
Phase: {current_phase}  

Current Values:  
Fx: {last_force_x:.3f} N  
Fy: {last_force_y:.3f} N  
Fz: {last_force_z:.3f} N  
Width: {last_gripper_width:.4f} m  

Status:  
Slip Detection: {'ACTIVE' if slip_detection_active else 'INACTIVE'}  
Control: {'ACTIVE' if control_active else 'INACTIVE'}  
Slip Corrections: {top_position_slip_corrections}  

Thresholds:  
Fx: +/-{SLIP_THRESHOLD_FX:.2f} N  
Fy: +/-{SLIP_THRESHOLD_FY:.2f} N  
Fz Target: {TARGET_FORCE:.2f} N  
        """  
        
        self.ax6.text(0.05, 0.95, info_text, transform=self.ax6.transAxes,  
                     fontsize=9, verticalalignment='top', fontfamily='monospace')  
        
        # Refresh canvas  
        self.draw_idle()  

class SlipDetectionGUI(QMainWindow):  
    def __init__(self):  
        super().__init__()  
        self.setWindowTitle("Slip Detection with Video Recording - Real-time Monitoring")  
        self.setGeometry(50, 50, 1800, 1100)  
        
        # Initialize screen recorder  
        self.recorder = None  
        
        # Setup UI  
        self.setup_ui()  
        
        # Setup timer  
        self.timer = QTimer()  
        self.timer.timeout.connect(self.update_display)  
        self.timer.start(UPDATE_INTERVAL)  
        
    def setup_ui(self):  
        central_widget = QWidget()  
        self.setCentralWidget(central_widget)  
        layout = QVBoxLayout(central_widget)  
        
        # Status bar with recording indicator  
        status_layout = QHBoxLayout()  
        status_layout.setSpacing(20)  
        
        self.phase_label = QLabel("Phase: Initializing")  
        self.phase_label.setFont(QFont("Arial", 14, QFont.Bold))  
        self.phase_label.setStyleSheet("""  
            color: blue;   
            padding: 15px;   
            border: 2px solid blue;   
            border-radius: 8px;  
            margin: 8px;  
        """)  
        
        self.status_label = QLabel("System Ready")  
        self.status_label.setFont(QFont("Arial", 12, QFont.Bold))  
        self.status_label.setStyleSheet("""  
            color: green;   
            padding: 12px;   
            border: 2px solid green;   
            border-radius: 8px;  
            margin: 8px;  
        """)  
        
        self.recording_label = QLabel("🔴 Recording")  
        self.recording_label.setFont(QFont("Arial", 12, QFont.Bold))  
        self.recording_label.setStyleSheet("""  
            color: red;   
            padding: 12px;   
            border: 2px solid red;   
            border-radius: 8px;  
            margin: 8px;  
        """)  
        
        status_layout.addWidget(self.phase_label)  
        status_layout.addWidget(self.status_label)  
        status_layout.addWidget(self.recording_label)  
        status_layout.addStretch()  
        
        layout.addLayout(status_layout)  
        
        # Matplotlib canvas  
        self.canvas = MatplotlibCanvas()  
        layout.addWidget(self.canvas)  
        
    def update_display(self):  
        self.canvas.update_plots()  
        
        # Update status labels  
        self.phase_label.setText(f"Phase: {current_phase}")  
        
        slip_detected = detect_slip_at_top() and slip_detection_active  
        if slip_detected:  
            self.status_label.setText("SLIP DETECTED!")  
            self.status_label.setStyleSheet("""  
                color: white;   
                background-color: red;   
                padding: 12px;   
                border: 2px solid red;   
                border-radius: 8px;   
                margin: 8px;  
                font-weight: bold;  
            """)  
        elif control_active:  
            self.status_label.setText("CONTROL ACTIVE")  
            self.status_label.setStyleSheet("""  
                color: white;   
                background-color: blue;   
                padding: 12px;   
                border: 2px solid blue;   
                border-radius: 8px;   
                margin: 8px;  
                font-weight: bold;  
            """)  
        else:  
            self.status_label.setText("System Running")  
            self.status_label.setStyleSheet("""  
                color: green;   
                padding: 12px;   
                border: 2px solid green;   
                border-radius: 8px;  
                margin: 8px;  
            """)  
    
    def start_recording(self):  
        """Start video recording"""  
        try:  
            self.recorder = ScreenRecorder(fps=VIDEO_FPS)  
            success = self.recorder.start()  
            if success:  
                self.recording_label.setVisible(True)  
            return success  
        except Exception as e:  
            rospy.logerr(f"Failed to start recording: {e}")  
            return False  
    
    def stop_recording(self):  
        """Stop video recording"""  
        if self.recorder:  
            self.recorder.stop()  
            self.recording_label.setVisible(False)  
    
    def closeEvent(self, event):  
        """Handle window close event"""  
        if self.recorder:  
            self.stop_recording()  
        event.accept()  


def gripper_state_callback(msg):  
    global last_gripper_width  
    try:  
        finger_indices = [i for i, n in enumerate(msg.name) if 'finger_joint1' in n]  
        if finger_indices:  
            width = 2 * msg.position[finger_indices[0]]  
            last_gripper_width = width  
    except Exception as e:  
        print("Fail to get gripper state:", e)  

def tactile_z_callback(msg):  
    global last_force_z  
    with lock:  
        last_force_z = msg.data  

def tactile_x_callback(msg):  
    global last_force_x, force_history_x, force_timestamps, slip_detection_active  
    with lock:  
        last_force_x = msg.data  
        
        # Only collect history when slip detection is active (at top position)  
        if slip_detection_active:  
            current_time = rospy.Time.now().to_sec()  
            force_history_x.append(msg.data)  
            force_timestamps.append(current_time)  
            
            # Keep only recent data within detection window  
            cutoff_time = current_time - SLIP_DETECTION_WINDOW  
            while force_timestamps and force_timestamps[0] < cutoff_time:  
                force_timestamps.pop(0)  
                force_history_x.pop(0)  

def tactile_y_callback(msg):  
    global last_force_y, force_history_y, slip_detection_active  
    with lock:  
        last_force_y = msg.data  
        
        # Only collect history when slip detection is active (at top position)  
        if slip_detection_active:  
            force_history_y.append(msg.data)  
            
            # Keep same length as x history  
            if len(force_history_y) > len(force_history_x):  
                force_history_y.pop(0)  

def detect_slip_at_top():  
    """  
    Detect slip at top position based on Fx and Fy forces  
    Returns True if slip is detected  
    """  
    global force_history_x, force_history_y, force_timestamps  
    
    if not slip_detection_active:  
        return False  
    
    with lock:  
        if len(force_history_x) < 3 or len(force_history_y) < 3:  
            return False  
        
        # Calculate force variations in recent window  
        recent_fx = force_history_x[-3:]  
        recent_fy = force_history_y[-3:]  
        
        # Check for sudden force changes  
        fx_variation = max(recent_fx) - min(recent_fx)  
        fy_variation = max(recent_fy) - min(recent_fy)  
        
        # Check absolute force values  
        current_fx_abs = abs(last_force_x)  
        current_fy_abs = abs(last_force_y)  
        
        slip_detected = (fx_variation > SLIP_THRESHOLD_FX or   
                        fy_variation > SLIP_THRESHOLD_FY or  
                        current_fx_abs > SLIP_THRESHOLD_FX or   
                        current_fy_abs > SLIP_THRESHOLD_FY)  
        
        if slip_detected:  
            rospy.logwarn(f"TOP POSITION SLIP DETECTED! Fx_var={fx_variation:.2f}, Fy_var={fy_variation:.2f}, "  
                         f"Fx_abs={current_fx_abs:.2f}, Fy_abs={current_fy_abs:.2f}")  
        
        return slip_detected  

def normal_p_control_gripper(gripper_open_client, target_force=TARGET_FORCE, kp=KP):  
    """  
    Stage 2: Normal P-control gripper without slip detection  
    """  
    global last_force_z, last_gripper_width, control_active, current_phase  
    
    current_phase = "P-Control Grasp"  
    control_active = True  
    
    width = START_WIDTH  
    rate = rospy.Rate(CONTROL_HZ)  
    
    rospy.loginfo(f"Stage 2: Starting normal P-control grasp - target_force={target_force}N, Kp={kp}")  
    
    iteration = 0  
    max_iterations = 150  

    while not rospy.is_shutdown() and iteration < max_iterations:  
        # Normal P-control only  
        force_error = target_force - abs(last_force_z)  
        d_width = kp * force_error  
        width -= d_width  
        width = max(WIDTH_MIN, min(WIDTH_MAX, width))  # Clamp  
        
        try:  
            goal = MoveGoal(width=width, speed=SPEED)  
            gripper_open_client.send_goal(goal)  
            gripper_open_client.wait_for_result(rospy.Duration(0.7))  
        except Exception as e:  
            rospy.logwarn(f"Gripper control error: {e}")  
        
        rospy.loginfo(f"Stage 2 - Iter {iteration}: Force_z: {last_force_z:.3f}N | Target: {target_force:.2f}N | "  
                     f"Error: {force_error:.3f} | Width: {width:.4f}m")  
        
        if abs(force_error) < FORCE_TOL:  
            rospy.loginfo("Stage 2: Target force reached - grasp complete!")  
            break  
            
        iteration += 1  
        rate.sleep()  
    
    control_active = False  
    rospy.loginfo(f"Stage 2 complete. Final width: {width:.4f}m after {iteration} iterations")  
    return width  

def top_position_slip_control(gripper_open_client, initial_width):  
    """  
    Stage 4: Slip detection and control ONLY at top position for specified time  
    """  
    global slip_detection_active, top_position_slip_corrections, force_history_x, force_history_y, force_timestamps  
    global control_active, current_phase  
    
    current_phase = "Top Position Slip Detection"  
    
    # Enable slip detection  
    slip_detection_active = True  
    control_active = True  
    top_position_slip_corrections = 0  
    current_width = initial_width  
    
    # Clear force history  
    with lock:  
        force_history_x.clear()  
        force_history_y.clear()  
        force_timestamps.clear()  
    
    rate = rospy.Rate(TOP_CONTROL_HZ)  
    
    rospy.loginfo("=" * 70)  
    rospy.loginfo("🔍 STAGE 4: TOP POSITION SLIP DETECTION AND CONTROL 🔍")  
    rospy.loginfo("=" * 70)  
    rospy.loginfo(f"Duration: {TOP_HOLD_TIME} seconds")  
    rospy.loginfo("=== WATCH THE GUI FOR REAL-TIME FORCE VISUALIZATION! ===")  
    rospy.loginfo("- Force X (Red): Horizontal slip forces")  
    rospy.loginfo("- Force Y (Green): Lateral slip forces")   
    rospy.loginfo("- System Status: Red fill = Slip detected!")  
    rospy.loginfo("- Gripper Width: Will decrease if slip corrected")  
    rospy.loginfo("=" * 70)  
    
    start_time = time.time()  
    
    while not rospy.is_shutdown():  
        current_time = time.time()  
        elapsed_time = current_time - start_time  
        
        # Stop after specified hold time  
        if elapsed_time >= TOP_HOLD_TIME:  
            rospy.loginfo("Top position hold time completed")  
            break  
        
        # Check for slip at top position  
        if detect_slip_at_top():  
            # Apply slip correction by decreasing width  
            new_width = max(WIDTH_MIN, current_width - SLIP_WIDTH_DECREASE)  
            
            if new_width != current_width:  # Only apply if change is possible  
                rospy.loginfo(f"🚨 SLIP CORRECTION #{top_position_slip_corrections + 1} 🚨")  
                rospy.loginfo(f"Width adjustment: {current_width:.4f} -> {new_width:.4f}m")  
                
                try:  
                    goal = MoveGoal(width=new_width, speed=SPEED * 1.5)  # Faster response  
                    gripper_open_client.send_goal(goal)  
                    gripper_open_client.wait_for_result(rospy.Duration(0.5))  
                    
                    current_width = new_width  
                    top_position_slip_corrections += 1  
                    
                    # Clear history after correction to avoid false positives  
                    with lock:  
                        force_history_x.clear()  
                        force_history_y.clear()  
                        force_timestamps.clear()  
                        
                    rospy.loginfo(f"✅ Slip correction applied successfully!")  
                    
                except Exception as e:  
                    rospy.logwarn(f"Slip correction error: {e}")  
            else:  
                rospy.logwarn("⚠️  Cannot decrease width further - already at minimum!")  
        
        # Display current status with remaining time (less frequent logging)  
        if int(elapsed_time * 2) % 2 == 0 and elapsed_time != int(elapsed_time / 0.5) * 0.5:  # Every 2 seconds  
            remaining_time = TOP_HOLD_TIME - elapsed_time  
            rospy.loginfo(f"Stage 4 - Fz: {last_force_z:.3f}N | Fx: {last_force_x:.2f}N | "  
                         f"Fy: {last_force_y:.2f}N | Width: {current_width:.4f}m | "  
                         f"Corrections: {top_position_slip_corrections} | Time: {remaining_time:.1f}s")  
        
        rate.sleep()  
    
    # Disable slip detection  
    slip_detection_active = False  
    control_active = False  
    rospy.loginfo("=" * 50)  
    rospy.loginfo(f"🏁 Stage 4 COMPLETE! Total slip corrections: {top_position_slip_corrections}")  
    rospy.loginfo("=" * 50)  

def control_gripper(gripper_action_client, width, speed=0.05, force=10.0, grasp=False):  
    """Closes or opens the gripper."""  
    try:  
        if grasp:  
            epsilon = GraspEpsilon(inner=0.04, outer=0.04)  
            goal = GraspGoal(width=width, epsilon=epsilon, speed=speed, force=force)  
            gripper_action_client.send_goal(goal)  
            gripper_action_client.wait_for_result(rospy.Duration(3.0))  
            gripper_action_client.get_result()  
        else:  
            goal = MoveGoal(width=width, speed=speed)  
            gripper_action_client.send_goal(goal)  
            gripper_action_client.wait_for_result(rospy.Duration(3.0))  
            gripper_action_client.get_result()  
    except Exception as e:  
        rospy.logwarn(f"Gripper control error: {e}")  

def print_current_pose(arm, msg):  
    try:  
        joint_values = arm.get_current_joint_values()  
        pose = arm.get_current_pose().pose  
        rospy.loginfo(f"{msg}:")  
        rospy.loginfo(f"  Joints: {[round(val, 4) for val in joint_values]}")  
        rospy.loginfo(f"  Position: x={pose.position.x:.3f}, y={pose.position.y:.3f}, z={pose.position.z:.3f}")  
    except Exception as e:  
        rospy.logwarn(f"Failed to print pose: {e}")  

def safe_cartesian_move(arm, dx=0, dy=0, dz=0, description=""):  
    """Safe cartesian movement with retry mechanism"""  
    max_retries = 3  
    
    for attempt in range(max_retries):  
        try:  
            rospy.loginfo(f"Attempt {attempt + 1}/{max_retries}: {description}")  
            
            current_pose = arm.get_current_pose().pose  
            target_pose = current_pose  
            target_pose.position.x += dx  
            target_pose.position.y += dy  
            target_pose.position.z += dz  
            
            arm.set_pose_target([target_pose.position.x, target_pose.position.y, target_pose.position.z,  
                                current_pose.orientation.x, current_pose.orientation.y,  
                                current_pose.orientation.z, current_pose.orientation.w])  
            
            success = arm.go(wait=True)  
            
            if success:  
                rospy.loginfo(f"✅ Successfully completed: {description}")  
                return True  
            else:  
                rospy.logwarn(f"❌ Movement failed for {description}")  
                
        except Exception as e:  
            rospy.logwarn(f"Error in attempt {attempt + 1}: {e}")  
        
        if attempt < max_retries - 1:  
            rospy.loginfo("Retrying in 1 second...")  
            rospy.sleep(1.0)  
    
    rospy.logerr(f"❌ Failed to complete movement: {description} after {max_retries} attempts")
    return False

def main():
    roscpp_initialize([])
    rospy.init_node('fr3_slip_detection_with_video_gui')

    app = QApplication(sys.argv)
    gui = SlipDetectionGUI()
    gui.show()
    
    # Start video recording
    rospy.loginfo("Starting video recording...")
    recording_started = gui.start_recording()
    if recording_started:
        rospy.loginfo("✅ Video recording started successfully!")
    else:
        rospy.logwarn("❌ Failed to start video recording")

    # ROS subscribers
    sensor = "tactip"  # can be changed to "AII" or another sensor
    gripper_state_sub = rospy.Subscriber('/franka_gripper/joint_states', JointState, gripper_state_callback)
    tactile_z_sub = rospy.Subscriber(f'/force/{sensor}/z', Float32, tactile_z_callback)
    tactile_x_sub = rospy.Subscriber(f'/force/{sensor}/x', Float32, tactile_x_callback)
    tactile_y_sub = rospy.Subscriber(f'/force/{sensor}/y', Float32, tactile_y_callback)
    
    # MoveIt
    try:
        arm = MoveGroupCommander("fr3_arm")
        gripper_client = actionlib.SimpleActionClient('/franka_gripper/grasp', GraspAction)
        gripper_client.wait_for_server(rospy.Duration(5.0))
        gripper_open_client = actionlib.SimpleActionClient('/franka_gripper/move', MoveAction)
        gripper_open_client.wait_for_server(rospy.Duration(5.0))
        
        rospy.loginfo("All action servers connected!")
    except Exception as e:
        rospy.logerr(f"Failed to initialize MoveIt/Gripper: {e}")
        gui.stop_recording()
        return

    rospy.loginfo("=" * 80)
    rospy.loginfo("🚀 SLIP DETECTION GUI WITH VIDEO RECORDING LAUNCHED! 🚀")
    rospy.loginfo("=" * 80)
    rospy.loginfo("📊 GUI Features:")
    rospy.loginfo("  - Force X/Y curves (red/green) with slip threshold lines")
    rospy.loginfo("  - Force Z curve (blue) showing grasp force control")
    rospy.loginfo("  - Gripper Width (purple) showing control actions")
    rospy.loginfo("  - System Status showing slip detection/control activity")
    rospy.loginfo("  - Real-time slip detection ONLY at top position")
    rospy.loginfo("📹 Video recording at 60 FPS with real-world timestamps")
    rospy.loginfo("=" * 80)

    def robot_execution():
        global current_phase
        
        try:
            # ========== Stage 1: Initial Setup ==========
            current_phase = "Initial Setup"
            rospy.loginfo("=" * 50)
            rospy.loginfo("STAGE 1: Initial Setup")
            rospy.loginfo("=" * 50)
            
            print_current_pose(arm, "Home")
            home = [0.5135, 1.0461, -0.7594, -1.598, 2.0161, 1.1814, -1.596]
            arm.go(home, wait=True)
            control_gripper(gripper_open_client, width=START_WIDTH, speed=0.1, grasp=False)
            
            rospy.loginfo("Waiting for sensors to stabilize...")
            rospy.sleep(3.0)
            rospy.loginfo(f"Initial sensor readings - Fx: {last_force_x:.3f}N, Fy: {last_force_y:.3f}N, Fz: {last_force_z:.3f}N")

            # ========== Stage 2: Move down to object ==========
            current_phase = "Moving Down"
            rospy.loginfo("=" * 50)
            rospy.loginfo("STAGE 2: Moving down to object")
            rospy.loginfo(f"Target movement: {MOVE_DOWN:.3f}m downward")
            rospy.loginfo("=" * 50)
            
            success = safe_cartesian_move(arm, dx=0, dy=0, dz=-MOVE_DOWN, 
                                        description="Moving down to object")
            
            if not success:
                rospy.logerr("Failed to move down to object")
                return
                
            print_current_pose(arm, "Down Position - Object Contact")
            
            rospy.loginfo("Checking contact forces...")
            rospy.sleep(1.0)
            rospy.loginfo(f"Contact forces - Fx: {last_force_x:.3f}N, Fy: {last_force_y:.3f}N, Fz: {last_force_z:.3f}N")

            # ========== Stage 3: Normal P-Control Grasp ==========
            rospy.loginfo("=" * 50)
            rospy.loginfo("STAGE 3: P-Control Force Grasp")
            rospy.loginfo("🎯 Watch GUI: Force Z curve approaching target!")
            rospy.loginfo("=" * 50)
            
            final_width = normal_p_control_gripper(gripper_open_client, TARGET_FORCE, KP)
            
            rospy.loginfo("Grasp completed - stabilizing...")
            rospy.sleep(3.0)
            rospy.loginfo(f"Post-grasp forces - Fx: {last_force_x:.3f}N, Fy: {last_force_y:.3f}N, Fz: {last_force_z:.3f}N")

            # ========== Stage 4: Lift to top position ==========
            current_phase = "Lifting Up"
            rospy.loginfo("=" * 50)
            rospy.loginfo("STAGE 4: Lifting object to top position")
            rospy.loginfo("🎯 Watch GUI: Force changes during lift!")
            rospy.loginfo("=" * 50)
            
            success = safe_cartesian_move(arm, dx=0, dy=0, dz=MOVE_DOWN, 
                                        description="Lifting to top position")
            
            if not success:
                rospy.logwarn("Lift failed, but continuing...")
                
            print_current_pose(arm, "Top Position Reached")
            
            rospy.loginfo("Top position reached - checking forces:")
            rospy.sleep(1.0)
            rospy.loginfo(f"Top position forces - Fx: {last_force_x:.3f}N, Fy: {last_force_y:.3f}N, Fz: {last_force_z:.3f}N")

            # ========== Stage 5: SLIP DETECTION AND CONTROL ==========
            rospy.loginfo("=" * 70)
            rospy.loginfo("🔍 STAGE 5: TOP POSITION SLIP DETECTION AND CONTROL 🔍")
            rospy.loginfo("=" * 70)
            rospy.loginfo("NOW WATCH THE GUI CAREFULLY!")
            rospy.loginfo("- Force X (Red): Horizontal slip forces")
            rospy.loginfo("- Force Y (Green): Lateral slip forces")
            rospy.loginfo("- System Status: Red fill = Slip detected!")
            rospy.loginfo("- Gripper Width: Will decrease if slip corrected")
            rospy.loginfo("=" * 70)
            
            top_position_slip_control(gripper_open_client, final_width)

            # ========== Stage 6: Move back down ==========
            current_phase = "Moving Down"
            rospy.loginfo("=" * 50)
            rospy.loginfo("STAGE 6: Moving back down")
            rospy.loginfo("=" * 50)
            
            success = safe_cartesian_move(arm, dx=0, dy=0, dz=-MOVE_DOWN, 
                                        description="Moving back down")
            
            if not success:
                rospy.logwarn("Down movement failed, but continuing...")
                
            print_current_pose(arm, "Back Down Position")

            # ========== Stage 7: Release object ==========
            current_phase = "Releasing"
            rospy.loginfo("=" * 50)
            rospy.loginfo("STAGE 7: Releasing object")
            rospy.loginfo("=" * 50)
            
            control_gripper(gripper_open_client, width=0.08, speed=0.1, grasp=False)
            rospy.loginfo("Object released!")
            
            rospy.sleep(2.0)
            rospy.loginfo(f"After release - Fx: {last_force_x:.3f}N, Fy: {last_force_y:.3f}N, Fz: {last_force_z:.3f}N")

            # ========== Stage 8: Return home ==========
            current_phase = "Returning Home"
            rospy.loginfo("=" * 50)
            rospy.loginfo("STAGE 8: Returning to home position")
            rospy.loginfo("=" * 50)
            
            arm.go(home, wait=True)
            print_current_pose(arm, "Final Home Position")

            # ========== Completion Summary ==========
            current_phase = "Completed"
            rospy.loginfo("=" * 80)
            rospy.loginfo("🎉 ALL STAGES COMPLETED SUCCESSFULLY! 🎉")
            rospy.loginfo("=" * 80)
            rospy.loginfo("📊 FINAL RESULTS:")
            rospy.loginfo(f"   - Total slip corrections: {top_position_slip_corrections}")
            rospy.loginfo(f"   - Final gripper width: {last_gripper_width:.4f}m")
            rospy.loginfo(f"   - Final force Z: {last_force_z:.3f}N")
            rospy.loginfo(f"   - Final force X: {last_force_x:.3f}N")
            rospy.loginfo(f"   - Final force Y: {last_force_y:.3f}N")
            rospy.loginfo("📹 Video recording will be saved when GUI closes")
            rospy.loginfo("📈 Check the GUI plots for complete slip detection data!")
            rospy.loginfo("=" * 80)
            
        except KeyboardInterrupt:
            rospy.loginfo("Execution interrupted by user")
            current_phase = "Interrupted"
        except Exception as e:
            rospy.logerr(f"Robot execution error: {e}")
            current_phase = "Error"
            import traceback
            traceback.print_exc()
        
        finally:
            try:
                arm.stop()
                arm.clear_pose_targets()
                rospy.loginfo("Arm stopped safely")
            except:
                pass

    robot_thread = threading.Thread(target=robot_execution)
    robot_thread.daemon = True
    robot_thread.start()

    try:
        rospy.loginfo("Starting GUI event loop...")
        app.exec_()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down GUI...")
    finally:
        rospy.loginfo("Stopping video recording...")
        gui.stop_recording()
        rospy.loginfo("Cleaning up...")
        roscpp_shutdown()

if __name__ == '__main__':
    main()