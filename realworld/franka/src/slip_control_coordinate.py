#!/usr/bin/env python3  

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
import sys  
import os  
import traceback

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import GUI module
from gui_recorder_double import SlipDetectionGUI

# --- Sensor names ---
SENSORS = ["tactip", "uskin"]  # dual sensors

# --- Force control parameters ---  
TARGET_FORCE = 1.0  # pen 1N  meat box 1.2/1.5 plum 1/1.2 banana1.2
KP = 0.0004  
WIDTH_MIN = 0.0  
WIDTH_MAX = 0.08  
START_WIDTH = 0.05 # meat_box 0.065 pen 0.045
FORCE_TOL = 0.1  
CONTROL_HZ = 10  
SPEED = 0.005  
MOVE_DOWN = 0.1 # plum 0.08 meat 0.03 pen 0.03

# --- Slip detection parameters (per sensor) ---  
SLIP_THRESHOLDS = {
    "tactip": {"fx": 0.2, "fy": 0.2},
    "uskin": {"fx": 0.3, "fy": 0.3}
}
SLIP_WIDTH_DECREASE = 0.001  
SLIP_DETECTION_WINDOW = 0.5  
TOP_HOLD_TIME = 10.0  
TOP_CONTROL_HZ = 20  

# --- GUI parameters ---  
WINDOW_LENGTH = 10  
VIDEO_FPS = 30  

# Global variables - dual-sensor data store
last_gripper_width = 0.0  
sensor_data = {
    "tactip": {"fx": 0.0, "fy": 0.0, "fz": 0.0, 
               "fx_history": [], "fy_history": [], "timestamps": []},
    "uskin": {"fx": 0.0, "fy": 0.0, "fz": 0.0,
              "fx_history": [], "fy_history": [], "timestamps": []}
}
lock = threading.Lock()  

# Control flags  
slip_detection_active = False  
top_position_slip_corrections = 0  
control_active = False  
current_phase = "Initializing"  

# GUI instance  
gui = None  

def gripper_state_callback(msg):  
    global last_gripper_width, gui  
    try:  
        finger_indices = [i for i, n in enumerate(msg.name) if 'finger_joint1' in n]  
        if finger_indices:  
            width = 2 * msg.position[finger_indices[0]]  
            last_gripper_width = width  
            
            if gui:  
                gui.update_gripper_data(width)  
                
    except Exception as e:  
        rospy.logwarn(f"Gripper state callback error: {e}")  

def create_tactile_callback(sensor_name, force_type):
    """Factory to create a tactile sensor callback function"""
    def callback(msg):
        global sensor_data, slip_detection_active, gui
        with lock:
            sensor_data[sensor_name][force_type] = msg.data
            
            # Only x and y need history for slip detection
            if force_type in ["fx", "fy"] and slip_detection_active:
                current_time = rospy.Time.now().to_sec()
                
                if force_type == "fx":
                    sensor_data[sensor_name]["fx_history"].append(msg.data)
                    sensor_data[sensor_name]["timestamps"].append(current_time)
                    
                    # Clean up old data
                    cutoff_time = current_time - SLIP_DETECTION_WINDOW
                    while (sensor_data[sensor_name]["timestamps"] and 
                           sensor_data[sensor_name]["timestamps"][0] < cutoff_time):
                        sensor_data[sensor_name]["timestamps"].pop(0)
                        sensor_data[sensor_name]["fx_history"].pop(0)
                        
                elif force_type == "fy":
                    sensor_data[sensor_name]["fy_history"].append(msg.data)
                    if len(sensor_data[sensor_name]["fy_history"]) > len(sensor_data[sensor_name]["fx_history"]):
                        sensor_data[sensor_name]["fy_history"].pop(0)
        
        # Update GUI on each data receipt (only update the corresponding sensor)
        if gui and force_type == "fz":  # only trigger GUI update on Fz to reduce frequency
            try:
                with lock:
                    fx = sensor_data[sensor_name]["fx"]
                    fy = sensor_data[sensor_name]["fy"]
                    fz = sensor_data[sensor_name]["fz"]
                gui.update_force_data(fx, fy, fz, sensor_name=sensor_name)
            except Exception as e:
                rospy.logwarn(f"GUI update error in callback: {e}")
    
    return callback

def get_fused_force_z():
    """Get fused Z-direction force (average)"""
    with lock:
        fz_tactip = sensor_data["tactip"]["fz"]
        fz_uskin = sensor_data["uskin"]["fz"]
        return (fz_tactip + fz_uskin) / 2.0

def update_gui_force_data():
    """Update GUI force sensor data - dual-sensor version"""
    global gui  
    if not gui:
        return
        
    try:
        # Update data for each sensor
        for sensor in SENSORS:
            with lock:
                fx = sensor_data[sensor]["fx"]
                fy = sensor_data[sensor]["fy"]
                fz = sensor_data[sensor]["fz"]
            
            gui.update_force_data(fx, fy, fz, sensor_name=sensor)
        
        # Update status
        slip_detected = detect_slip_dual_sensor() and slip_detection_active  
        gui.update_status_data(slip_detected, control_active)  
        gui.update_system_status(current_phase, top_position_slip_corrections)
        
    except Exception as e:
        rospy.logwarn(f"GUI force data update error: {e}")

def detect_slip_dual_sensor():
    """Dual-sensor collaborative slip detection - triggers if either sensor detects slip"""
    if not slip_detection_active:
        return False
    
    slip_results = {}
    
    with lock:
        for sensor in SENSORS:
            data = sensor_data[sensor]
            
            if len(data["fx_history"]) < 3 or len(data["fy_history"]) < 3:
                slip_results[sensor] = False
                continue
            
            # Compute force variation
            recent_fx = data["fx_history"][-3:]
            recent_fy = data["fy_history"][-3:]
            
            fx_variation = max(recent_fx) - min(recent_fx)
            fy_variation = max(recent_fy) - min(recent_fy)
            
            # Use sensor-specific thresholds
            threshold_fx = SLIP_THRESHOLDS[sensor]["fx"]
            threshold_fy = SLIP_THRESHOLDS[sensor]["fy"]
            
            slip_detected = (fx_variation > threshold_fx or fy_variation > threshold_fy)
            slip_results[sensor] = slip_detected
            
            if slip_detected:
                rospy.logwarn(f"[{sensor.upper()}] SLIP DETECTED! "
                            f"Fx_var={fx_variation:.2f}, Fy_var={fy_variation:.2f}")
    
    # Return True if any sensor detects slip
    overall_slip = any(slip_results.values())
    
    if overall_slip:
        rospy.logwarn(f"🚨 DUAL SENSOR SLIP: tactip={slip_results['tactip']}, "
                     f"uskin={slip_results['uskin']}")
    
    return overall_slip

def calculate_slip_correction_amount():
    """Calculate slip correction amount - weighted by both sensors' force changes"""
    with lock:
        corrections = []
        
        for sensor in SENSORS:
            data = sensor_data[sensor]
            if len(data["fx_history"]) < 3:
                continue
                
            # Compute force variation rate
            recent_fx = data["fx_history"][-3:]
            recent_fy = data["fy_history"][-3:]
            
            fx_variation = max(recent_fx) - min(recent_fx)
            fy_variation = max(recent_fy) - min(recent_fy)
            
            # Compute suggested correction for this sensor (proportional to force variation)
            max_variation = max(fx_variation, fy_variation)
            threshold = max(SLIP_THRESHOLDS[sensor]["fx"], SLIP_THRESHOLDS[sensor]["fy"])
            
            if max_variation > threshold:
                # The more it exceeds the threshold, the larger the correction
                correction = SLIP_WIDTH_DECREASE * (1 + (max_variation - threshold) / threshold)
                corrections.append(correction)
        
        # Take the average of suggested corrections from both sensors
        if corrections:
            return sum(corrections) / len(corrections)
        else:
            return SLIP_WIDTH_DECREASE

def normal_p_control_gripper(gripper_open_client, target_force=TARGET_FORCE, kp=KP):  
    """P-control grasp - uses fused Fz"""
    global control_active, current_phase  
    
    current_phase = "P-Control Grasp (Dual Sensor)"  
    control_active = True  
    
    width = START_WIDTH  
    rate = rospy.Rate(CONTROL_HZ)  
    
    rospy.loginfo(f"Starting DUAL-SENSOR P-control grasp - target_force={target_force}N, Kp={kp}")  
    
    iteration = 0  
    max_iterations = 150  

    while not rospy.is_shutdown() and iteration < max_iterations:  
        fused_fz = get_fused_force_z()
        
        force_error = target_force - abs(fused_fz)  
        d_width = kp * force_error  
        width -= d_width  
        width = max(WIDTH_MIN, min(WIDTH_MAX, width))  
        
        try:  
            goal = MoveGoal(width=width, speed=SPEED)  
            gripper_open_client.send_goal(goal)  
            gripper_open_client.wait_for_result(rospy.Duration(0.7))  
        except Exception as e:  
            rospy.logwarn(f"Gripper control error: {e}")  
        
        # Update GUI
        update_gui_force_data()  
        
        with lock:
            tactip_fz = sensor_data["tactip"]["fz"]
            uskin_fz = sensor_data["uskin"]["fz"]
        
        rospy.loginfo(f"P-Control Iter {iteration}: Fz_fused: {fused_fz:.3f}N | "
                     f"Tactip: {tactip_fz:.3f}N | Uskin: {uskin_fz:.3f}N | "
                     f"Target: {target_force:.2f}N | Width: {width:.4f}m")  
        
        if abs(force_error) < FORCE_TOL:  
            rospy.loginfo("✅ Target force reached - grasp complete!")  
            break  
            
        iteration += 1  
        rate.sleep()  
    
    control_active = False  
    rospy.loginfo(f"P-control complete. Final width: {width:.4f}m after {iteration} iterations")  
    return width  

def top_position_slip_control(gripper_open_client, initial_width):
    """Top-position slip detection and control - dual-sensor collaborative version"""
    global slip_detection_active, top_position_slip_corrections  
    global control_active, current_phase  
    
    current_phase = "Top Position Slip Detection (Dual Sensor)"  
    
    slip_detection_active = True  
    control_active = True  
    top_position_slip_corrections = 0  
    current_width = initial_width  
    
    # Clear all sensors' history data
    with lock:
        for sensor in SENSORS:
            sensor_data[sensor]["fx_history"].clear()
            sensor_data[sensor]["fy_history"].clear()
            sensor_data[sensor]["timestamps"].clear()
    
    rate = rospy.Rate(TOP_CONTROL_HZ)  
    
    rospy.loginfo("=" * 70)  
    rospy.loginfo("🔍 DUAL-SENSOR TOP POSITION SLIP DETECTION 🔍")  
    rospy.loginfo("=" * 70)  
    rospy.loginfo(f"Duration: {TOP_HOLD_TIME} seconds")  
    rospy.loginfo("=== BOTH TACTIP AND USKIN MONITORING ACTIVE! ===")  
    rospy.loginfo("=" * 70)  
    
    start_time = time.time()  
    last_log_time = 0
    
    while not rospy.is_shutdown():  
        current_time = time.time()  
        elapsed_time = current_time - start_time  
        
        if elapsed_time >= TOP_HOLD_TIME:  
            rospy.loginfo("✅ Top position hold time completed")  
            break  
        
        update_gui_force_data()  
        
        # Dual-sensor slip detection
        if detect_slip_dual_sensor():  
            correction_amount = calculate_slip_correction_amount()
            new_width = max(WIDTH_MIN, current_width - correction_amount)  
            
            if new_width != current_width:
                rospy.loginfo(f"🚨 DUAL-SENSOR SLIP CORRECTION #{top_position_slip_corrections + 1} 🚨")  
                
                with lock:
                    tactip_fx = sensor_data["tactip"]["fx"]
                    tactip_fy = sensor_data["tactip"]["fy"]
                    uskin_fx = sensor_data["uskin"]["fx"]
                    uskin_fy = sensor_data["uskin"]["fy"]
                
                rospy.loginfo(f"  Tactip: Fx={tactip_fx:.2f}N, Fy={tactip_fy:.2f}N")
                rospy.loginfo(f"  Uskin:  Fx={uskin_fx:.2f}N, Fy={uskin_fy:.2f}N")
                rospy.loginfo(f"  Width: {current_width:.4f}m -> {new_width:.4f}m (Δ={correction_amount:.4f}m)")  
                
                try:  
                    goal = MoveGoal(width=new_width, speed=SPEED * 1.5)  
                    gripper_open_client.send_goal(goal)  
                    gripper_open_client.wait_for_result(rospy.Duration(0.5))  
                    
                    current_width = new_width  
                    top_position_slip_corrections += 1  
                    
                    # Clear history to avoid false positives
                    with lock:
                        for sensor in SENSORS:
                            sensor_data[sensor]["fx_history"].clear()
                            sensor_data[sensor]["fy_history"].clear()
                            sensor_data[sensor]["timestamps"].clear()
                        
                    rospy.loginfo(f"✅ Slip correction #{top_position_slip_corrections} applied successfully!")  
                    
                except Exception as e:  
                    rospy.logwarn(f"❌ Slip correction error: {e}")  
            else:  
                rospy.logwarn("⚠️  Cannot decrease width further (at minimum)!")  
        
        # Periodically display status (every 2 seconds)
        if current_time - last_log_time >= 2.0:
            remaining_time = TOP_HOLD_TIME - elapsed_time
            fused_fz = get_fused_force_z()
            
            with lock:
                tactip_fx = sensor_data["tactip"]["fx"]
                tactip_fy = sensor_data["tactip"]["fy"]
                uskin_fx = sensor_data["uskin"]["fx"]
                uskin_fy = sensor_data["uskin"]["fy"]
            
            rospy.loginfo(f"📊 Status - Time: {remaining_time:.1f}s | Width: {current_width:.4f}m | Corrections: {top_position_slip_corrections}")
            rospy.loginfo(f"   Fused Fz: {fused_fz:.3f}N | Tactip Fx/Fy: {tactip_fx:.2f}/{tactip_fy:.2f}N | Uskin Fx/Fy: {uskin_fx:.2f}/{uskin_fy:.2f}N")
            
            last_log_time = current_time
        
        rate.sleep()  
    
    slip_detection_active = False  
    control_active = False  
    rospy.loginfo("=" * 70)  
    rospy.loginfo(f"🏁 Dual-sensor slip control COMPLETE!")
    rospy.loginfo(f"   Total corrections: {top_position_slip_corrections}")  
    rospy.loginfo("=" * 70)  

def control_gripper(gripper_action_client, width, speed=0.05, force=10.0, grasp=False):
    """Control gripper open/close"""
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
    """Print current robot pose"""
    try:  
        joint_values = arm.get_current_joint_values()  
        pose = arm.get_current_pose().pose  
        rospy.loginfo(f"{msg}:")  
        rospy.loginfo(f"  Joints: {[round(val, 4) for val in joint_values]}")  
        rospy.loginfo(f"  Position: x={pose.position.x:.3f}, y={pose.position.y:.3f}, z={pose.position.z:.3f}")  
    except Exception as e:  
        rospy.logwarn(f"Failed to print pose: {e}")  

def safe_cartesian_move(arm, dx=0, dy=0, dz=0, description=""):
    """Safe Cartesian movement"""
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

def gui_update_thread():
    """GUI data update thread"""
    global gui
    rate = rospy.Rate(30)  # 30Hz update rate
    
    while not rospy.is_shutdown():  
        try:  
            update_gui_force_data()  
            rate.sleep()  
        except Exception as e:  
            rospy.logwarn(f"GUI update thread error: {e}")
            traceback.print_exc()
            break  

def main():  
    global gui, current_phase  
    
    roscpp_initialize([])  
    rospy.init_node('fr3_dual_sensor_slip_detection')  
    
    # Create GUI
    rospy.loginfo("Initializing Dual-Sensor GUI...")
    gui = SlipDetectionGUI(  
        window_length=WINDOW_LENGTH,  
        video_fps=VIDEO_FPS,  
        video_width=1500,  
        video_height=1200  
    )  
    
    gui.setup_window()  
    rospy.loginfo("✅ GUI window setup complete")
    
    # Start video recording
    video_filename = "video/dual_sensor_slip_detection"  
    rospy.loginfo("Starting video recording...")  
    recording_started = gui.start_recording(video_filename)  
    if recording_started:  
        rospy.loginfo("✅ Video recording started successfully!")  
    else:  
        rospy.logwarn("❌ Failed to start video recording")  
    
    # Start data saving
    data_filename = "data/dual_sensor_experiment"
    rospy.loginfo("Starting data saving...")
    data_saving_started = gui.start_data_saving(data_filename)
    if data_saving_started:
        rospy.loginfo("✅ Data saving started successfully!")
    else:
        rospy.logwarn("❌ Failed to start data saving")

    # ROS subscribers - dual sensors
    gripper_state_sub = rospy.Subscriber('/franka_gripper/joint_states', JointState, gripper_state_callback)  
    
    # Create subscriptions for each sensor
    for sensor in SENSORS:
        rospy.Subscriber(f'/force/{sensor}/x', Float32, create_tactile_callback(sensor, "fx"))
        rospy.Subscriber(f'/force/{sensor}/y', Float32, create_tactile_callback(sensor, "fy"))
        rospy.Subscriber(f'/force/{sensor}/z', Float32, create_tactile_callback(sensor, "fz"))
        rospy.loginfo(f"✅ Subscribed to {sensor} force topics (/force/{sensor}/x/y/z)")
    
    # Initialize MoveIt and gripper clients
    try:  
        rospy.loginfo("Initializing MoveIt and Gripper clients...")
        arm = MoveGroupCommander("fr3_arm")  
        gripper_client = actionlib.SimpleActionClient('/franka_gripper/grasp', GraspAction)  
        gripper_client.wait_for_server(rospy.Duration(5.0))  
        gripper_open_client = actionlib.SimpleActionClient('/franka_gripper/move', MoveAction)  
        gripper_open_client.wait_for_server(rospy.Duration(5.0))  
        
        rospy.loginfo("✅ All action servers connected!")  
    except Exception as e:  
        rospy.logerr(f"❌ Failed to initialize MoveIt/Gripper: {e}")
        traceback.print_exc()
        if gui:  
            gui.stop_recording()
            gui.stop_data_saving()
        return  

    rospy.loginfo("=" * 80)  
    rospy.loginfo("🚀 DUAL-SENSOR SLIP DETECTION SYSTEM LAUNCHED! 🚀")  
    rospy.loginfo("=" * 80)  
    rospy.loginfo("📊 System Features:")  
    rospy.loginfo("  ✓ Tactip + Uskin dual sensor fusion")
    rospy.loginfo("  ✓ Intelligent slip detection and correction")  
    rospy.loginfo("  ✓ Real-time force monitoring (6 channels + fused)")
    rospy.loginfo("  ✓ Video recording with timestamps")
    rospy.loginfo("  ✓ CSV data export with statistics")
    rospy.loginfo("=" * 80)  

    # Start GUI update thread
    gui_thread = threading.Thread(target=gui_update_thread, daemon=True)  
    gui_thread.start()  
    rospy.loginfo("✅ GUI update thread started")

    # Robot control logic
    def robot_execution():  
        global current_phase  
        
        try:  
            # ========== Stage 1: Initial Setup ==========  
            current_phase = "Initial Setup"  
            rospy.loginfo("=" * 60)  
            rospy.loginfo("STAGE 1: Initial Setup & Sensor Calibration")  
            rospy.loginfo("=" * 60)  
            
            print_current_pose(arm, "Current Position")  
            home = [0.5135, 1.0461, -0.7594, -1.598, 2.0161, 1.1814, -1.596]  
            rospy.loginfo("Moving to home position...")
            arm.go(home, wait=True)  
            rospy.loginfo("Opening gripper...")
            control_gripper(gripper_open_client, width=START_WIDTH, speed=0.1, grasp=False)  
            
            rospy.loginfo("Waiting for dual sensors to stabilize (3 seconds)...")  
            rospy.sleep(3.0)
            
            # Show initial sensor readings
            with lock:
                rospy.loginfo("Initial sensor readings:")
                rospy.loginfo(f"  Tactip - Fx: {sensor_data['tactip']['fx']:.3f}N, "
                            f"Fy: {sensor_data['tactip']['fy']:.3f}N, "
                            f"Fz: {sensor_data['tactip']['fz']:.3f}N")
                rospy.loginfo(f"  Uskin  - Fx: {sensor_data['uskin']['fx']:.3f}N, "
                            f"Fy: {sensor_data['uskin']['fy']:.3f}N, "
                            f"Fz: {sensor_data['uskin']['fz']:.3f}N")

            # ========== Stage 2: Move down to object ==========  
            current_phase = "Moving Down"  
            rospy.loginfo("=" * 60)  
            rospy.loginfo("STAGE 2: Approaching Object")  
            rospy.loginfo("=" * 60)  
            
            success = safe_cartesian_move(arm, dx=0, dy=0, dz=-MOVE_DOWN, 
                                        description=f"Moving down {MOVE_DOWN}m to object")
            
            if not success:
                rospy.logerr("❌ Failed to move down to object - ABORTING")
                return
                
            print_current_pose(arm, "Contact Position")
            rospy.sleep(1.0)

            # ========== Stage 3: P-Control Grasp ==========
            rospy.loginfo("=" * 60)
            rospy.loginfo("STAGE 3: Dual-Sensor Force-Controlled Grasping")
            rospy.loginfo("=" * 60)
            
            final_width = normal_p_control_gripper(gripper_open_client, TARGET_FORCE, KP)
            
            rospy.loginfo(f"Grasp achieved at width: {final_width:.4f}m - Stabilizing...")
            rospy.sleep(3.0)

            # ========== Stage 4: Lift Object ==========
            current_phase = "Lifting Up"
            rospy.loginfo("=" * 60)
            rospy.loginfo("STAGE 4: Lifting Object")
            rospy.loginfo("=" * 60)
            
            success = safe_cartesian_move(arm, dx=0, dy=0, dz=MOVE_DOWN, 
                                        description=f"Lifting {MOVE_DOWN}m to top position")
            
            if not success:
                rospy.logwarn("⚠️  Lift partially failed, but continuing...")
                
            print_current_pose(arm, "Top Position")
            rospy.sleep(1.0)

            # ========== Stage 5: Slip Detection ==========
            rospy.loginfo("=" * 70)
            rospy.loginfo("🔍 STAGE 5: DUAL-SENSOR SLIP DETECTION & CONTROL 🔍")
            rospy.loginfo("=" * 70)
            
            top_position_slip_control(gripper_open_client, final_width)

            # ========== Stage 6: Return Object ==========
            current_phase = "Placing Down"
            rospy.loginfo("=" * 60)
            rospy.loginfo("STAGE 6: Returning Object")
            rospy.loginfo("=" * 60)
            safe_cartesian_move(arm, dx=0, dy=0, dz=-MOVE_DOWN, 
                              description="Moving back down to original position")

            # ========== Stage 7: Release ==========
            current_phase = "Releasing"
            rospy.loginfo("=" * 60)
            rospy.loginfo("STAGE 7: Releasing Object")
            rospy.loginfo("=" * 60)
            control_gripper(gripper_open_client, width=0.08, speed=0.1, grasp=False)
            rospy.sleep(2.0)

            # ========== Stage 8: Return Home ==========
            current_phase = "Returning Home"
            rospy.loginfo("=" * 60)
            rospy.loginfo("STAGE 8: Returning to Home Position")
            rospy.loginfo("=" * 60)
            arm.go(home, wait=True)

            # ========== Experiment Complete ==========
            current_phase = "Completed"
            rospy.loginfo("=" * 80)
            rospy.loginfo("🎉 DUAL-SENSOR EXPERIMENT COMPLETED SUCCESSFULLY! 🎉")
            rospy.loginfo("=" * 80)
            rospy.loginfo(f"📊 Total slip corrections applied: {top_position_slip_corrections}")
            rospy.loginfo(f"🎥 Video and data saved to video/ and data/ directories")
            rospy.loginfo("=" * 80)
            
        except KeyboardInterrupt:
            rospy.loginfo("⚠️  Execution interrupted by user")
            current_phase = "Interrupted"
        except Exception as e:
            rospy.logerr(f"❌ Robot execution error: {e}")
            current_phase = "Error"
            traceback.print_exc()
        
        finally:
            try:
                arm.stop()
                arm.clear_pose_targets()
                rospy.loginfo("✅ Arm stopped safely")
            except:
                pass

    # Start robot control thread
    robot_thread = threading.Thread(target=robot_execution, daemon=True)
    robot_thread.start()
    rospy.loginfo("✅ Robot execution thread started")

    try:
        rospy.loginfo("Starting GUI main loop...")
        gui.run()
    except KeyboardInterrupt:
        rospy.loginfo("⚠️  Shutting down due to KeyboardInterrupt...")
    except Exception as e:
        rospy.logerr(f"❌ GUI error: {e}")
        traceback.print_exc()
    finally:
        rospy.loginfo("Cleaning up resources...")
        if gui:
            gui.stop_recording()
            gui.stop_data_saving()
        rospy.loginfo("Shutting down ROS...")
        roscpp_shutdown()
        rospy.loginfo("✅ Shutdown complete.")
        

if __name__ == '__main__':
    main()