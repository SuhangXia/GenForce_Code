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

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import GUI module
from gui_recorder_single import SlipDetectionGUI


# sensor = "tactip"  # can be changed to another sensor name
sensor = "uskin"  # can be changed to another sensor name

# --- Force control parameters ---  
# TARGET_FORCE = 1.5  # pen 1N  meat box 1.2/1.5 plum 1/1.2 banana1.2
TARGET_FORCE = 1.5  # pen 1N  meat box 1.2/1.5 plum 1/1.2 banana1.2
KP = 0.0004  
WIDTH_MIN = 0.0  
WIDTH_MAX = 0.08  
START_WIDTH = 0.05 # meat_box 0.065 pen 0.045
FORCE_TOL = 0.1  
CONTROL_HZ = 10  
SPEED = 0.005  
MOVE_DOWN = 0.1  # plum 0.08 meat 0.03 pen 0.03 banana 0.094

# --- Slip detection parameters ---  
SLIP_THRESHOLD_FX = 0.3  #tactip 0.2 uskin 0.3
SLIP_THRESHOLD_FY = 0.3 
SLIP_WIDTH_DECREASE = 0.001  
SLIP_DETECTION_WINDOW = 0.5  
TOP_HOLD_TIME = 10.0  
TOP_CONTROL_HZ = 20  

# --- GUI parameters ---  
WINDOW_LENGTH = 10  
VIDEO_FPS = 30  

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

# GUI instance  
gui = None  

# === Sensor callbacks ===
def gripper_state_callback(msg):  
    global last_gripper_width, gui  
    try:  
        finger_indices = [i for i, n in enumerate(msg.name) if 'finger_joint1' in n]  
        if finger_indices:  
            width = 2 * msg.position[finger_indices[0]]  
            last_gripper_width = width  
            
            # Update GUI
            if gui:
                gui.update_gripper_data(width)
                
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
        
        # Only collect history when slip detection is active  
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
        
        # Only collect history when slip detection is active  
        if slip_detection_active:  
            force_history_y.append(msg.data)  
            
            # Keep same length as x history  
            if len(force_history_y) > len(force_history_x):  
                force_history_y.pop(0)  

def update_gui_force_data():
    """Update GUI force sensor data"""
    global gui  
    if gui:  
        gui.update_force_data(last_force_x, last_force_y, last_force_z)  
        # 更新状态  
        slip_detected = detect_slip_at_top() and slip_detection_active  
        gui.update_status_data(slip_detected, control_active)  
        gui.update_system_status(current_phase, top_position_slip_corrections)  

# === Slip detection functions ===
def detect_slip_at_top():
    """Detect slip at the top position"""
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
        
        # slip_detected = (fx_variation > SLIP_THRESHOLD_FX or   
        #                 fy_variation > SLIP_THRESHOLD_FY or  
        #                 current_fx_abs > SLIP_THRESHOLD_FX or   
        #                 current_fy_abs > SLIP_THRESHOLD_FY)  
        
        slip_detected = (fx_variation > SLIP_THRESHOLD_FX or   
                        fy_variation > SLIP_THRESHOLD_FY)
        
        if slip_detected:  
            rospy.logwarn(f"SLIP DETECTED! Fx_var={fx_variation:.2f}, Fy_var={fy_variation:.2f}, "  
                         f"Fx_abs={current_fx_abs:.2f}, Fy_abs={current_fy_abs:.2f}")  
        
        return slip_detected  

def normal_p_control_gripper(gripper_open_client, target_force=TARGET_FORCE, kp=KP):
    """P-control grasp"""
    global last_force_z, last_gripper_width, control_active, current_phase  
    
    current_phase = "P-Control Grasp"  
    control_active = True  
    
    width = START_WIDTH  
    rate = rospy.Rate(CONTROL_HZ)  
    
    rospy.loginfo(f"Starting P-control grasp - target_force={target_force}N, Kp={kp}")  
    
    iteration = 0  
    max_iterations = 150  

    while not rospy.is_shutdown() and iteration < max_iterations:  
        # P-control logic  
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
        
        # Update GUI data
        update_gui_force_data()
        
        rospy.loginfo(f"P-Control - Iter {iteration}: Fz: {last_force_z:.3f}N | Target: {target_force:.2f}N | "  
                     f"Error: {force_error:.3f} | Width: {width:.4f}m")  
        
        if abs(force_error) < FORCE_TOL:  
            rospy.loginfo("Target force reached - grasp complete!")  
            break  
            
        iteration += 1  
        rate.sleep()  
    
    control_active = False  
    rospy.loginfo(f"P-control complete. Final width: {width:.4f}m after {iteration} iterations")  
    return width  

def top_position_slip_control(gripper_open_client, initial_width):
    """Top-position slip detection and control"""
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
    rospy.loginfo("🔍 TOP POSITION SLIP DETECTION AND CONTROL 🔍")  
    rospy.loginfo("=" * 70)  
    rospy.loginfo(f"Duration: {TOP_HOLD_TIME} seconds")  
    rospy.loginfo("=== WATCH THE GUI FOR REAL-TIME VISUALIZATION! ===")  
    rospy.loginfo("- Red curve: Fx (horizontal forces)")  
    rospy.loginfo("- Green curve: Fy (lateral forces)")  
    rospy.loginfo("- Purple curve: Gripper width changes")  
    rospy.loginfo("- Slip status: Red fill when slip detected")  
    rospy.loginfo("=" * 70)  
    
    start_time = time.time()  
    
    while not rospy.is_shutdown():  
        current_time = time.time()  
        elapsed_time = current_time - start_time  
        
        # Stop after specified hold time  
        if elapsed_time >= TOP_HOLD_TIME:  
            rospy.loginfo("Top position hold time completed")  
            break  
        
        # Update GUI data
        update_gui_force_data()
        
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
        
        # Display current status periodically  
        if int(elapsed_time) % 2 == 0 and elapsed_time != int(elapsed_time):  # Every 2 seconds  
            remaining_time = TOP_HOLD_TIME - elapsed_time  
            rospy.loginfo(f"Slip Control - Fz: {last_force_z:.3f}N | Fx: {last_force_x:.2f}N | "  
                         f"Fy: {last_force_y:.2f}N | Width: {current_width:.4f}m | "  
                         f"Corrections: {top_position_slip_corrections} | Time: {remaining_time:.1f}s")  
        
        rate.sleep()  
    
    # Disable slip detection  
    slip_detection_active = False  
    control_active = False  
    rospy.loginfo("=" * 50)  
    rospy.loginfo(f"🏁 Slip control COMPLETE! Total corrections: {top_position_slip_corrections}")  
    rospy.loginfo("=" * 50)  

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
            break  

def main():  
    global gui, current_phase  
    
    roscpp_initialize([])  
    rospy.init_node('fr3_slip_detection_gui')  
    
    # Create GUI
    gui = SlipDetectionGUI(
        window_length=WINDOW_LENGTH,  
        video_fps=VIDEO_FPS,  
        video_width=1500,  
        video_height=1000  
    )  
    
    # Set up the GUI window
    gui.setup_window()  
    
    # Start video recording
    video_filename = f"video/slip_detection_experiment.mp4"  
    rospy.loginfo("Starting video recording...")  
    recording_started = gui.start_recording(video_filename)  
    if recording_started:  
        rospy.loginfo("✅ Video recording started successfully!")  
    else:  
        rospy.logwarn("❌ Failed to start video recording")  
    gui.start_data_saving("data/experiment_data")  # 新增！

    # ROS subscribers
    gripper_state_sub = rospy.Subscriber('/franka_gripper/joint_states', JointState, gripper_state_callback)  
    tactile_z_sub = rospy.Subscriber(f'/force/{sensor}/z', Float32, tactile_z_callback)  
    tactile_x_sub = rospy.Subscriber(f'/force/{sensor}/x', Float32, tactile_x_callback)  
    tactile_y_sub = rospy.Subscriber(f'/force/{sensor}/y', Float32, tactile_y_callback)  
    
    # Initialize MoveIt and gripper clients
    try:  
        arm = MoveGroupCommander("fr3_arm")  
        gripper_client = actionlib.SimpleActionClient('/franka_gripper/grasp', GraspAction)  
        gripper_client.wait_for_server(rospy.Duration(5.0))  
        gripper_open_client = actionlib.SimpleActionClient('/franka_gripper/move', MoveAction)  
        gripper_open_client.wait_for_server(rospy.Duration(5.0))  
        
        rospy.loginfo("All action servers connected!")  
    except Exception as e:  
        rospy.logerr(f"Failed to initialize MoveIt/Gripper: {e}")  
        if gui:  
            gui.stop_recording()  
        return  

    rospy.loginfo("=" * 80)  
    rospy.loginfo("🚀 SLIP DETECTION GUI WITH VIDEO RECORDING LAUNCHED! 🚀")  
    rospy.loginfo("=" * 80)  
    rospy.loginfo("📊 GUI Features:")  
    rospy.loginfo("  - Real-time force curves (Fx, Fy, Fz)")  
    rospy.loginfo("  - Gripper width monitoring")  
    rospy.loginfo("  - Slip detection status visualization")  
    rospy.loginfo("  - Control activity monitoring")  
    rospy.loginfo("📹 Video recording with real-world timestamps")  
    rospy.loginfo("=" * 80)  

    # Start GUI update thread
    gui_thread = threading.Thread(target=gui_update_thread, daemon=True)  
    gui_thread.start()  

    # Robot control logic
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
            rospy.loginfo(f"Initial readings - Fx: {last_force_x:.3f}N, Fy: {last_force_y:.3f}N, Fz: {last_force_z:.3f}N")

            # ========== Stage 2: Move down to object ==========  
            current_phase = "Moving Down"  
            rospy.loginfo("=" * 50)  
            rospy.loginfo("STAGE 2: Moving down to object")  
            rospy.loginfo("=" * 50)  
            
            success = safe_cartesian_move(arm, dx=0, dy=0, dz=-MOVE_DOWN,   
                                        description="Moving down to object")  
            
            if not success:  
                rospy.logerr("Failed to move down to object")  
                return  
                
            print_current_pose(arm, "Down Position - Object Contact")  
            rospy.sleep(1.0)  

            # ========== Stage 3: P-Control Grasp ==========  
            rospy.loginfo("=" * 50)  
            rospy.loginfo("STAGE 3: P-Control Force Grasp")  
            rospy.loginfo("🎯 Watch GUI: Blue curve (Fz) approaching target!")  
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
            rospy.sleep(1.0)

            # ========== Stage 5: SLIP DETECTION AND CONTROL ==========
            rospy.loginfo("=" * 70)
            rospy.loginfo("🔍 STAGE 5: TOP POSITION SLIP DETECTION 🔍")
            rospy.loginfo("=" * 70)
            rospy.loginfo("NOW WATCH THE GUI CAREFULLY!")
            rospy.loginfo("- Red/Green curves: Slip forces (Fx/Fy)")
            rospy.loginfo("- Purple curve: Gripper width adjustments")
            rospy.loginfo("- Red fill area: Slip detection events")
            rospy.loginfo("- Blue fill area: Control active periods")
            rospy.loginfo("=" * 70)
            
            top_position_slip_control(gripper_open_client, final_width)

            # ========== Stage 6: Move back down ==========
            current_phase = "Moving Down"
            rospy.loginfo("STAGE 6: Moving back down")
            safe_cartesian_move(arm, dx=0, dy=0, dz=-MOVE_DOWN, 
                              description="Moving back down")

            # ========== Stage 7: Release object ==========
            current_phase = "Releasing"
            rospy.loginfo("STAGE 7: Releasing object")
            control_gripper(gripper_open_client, width=0.08, speed=0.1, grasp=False)
            rospy.sleep(2.0)

            # ========== Stage 8: Return home ==========
            current_phase = "Returning Home"
            rospy.loginfo("STAGE 8: Returning to home position")
            arm.go(home, wait=True)

            # ========== Completion ==========
            current_phase = "Completed"
            rospy.loginfo("=" * 80)
            rospy.loginfo("🎉 ALL STAGES COMPLETED SUCCESSFULLY! 🎉")
            rospy.loginfo("=" * 80)
            rospy.loginfo("📊 FINAL RESULTS:")
            rospy.loginfo(f"   - Total slip corrections: {top_position_slip_corrections}")
            rospy.loginfo(f"   - Final gripper width: {last_gripper_width:.4f}m")
            rospy.loginfo(f"   - Final forces: Fx={last_force_x:.3f}N, Fy={last_force_y:.3f}N, Fz={last_force_z:.3f}N")
            rospy.loginfo("📹 Video saved with detailed frame timestamps")
            rospy.loginfo("📈 GUI shows complete experimental data!")
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

    # Start robot control thread
    robot_thread = threading.Thread(target=robot_execution, daemon=True)
    robot_thread.start()

    # 运行GUI主循环
    try:
        rospy.loginfo("Starting GUI main loop...")
        gui.run()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down due to KeyboardInterrupt...")
    except Exception as e:
        rospy.logerr(f"GUI error: {e}")
    finally:
        rospy.loginfo("Stopping video recording...")
        if gui:
            gui.stop_recording()
            gui.stop_data_saving()
        rospy.loginfo("Cleaning up...")
        roscpp_shutdown()
        rospy.loginfo("Shutdown complete.")
        

if __name__ == '__main__':
    main()