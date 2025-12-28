#!/usr/bin/env python

import rospy
from moveit_commander import MoveGroupCommander, roscpp_initialize, roscpp_shutdown
from franka_gripper.msg import GraspAction, GraspGoal, MoveAction, MoveGoal, GraspEpsilon, MoveActionFeedback
import actionlib
import numpy as np
from std_msgs.msg import Float32
from sensor_msgs.msg import JointState

# --- Proportional force control parameters ---
TARGET_FORCE =0.6  # meat_box 1.2; tea_box 1.2 ; plum 1.2 ; orange 1.2 ; pen 1.2 ; wood 1.2 ; staw 0.8 ; grape 0.6 ; chips 0.6
KP = 0.0004       # meat_box 0.0008 ; tea_box 0.0008; plum 0.0004 ; orange 0.0002 ; pen 0.0004 ; wood 0.0004 ; staw 0.0004 ; grape 0.0004  ; chips 0.0004
WIDTH_MIN = 0.0       
WIDTH_MAX = 0.08
START_WIDTH = 0.045 # meat_box 0.065 ; tea_box 0.075; plum 0.055 ; orange 0.075 ; pen 0.036  ; wood 0.045  ; staw 0.032  ; grape 0.025   ; chips 0.055
FORCE_TOL = 0.1      
CONTROL_HZ = 10 
SPEED = 0.005  # meat_box 0.005 ; tea_box 0.005; plum 0.005 ; orange 0.005 ; pen 0.005 ; wood 0.005  ; staw 0.005 ; grape 0.005 ; chips 0.005
MOVE_DOWN = 0.068  # meat_box tea_box 0.045;  plum 0.1 ;  orange 0.1 ;  pen 0.02 ;  wood 0.1  ;  staw 0.068 ;  grape 0.068 ;  chips 0.067

last_gripper_width = 0.0
last_force_z = 0.0  # Shared variable for tactile reading

def gripper_state_callback(msg):
    global last_gripper_width
    try:
        finger_indices = [i for i, n in enumerate(msg.name) if 'finger_joint1' in n]
        if finger_indices:
            width = 2 * msg.position[finger_indices[0]]
            last_gripper_width = width
    except Exception as e:
        print("Fail to get gripper state:", e)

def print_current_pose(arm, msg):
    joint_values = arm.get_current_joint_values()
    print(f"{msg} joint angle: {[round(val, 4) for val in joint_values]}")

def tactile_callback(msg):
    global last_force_z
    last_force_z = msg.data

def control_gripper(gripper_action_client, width, speed=0.05, force=10.0, grasp=False):
    """Closes or opens the gripper."""
    if grasp:
        epsilon = GraspEpsilon(inner=0.04, outer=0.04)
        goal = GraspGoal(width=width, epsilon=epsilon, speed=speed, force=force)
        gripper_action_client.send_goal(goal)
        gripper_action_client.wait_for_result()
        gripper_action_client.get_result()
    else:
        goal = MoveGoal(width=width, speed=speed)
        gripper_action_client.send_goal(goal)
        gripper_action_client.wait_for_result()
        gripper_action_client.get_result()

def p_control_gripper_to_force(gripper_open_client, target_force=TARGET_FORCE, kp=KP):
    """
    Close the gripper until tactile z-force reaches the setpoint, using P control loop.
    """
    global last_force_z, last_gripper_width
    width = START_WIDTH
    rate = rospy.Rate(CONTROL_HZ)
    rospy.loginfo(f"Starting P-gripper: target force={target_force}N, Kp={kp}")

    while not rospy.is_shutdown():
        force_error = target_force - abs(last_force_z)
        d_width = kp * force_error
        width -= d_width
        width = max(WIDTH_MIN, min(WIDTH_MAX, width))  # Clamp
        goal = MoveGoal(width=width, speed=SPEED) #0.05
        gripper_open_client.send_goal(goal)
        gripper_open_client.wait_for_result(rospy.Duration(0.7))  # Wait for completion or timeout

        rospy.loginfo(f"Curr force: {last_force_z:.3f}N | err={force_error:.3f} | width_cmd={width:.4f}m | actual_width={last_gripper_width:.4f}m")
        if abs(force_error) < FORCE_TOL:
            rospy.loginfo("Target force reached (within tolerance).")
            break
        rate.sleep()
    rospy.loginfo("P-control gripper complete. Final width: {:.4f}m".format(width))

if __name__ == '__main__':
    roscpp_initialize([])
    rospy.init_node('fr3_pick_fruit')

    gripper_state_sub = rospy.Subscriber('/franka_gripper/joint_states', JointState, gripper_state_callback)

    arm = MoveGroupCommander("fr3_arm")
    gripper_client = actionlib.SimpleActionClient('/franka_gripper/grasp', GraspAction)
    gripper_client.wait_for_server()
    gripper_open_client = actionlib.SimpleActionClient('/franka_gripper/move', MoveAction)
    gripper_open_client.wait_for_server()

    # --- Tactile force subscriber (persistent global) ---
    tactile_sub = rospy.Subscriber('/force/AII/z', Float32, tactile_callback)
    
    # tactile_sub = rospy.Subscriber('/force/uskin/z', Float32, tactile_callback)

    print_current_pose(arm,  "Home")
    home = [0.5135, 1.0461, -0.7594, -1.598, 2.0161, 1.1814, -1.596]
    arm.go(home, wait=True)
    control_gripper(gripper_open_client, width=START_WIDTH, speed=0.1, grasp=False)
    

    current_pose = arm.get_current_pose().pose
    target1 = current_pose
    target1.position.z -= MOVE_DOWN
    # Explicitly set the target with the same orientation
    arm.set_pose_target([target1.position.x, target1.position.y, target1.position.z,
                        current_pose.orientation.x, current_pose.orientation.y,
                        current_pose.orientation.z, current_pose.orientation.w])
    arm.go(wait=True)


    # ======= Replace direct gripper closing with force P-control: =======
    p_control_gripper_to_force(gripper_open_client, target_force=TARGET_FORCE, kp=KP)

    current_pose = arm.get_current_pose().pose
    target1 = current_pose
    target1.position.z += MOVE_DOWN 
    # Explicitly set the target with the same orientation
    arm.set_pose_target([target1.position.x, target1.position.y, target1.position.z,
                        current_pose.orientation.x, current_pose.orientation.y,
                        current_pose.orientation.z, current_pose.orientation.w])
    arm.go(wait=True)
    print_current_pose(arm,  "UP")
    # rospy.sleep(10)
    
    current_pose = arm.get_current_pose().pose
    target1 = current_pose
    target1.position.z -= MOVE_DOWN
    # Explicitly set the target with the same orientation
    arm.set_pose_target([target1.position.x, target1.position.y, target1.position.z,
                        current_pose.orientation.x, current_pose.orientation.y,
                        current_pose.orientation.z, current_pose.orientation.w])
    arm.go(wait=True)
    print_current_pose(arm,  "Down")

    control_gripper(gripper_open_client, width=0.08, speed=0.1, grasp=False)

    arm.go(home, wait=True)
    print_current_pose(arm,  "Home")

    arm.stop()
    arm.clear_pose_targets()
    roscpp_shutdown()