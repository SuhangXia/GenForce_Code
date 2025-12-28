#!/usr/bin/env python

import rospy
from moveit_commander import MoveGroupCommander, roscpp_initialize, roscpp_shutdown
from franka_gripper.msg import GraspAction, GraspGoal, MoveAction, MoveGoal, GraspEpsilon
import actionlib
import numpy as np


def print_current_pose(arm, msg):
    joint_values = arm.get_current_joint_values()
    print(f"{msg} joint angle: {[round(val, 4) for val in joint_values]}")

def control_gripper(gripper_action_client, width, speed=0.05, force=10.0, grasp=False):
    """Closes or opens the gripper."""
    if grasp:
        epsilon = GraspEpsilon(inner=0.04, outer=0.04)
        goal = GraspGoal(width=width, epsilon=epsilon, speed=speed, force=force)
        gripper_action_client.send_goal(goal)
        gripper_action_client.wait_for_result()
        result = gripper_action_client.get_result()
        print(result.success)
    else:
        goal = MoveGoal(width=width, speed=speed)
        gripper_action_client.send_goal(goal)
        gripper_action_client.wait_for_result()
        result = gripper_action_client.get_result()
        print(result.success)

if __name__ == '__main__':
    roscpp_initialize([])
    rospy.init_node('fr3_pick_fruit')

    arm = MoveGroupCommander("fr3_arm")
    gripper_client = actionlib.SimpleActionClient('/franka_gripper/grasp', GraspAction)
    gripper_client.wait_for_server()
    gripper_open_client = actionlib.SimpleActionClient('/franka_gripper/move', MoveAction)
    gripper_open_client.wait_for_server()

    print_current_pose(arm,  "Home")
    
    home = [0.5135, 1.0461, -0.7594, -1.598, 2.0161, 1.1814, -1.596]
    arm.go(home, wait=True)

    target1 = arm.get_current_pose().pose
    # # target1.position.z -= 0.065
    # target1.position.z -= 0.045 #meat_box
    target1.position.z -= 0.1
    arm.set_pose_target(target1)
    arm.go(wait=True)
    
    # Use this approach to separate position and orientation:
    # current_pose = arm.get_current_pose().pose
    # target1 = current_pose
    # target1.position.z -= 0.045  #meat_box tea_box
    # target1.position.z -= 0.1  # plum orange
    # target1.position.z -= 0.01  # pen
    # target1.position.z -= 0.1  # wood
    # target1.position.z -= 0.068  # straw
    # target1.position.z -= 0.067  # grape
    # Explicitly set the target with the same orientation
    # arm.set_pose_target([target1.position.x, target1.position.y, target1.position.z,
    #                     current_pose.orientation.x, current_pose.orientation.y,
    #                     current_pose.orientation.z, current_pose.orientation.w])
    # arm.go(wait=True)

    # rospy.sleep(3)

    # control_gripper(gripper_open_client, width=0.05, speed=0.1, grasp=False) #meat_box
    # control_gripper(gripper_open_client, width=0.075, speed=0.1, grasp=False) #tea_box
    # control_gripper(gripper_open_client, width=0.055, speed=0.1, grasp=False) #plum
    # control_gripper(gripper_open_client, width=0.075, speed=0.1, grasp=False) #orange
    # control_gripper(gripper_open_client, width=0.036, speed=0.1, grasp=False) #pen
    # control_gripper(gripper_open_client, width=0.035, speed=0.1, grasp=False) #straw
    # control_gripper(gripper_open_client, width=0.025, speed=0.1, grasp=False) #grape
    # control_gripper(gripper_open_client, width=0.05, speed=0.1, grasp=False) #chips
    # control_gripper(gripper_client, width=0.05, speed=0.01, force=1, grasp=True) #0.05 orange 0.06
    # target2 = arm.get_current_pose().pose
    # target2.position.z += 0.1
    # arm.set_pose_target(target2)
    # arm.go(wait=True)
    # print_current_pose(arm,  "UP")
    # rospy.sleep(10)
    
    # target3 = arm.get_current_pose().pose
    # # target3.position.z -= 0.068
    # target3.position.z -= 0.04
    # arm.set_pose_target(target3)
    # arm.go(wait=True)
    # print_current_pose(arm,  "Down")
    
    # target4 = arm.get_current_pose().pose
    # target4.position.z -= 0.12
    # arm.set_pose_target(target4)
    # arm.go(wait=True)
    
    # control_gripper(gripper_open_client, width=0.04, speed=0.1, grasp=False)
    # control_gripper(gripper_open_client, width=0.08, speed=0.1, grasp=False)
    # control_gripper(gripper_open_client, width=0.017, speed=0.1, grasp=False)
    # control_gripper(gripper_open_client, width=0.04, speed=0.1, grasp=False)
    control_gripper(gripper_open_client, width=0.08, speed=0.1, grasp=False)
    

    # target5 = arm.get_current_pose().pose
    # target5.position.z += 0.12
    # arm.set_pose_target(target5)
    # arm.go(wait=True)

    arm.go(home, wait=True)
    print_current_pose(arm,  "Home")
    

    arm.stop()
    arm.clear_pose_targets()
    roscpp_shutdown()