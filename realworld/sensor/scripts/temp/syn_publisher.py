#!/usr/bin/env python3

import rospy
import numpy as np
import message_filters
from sensor_msgs.msg import Image
from genforce.msg import StampedFloat64MultiArray  # Custom message: std_msgs/Header + float64[] data
from geometry_msgs.msg import PoseStamped

# Topic name for a single ArUco marker pose (published by aruco_ros single.launch)
POSE_TOPIC = '/aruco_single/pose'

def image_msg_to_float64(image_msg):
    """
    Converts a ROS Image message into a normalized 1D float64 vector (RGB, range [0,1]).
    """
    array = np.frombuffer(image_msg.data, dtype=np.uint8)
    img = array.reshape((image_msg.height, image_msg.width, 3))
    img = img.astype(np.float64) / 255.0
    return img.flatten()

def marker_to_list(pose_stamped):
    """
    Converts a geometry_msgs/Pose or PoseStamped into a 7-element float list:
    [x, y, z, ox, oy, oz, ow]
    """
    pose = pose_stamped.pose if hasattr(pose_stamped, 'pose') else pose_stamped
    return [
        pose.position.x, pose.position.y, pose.position.z,
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
    ]

def sync_publisher(uskin_data, robot_state_data, gelsight_data, pose_msg):
    """
    Time-synchronized callback: merges sensor, robot, image, and single marker pose into one vector.
    The vector layout is:
        [uskin_data..., robot_state_data..., marker_pose_7..., gelsight_pixels...]
    If the ArUco marker is missing, marker_pose is filled with zeros.
    """
    gelsight_flat = image_msg_to_float64(gelsight_data)
    n_uskin = len(uskin_data.data)
    n_robot = len(robot_state_data.data)
    n_marker = 7  # Single marker pose
    n_gelsight = len(gelsight_flat)
    total_len = n_uskin + n_robot + n_marker + n_gelsight

    sync_vec = np.zeros((total_len,), dtype=np.float64)
    sync_vec[:n_uskin] = uskin_data.data
    sync_vec[n_uskin:n_uskin + n_robot] = robot_state_data.data

    # Fill marker pose (use zeros if no marker detected)
    pose_vals = marker_to_list(pose_msg) if pose_msg is not None else [0.0] * 7
    sync_vec[n_uskin + n_robot : n_uskin + n_robot + 7] = pose_vals

    # Fill gelsight features
    sync_vec[n_uskin + n_robot + n_marker :] = gelsight_flat

    # Prepare and publish the output message
    sync_msg = StampedFloat64MultiArray()
    # Use the minimum timestamp for strict sync
    sync_msg.header.stamp = min([
        uskin_data.header.stamp,
        robot_state_data.header.stamp,
        gelsight_data.header.stamp,
        pose_msg.header.stamp if hasattr(pose_msg, "header") else uskin_data.header.stamp
    ])

    # print("uskin:", n_uskin, "robot:", n_robot, "marker:", n_marker, "gelsight:", n_gelsight, "total:", total_len)
    # print("uskin_data.data:", len(uskin_data.data), "robot_state_data.data:", len(robot_state_data.data), "gelsight_flat:", len(gelsight_flat))

    sync_msg.data = sync_vec.tolist()
    sync_data_pub.publish(sync_msg)

if __name__ == '__main__':
    rospy.init_node('sync_publisher_single_marker', anonymous=True, disable_signals=True)

    # Subscribers for time-synchronized messages
    uskin_sub       = message_filters.Subscriber('/uskin_data', StampedFloat64MultiArray)
    robot_state_sub = message_filters.Subscriber('/robot_state', StampedFloat64MultiArray)
    gelsight_sub    = message_filters.Subscriber('/gelsight_data', Image)
    pose_sub        = message_filters.Subscriber(POSE_TOPIC, PoseStamped)  # Single marker pose

    # Publisher for the merged and synchronized data
    sync_data_pub = rospy.Publisher('/syn_data', StampedFloat64MultiArray, queue_size=1)

    # Strict time synchronization (all messages must be present)
    ts = message_filters.ApproximateTimeSynchronizer(
        [uskin_sub, robot_state_sub, gelsight_sub, pose_sub],
        queue_size=6, slop=0.1, allow_headerless=False
    )
    ts.registerCallback(sync_publisher)

    rospy.loginfo('Strict synchronized publisher started (single marker pose from /aruco_single/pose, zeros if missing).')
    rospy.spin()