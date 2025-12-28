#!/usr/bin/env python3
import rospy
import cv2
import os
import time

def list_all_video_devices():
    devices = []
    for name in os.listdir('/dev'):
        if name.startswith('video') and name[5:].isdigit():
            devices.append(f'/dev/{name}')
    return sorted(devices, key=lambda x:int(x.split('video')[1]))

def test_device(path):
    cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
    time.sleep(0.2)
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        return ret
    cap.release()
    return False

if __name__ == "__main__":
    rospy.init_node("cam_port")
    all_devices = list_all_video_devices()
    rospy.loginfo(f"Found video devices: {all_devices}")
    cam_port = []
    for dev in all_devices:
        if test_device(dev):
            rospy.loginfo(f"{dev} is available and working.")
            cam_port.append(dev)
        else:
            rospy.logwarn(f"{dev} cannot be opened or no frame.")
    rospy.loginfo(f"Usable camera devices: {cam_port}")

    # If you want to publish the result as a ROS param (optional):
    rospy.set_param('~cam_port', cam_port)