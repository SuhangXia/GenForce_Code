#! /usr/bin/env python3
import rospy
import json
import websocket
import time
from genforce.msg import StampedFloat64MultiArray  # Custom message: Header header; float64[] data

ip = "10.70.110.32"
port = 5000

rospy.init_node('uskin')
uskin_pub = rospy.Publisher('/uskin_data', StampedFloat64MultiArray, queue_size=1)
rate = rospy.Rate(100)  # 100Hz
time.sleep(5)

def publisher(uskin_pub, data):
    """
    Publishes the force/pressure array with a precise ROS timestamp in the header.
    """
    xela_msg = StampedFloat64MultiArray()
    xela_msg.header.stamp = rospy.Time.now()  # Set current ROS time as the message timestamp
    xela_msg.data = data
    uskin_pub.publish(xela_msg)

def on_message(wsapp, message):
    """
    Callback for each websocket message, parses hex strings, converts to float,
    and publishes the result with timestamp.
    """
    data = json.loads(message)
    sensor = data['1']['data'].split(",")
    txls = int(len(sensor) / 3)  # Each taxel has 3 values (x, y, z)
    data_row = []
    for i in range(txls):
        x = int(sensor[i * 3], 16)
        y = int(sensor[i * 3 + 1], 16)
        z = int(sensor[i * 3 + 2], 16)
        data_row.append(float(x))
        data_row.append(float(y))
        data_row.append(float(z))

    publisher(uskin_pub, data_row)
    rate.sleep()

wsapp = websocket.WebSocketApp("ws://{}:{}".format(ip, port), on_message=on_message)
print("uskin running")
wsapp.run_forever()