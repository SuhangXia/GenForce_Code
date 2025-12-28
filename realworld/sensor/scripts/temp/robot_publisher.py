#!/usr/bin/env python3
import zmq
import rospy
import json
from genforce.msg import StampedFloat64MultiArray

def flatten(l):
    """
    Flattens nested lists into a one-dimensional list.
    """
    if isinstance(l, list):
        return [item for sublist in l for item in flatten(sublist)] if any(isinstance(i, list) for i in l) else l
    else:
        return [l]

def get_ros_time_from_observation(observation):
    """
    Extracts the timestamp (in seconds) from the observation and converts it to ROS Time.
    Supports int, float, or two-element list/tuple [secs, nsecs].
    """
    try:
        ts = observation.get('timestamp', {}).get('robot_state', None)
        # Handle most common ROS Time tuple or float seconds
        if isinstance(ts, (list, tuple)) and len(ts) == 2:  # [secs, nsecs]
            return rospy.Time(secs=int(ts[0]), nsecs=int(ts[1]))
        elif isinstance(ts, (int, float)):
            secs = int(ts)
            nsecs = int((ts - secs) * 1e9)
            return rospy.Time(secs, nsecs)
        else:
            return rospy.Time.now()
    except Exception:
        return rospy.Time.now()

def main():
    # Initialize the ROS node
    rospy.init_node("robot_state")
    pub = rospy.Publisher("/robot_state", StampedFloat64MultiArray, queue_size=10)

    # Setup ZeroMQ subscriber
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.connect("tcp://localhost:5560")
    sock.setsockopt_string(zmq.SUBSCRIBE, "")
    print("Connected to robot")

    while not rospy.is_shutdown():
        try:
            # Receive and parse the message from ZeroMQ
            msg = sock.recv_string(flags=zmq.NOBLOCK)
            data = json.loads(msg)

            # Extract observation and robot_state fields
            observation = data.get("observation", {})
            robot_state = observation.get("robot_state", {})

            # Extract positions and velocities from robot_state
            cartesian_position = robot_state.get('cartesian_position', [])
            joint_positions = robot_state.get('joint_positions', [])
            gripper_position = robot_state.get('gripper_position', [])
            joint_velocities = robot_state.get('joint_velocities', [])
            cartesian_velocity = robot_state.get('cartesian_velocity', [])

            # Create the custom message and set the timestamp from observation
            msg_out = StampedFloat64MultiArray()
            msg_out.header.stamp = get_ros_time_from_observation(observation)

            # Flatten and compile all relevant data fields into the msg data list
            float_data = []
            float_data += flatten(cartesian_position)
            float_data += flatten(joint_positions)
            float_data += flatten(gripper_position)
            float_data += flatten(joint_velocities)
            float_data += flatten(cartesian_velocity)
            msg_out.data = [float(x) for x in float_data if x is not None]

            # Publish the message
            pub.publish(msg_out)

        except zmq.Again:
            rospy.sleep(0.001)  # No new message this cycle
        except Exception as e:
            rospy.logwarn("Parse or publish error: {}".format(e))

if __name__ == '__main__':
    main()