cmake_minimum_required(VERSION 3.0.2)
project(genforce)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  actionlib
  moveit_commander
  franka_gripper
  geometry_msgs
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs actionlib moveit_commander franka_gripper sensor_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

# Install the Python pick_fruit node
catkin_install_python(PROGRAMS
  src/pick_fruit_move.py src/pick_fruit_control.py src/slip_control.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)