cmake_minimum_required(VERSION 3.0.2)
project(genforce)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  message_generation
)
find_package(Eigen3 REQUIRED)
find_package(Franka REQUIRED)

add_message_files(
  FILES
  StampedFloat64MultiArray.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS roscpp std_msgs rospy sensor_msgs message_runtime
  INCLUDE_DIRS include
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${Franka_INCLUDE_DIRS} 
)




catkin_install_python(PROGRAMS
  scripts/force_slip.py
  scripts/force.py
  scripts/gelsight_publisher.py
  scripts/uskin_publisher.py
  scripts/tactip_publisher.py
  scripts/cam_port.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)



install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  FILES_MATCHING PATTERN "*.launch"
)
