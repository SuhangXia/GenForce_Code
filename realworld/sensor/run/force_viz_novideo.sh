#!/bin/bash
sensor1="AII"
sensor2="uskin"
cam_id="11"
realsense_id=9
episode_name="tea_box"

roslaunch genforce gu_force.launch sensor1:=$sensor1 sensor2:=$sensor2 cam_id:=$cam_id&
sleep 2    # let it initialize, adjust as needed

roslaunch rosbridge_server rosbridge_websocket.launch &
sleep 2    # give it a moment too

python scripts/img_show.py --sensor1 "${sensor1}" --sensor2 "${sensor2}" --cam-port ${realsense_id} --fps 30 --video_name ${episode_name}&
python scripts/plot_show.py --sensor1 "${sensor1}" --sensor2 "${sensor2}" --video_name ${episode_name}&

wait       # Wait for all jobs to finish (your shell stays open)