#!/bin/bash
sensor1="timi_ink"
cam_id="4"
checkpoint1="timi_ink"
# realsense_id=9
episode_name="timi_ink"

roslaunch genforce force_compare_timi.launch sensor1:=$sensor1 cam_id:=$cam_id checkpoint1:=$checkpoint1&
sleep 2    # let it initialize, adjust as needed

roslaunch rosbridge_server rosbridge_websocket.launch &
sleep 2    # give it a moment too


# video
python scripts/img_show_force_compare.py --sensor "${sensor1}" --fps 60 --video_name ${episode_name} --save-video &
# python scripts/img_show_force_compare.py --sensor "${sensor1}" --fps 60 --video_name ${episode_name}  &
python scripts/plot_force_compare_video_timi.py --sensor "${sensor1}" --video_name ${episode_name} --width 3200 --height 1000 &

# #no video
# python scripts/img_show_force_compare.py --sensor "${sensor1}" &
# python scripts/plot_force_compare.py --sensor "${sensor1}" &


wait       # Wait for all jobs to finish (your shell stays open)