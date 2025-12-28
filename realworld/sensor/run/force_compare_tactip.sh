#!/bin/bash
sensor2="tactip"
episode_name="tactip"

roslaunch genforce force_compare_tactip.launch sensor2:=$sensor2 &
sleep 2    # let it initialize, adjust as needed

roslaunch rosbridge_server rosbridge_websocket.launch &
sleep 2    # give it a moment too


# video
# python scripts/img_show_force_compare_tactip.py --sensor "${sensor2}" --fps 60 --video_name ${episode_name} &
python scripts/img_show_force_compare_tactip.py --sensor "${sensor2}" --fps 60 --video_name ${episode_name} --save-video &
python scripts/plot_force_compare_video_tactip.py --sensor "${sensor2}" --width 3200 --height 1000 &


wait       # Wait for all jobs to finish (your shell stays open)