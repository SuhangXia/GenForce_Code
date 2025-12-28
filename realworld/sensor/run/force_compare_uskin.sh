#!/bin/bash
sensor2="uskin"
episode_name="tactip_uskin"

roslaunch genforce force_compare_uskin.launch sensor2:=$sensor2 &
sleep 2    # let it initialize, adjust as needed

roslaunch rosbridge_server rosbridge_websocket.launch &
sleep 2    # give it a moment too


# video
python scripts/img_show_force_compare_uskin.py --sensor "${sensor2}" --fps 60 --video_name ${episode_name} --save-video &
python scripts/plot_force_compare_video_uskin.py --sensor "${sensor2}" --width 3200 --height 1000 --video_name ${episode_name}&


wait       # Wait for all jobs to finish (your shell stays open) 