#!/bin/bash
nohup python3 -u extract_visual_features.py --source_video_list './source_video/train_5.txt' >./log/train5.log 2>&1 &
tail -f ./log/train5.log
