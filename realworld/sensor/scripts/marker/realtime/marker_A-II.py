#!/usr/bin/env python3

import cv2
import numpy as np
import os
import sys
from functools import partial

# ===== Import local utils =====
script_dir = os.path.dirname(os.path.realpath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from utils import find_marker, find_marker_centers, plot_marker_delta, refresh_dir

def clean_union_mask(union_mask, min_area=25):
    contours, _ = cv2.findContours(union_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_clean = np.zeros_like(union_mask)
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            cv2.drawContours(mask_clean, [cnt], -1, 255, cv2.FILLED)
    return mask_clean

def plot_marker_delta(image, pts, shifts, scale=6, arrow_color=(0,0,255)):
    img_vis = image.copy()
    for (pt, delta) in zip(pts.astype(int), shifts):
        end_pt = (int(pt[0] + delta[0]*scale), int(pt[1] + delta[1]*scale))
        cv2.arrowedLine(img_vis, tuple(pt), end_pt, arrow_color, 2, tipLength=0.25)
        cv2.circle(img_vis, tuple(pt), 3, (0,255,0), -1)
    return img_vis

def main():
    cam_id = 4
    roi_size = 40
    outdir = "marker_shift"
    os.makedirs(outdir, exist_ok=True)

    # Marker extraction params
    morphop_size = 3
    morphop_iter = 1
    morphclose_size = 3
    morphclose_iter = 1
    dilate_size = 3
    dilate_iter = 1
    marker_range = (148, 255)
    value_threshold = 90
    min_area = 30

    stream = cv2.VideoCapture(cam_id)
    if not stream.isOpened():
        print(f"[ERROR] Camera {cam_id} could not be opened.")
        sys.exit(1)

    # Prepare marker extraction function
    calib_find_marker = partial(
        find_marker,
        morphop_kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morphop_size, morphop_size)),
        morphclose_kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morphclose_size, morphclose_size)),
        dilate_kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size)),
        mask_range=marker_range,
        min_value=value_threshold,
        morphop_iter=morphop_iter,
        morphclose_iter=morphclose_iter,
        dilate_iter=dilate_iter
    )

    img_cnt = 0
    ret, ref_img = stream.read()
    if not ret:
        print("[ERROR] Initial camera image failed to grab.")
        sys.exit(1)
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    ref_marker = calib_find_marker(ref_img)
    p0 = np.array(find_marker_centers(ref_marker), dtype=np.float32)

    # For display and step debug
    print("[INFO] Marker tracker running. Press ESC to exit.")

    while True:
        ret, track_img = stream.read()
        if not ret:
            print('[WARNING] Failed to get frame from camera.')
            break

        # Optical flow tracking
        track_gray = cv2.cvtColor(track_img, cv2.COLOR_BGR2GRAY)
        p1, st, err = cv2.calcOpticalFlowPyrLK(ref_gray, track_gray, p0, None)
        st = st.reshape(-1)
        kpt0 = p0[st == 1]
        kpt1 = p1[st == 1]
        shifts = kpt1 - kpt0

        # 1. Show arrow plot of displacement
        plot_img = plot_marker_delta(track_img, kpt1, shifts, scale=6, arrow_color=(0, 0, 255))

        # 2. Mask union: 
        frame_rgb = cv2.cvtColor(track_img, cv2.COLOR_BGR2RGB)
        global_mask = calib_find_marker(frame_rgb)
        global_mask = clean_union_mask(global_mask, min_area=int(1.5*min_area))
        global_mask = cv2.dilate(global_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)), iterations=2)  
        global_mask = cv2.erode(global_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)), iterations=2) 

        # 3. Visualize masks and trackers
        cv2.imshow("Raw", track_img)
        cv2.imshow("Arrow plot", plot_img)
        cv2.imshow("Global Mask", global_mask)


        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

        img_cnt += 1

    stream.release()
    cv2.destroyAllWindows()
    print("[INFO] Shutting down.")

if __name__ == '__main__':
    main()