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

# def extract_roi_mask(frame, center, roi_size=30):
#     """Extract marker mask in ROI using thresholding and morphology."""
#     h, w = frame.shape[:2]
#     x0, y0 = int(center[0]), int(center[1])
#     half = roi_size // 2
#     x1, y1 = max(x0 - half, 0), max(y0 - half, 0)
#     x2, y2 = min(x0 + half, w), min(y0 + half, h)
#     roi = frame[y1:y2, x1:x2]
#     mask_roi = np.zeros((y2 - y1, x2 - x1), np.uint8)
#     if roi.size == 0 or roi.shape[0] < 5 or roi.shape[1] < 5:
#         return mask_roi, (x1, y1)
#     # roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#     # mask_roi = cv2.adaptiveThreshold(
#     #     roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#     #     cv2.THRESH_BINARY_INV, 11, 2
#     # )
#     #usr HSV
#     value = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)[...,-1]
#     _, mask_img = cv2.threshold(value, 70, 255, cv2.THRESH_BINARY_INV)  
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
#     mask_roi = cv2.dilate(mask_img, kernel, iterations=1)
#     mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_OPEN, kernel, iterations=1)
#     cv2.imshow("roi", value)
#     cv2.imshow("mask_roi", mask_roi)
#     num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_roi, connectivity=8)
#     if num_labels > 1:
#         max_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  
#         mask_roi = np.where(labels == max_idx, 255, 0).astype(np.uint8)

#     return mask_roi, (x1, y1)

# def extract_roi_mask(frame, center, roi_size=25):
#     """Extract marker mask in ROI,connectedComponentsWithStats"""
#     h, w = frame.shape[:2]
#     x0, y0 = int(center[0]), int(center[1])
#     half = roi_size // 2
#     x1, y1 = max(x0 - half, 0), max(y0 - half, 0)
#     x2, y2 = min(x0 + half, w), min(y0 + half, h)
#     roi = frame[y1:y2, x1:x2]
#     mask_roi = np.zeros((y2 - y1, x2 - x1), np.uint8)
#     if roi.size == 0 or roi.shape[0] < 5 or roi.shape[1] < 5:
#         return mask_roi, (x1, y1)

#     roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#     mask_roi = cv2.adaptiveThreshold(
#         roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY_INV, 11, 2
#     )
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
#     mask_roi = cv2.dilate(mask_roi, kernel, iterations=1)
#     mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_OPEN, kernel, iterations=1)

#     num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_roi, connectivity=8)
#     if num_labels > 1:
#         max_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  
#         mask_roi = np.where(labels == max_idx, 255, 0).astype(np.uint8)

#     return mask_roi, (x1, y1)

def extract_roi_mask(frame, center, roi_size=30):
    """Extract marker mask in ROI, only keep the largest closed region."""
    h, w = frame.shape[:2]
    x0, y0 = int(center[0]), int(center[1])
    half = roi_size // 2
    x1, y1 = max(x0 - half, 0), max(y0 - half, 0)
    x2, y2 = min(x0 + half, w), min(y0 + half, h)
    roi = frame[y1:y2, x1:x2]
    mask_roi = np.zeros((y2 - y1, x2 - x1), np.uint8)
    if roi.size == 0 or roi.shape[0] < 5 or roi.shape[1] < 5:
        return mask_roi, (x1, y1)

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    mask_roi = cv2.adaptiveThreshold(
        roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask_roi = cv2.dilate(mask_roi, kernel, iterations=1)
    mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        max_contour = max(contours, key=cv2.contourArea)
        mask_clean = np.zeros_like(mask_roi)
        cv2.drawContours(mask_clean, [max_contour], -1, 255, thickness=cv2.FILLED)
        mask_roi = mask_clean
    else:
        mask_roi[:] = 0

    return mask_roi, (x1, y1)


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
    morphop_size = 5
    morphop_iter = 1
    morphclose_size = 5
    morphclose_iter = 1
    dilate_size = 3
    dilate_iter = 0
    marker_range = (145, 255)
    value_threshold = 90

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

    lk_params = dict(
        winSize=(15,15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

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

        union_mask = global_mask.copy()
        for idx, pt in enumerate(kpt1):
            x0, y0 = int(pt[0]), int(pt[1])
            half = roi_size // 2

            roi_mask, (x1, y1) = extract_roi_mask(track_img, pt, roi_size=roi_size)
            cv2.rectangle(track_img, 
            (x0 - half, y0 - half), 
            (x0 + half, y0 + half), 
            (0,255,255), 2)
            if idx == 1:
                cv2.imshow("mask_roi2", roi_mask)
            y2, x2 = y1+roi_mask.shape[0], x1+roi_mask.shape[1]
            if roi_mask.shape[0]>0 and roi_mask.shape[1]>0:
                union_mask[y1:y2, x1:x2] = cv2.bitwise_or(union_mask[y1:y2, x1:x2], roi_mask)

        # 3. Visualize masks and trackers
        cv2.imshow("Raw", track_img)
        cv2.imshow("Arrow plot", plot_img)
        cv2.imshow("Global Mask", global_mask)
        cv2.imshow("Union Mask", union_mask)

        # 4. Optionally save results
        # cv2.imwrite(os.path.join(outdir, f"arrow_{img_cnt:04d}.png"), plot_img)
        # cv2.imwrite(os.path.join(outdir, f"mask_{img_cnt:04d}.png"), union_mask)

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

        img_cnt += 1

    stream.release()
    cv2.destroyAllWindows()
    print("[INFO] Shutting down.")

if __name__ == '__main__':
    main()