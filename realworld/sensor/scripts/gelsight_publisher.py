#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import sys

from functools import partial

# ===== Import utils from user code =====
script_dir = os.path.dirname(os.path.realpath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from utils import find_marker, find_marker_centers, plot_marker_delta, refresh_dir

# def extract_roi_mask(frame, center, roi_size=25):
#     """Extract the marker mask in a small ROI centered at center."""
#     h, w = frame.shape[:2]
#     x0, y0 = int(center[0]), int(center[1])
#     half = roi_size // 2
#     x1, y1 = max(x0 - half, 0), max(y0 - half, 0)
#     x2, y2 = min(x0 + half, w), min(y0 + half, h)
#     roi = frame[y1:y2, x1:x2]
#     mask_roi = np.zeros((y2-y1, x2-x1), np.uint8)
#     if roi.size == 0 or roi.shape[0] < 5 or roi.shape[1] < 5:
#         return mask_roi, (x1, y1) # fallback -- empty mask

#     roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#     mask_roi = cv2.adaptiveThreshold(
#         roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY_INV, 11, 2
#     )
#     return mask_roi, (x1, y1)


# def extract_roi_mask(frame, center, roi_size=25):
#     """Extract the marker mask in a small ROI centered at center, dilate first then open to reduce noise and fill small holes."""
#     h, w = frame.shape[:2]
#     x0, y0 = int(center[0]), int(center[1])
#     half = roi_size // 2
#     x1, y1 = max(x0 - half, 0), max(y0 - half, 0)
#     x2, y2 = min(x0 + half, w), min(y0 + half, h)
#     roi = frame[y1:y2, x1:x2]
#     mask_roi = np.zeros((y2 - y1, x2 - x1), np.uint8)
#     if roi.size == 0 or roi.shape[0] < 5 or roi.shape[1] < 5:
#         return mask_roi, (x1, y1)  # fallback -- empty mask

#     roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#     mask_roi = cv2.adaptiveThreshold(
#         roi_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY_INV, 11, 2
#     )
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#     mask_roi = cv2.dilate(mask_roi, kernel, iterations=1)
#     mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_OPEN, kernel, iterations=1)
#     return mask_roi, (x1, y1)


def extract_roi_mask(frame, center, roi_size=25):  # original roi_size=25

    """Extract marker mask in ROI,connectedComponentsWithStats"""
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

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_roi, connectivity=8)
    if num_labels > 1:
        max_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  
        mask_roi = np.where(labels == max_idx, 255, 0).astype(np.uint8)

    return mask_roi, (x1, y1)

def plot_marker_delta(image, pts, shifts, scale=6, arrow_color=(0,0,255)):
    img_vis = image.copy()
    for (pt, delta) in zip(pts.astype(int), shifts):
        end_pt = (int(pt[0] + delta[0]*scale), int(pt[1] + delta[1]*scale))
        cv2.arrowedLine(img_vis, tuple(pt), end_pt, arrow_color, 2, tipLength=0.25)
        cv2.circle(img_vis, tuple(pt), 3, (0,255,0), -1)
    return img_vis

def main():
    rospy.init_node('gelsight_marker_union_true_area')
    cam_id = rospy.get_param('~cam_id', 4)
    roi_size = rospy.get_param('~roi_size', 20) #30
    outdir = rospy.get_param('~output_dir', 'marker_shift')
    os.makedirs(outdir, exist_ok=True)

    # Marker extraction params
    morphop_size = rospy.get_param('~morphop_size', 5)
    morphop_iter = rospy.get_param('~morphop_iter', 1)
    morphclose_size = rospy.get_param('~morphclose_size', 5)
    morphclose_iter = rospy.get_param('~morphclose_iter', 1)
    dilate_size = rospy.get_param('~dilate_size', 3)
    dilate_iter = rospy.get_param('~dilate_iter', 0)
    marker_range = tuple(rospy.get_param('~marker_range', [145, 255]))
    value_threshold = rospy.get_param('~value_threshold', 90)

    bridge = CvBridge()
    raw_pub = rospy.Publisher('/gelsight/raw', Image, queue_size=1)
    marker_pub = rospy.Publisher('/gelsight/marker', Image, queue_size=1)
    plot_pub = rospy.Publisher('/gelsight/arrow', Image, queue_size=1)
    global_mask_pub = rospy.Publisher('/gelsight/marker_global', Image, queue_size=1)

    stream = cv2.VideoCapture(cam_id)
    if not stream.isOpened():
        rospy.logerr(f"Camera {cam_id} could not be opened.")
        sys.exit(1)

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
        rospy.logerr('Initial camera image failed to grab.')
        sys.exit(1)
    # Use BGR2GRAY just like in the script
    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    ref_marker = calib_find_marker(ref_img)
    p0 = np.array(find_marker_centers(ref_marker), dtype=np.float32)

    rate = rospy.Rate(30)
    rospy.loginfo("Gelsight marker union node running with true mask extraction.")

        # Tracking parameters as before
    lk_params = dict(
        winSize=(15,15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    while not rospy.is_shutdown():
        ret, track_img = stream.read()
        if not ret:
            rospy.logwarn('Failed to get frame from camera.')
            rate.sleep()
            continue

        # --- 1. Publish raw frame ---
        try:
            raw_pub.publish(bridge.cv2_to_imgmsg(track_img, encoding='bgr8'))
        except Exception as e:
            rospy.logerr(f"Failed to publish raw image: {e}")


        track_gray = cv2.cvtColor(track_img, cv2.COLOR_BGR2GRAY)
        p1, st, err = cv2.calcOpticalFlowPyrLK(ref_gray, track_gray, p0, None)
        st = st.reshape(-1)
        kpt0 = p0[st == 1]
        kpt1 = p1[st == 1]
        res = np.hstack((kpt0, kpt1 - kpt0))
        # np.savetxt(os.path.join(output_dir, "%04d.txt" % img_cnt), res, fmt="%.4f")
        img_cnt += 1

        plot_img = plot_marker_delta(track_img, kpt1, kpt1 - kpt0, scale=6, arrow_color=(0, 0, 255))

        try:
            plot_pub.publish(bridge.cv2_to_imgmsg(plot_img, encoding='bgr8'))
        except Exception as e:
            rospy.logerr(f"Failed to publish plot image: {e}")

        frame_rgb = cv2.cvtColor(track_img, cv2.COLOR_BGR2RGB)
        # --- 3. Mask union: global marker + all ROI marker patches ---
        # (a) Global mask (whole image)
        global_mask = find_marker(frame_rgb)

        # (b) Local ROI segmentation at each tracked marker (guided by optical flow)
        h, w = track_img.shape[:2]
        union_mask = global_mask.copy()
        for pt in kpt1:
            roi_mask, (x1, y1) = extract_roi_mask(track_img, pt, roi_size=roi_size)
            y2, x2 = y1+roi_mask.shape[0], x1+roi_mask.shape[1]
            if roi_mask.shape[0]>0 and roi_mask.shape[1]>0:
                union_mask[y1:y2, x1:x2] = cv2.bitwise_or(union_mask[y1:y2, x1:x2], roi_mask)

        # --- 4. Publish the true area binary marker mask ---
        try:
            marker_pub.publish(bridge.cv2_to_imgmsg(union_mask, encoding='mono8'))
            global_mask_pub.publish(bridge.cv2_to_imgmsg(global_mask, encoding='mono8'))
        except Exception as e:
            rospy.logerr(f"Failed to publish marker image: {e}")

        img_cnt += 1

        rate.sleep()

    stream.release()
    rospy.loginfo("Shutting down gelsight marker union (true area) node.")

if __name__ == '__main__':
    main()