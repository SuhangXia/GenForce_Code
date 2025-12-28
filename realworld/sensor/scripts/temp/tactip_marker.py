
#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import sys

def normalize_brightness(image):
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    yuv[..., 0] = cv2.equalizeHist(yuv[..., 0])
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

def poly_mask(img):
    # Adjust these points to your setup (image coordinates)
    mask = np.zeros_like(img)
    points = np.array(
        [[148, 36], [470, 36], [620, 236], [505, 430], [165, 430], [13, 244]],
        np.int32
    ).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [points], color=(255, 255, 255))
    return cv2.bitwise_and(mask, img)

def find_marker(cur,
                morphop_kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)),
                morphclose_kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)),
                dilate_kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)),
                mask_range=(100, 255),  # kept for signature compatibility
                min_value: int = 70,
                morphop_iter=2, morphclose_iter=1, dilate_iter=2):
    """
    Your original threshold + morphology pipeline
    Input: cur is expected BGR after normalization/inversion/polymask
    Output: 3-channel mask (uint8) matching original code
    """
    value = ((cur[..., 2] < 100) & (cur[..., 1] < 100) & (cur[..., 0] < 30)).astype(np.uint8) * 255
    value = np.stack((value, value, value), axis=2)
    mask255_op = cv2.morphologyEx(value, cv2.MORPH_OPEN, morphop_kernel, iterations=morphop_iter)
    if dilate_iter > 0:
        dilate_mask = cv2.dilate(mask255_op, dilate_kernel, iterations=dilate_iter)
    else:
        dilate_mask = mask255_op
    morph_close = cv2.morphologyEx(dilate_mask, cv2.MORPH_CLOSE, morphclose_kernel, iterations=morphclose_iter)
    return morph_close

class TacTipMarkerNode:
    def __init__(self):
        rospy.init_node('tactip_publisher_detection')
        
        # ROS Parameters
        self.cam_id = rospy.get_param('~cam_id', 4)
        self.min_area = rospy.get_param('~min_area', 200)
        self.sensor_id = rospy.get_param('~sensor_id', "tactip")
        self.morphop_iter = rospy.get_param('~morphop_iter', 2)
        self.morphclose_iter = rospy.get_param('~morphclose_iter', 1)
        self.dilate_iter = rospy.get_param('~dilate_iter', 2)
        self.crop_top = rospy.get_param('~crop_top', 50)
        self.crop_bottom = rospy.get_param('~crop_bottom', 380)
        self.crop_left = rospy.get_param('~crop_left', 180)
        self.crop_right = rospy.get_param('~crop_right', 510)
        self.resize_width = rospy.get_param('~resize_width', 640)
        self.resize_height = rospy.get_param('~resize_height', 480)
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Publishers
        self.raw_pub = rospy.Publisher(f'{self.sensor_id}/raw', Image, queue_size=1)
        # self.roi_pub = rospy.Publisher(f'{self.sensor_id}/roi', Image, queue_size=1)
        # self.normalized_pub = rospy.Publisher(f'{self.sensor_id}/normalized', Image, queue_size=1)
        # self.inverted_pub = rospy.Publisher(f'{self.sensor_id}/inverted', Image, queue_size=1)
        self.marker_pub = rospy.Publisher(f'{self.sensor_id}/marker', Image, queue_size=1)
        
        # Initialize camera
        self.cap = cv2.VideoCapture(self.cam_id)
        if not self.cap.isOpened():
            rospy.logerr(f"Cannot open camera {self.cam_id}")
            sys.exit(1)
        
        # Prebuilt kernels to match your defaults
        self.morphop_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        self.morphclose_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        self.dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        
        rospy.loginfo(f"TacTip marker detection node initialized with camera {self.cam_id}")
        
    def process_frame(self):
        """Process a single frame and publish results"""
        ok, frame = self.cap.read()
        if not ok:
            rospy.logwarn("Failed to read frame from camera")
            return False
            
        # Crop and resize frame
        cropped_frame = frame[self.crop_top:self.crop_bottom, self.crop_left:self.crop_right]
        resized_frame = cv2.resize(cropped_frame, (self.resize_width, self.resize_height), interpolation=cv2.INTER_LINEAR)
        
        # Apply polygon ROI first to constrain region
        roi = poly_mask(resized_frame)
        
        # Normalize brightness for more consistent thresholding
        norm = normalize_brightness(roi)
        
        # Invert as in your original flow
        inv = cv2.bitwise_not(norm)
        
        # Run your marker pipeline
        marker = find_marker(
            inv,
            morphop_kernel=self.morphop_kernel,
            morphclose_kernel=self.morphclose_kernel,
            dilate_kernel=self.dilate_kernel,
            morphop_iter=self.morphop_iter,
            morphclose_iter=self.morphclose_iter,
            dilate_iter=self.dilate_iter
        )
        
        # Publish all images
        try:
            self.raw_pub.publish(self.bridge.cv2_to_imgmsg(resized_frame, encoding='bgr8'))
            # self.roi_pub.publish(self.bridge.cv2_to_imgmsg(roi, encoding='bgr8'))
            # self.normalized_pub.publish(self.bridge.cv2_to_imgmsg(norm, encoding='bgr8'))
            # self.inverted_pub.publish(self.bridge.cv2_to_imgmsg(inv, encoding='bgr8'))
            self.marker_pub.publish(self.bridge.cv2_to_imgmsg(marker, encoding='bgr8'))
        except CvBridgeError as e:
            rospy.logerr(f"Failed to publish images: {e}")
            
        return True
    
    def run(self):
        """Main execution loop"""
        rate = rospy.Rate(30)  # 30 Hz
        
        while not rospy.is_shutdown():
            if not self.process_frame():
                rospy.logwarn("Frame processing failed, retrying...")
            rate.sleep()
    
    def shutdown(self):
        """Clean shutdown"""
        if self.cap.isOpened():
            self.cap.release()
        rospy.loginfo("TacTip marker detection node shutting down")

def main():
    try:
        node = TacTipMarkerNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        if 'node' in locals():
            node.shutdown()

if __name__ == "__main__":
    main()
