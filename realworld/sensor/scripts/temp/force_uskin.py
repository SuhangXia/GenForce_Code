#!/usr/bin/env python3
import rospy
import torch
from torchvision import transforms  
import numpy as np
import os
import sys
import json

from PIL import Image as ImagePIL
from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float32
from cv_bridge import CvBridge

# Add model path
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))

from model import TemporalForce
import cv2

def to_normal(m, y):
    return y * (m[:, 1] - m[:, 0]) + m[:, 0]

class ForcePredictorNode:
    def __init__(self):
        rospy.init_node('force_predictor', anonymous=True)
        # ROS topics
        self.img_topic = rospy.get_param('~img_topic', '/uskin/marker')
        self.fz_topic = rospy.get_param('~fz', '/force/uskin/z')
        self.fy_topic = rospy.get_param('~fy', '/force/uskin/y')
        self.fx_topic = rospy.get_param('~fx', '/force/uskin/x')

        # Model/config paths (edit these if needed)
        checkpoint_path = rospy.get_param('~checkpoint', '/home/zhuo/catkin_ws/src/genforce/checkpoints/force/force/heter/uskin/uskin.pth')
        minmax_path = rospy.get_param('~minmax', '/home/zhuo/catkin_ws/src/genforce/scripts/config/min_max.json')

        self.cv_bridge = CvBridge()

        # Model setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = TemporalForce().to(self.device)
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        with open(minmax_path, "r") as file:
            global_min_max = json.load(file)
        global_min = torch.tensor(global_min_max['min']).to(self.device)
        global_max = torch.tensor(global_min_max['max']).to(self.device)
        self.normalization_stats = torch.stack([global_min, global_max], dim=1)

        self.BATCH_SIZE = 1
        self.SEQ_LEN = 1
        self.hidden_state = None

        self.fz_pub = rospy.Publisher(self.fz_topic, Float32, queue_size=10)
        self.fy_pub = rospy.Publisher(self.fy_topic, Float32, queue_size=10)
        self.fx_pub = rospy.Publisher(self.fx_topic, Float32, queue_size=10)
        rospy.Subscriber(self.img_topic, Image, self.image_callback, queue_size=1, buff_size=2**24)
        rospy.loginfo(f"Listening to {self.img_topic}")

    # def preprocess(self, cv_img):
    #     frame_resized = cv2.resize(cv_img, (256, 256))
    #     img = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    #     img = img / 255.0  # Normalize to [0, 1]
    #     img = torch.from_numpy(img).permute(2, 0, 1).float()  # (C, H, W)
    #     img = img.unsqueeze(0).unsqueeze(0)  # (S, B, C, H, W)
    #     img = img.to(self.device)
    #     return img
    
    def preprocess(self, cv_img):
       transform = transforms.Compose([  
            transforms.Resize([256, 256]),  
            transforms.ToTensor(),  
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
        ])  
       cvimage_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
       image = ((np.array(cvimage_rgb)>50)*255).astype(np.float32)
       image = ImagePIL.fromarray((image).astype(np.uint8))
       image = transform(image).to(self.device)
       return image

    def predict_force(self, img):
        with torch.no_grad():
            features = self.model.base_network(img.view(1, *img.shape))
            # print(img.view(-1, *img.shape[2:]).shape)
            # features = self.model.base_network(img.view(-1, *img.shape[2:]))
            features = features.view(self.SEQ_LEN, self.BATCH_SIZE, self.model.feature_dim, 32, 32)
            outputs, self.hidden_state = self.model.convgru(features, self.hidden_state)
            outputs_pp = outputs.view(-1, self.model.feature_dim, 32, 32)
            outputs_pp = self.model.post_processing(outputs_pp)
            force_pred = self.model.reg_layer(outputs_pp.view(-1, 512))
            force_pred = force_pred.view(-1,3)
            force_real = to_normal(self.normalization_stats, force_pred).cpu().numpy()
        return force_real

    def image_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV image (BGR)
            cv_img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logwarn(f"cv_bridge error: {e}")
            return

        img_torch = self.preprocess(cv_img)
        force = self.predict_force(img_torch)
        force = np.squeeze(force)
        # Publish force as Vector3
        force_msg = Vector3()
        force_msg.x = float(force[0])
        force_msg.y = float(force[1])
        force_msg.z = float(force[2])
        self.fz_pub.publish(force_msg.z)
        self.fy_pub.publish(force_msg.y)
        self.fx_pub.publish(force_msg.x )

        rospy.loginfo_throttle(1, f"Predicted force: x={force[0]:.3f}, y={force[1]:.3f}, z={force[2]:.3f}")

    def spin(self):
        rospy.spin()

if __name__ == '__main__':
    node = ForcePredictorNode()
    node.spin()