
#!/usr/bin/env python3
import rospy
import torch
from torchvision import transforms  
import numpy as np
import os
import sys
import json
import yaml
import time

from PIL import Image as ImagePIL
from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float32
from cv_bridge import CvBridge
from std_srvs.srv import Empty, EmptyResponse

# Add model path
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))

from model import TemporalForce
import cv2

def to_normal(m, y):
    return y * (m[:, 1] - m[:, 0]) + m[:, 0]

class ForcePredictorNode:
    def __init__(self):
        rospy.init_node('force_predictor', anonymous=True)
        # Model/config paths (edit these if needed)
        self.config = rospy.get_param('~config', '/home/zhuo/catkin_ws/src/genforce/checkpoints/force/config.yaml')
        self.sensor_id = rospy.get_param('~sensor_id', 'DI')
        self.checkpoint = rospy.get_param('~checkpoint', 'DI')
        with open(self.config, 'r') as f:
            config = yaml.safe_load(f)
        checkpoint_path = config[self.checkpoint]

        # ROS topics
        self.img_topic = rospy.get_param('~img_topic', f'{self.sensor_id}/raw')
        self.fz_topic = rospy.get_param('~fz', f'/force/{self.sensor_id}/z')
        self.fy_topic = rospy.get_param('~fy', f'/force/{self.sensor_id}/y')
        self.fx_topic = rospy.get_param('~fx', f'/force/{self.sensor_id}/x')

        # checkpoint_path = rospy.get_param('~checkpoint', '/home/zhuo/catkin_ws/src/genforce/checkpoints/force/force/homo/Array-II/D-I_A-II.pth')
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
        
        # 基线漂移消除相关设置
        self.calibration_time = rospy.get_param('~calibration_time', 3.0)  # 30秒校准时间
        self.baseline_buffer_size = rospy.get_param('~baseline_buffer_size', 50)  # 用最后50帧计算基线
        self.baseline_buffer = []
        self.baseline = np.zeros(3, dtype=np.float32)
        self.baseline_ready = False
        self.start_time = time.time()
        
        # 添加服务以重置基线
        self.reset_srv = rospy.Service('~reset_baseline', Empty, self._handle_reset_baseline)
        
        # rospy.loginfo(f"Force predictor initialized with 30-second baseline calibration.")
        # rospy.loginfo(f"No force values will be published during the first 30 seconds.")
        # rospy.loginfo(f"After 30 seconds, the last 50 frames will be used to calculate baseline.")
        
        # 最后启动订阅，确保所有初始化完成
        rospy.Subscriber(self.img_topic, Image, self.image_callback, queue_size=1, buff_size=2**24)
        rospy.loginfo(f"Listening to {self.img_topic}")

    def _handle_reset_baseline(self, req):
        """处理基线重置请求"""
        self.baseline_ready = False
        self.baseline_buffer.clear()
        self.baseline[:] = 0.0
        self.start_time = time.time()
        rospy.loginfo("Baseline reset via service. Starting new 30-second calibration period...")
        return EmptyResponse()
    
    def preprocess(self, cv_img):
       transform = transforms.Compose([  
            transforms.Resize([256, 256]),  
            transforms.ToTensor(),  
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
        ])  
       cvimage_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    #    image = ((np.array(cvimage_rgb)>50)*255).astype(np.float32)
       image = ImagePIL.fromarray((cvimage_rgb).astype(np.uint8))
       image = transform(image).to(self.device)
       return image

    def predict_force(self, img):
        with torch.no_grad():
            features = self.model.base_network(img.view(1, *img.shape))
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
        force_raw = self.predict_force(img_torch)
        force_raw = np.squeeze(force_raw).astype(np.float32)  # shape (3,)

        # 如果还没建立基线
        if not self.baseline_ready:
            elapsed_time = time.time() - self.start_time
            
            # 将帧添加到缓冲区，保持最多 baseline_buffer_size 帧
            self.baseline_buffer.append(force_raw.copy())
            if len(self.baseline_buffer) > self.baseline_buffer_size:
                self.baseline_buffer.pop(0)  # 移除最旧的帧
            
            # 显示剩余校准时间
            remaining_time = max(0, self.calibration_time - elapsed_time)
            rospy.loginfo_throttle(1, f"Calibration in progress... {remaining_time:.1f} seconds remaining")
            
            # 检查是否已经达到校准时间
            if elapsed_time >= self.calibration_time:
                # 使用缓冲区中的所有帧计算基线
                self.baseline = np.mean(np.stack(self.baseline_buffer, axis=0), axis=0).astype(np.float32)
                self.baseline_ready = True
                rospy.loginfo(f"Calibration complete after 30 seconds.")
                rospy.loginfo(f"Baseline established from last {len(self.baseline_buffer)} frames: "
                              f"x={self.baseline[0]:.3f}, y={self.baseline[1]:.3f}, z={self.baseline[2]:.3f}")
                rospy.loginfo(f"Beginning force publication with drift correction.")
            
            # 校准期间不发布力值
            return

        # 已完成校准，扣除基线并发布
        force = force_raw - self.baseline

        # 发布校正后的力值
        self.fz_pub.publish(float(force[2]))
        self.fy_pub.publish(float(force[1]))
        self.fx_pub.publish(float(force[0]))

        rospy.loginfo_throttle(1, f"Predicted force: x={force[0]:.3f}, y={force[1]:.3f}, z={force[2]:.3f}")

    def spin(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = ForcePredictorNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
