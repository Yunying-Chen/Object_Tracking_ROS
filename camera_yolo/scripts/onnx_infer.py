#!/usr/bin/env python

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
# from ultralytics import YOLO
import numpy as np
from camera_yolo import Filters
# from scipy.signal import savgol_filter
# import torch
from geometry_msgs.msg import Vector3
# from deep_sort_realtime.deepsort_tracker import DeepSort
import sys
import os
from camera_yolo import Filters

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from camera_yolo import onnx
# from onnx import ONNXDetect
#pip install deep-sort-realtime
#pip install onnxruntime
         
class YOLODetector:
    def __init__(self):

        self.dif_x = None
        self.dif_y = None
        self.dif_z = None
        self.camera_detected = False
        self.kf = Filters.KalmanFilter(dt=1,std_acc=0.1,std_meas=0.3)
        
        self.model = onnx.ONNXDetect()
        # self.model = YOLO('./catkin_ws/src/camera_yolo/scripts/yolov8.engine')  	
        # self.model.eval()
        self.bridge = CvBridge()
        self.result = None
        self.do_inference = True
        self.x_min = None
        self.x_max = None
        self.y_min = None
        self.y_max = None
        self.image_sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.image_callback)
        self.position_pub = rospy.Publisher('/luma_position', Vector3, queue_size=10)
        self.times = 0
        self.image = None
        # self.tracker = DeepSort(max_age  = 50)
        rospy.Timer(rospy.Duration(0.001),self.yolo_detection)
        rospy.Timer(rospy.Duration(0.1),self.message_pub)
        # rospy.Timer(rospy.Duration(0.1),self.tracking)
        
    def image_callback(self, msg):

        try:
            
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
            # img = cv_image.copy()
            self.img_h,self.img_w,img_c = cv_image.shape
            self.image = cv_image
            self.img = cv_image.copy()
        except CvBridgeError as e:
            print(e)
            # rospy.logerr(e)
            return
        updated = None
        if self.x_min is not None and self.score>0.9:
            center_x = int((self.x_min + self.x_max) / 2)
            center_y = int((self.y_min + self.y_max) / 2)
            # self.img = cv2.rectangle(self.img, (self.x_min, self.y_min), (self.x_max, self.y_max), (0, 255, 0), 5)
            self.img = cv2.circle(self.img, (center_x,center_y), radius=0, color=(255, 0, 0), thickness=10)
            # self.img = cv2.putText(self.img, str(f'{self.score:.2f}'), (self.x_min, self.y_max), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            z = np.array([center_x,center_y])
            updated = self.kf.update(z)
        # if self.img is not None:
        #     cv2.imshow('Detection', cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
        #     cv2.waitKey(1)


        pred = self.kf.predict()
        self.pred_x,self.pred_y = pred[0,0],pred[1,0]
        
        if updated is not None:
            self.pred_x,self.pred_y = updated
     



        self.dif_x = self.pred_x - int(self.img_w/2)
        self.dif_y = self.pred_y - int(self.img_h/2)
        self.dif_z = 0
        
        # rospy.loginfo(f'{int(self.pred_x),int(self.pred_y)}')
        if self.pred_x>0 and self.pred_x<self.img_w and self.pred_y>0 and self.pred_y<self.img_h:
            # print(f'updated:{int(self.pred_x),int(self.pred_y)}')
            self.img = cv2.circle(self.img, (int(self.pred_x),int(self.pred_y)), radius=0, color=(0, 255, 0), thickness=10)		

        cv2.imshow('Luma Detection', cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)

    def yolo_detection(self,event):

        if self.do_inference and self.image is not None:
            self.do_inference =False
            outputs = self.model(self.image)
            # for output in outputs:
            if outputs is not None:
                output = outputs[0]
                print(outputs)
                x, y, w, h, self.score, index = outputs
                self.x_min = x
                self.y_min = y
                self.x_max = x+w
                self.y_max = y+h
                self.times = 0
            else:
                self.times+=1
                
            self.do_inference =True        

            
    
    def message_pub(self,event):
        msg_dis = Vector3()
        if self.times>10:
            msg_dis.x = None
            msg_dis.y = None
            msg_dis.z = None
        else:
            msg_dis.x = self.dif_x
            msg_dis.y = self.dif_y
            msg_dis.z = self.dif_z

        self.position_pub.publish(msg_dis)


if __name__ == '__main__':
    rospy.init_node('yolo_detector', anonymous=True)

    try:
        detector = YOLODetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
