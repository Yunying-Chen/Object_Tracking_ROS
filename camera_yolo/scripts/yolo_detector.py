#!/usr/bin/env python

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO
import numpy as np
from camera_yolo import Filters
from scipy.signal import savgol_filter


         
class YOLODetector:
    def __init__(self):

    
        rospy.init_node('yolo_detector', anonymous=True)
        self.camera_detected = False
        self.kf = Filters.KalmanFilter(dt=1,std_acc=0.1,std_meas=1)

        self.model = YOLO('./yolov8.pt',verbose=False)  	
  
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.image_callback)
        self.image_pub = rospy.Publisher('/yolo_detection_result', Image, queue_size=10)

        rospy.Timer(rospy.Duration(5),self.camera_check)

    def image_callback(self, msg):
        try:
            
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
            img = cv_image.copy()
            img_h,img_w,img_c = cv_image.shape
        except CvBridgeError as e:
            print(e)
            # rospy.logerr(e)
            return

        results = self.model(cv_image,verbose=False)

        updated = None
        boxes = results[0].boxes
        if boxes is not None and len(boxes)>0:
            best_idx = boxes.conf.argmax()
            best_conf = boxes.conf.max()
            if best_conf>0.5:
                box = boxes[best_idx]
                x_min, y_min, x_max, y_max = box.xyxy.tolist()[0]
                x_min = int(x_min)
                y_min = int(y_min)
                x_max = int(x_max)
                y_max = int(y_max)
                center_x = int((x_min + x_max) / 2)
                center_y = int((y_min + y_max) / 2)
                z = np.array([center_x,center_y])
                updated = self.kf.update(z)
                img = cv2.circle(img, (center_x,center_y), radius=0, color=(255, 0, 0), thickness=10)
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                conf = box.conf.item()
                text = f"Conf: {conf:.2f}"
                cv2.putText(img, text, (x_max, y_max), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

 
        pred = self.kf.predict()
        self.pred_x,self.pred_y = pred[0,0],pred[1,0]
        # print(pred.shape)
        print(f'pred:{int(self.pred_x),int(self.pred_y)}')
        
        if updated is not None:
            self.pred_x,self.pred_y = updated
            # print(f'update:{int(pred_x),int(pred_y)}')
            self.camera_detected=True

        if self.pred_x>0 and self.pred_x<img_w and self.pred_y>0 and self.pred_y<img_h:
            print(f'updated:{int(self.pred_x),int(self.pred_y)}')
            img = cv2.circle(img, (int(self.pred_x),int(self.pred_y)), radius=0, color=(0, 255, 0), thickness=10)		

        cv2.imshow('YOLOv8 Detection', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)

    def camera_check(self,event):
        if self.camera_deteced:
            self.camera_deteced = False
            return 0
        else:
            self.pred_x,self.pred_y = -1,-1
            return 0


if __name__ == '__main__':
    try:
        detector = YOLODetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
