#!/usr/bin/env python

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO
import numpy as np
class YOLODetector:
    def __init__(self):

        # 初始化ROS节点
        rospy.init_node('yolo_detector', anonymous=True)
        print('Initialized')
        # 加载YOLOv8模型
        self.model = YOLO('/home/yunying/udg/PanAndTilt/yolo_ws/src/yolo_ros/yolov8.pt')  	
 
        # 初始化CV Bridge
        self.bridge = CvBridge()

        # 订阅摄像头图像话题
        self.image_sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.image_callback)

        # 发布检测结果图像话题
        self.image_pub = rospy.Publisher('/yolo_detection_result', Image, queue_size=10)

    def image_callback(self, msg):
        try:
            # 将ROS图像消息转换为OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
            img = cv_image.copy()
        except CvBridgeError as e:
            print(e)
            # rospy.logerr(e)
            return
        # cv2.imshow('original image', cv_image)
        # cv2.waitKey(1)
        # 使用YOLOv8进行检测
        results = self.model(cv_image)
        # print(f'result:{results}')
        # 绘制检测结果
        # for result in results:
           
        #     # Get the predictions for this image (boxes, labels, confidences)
        #     boxes = result.boxes.xywh  # Get the bounding boxes (x_center, y_center, width, height)
        #     labels = result.boxes.cls  # Get the class labels
        #     confidences = result.boxes.conf
      
        #     result_list = []
        #     # Loop through the detections and prepare them for the output
        #     for box, label,conf in zip(boxes, labels,confidences):
        #         if conf < 0.8:
        #             continue
        # # Convert normalized coordinates to absolute pixel values
        #         x_center, y_center, width, height = box
        

        #         # Convert to pixel coordinates
        #         x_min = (x_center - width / 2)
        #         y_min = (y_center - height / 2)
        #         x_max = (x_center + width / 2)
        #         y_max = (y_center + height / 2)
        #         # label = f"Class {class_id} {conf:.2f}"
        #         x_min = int(x_min.item())
        #         y_min = int(y_min.item())
        #         x_max = int(x_max.item())
        #         y_max = int(y_max.item())
        #         #print(x_min,y_min,x_max,y_max)
        #         cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        #         text = f"Conf: {conf:.2f}"
        #         cv2.putText(img, text, (x_max, y_max), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        boxes = results[0].boxes
        if boxes is not None and len(boxes)>0:
            best_idx = boxes.conf.argmax()
            box = boxes[best_idx]

            # Convert normalized coordinates to absolute pixel values
            # best_xyxy = box.xyxy.tolist()[0]
            # x_center, y_center, width, height = box
    

            # Convert to pixel coordinates
            # x_min = (x_center - width / 2)
            # y_min = (y_center - height / 2)
            # x_max = (x_center + width / 2)
            # y_max = (y_center + height / 2)
            # label = f"Class {class_id} {conf:.2f}"
            x_min, y_min, x_max, y_max = box.xyxy.tolist()[0]
            x_min = int(x_min)
            y_min = int(y_min)
            x_max = int(x_max)
            y_max = int(y_max)
            #print(x_min,y_min,x_max,y_max)
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            conf = box.conf.item()
            text = f"Conf: {conf:.2f}"
            cv2.putText(img, text, (x_max, y_max), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        # annotated_frame = results[0].plot()

        # 显示检测结果
        cv2.imshow('YOLOv8 Detection', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)



if __name__ == '__main__':
    try:
        detector = YOLODetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
