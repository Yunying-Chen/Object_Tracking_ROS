#!/usr/bin/env python

import rospy
import cv2
from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
class LumaDetector:
    def __init__(self):
        # print('initializing')
        # 初始化ROS节点
        rospy.init_node('luma_detector', anonymous=True)
    
 
        # 初始化CV Bridge
        self.bridge = CvBridge()

        # 订阅摄像头图像话题
        self.image_sub = rospy.Subscriber('/dummy/camera/image_color', Image, self.image_callback)
        # print(f'Subscribered')
        # 发布检测结果图像话题
        self.dis_pub = rospy.Publisher('/dummy/luma_position', Vector3, queue_size=10)
        self.rate = rospy.Rate(1) 
    def image_callback(self, msg):
        # print(f'calling back')
        try:
            # 将ROS图像消息转换为OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
            
            img = cv_image.copy()
        except CvBridgeError as e:
            print(e)
            # rospy.logerr(e)
            return
        h,w,c = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, img_thres = cv2.threshold(img_gray,240,255,cv2.THRESH_BINARY)
        moments = cv2.moments(img_thres)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
        else:
            cx, cy = 0, 0 

        img = cv2.circle(img,(cx,cy),5,(255,0,0),-1)
        img = cv2.circle(img,(int(w/2),int(h/2)),5,(0,255,255),-1)
        # 显示检测结果
        cv2.imshow('Luma Detection', img)
        cv2.waitKey(1)

        dis_x = cx - int(w/2)
        dis_y = cy - int(h/2)
        dis_z = 0
     
        msg_dis = Vector3()
        msg_dis.x = dis_x
        msg_dis.y = dis_y
        msg_dis.z = dis_z

        self.dis_pub.publish(msg_dis)
    


if __name__ == '__main__':
    try:
        detector = LumaDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
