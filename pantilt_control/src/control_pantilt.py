#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float64MultiArray
import numpy as np

class PIDController:
    def __init__(self,Kp=0.005,Ki=0.01,Kd=0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error_x = 0
        self.prev_error_y = 0
        self.dt = 0.1
    
    def update(self,error_x,error_y):
        derivative_x = (error_x - self.prev_error_x)/self.dt
        derivative_y = (error_y - self.prev_error_y)/self.dt
        control_x = (self.Kp * error_x) + (self.Kd * derivative_x)
        control_y = (self.Kp * error_y) + (self.Kd * derivative_y)

        self.prev_error_x = error_x
        self.prev_error_y = error_y

        return control_x,control_y


class PanTiltController:
    def __init__(self):

        rospy.init_node('pantilt_controller', anonymous=True)

        # defined variables 
        self.control_msg = Float64MultiArray()
        self.control_msg.data = [0.3,0.3]
        self.velocity_updated=False

        self.control_sub = rospy.Subscriber('/dummy/luma_position', Vector3, self.control_callback)
        self.velocity_pub = rospy.Publisher('/dummy/pan_tilt/joint_velocity_controller/command', Float64MultiArray, queue_size=10)
        self.pid_controller = PIDController()

        rospy.Timer(rospy.Duration(5),self.update_check)

    def control_callback(self, msg):
        dis_x = msg.x
        dis_y = msg.y
        control_x,control_y = self.pid_controller.update(dis_x,dis_y)
        self.control_msg.data = [control_x,-control_y]
        self.velocity_updated=True
        self.velocity_pub.publish(self.control_msg)
        

    def update_check(self,event):
        if self.velocity_updated:
            self.velocity_updated = False
            return 0
        else:
            self.control_msg.data = [0.3,0.3]
            self.velocity_pub.publish(self.control_msg)
            return 0
            
            


if __name__ == '__main__':
    try:
        controler = PanTiltController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


