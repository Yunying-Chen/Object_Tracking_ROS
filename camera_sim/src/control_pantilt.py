
#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float64MultiArray
import numpy as np

class PIDController:
    def __init__(self,Kp=0.1,Ki=0,Kd=0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
    
    def update(self,error_x,error_y):
        control_x = self.Kp * error_x
        control_y = self.Kp * error_y

        return control_x,control_y


class PanTiltController:
    def __init__(self):
        rospy.init_node('pantilt_controller', anonymous=True)
        self.control_sub = rospy.Subscriber('/dummy/luma_position', Vector3, self.control_callback)
        self.velocity_pub = rospy.Publisher('/dummy/PID_velocity', Float64MultiArray, queue_size=10)
        self.pid_controller = PIDController()

    def control_callback(self, msg):
        dis_x = msg.x
        dis_y = msg.y
        control_x,control_y = self.pid_controller.update(dis_x,dis_y)
        control_msg = Float64MultiArray()
        control_msg.data = [control_x,control_y]
        self.velocity_pub.publish(control_msg)

if __name__ == '__main__':
    try:
        controler = PanTiltController()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


