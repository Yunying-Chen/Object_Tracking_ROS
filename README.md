# Pan-And-Tilt Realtime Object Tracking

## Introduction
This project demonstrates real-time object tracking using a pan-and-tilt system controlled via PID control. The object detection is powered by a deep neural network in ONNX format, integrated with ROS for real-time communication.    


## Requirements
1. ROS   
2. opencv   
3. onnxruntime

## Project Structure
PanTilt-Tracking/          
├── Camera_yolo/           
│   ├── Performs ONNX-based object detection     
│   ├── Uses a static Kalman Filter to stabilize bounding box detection            
│   └── Publishes object distance and image center as ROS topics              
│               
├── Pantilt_control/             
│   ├── Implements PID control logic                          
│   └── Subscribes to detection info to command pan-and-tilt motors             


## Performance
A real-time example of the system in action:   
<p align="center"> <img src="./images/tracking.gif" width="480" alt="Pan and Tilt Object Tracking Demo"/> </p>
