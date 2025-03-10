# Pan-And-Tilt Realtime Object Tracking

## Introduction
In this project, it uses PID control to control the pan-and-tilt to tracking on an object. The object detection is based on a deep neural network in the ONNX format.

## Requirements
1. ROS   
2. opencv   
3. onnxruntime

## Folder
- Camera_yolo:   
It includes the script to do the inference for object detection and a static Kalman Filter to stablize the detection. It pubishes the topic of the distance of the object and the center fo the images.    
- Pantilt_control:    
It includes the script to use PID control to control the pan-and-tilt.

## Performance
![Animation](./images/tracking.gif)