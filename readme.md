# Driver Fatigue Detection | OpenCV
This repository uses OpenCV and dlib to detect and track the user's eyes and mouth in real-time video, alerting when drowsiness (such as prolonged eye closure or yawning) is detected.

## Applications
Drowsy driving is a major cause of serious traffic accidents.  
- In the United States, drowsy driving is estimated to contribute to approximately **17.6%** of fatal crashes (AAA Foundation for Traffic Safety analysis of 2017–2021 data, published 2024).  
  (Official NHTSA reports list lower figures around 1–2%, but experts widely agree that drowsiness is significantly under-reported.)  
- In countries like India with higher road accident rates, long-haul drivers, truck operators, and motorcycle riders face even greater risk.

This system can serve as a simple, non-intrusive safety aid — using only a webcam to monitor the driver and sound an alert to help prevent fatigue-related accidents.



### Dependencies

- import cv2
- import immutils
- import dlib
- import scipy

### Description
A computer vision system that automatically detects driver drowsiness in a real-time video stream by tracking eye closure (via Eye Aspect Ratio) and yawning behavior, then triggers an audible and visual alert when the driver appears to be drowsy.

### Execution
To run the code, perform the below command
```
python3 driverFatigue_v2.py
```
or
```
python driverFatigue_v2.py
```
