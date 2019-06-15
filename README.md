# 3dVolumeReconstruction
2.5 Depth Image to Complete Volumetric Representation of the human body

## Prerequisites
* ROS Kinetic (full)
* freenect-dev for Kinetic
* Python 2.7
* Dependencies folder of this repo
* PCL C++ Library
* Tensorflow >=1.12 w/ GPU support
* CUDA 7.0, cDNN 7.5
* Jupyter Notebook
* gcc >=4.8



## Data collection
1. Build the **camera_pose** ROS package in your catkin workspace.  

2. Run the **calibration1.launch** in terminal.  

3. In the second terminal run:

```
rosbag record --limit=300 -b 0 -O name.bag /kinect1/depth_registered/image_raw /kinect1/depth_registered/camera_info /kinect2/depth_registered/image_raw /kinect2/depth_registered/camera_info /kinect3/depth_registered/image_raw /kinect3/depth_registered/camera_info /kinect4/depth_registered/image_raw /kinect4/depth_registered/camera_info
```

## Prerprocessing

1. From the **camera_pose** package run **converter.launch**  

2. Record the published topics with pointclouds using the command `rosbag record ...` from the Data Collection step, though change the topic `image_raw` to `points`  

3. Play the prerecorded .bag files using:
```
rosbag play -r 0.1 name.bag
```

4. Build the **segment** package and run
```
rosrun segment segment_node name_of_the_bag_file_with_pointclouds.bag
```

5. Run the **cloud_preprocessing.py** script to convert .ply files into .txt with occupied coordinates in range from 0 to 63. Change the path string in .py file to the one, where your .ply files are located.  

## Training

Run the **train.py** script with the prefered config strings, batch size and the GPU ID of your choice. Don't forget to make devices visible to CUDA wrappers with CUDA_VISIBLE_DEVISES.  

After training, freeze the pretrained graph in order to deploy it in future.

