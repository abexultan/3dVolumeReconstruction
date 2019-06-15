#!/usr/bin/env python
import tensorflow as tf
import argparse
import time
import rospy
import std_msgs.msg
from std_msgs.msg import Int32MultiArray
import numpy as np
import math
import matplotlib.pyplot as plt
import sensor_msgs.point_cloud2 as pcl2
from sensor_msgs.msg import PointCloud2
from open3d import Vector3dVector
k=0
GPU0 = '0'
def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph

# def fromGridToPLY(grid):
#     grid = (grid>0.5)
#     ind = 0
#     counter = 0
#     for i in range(grid.shape[0]):
#         for j in range(grid.shape[1]):
#             for k in range(grid.shape[2]):
#                 if(grid[i,j,k]==1):
#                     counter = counter + 1
#     xyz = np.zeros((counter,3))
#     for i in range(grid.shape[0]):
#         for j in range(grid.shape[1]):
#             for k in range(grid.shape[2]):
#                 if(grid[i,j,k]==1):
#                     xyz[ind, :] = (i,j,k)
#                     ind = ind + 1
#                 else:
#                     continue
    # return Vector3dVector(xyz)
def fromGridToPLY(grid):
	for i in range(3):
		if i==0:
			temp = np.where(grid>0.5)[i]
			x = temp.reshape(temp.shape[0],1)
		if i==1:
			temp = np.where(grid>0.5)[i]
			y = temp.reshape(temp.shape[0],1)
        if i==2:
        	temp = np.where(grid>0.5)[i]
        	z = temp.reshape(temp.shape[0],1)

	pcl = np.c_[x,y,z]
	return Vector3dVector(pcl)

def callback(data, args):
	sess = args[0]
	graph = args[1]
	cloud_pub = args[2]
	x_tensor = graph.get_tensor_by_name('prefix/Placeholder:0')
	y_tensor = graph.get_tensor_by_name('prefix/ae/Sigmoid:0')
	grid = np.zeros((64,64,64))
	
	start = time.time()
	for i in data.data:
		x=int(math.floor((i/64)/64))
		y=int(math.floor((i-x*64*64)/64))
		z=int(math.floor(i-x*64*64-y*64))
		grid[x,y,z]=1
	end = time.time()
	print('Time for one grid to ged decoded: ' + str(end-start))


	temp = grid.reshape(1,64,64,64,1)
	inp = temp.astype(float)

	start = time.time()
	y_out = sess.run(y_tensor, feed_dict={x_tensor: inp})
	end = time.time()
	print('Time for one feed forward: ' + str(end-start))

	temp = np.zeros((1,64,64,64,1))
	temp = y_out
	y_out = temp.reshape(64,64,64)

	start = time.time()
	data = fromGridToPLY(y_out)
	end = time.time()
	print('Time for one grid to pointcloud conversion: ' + str(end-start))

	header = std_msgs.msg.Header()
	header.stamp = rospy.Time.now()
	header.frame_id = 'pcl'
 	#create pcl from points
	pcl = pcl2.create_cloud_xyz32(header, data)

	start = time.time()
	cloud_pub.publish(pcl)
	end = time.time()
	print('Time for one point cloud to publish: ' + str(end-start))
	# rate = rospy.Rate(10).sleep()



	# np.save("output_demo.npy", y_out)
	
def listener(session, graph):

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)
    publisher = rospy.Publisher('pointcloud_debug',PointCloud2,queue_size=1)
    print('Listening......')
    rospy.Subscriber("matrix_pub", Int32MultiArray, callback, (session, graph, publisher))

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
	graph = load_graph('/home/bexultan/Desktop/demo/OptimizedGraph.pb')
	x = graph.get_tensor_by_name('prefix/Placeholder:0')
	y = graph.get_tensor_by_name('prefix/ae/Sigmoid:0')
	config = tf.ConfigProto(log_device_placement=True)
	config.gpu_options.visible_device_list = GPU0
	config.gpu_options.allow_growth = True
	sess = tf.Session(graph = graph, config = config)
	sample = np.load("/home/bexultan/Desktop/demo/sample.npy")
	placeholder = sample[0].reshape(1,64,64,64,1)
	y_out = sess.run(y, feed_dict={x: placeholder})
	print('Initialization is done.')
	listener(sess, graph)