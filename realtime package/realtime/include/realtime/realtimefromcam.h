#pragma once
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
// Fixes undefined reference to pcl::Search
#include <pcl/search/impl/search.hpp>
#include <iostream>
#include <fstream>
#include <math.h>
#include <ros/ros.h>
#include "std_msgs/Int16MultiArray.h"
#include "std_msgs/MultiArrayDimension.h"
#include <vector>
#include <array>

#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/features/normal_3d.h>

#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/PointCloud2.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <Eigen/Dense>

#include <octomap/octomap.h>

#include <cilantro/icp_common_instances.hpp>
#include <cilantro/point_cloud.hpp>
#include <cilantro/timer.hpp>

#include <algorithm>
#include <chrono>
#include <memory>
#include <iostream>
#include <list>
#include <unordered_map>

Eigen::Matrix3f R2, R3, R4;
Eigen::Array3f t2, t3, t4;
Eigen::Matrix4f T2, T3, T4;

auto const P1 = Eigen::Vector3f(0.111783, 1.85, 1.91526);
auto const P2 = Eigen::Vector3f(-1.0, 1.3, 3.5);
auto const P4 = Eigen::Vector3f(1.403284, 0.953331, 2.052759);
auto const P5 = Eigen::Vector3f(0.3, 0.0, 0.2);

auto const i = P2 - P1;
auto const j = P4 - P1;
auto const k = P5 - P1;

int count = 0;

inline auto timestamp() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch());
}

template <class M>
class BagSubscriber : public message_filters::SimpleFilter<M>
{
public:
    void newMessage(boost::shared_ptr<M const> const& msg)
    {
        this->signalMessage(msg);
    }
};




bool inside(pcl::PointXYZ const& point)
{
    if (!pcl::isFinite(point)) {
        return false;
    }

    auto const x = Eigen::Vector3f(point.x, point.y, point.z);
    auto const v = x - P1;

    return (0 < v.dot(i) && v.dot(i) < i.dot(i))
        && (0 < v.dot(j) && v.dot(j) < j.dot(j))
        && (0 < v.dot(k) &&  v.dot(k) < k.dot(k));
}

void box_filter(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
    // printf("Total\t%ld\n", cloud->size());

    cloud->erase(std::remove_if(cloud->begin(), cloud->end(),
            [](auto const& x)
            {
                return !inside(x);
            }), cloud->end());
    // printf("After\t%ld\n", cloud->size());
    // pcl::PLYWriter plywriter;
    // plywriter.write<pcl::PointXYZ>(filename, *cloud, false);
}
void segment(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,std_msgs::Int16MultiArray dat,ros::Publisher pub)
{
  
  pcl::PointCloud<pcl::PointXYZ>::Ptr  cloud_f (new pcl::PointCloud<pcl::PointXYZ>);
  // reader.read (filename, *cloud);
  // std::cout << "PointCloud before filtering has: " << cloud->points.size () << " data points." << std::endl; //*

  // Create the filtering object: downsample the dataset using a leaf size of 1cm
  pcl::VoxelGrid<pcl::PointXYZ> vg;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
  vg.setInputCloud (cloud);
  vg.setLeafSize (0.02f, 0.02f, 0.02f);
  vg.filter (*cloud_filtered);
  // std::cout << "PointCloud after filtering has: " << cloud_filtered->points.size ()  << " data points." << std::endl; //*

  // Create the segmentation object for the planar model and set all the parameters
  pcl::SACSegmentation<pcl::PointXYZ> seg;
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane (new pcl::PointCloud<pcl::PointXYZ> ());
  pcl::PLYWriter plywriter;
  seg.setOptimizeCoefficients (true);
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setMaxIterations (100);
  seg.setDistanceThreshold (0.04); //def 0.02

  int i=0, nr_points = (int) cloud_filtered->points.size ();
  while (cloud_filtered->points.size () > 0.4 * nr_points)
  {
    // Segment the largest planar component from the remaining cloud
    seg.setInputCloud (cloud_filtered);
    seg.segment (*inliers, *coefficients);
    if (inliers->indices.size () == 0)
    {
      std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
      break;
    }

    // Extract the planar inliers from the input cloud
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud (cloud_filtered);
    extract.setIndices (inliers);
    extract.setNegative (false);

    // Get the points associated with the planar surface
    extract.filter (*cloud_plane);
    // std::cout << "PointCloud representing the planar component: " << cloud_plane->points.size () << " data points." << std::endl;

    // Remove the planar inliers, extract the rest
    extract.setNegative (true);
    extract.filter (*cloud_f);
    *cloud_filtered = *cloud_f;
  }

  // Creating the KdTree object for the search method of the extraction
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
  tree->setInputCloud (cloud_filtered);

  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
  ec.setClusterTolerance (0.2); // 2cm
  ec.setMinClusterSize (1000);
  ec.setMaxClusterSize (30000);
  ec.setSearchMethod (tree);
  ec.setInputCloud (cloud_filtered);
  ec.extract (cluster_indices);
  

  int j = 0;
  for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
    for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
      cloud_cluster->points.push_back (cloud_filtered->points[*pit]); //*
    cloud_cluster->width = cloud_cluster->points.size ();
    cloud_cluster->height = 1;
    cloud_cluster->is_dense = true;

    // std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size () << " data points." << std::endl;
    // std::stringstream ss;
    // ss << "cloud_cluster_" << j << ".pcd";
    if(j==0){
      int l = cloud_cluster->points.size ();
      float coord[l][3];
      int m=0;
      float minx=500;
      float miny=500;
      float minz=500;
      
      dat.data.clear();
      // dat.data = new int[64*64*64];
      
      for(pcl::PointCloud<pcl::PointXYZ>::iterator it = cloud_cluster->begin(); it!= cloud_cluster->end(); it++){
        coord[m][0]=it->x*25;
        coord[m][1]=it->y*25;
        coord[m][2]=it->z*25;
        if(coord[m][0]<minx){minx=coord[m][0];}
        if(coord[m][1]<miny){miny=coord[m][1];}
        if(coord[m][2]<minz){minz=coord[m][2];}

        m++;
      }
      // for(int h=0;h<64*64*64;h++){
      //       dat.data.push_back(0);
      // }
      // std::vector<int> vec(64*64*64, 0);
      int s=0;
      for(int n=0;n<l;n++){
        coord[n][0]=(int)round(coord[n][0]-minx);
        coord[n][1]=(int)round(coord[n][1]-miny);
        coord[n][2]=(int)round(coord[n][2]-minz);
        if(coord[n][0]<64 && coord[n][1]<64 && coord[n][2]<64){
          // vec[coord[n][0]*64*64+coord[n][1]*64+coord[n][2]]=1;
          dat.data.push_back(coord[n][0]*64*64+coord[n][1]*64+coord[n][2]);
          s=n;
        }
        
      }
      // dat.data.insert(dat.data.end(),vec.begin(), vec.end());
      // vec.clear();
      // int s=dat.data.size();
      // while (ros::ok())
      // {
          pub.publish(dat);
          ros::spinOnce ();
      // }
      std::cout << dat.data.size()<<" "<< coord[s-1][0]<< " " << coord[s-1][1]<< " " << coord[s-1][2]<< endl;
    }
    // plywriter.write<pcl::PointXYZ> (btfilename+std::to_string(count) +std::to_string(j)+".ply", *cloud_cluster, false); //*
    j++;
  }
  
    
}


void callback(sensor_msgs::PointCloud2ConstPtr const& pcloud2,std_msgs::Int16MultiArray dat,ros::Publisher pub)
{
    printf("--------------------------------------------\n");
    printf("Cloud number %i\n", count);
    printf("--------------------------------------------\n");
   
    auto prep = timestamp();
    pcl::PointCloud<pcl::PointXYZ> cloudb;
    // clouda, cloudb, cloudc, cloudd;
    // pcl::fromROSMsg(*pcloud1, clouda);
    pcl::fromROSMsg(*pcloud2, cloudb);
    // pcl::fromROSMsg(*pcloud3, cloudc);
    // pcl::fromROSMsg(*pcloud4, cloudd);

    // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1 = clouda.makeShared();
    pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud2 = cloudb.makeShared();
    // pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud3 = cloudc.makeShared();
    // pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud4 = cloudd.makeShared();

    // Transform prep
    pcl::PointCloud<pcl::PointXYZ> c2;
    // , c3, c4;
    pcl::PointCloud<pcl::PointXYZ>::Ptr tcloud2 = c2.makeShared();
    // pcl::PointCloud<pcl::PointXYZ>::Ptr tcloud3 = c3.makeShared();
    // pcl::PointCloud<pcl::PointXYZ>::Ptr tcloud4 = c4.makeShared();
    // auto prepend = timestamp() - prep;
    // printf("Conversion\t%s ms\n", std::to_string(prepend.count()).c_str());

    // Transform
    // auto transform = timestamp();
    pcl::transformPointCloud(*cloud2, *tcloud2, T2);
    // pcl::transformPointCloud(*cloud3, *tcloud3, T3);
    // pcl::transformPointCloud(*cloud4, *tcloud4, T4);
    // auto transformend = timestamp() - transform;
    // printf("Transform\t%s ms\n", std::to_string(transformend.count()).c_str());

    // Passthrough (NaN and extra points) filter
    // auto removeextra = timestamp();
    // box_filter(cloud1, std::string("random1.ply"));
    box_filter(tcloud2);
    // box_filter(tcloud3, std::string("random3.ply"));
    // box_filter(tcloud4, std::string("random4.ply"));
    // auto removeextraend = timestamp() - removeextra;
    // printf("Removing NaN and extra points\t%s ms\n", std::to_string(removeextraend.count()).c_str());

    

    // pcl::PLYWriter plywriter;
    // plywriter.write<pcl::PointXYZRGBNormal>("cloud1.ply", *cloud1normal, true);
    // plywriter.write<pcl::PointXYZ>("cloud2.ply", *tcloud2, true);
    // plywriter.write<pcl::PointXYZRGBNormal>("cloud3.ply", *cloud3normal, true);
    // plywriter.write<pcl::PointXYZRGBNormal>("cloud4.ply", *cloud4normal, true);

    // auto segmenttime = timestamp();

    // // segment("cloud1.ply");
    segment(tcloud2,dat,pub);
    // segment("cloud3.ply");
    // // segment("cloud4.ply");

    auto segmenttimeend = timestamp() - prep;
    printf("Segmentation and saving\t%s ms\n", std::to_string(segmenttimeend.count()).c_str());

 

    count++;
}

void process_clouds(std::string const& filename)
{
    // std::string txtfilename = filename;
    // txtfilename.erase(txtfilename.end()-4,txtfilename.end());
    // txtfilename = txtfilename + ".txt";
    // rosbag::Bag bag;
    // bag.open(filename, rosbag::bagmode::Read);
    const std::string base = "/kinect";
    const std::string base_end = "/depth_registered/points";
    // const auto cam1 = base + "1" + base_end;
    const auto cam2 = base + "3" + base_end;
    // const auto cam3 = base + "3" + base_end;
    // const auto cam4 = base + "4" + base_end;

    // std::vector<std::string> topics;
    // topics.emplace_back(cam1);
    // topics.emplace_back(cam2);
   

    // rosbag::View view(bag, rosbag::TopicQuery(topics));
    // BagSubscriber<sensor_msgs::PointCloud2> sub1, sub2, sub3, sub4;

  
    int t1, t2, t3, t4;
    t1=0;
    t2=0;
    t3=0;
    t4=0;
    ros::NodeHandle n;
    ros::Publisher pub = n.advertise<std_msgs::Int16MultiArray>("matrix_pub", 1);
    ros::Rate loop_rate(100);
    std_msgs::Int16MultiArray dat;
    dat.layout.dim.push_back(std_msgs::MultiArrayDimension());
    dat.layout.dim.push_back(std_msgs::MultiArrayDimension());
    dat.layout.dim.push_back(std_msgs::MultiArrayDimension());
    dat.layout.dim[0].label = "length";
    dat.layout.dim[1].label = "width";
    dat.layout.dim[2].label = "height";
    dat.layout.dim[0].size = 64;
    dat.layout.dim[1].size = 64;
    dat.layout.dim[2].size = 64;
    dat.layout.dim[0].stride = 64*64*64;
    dat.layout.dim[1].stride = 64*64;
    dat.layout.dim[2].stride = 64;
    dat.layout.data_offset = 0;

    ros::Subscriber sub = n.subscribe<sensor_msgs::PointCloud2> (cam2,1,boost::bind(callback,_1,dat,pub));

    // for (auto const& m : view) {
        

    //     if (m.getTopic() == cam2 || ("/" + m.getTopic() == cam2))
    //     {
    //         sensor_msgs::PointCloud2ConstPtr t = m.instantiate<sensor_msgs::PointCloud2>();
    //         if (t != NULL)
    //             // camera2.insert({t->header.stamp.toSec(),t2});
    //             // sub2.newMessage(t);
    //             callback(t,dat, pub);
    //             t2++;
    //     }


    // }
   



    // bag.close();
    printf("Done! Processed: %i samples\n", count);
}