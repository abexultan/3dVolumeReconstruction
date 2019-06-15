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

#include <ros/ros.h>

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


void addNormal(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_with_normals)
{
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr searchTree(new pcl::search::KdTree<pcl::PointXYZ>);

    searchTree->setInputCloud(cloud);

    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimator;
    normalEstimator.setInputCloud(cloud);
    normalEstimator.setSearchMethod(searchTree);
    normalEstimator.setKSearch(15);
    normalEstimator.compute(*normals);

    pcl::concatenateFields(*cloud, *normals, *cloud_with_normals);
}

auto nonRigidICP(std::string const& first, std::string const& second)
{
    cilantro::PointCloud3f dst(first), src(second);
    // Compute a sparsely supported warp field (compute transformations for a sparse set of control nodes)
    // Neighborhood parameters
    float const control_res = 0.025f;
    float const src_to_control_sigma = 0.5f * control_res;
    float const regularization_sigma = 3.0f * control_res;
    float const max_correspondence_dist_sq = 0.02f*0.02f;

    // Get a sparse set of control nodes by downsampling
    cilantro::VectorSet<float,3> control_points = cilantro::PointsGridDownsampler3f(src.points, control_res).getDownsampledPoints();
    cilantro::KDTree<float,3> control_tree(control_points);

    // Find which control nodes affect each point in src
    std::vector<cilantro::NeighborSet<float>> src_to_control_nn;
    control_tree.search(src.points, cilantro::kNNNeighborhood<float>(4), src_to_control_nn);

    // Get regularization neighborhoods for control nodes
    std::vector<cilantro::NeighborSet<float>> regularization_nn;
    control_tree.search(control_points, cilantro::kNNNeighborhood<float>(8), regularization_nn);

    // Perform ICP registration
    cilantro::Timer timer;
    timer.start();
    cilantro::SimpleSparseCombinedMetricNonRigidICP3f icp(dst.points, dst.normals, src.points, src_to_control_nn, control_points.cols(), regularization_nn);

    // Parameter setting
    icp.correspondenceSearchEngine().setMaxDistance(max_correspondence_dist_sq);
    icp.controlWeightEvaluator().setSigma(src_to_control_sigma);
    icp.regularizationWeightEvaluator().setSigma(regularization_sigma);
    icp.setMaxNumberOfIterations(20).setConvergenceTolerance(2.5e-3f);
    icp.setMaxNumberOfGaussNewtonIterations(1).setGaussNewtonConvergenceTolerance(5e-4f);
    icp.setMaxNumberOfConjugateGradientIterations(500).setConjugateGradientConvergenceTolerance(1e-5f);
    icp.setPointToPointMetricWeight(0.0f).setPointToPlaneMetricWeight(1.0f).setStiffnessRegularizationWeight(200.0f);
    icp.setHuberLossBoundary(1e-2f);

    auto tf_est = icp.estimateTransformation().getPointTransformations();
    timer.stop();

    std::cout << "Registration time: " << timer.getElapsedTime() << "ms" << std::endl;
    std::cout << "Iterations performed: " << icp.getNumberOfPerformedIterations() << std::endl;
    std::cout << "Has converged: " << icp.hasConverged() << std::endl;

    return src.transformed(tf_est);
}

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

void box_filter(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, std::string const& filename)
{
    printf("Total\t%ld\n", cloud->size());

    cloud->erase(std::remove_if(cloud->begin(), cloud->end(),
            [](auto const& x)
            {
                return !inside(x);
            }), cloud->end());
    printf("After\t%ld\n", cloud->size());
    pcl::PLYWriter plywriter;
    plywriter.write<pcl::PointXYZ>(filename, *cloud, false);
}

void segment(std::string const& filename)
{
  std::string btfilename = filename;
  btfilename.erase(btfilename.end()-4,btfilename.end());
  btfilename = btfilename + "_";
  pcl::PLYReader reader;
  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGBNormal>),  cloud_f (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
  reader.read (filename, *cloud);//
  
  std::cout << "PointCloud before filtering has: " << cloud->points.size () << " data points." << std::endl; //*

  // Create the filtering object: downsample the dataset using a leaf size of 1cm
  pcl::VoxelGrid<pcl::PointXYZRGBNormal> vg;
  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
  vg.setInputCloud (cloud);
  vg.setLeafSize (0.02f, 0.02f, 0.02f);
  vg.filter (*cloud_filtered);
  std::cout << "PointCloud after filtering has: " << cloud_filtered->points.size ()  << " data points." << std::endl; //*

  // Create the segmentation object for the planar model and set all the parameters
  pcl::SACSegmentation<pcl::PointXYZRGBNormal> seg;
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
  pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_plane (new pcl::PointCloud<pcl::PointXYZRGBNormal> ());
  pcl::PLYWriter plywriter;
  seg.setOptimizeCoefficients (true);
  seg.setModelType (pcl::SACMODEL_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setMaxIterations (100);
  seg.setDistanceThreshold (0.02);

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
    pcl::ExtractIndices<pcl::PointXYZRGBNormal> extract;
    extract.setInputCloud (cloud_filtered);
    extract.setIndices (inliers);
    extract.setNegative (false);

    // Get the points associated with the planar surface
    extract.filter (*cloud_plane);
    std::cout << "PointCloud representing the planar component: " << cloud_plane->points.size () << " data points." << std::endl;

    // Remove the planar inliers, extract the rest
    extract.setNegative (true);
    extract.filter (*cloud_f);
    *cloud_filtered = *cloud_f;
  }

  // Creating the KdTree object for the search method of the extraction
  pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr tree1 (new pcl::search::KdTree<pcl::PointXYZRGBNormal>);
  tree1->setInputCloud (cloud_filtered);

  std::vector<pcl::PointIndices> cluster_indices;
  pcl::EuclideanClusterExtraction<pcl::PointXYZRGBNormal> ec;
  ec.setClusterTolerance (0.04); // 2cm
  ec.setMinClusterSize (1000);
  ec.setMaxClusterSize (30000);
  ec.setSearchMethod (tree1);
  ec.setInputCloud (cloud_filtered);
  ec.extract (cluster_indices);

  int j = 0;
  for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
  {
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZRGBNormal>);
    for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
      cloud_cluster->points.push_back (cloud_filtered->points[*pit]); //*
    cloud_cluster->width = cloud_cluster->points.size ();
    cloud_cluster->height = 1;
    cloud_cluster->is_dense = true;

    std::cout << "PointCloud representing the Cluster: " << cloud_cluster->points.size () << " data points." << std::endl;
    std::stringstream ss;
    // ss << "cloud_cluster_" << j << ".ply";
    plywriter.write<pcl::PointXYZRGBNormal> (btfilename+std::to_string(count) +std::to_string(j)+".ply", *cloud_cluster, false); //*
    j++;
  }
  // pcl::PointCloud<pcl::PointXYZRGBNormal> save;
  //   pcl::io::loadPCDFile(std::string("cloud_cluster_0.pcd"), save);
  //   octomap::OcTree tree(0.03);
  //   for (auto const& p : save.points)
  //   {
  //       tree.updateNode(octomap::point3d(p.x, p.y, p.z), true);
  //   }
  //   tree.updateInnerOccupancy();
  //   tree.writeBinary(btfilename+std::to_string(count) + ".bt");

}

void callback(sensor_msgs::PointCloud2ConstPtr const& pcloud1,
              sensor_msgs::PointCloud2ConstPtr const& pcloud2,
              sensor_msgs::PointCloud2ConstPtr const& pcloud3,
              sensor_msgs::PointCloud2ConstPtr const& pcloud4,
              std::vector<double> *kinect1,
              std::vector<double> *kinect2,
              std::vector<double> *kinect3,
              std::vector<double> *kinect4)
{
    printf("--------------------------------------------\n");
    printf("Cloud number %i\n", count);
    printf("--------------------------------------------\n");
    // std::cout << "camera 1 " << pcloud1->header.stamp << std::endl;
    // std::cout << "camera 2 " << pcloud2->header.stamp << std::endl;
    // std::cout << "camera 3 " << pcloud3->header.stamp << std::endl;
    // std::cout << "camera 4 " << pcloud4->header.stamp << std::endl;
    // std::cout << "camera 4 " << pcloud4->header.stamp.toSec() << std::endl;
    // Prepare and convert
    kinect1->push_back(pcloud1->header.stamp.toSec());
    kinect2->push_back(pcloud2->header.stamp.toSec());
    kinect3->push_back(pcloud3->header.stamp.toSec());
    kinect4->push_back(pcloud4->header.stamp.toSec());
    //std::cout << "kinect " << kinect1->size() << std::endl;
    auto prep = timestamp();
    pcl::PointCloud<pcl::PointXYZ> clouda, cloudb, cloudc, cloudd;
    pcl::fromROSMsg(*pcloud1, clouda);
    pcl::fromROSMsg(*pcloud2, cloudb);
    pcl::fromROSMsg(*pcloud3, cloudc);
    pcl::fromROSMsg(*pcloud4, cloudd);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1 = clouda.makeShared();
    pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud2 = cloudb.makeShared();
    pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud3 = cloudc.makeShared();
    pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud4 = cloudd.makeShared();

    // Transform prep
    pcl::PointCloud<pcl::PointXYZ> c2, c3, c4;
    pcl::PointCloud<pcl::PointXYZ>::Ptr tcloud2 = c2.makeShared();
    pcl::PointCloud<pcl::PointXYZ>::Ptr tcloud3 = c3.makeShared();
    pcl::PointCloud<pcl::PointXYZ>::Ptr tcloud4 = c4.makeShared();
    auto prepend = timestamp() - prep;
    printf("Conversion\t%s ms\n", std::to_string(prepend.count()).c_str());

    // Transform
    auto transform = timestamp();
    pcl::transformPointCloud(*cloud2, *tcloud2, T2);
    pcl::transformPointCloud(*cloud3, *tcloud3, T3);
    pcl::transformPointCloud(*cloud4, *tcloud4, T4);
    auto transformend = timestamp() - transform;
    printf("Transform\t%s ms\n", std::to_string(transformend.count()).c_str());

    // Passthrough (NaN and extra points) filter
    auto removeextra = timestamp();
    box_filter(cloud1, std::string("random1.ply"));
    box_filter(tcloud2, std::string("random2.ply"));
    box_filter(tcloud3, std::string("random3.ply"));
    box_filter(tcloud4, std::string("random4.ply"));
    auto removeextraend = timestamp() - removeextra;
    printf("Removing NaN and extra points\t%s ms\n", std::to_string(removeextraend.count()).c_str());

    // Removing outliers
    auto removeoutliers = timestamp();
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(cloud1);
    sor.setMeanK(50);
    sor.setStddevMulThresh(1.0);

    sor.filter(*cloud1); // *cloud_filtered
    sor.setInputCloud(tcloud2);
    sor.filter(*tcloud2);
    sor.setInputCloud(tcloud3);
    sor.filter(*tcloud3);
    sor.setInputCloud(tcloud4);
    sor.filter(*tcloud4);
    auto removeoutliersend = timestamp() - removeoutliers;
    printf("Removing outliers\t%s ms\n", std::to_string(removeoutliersend.count()).c_str());

    pcl::PointCloud<pcl::PointXYZRGBNormal> cloud1norm, cloud2norm, cloud3norm, cloud4norm;
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud1normal = cloud1norm.makeShared();
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud2normal = cloud2norm.makeShared();
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud3normal = cloud3norm.makeShared();
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud4normal = cloud4norm.makeShared();

    auto normaltime = timestamp();

    addNormal(cloud1, cloud1normal);
    addNormal(tcloud2, cloud2normal);
    addNormal(tcloud3, cloud3normal);
    addNormal(tcloud4, cloud4normal);

    auto normaltimeend = timestamp() - normaltime;
    printf("Computing normals\t%s ms\n", std::to_string(normaltimeend.count()).c_str());

    pcl::PLYWriter plywriter;
    plywriter.write<pcl::PointXYZRGBNormal>("cloud1.ply", *cloud1normal, true);
    plywriter.write<pcl::PointXYZRGBNormal>("cloud2.ply", *cloud2normal, true);
    plywriter.write<pcl::PointXYZRGBNormal>("cloud3.ply", *cloud3normal, true);
    plywriter.write<pcl::PointXYZRGBNormal>("cloud4.ply", *cloud4normal, true);

    auto segmenttime = timestamp();

    segment("cloud1.ply");
    segment("cloud2.ply");
    segment("cloud3.ply");
    segment("cloud4.ply");

    auto segmenttimeend = timestamp() - segmenttime;
    printf("Segmentation and saving\t%s ms\n", std::to_string(segmenttimeend.count()).c_str());

    // Saving
    auto icptime = timestamp();

    auto warped = nonRigidICP("cloud1.ply", "cloud2.ply");
    cilantro::PointCloud3f dst("cloud1.ply");
    dst.append(warped);
    dst.toPLYFile("cloud_merged.ply");

    warped = nonRigidICP("cloud_merged.ply", "cloud3.ply");
    dst.append(warped);
    dst.toPLYFile("cloud_merged.ply");

    warped = nonRigidICP("cloud_merged.ply", "cloud4.ply");
    dst.append(warped);
    dst.toPLYFile("cloud_merged.ply");

    segment("cloud_merged.ply");

    count++;
}

void process_clouds(std::string const& filename)
{
    std::string txtfilename = filename;
    txtfilename.erase(txtfilename.end()-4,txtfilename.end());
    txtfilename = txtfilename + ".txt";
    rosbag::Bag bag;
    bag.open(filename, rosbag::bagmode::Read);
    const std::string base = "/kinect";
    const std::string base_end = "/depth_registered/points";
    const auto cam1 = base + "1" + base_end;
    const auto cam2 = base + "2" + base_end;
    const auto cam3 = base + "3" + base_end;
    const auto cam4 = base + "4" + base_end;

    std::vector<std::string> topics;
    topics.emplace_back(cam1);
    topics.emplace_back(cam2);
    topics.emplace_back(cam3);
    topics.emplace_back(cam4);

    std::unordered_map<double,uint32_t> camera1;
    std::unordered_map<double,uint32_t> camera2;
    std::unordered_map<double,uint32_t> camera3;
    std::unordered_map<double,uint32_t> camera4;
    std::vector<double> kinect1;
    std::vector<double> kinect2;
    std::vector<double> kinect3;
    std::vector<double> kinect4;


    rosbag::View view(bag, rosbag::TopicQuery(topics));
    BagSubscriber<sensor_msgs::PointCloud2> sub1, sub2, sub3, sub4;

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2,
    sensor_msgs::PointCloud2,
    sensor_msgs::PointCloud2,
    sensor_msgs::PointCloud2> MySyncPolicy;
    message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), sub1, sub2, sub3, sub4);
    sync.registerCallback(boost::bind(&callback, _1, _2, _3, _4, &kinect1, &kinect2, &kinect3, &kinect4));
    int t1, t2, t3, t4;
    t1=0;
    t2=0;
    t3=0;
    t4=0;
    camera1.clear();
    camera2.clear();
    camera3.clear();
    camera4.clear();
    kinect1.clear();
    kinect2.clear();
    kinect3.clear();
    kinect4.clear();

    for (auto const& m : view) {
        if (m.getTopic() == cam1 || ("/" + m.getTopic() == cam1))
        {
            sensor_msgs::PointCloud2ConstPtr t = m.instantiate<sensor_msgs::PointCloud2>();
            if (t != NULL)
                camera1.insert({t->header.stamp.toSec(),t1});
                sub1.newMessage(t);
                t1++;
        }

        if (m.getTopic() == cam2 || ("/" + m.getTopic() == cam2))
        {
            sensor_msgs::PointCloud2ConstPtr t = m.instantiate<sensor_msgs::PointCloud2>();
            if (t != NULL)
                camera2.insert({t->header.stamp.toSec(),t2});
                sub2.newMessage(t);
                t2++;
        }

        if (m.getTopic() == cam3 || ("/" + m.getTopic() == cam3))
        {
            sensor_msgs::PointCloud2ConstPtr t = m.instantiate<sensor_msgs::PointCloud2>();
            if (t != NULL)
                camera3.insert({t->header.stamp.toSec(),t3});
                t3++;
                sub3.newMessage(t);
        }

        if (m.getTopic() == cam4 || ("/" + m.getTopic() == cam4))
        {
            sensor_msgs::PointCloud2ConstPtr t = m.instantiate<sensor_msgs::PointCloud2>();
            if (t != NULL)
                camera4.insert({t->header.stamp.toSec(),t4});
                t4++;
                sub4.newMessage(t);
        }
    }
    for (auto const& m: kinect1){
        camera1.erase(m);
    }
    for (auto const& m: kinect2){
        camera2.erase(m);
    }
    for (auto const& m: kinect3){
        camera3.erase(m);
    }
    for (auto const& m: kinect4){
        camera4.erase(m);
    } 

    std::ofstream outfile (txtfilename);
    outfile.precision(20);

    outfile << "camera1 last index " << t1-1 << std::endl;
    for (auto const& m: camera1){
        outfile << "camera1 timestamp: " << m.first << "; index:" << m.second << std::endl;
    }
    outfile << "camera2 last index " << t2-1 << std::endl;
    for (auto const& m: camera2){
        outfile << "camera2 timestamp: " << m.first << "; index:" << m.second << std::endl;
    }
    outfile << "camera3 last index " << t3-1 << std::endl;
    for (auto const& m: camera3){
        outfile << "camera3 timestamp: " << m.first << "; index:" << m.second << std::endl;
    }
    outfile << "camera4 last index " << t4-1 << std::endl;
    for (auto const& m: camera4){
        outfile << "camera4 timestamp: " << m.first << "; index:" << m.second << std::endl;
    }




    bag.close();
    printf("Done! Processed: %i samples\n", count);
}