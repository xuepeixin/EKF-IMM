#include <cstdlib>
#include <iostream>
#include <memory>
#include <algorithm>
#include <queue>
#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>
// #include "vec_map/geometry.h"
// #include "vec_map/vec_map_ros.h"
// #include "vec_map/vec_map_navigation.h"
// #include "vec_map/vec_map_generator.h"
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>
#include <vec_map_cpp_msgs/GetPredictedTrajectory.h>
#include <ibeo_msgs/ObjectData2280.h>
#include <tf/transform_listener.h>
#include <nav_msgs/Path.h>
#include "EKF.h"
#include "IMM.h"
#include "ModelGenerator.h"
using namespace std;

ros::Publisher pub_trajectory_predicted;
ros::Publisher pub_obstacle_paths;
ros::Publisher pub_obstacle_paths_estimated;
static std::unique_ptr<tf::TransformListener> g_tf_listener;
ros::ServiceClient raw_map_client;
std::map<int, nav_msgs::Path>  g_obstacle_paths;
std::map<int, nav_msgs::Path>  g_obstacle_paths_estimated;
// std::map<int, CTRA> g_obstacle_models;
// std::map<int, std::shared_ptr<KFBase>> g_obstacle_models;
const bool IMM_MODE = false;
std::map<int, std::shared_ptr<KFBase>> g_obstacle_models;





visualization_msgs::MarkerArray getTrajVis(vec_map_cpp_msgs::GetPredictedTrajectoryResponse response, std_msgs::ColorRGBA color)
{
    visualization_msgs::MarkerArray markers;
    // visualization_msgs::Marker marker;
    // marker.id = 0;
    // marker.action = visualization_msgs::Marker::DELETEALL;
    // markers.markers.push_back(marker);
    static size_t index = 0;
    for (auto curve : response.paths)
    {
        visualization_msgs::Marker curve_marker;
        curve_marker.id = index;
        curve_marker.header = curve.header;
        curve_marker.type = visualization_msgs::Marker::LINE_STRIP;
        curve_marker.color = color;
        curve_marker.lifetime = ros::Duration(0.3);
        curve_marker.scale.x = 0.4;
        index += 1;
        for (auto curve_point : curve.points)
        {
            geometry_msgs::Point p;
            p.x = curve_point.x;
            p.y = curve_point.y;
            curve_marker.points.push_back(p);
        }
        markers.markers.push_back(curve_marker);
    }
    return markers;
}


void generatePathFromState(
    const std::shared_ptr<KFBase>& model, 
    const double& predict_time, 
    path_planning_msgs::Curve* path_ptr) {
    
    std::shared_ptr<KFBase> model_copy = std::shared_ptr<KFBase>(model->clone());
    const double t0 = model_copy->stamp();
    const double dt = 0.1;
    auto state = model_copy->x();
    if (state(4) > 1 || state(4) < -1 || state(5) > 10 || state(5) < -10) {
        state(4) = 0;
        state(5) = 0;
        model_copy->setState(state);
    }
    while (model_copy->stamp() < t0 + predict_time) {
        path_planning_msgs::CurvePoint point;
        auto state = model_copy->x();
        point.x = state(0);
        point.y = state(1);
        point.theta = state(2);
        point.velocity = state(3);
        point.time = model_copy->stamp();

        path_ptr->points.push_back(point);
        model_copy->updateOnce(model_copy->stamp() + dt);
    }
}

void generatePathFromState(
    const std::shared_ptr<IMM>& model, 
    const double& predict_time, 
    path_planning_msgs::Curve* path_ptr) {
    
    std::shared_ptr<IMM> model_copy = std::shared_ptr<IMM>(model->clone());
    const double t0 = model_copy->stamp();
    const double dt = 0.1;
    while (model_copy->stamp() < t0 + predict_time) {
        path_planning_msgs::CurvePoint point;
        auto state = model_copy->x();

        point.x = state(0);
        point.y = state(1);
        point.theta = atan2(state(3), state(2));
        point.velocity = sqrt(state(2) * state(2) + state(3) * state(3));
        point.time = model_copy->stamp();
        path_ptr->points.push_back(point);
        model_copy->updateOnce(model_copy->stamp() + dt);
    }
}

void updatePathsEstimated(const geometry_msgs::PoseStamped& geo_pose, const ibeo_msgs::Object2280& object) {
    double speed = std::sqrt(object.absolute_velocity.x * object.absolute_velocity.x + object.absolute_velocity.y * object.absolute_velocity.y);
    double vx = speed * std::cos(tf::getYaw(geo_pose.pose.orientation));
    double vy = speed * std::sin(tf::getYaw(geo_pose.pose.orientation));
    
    Eigen::VectorXd z;
    Eigen::VectorXd x;
    z.resize(4);
    x.resize(6);
    if (IMM_MODE == true) {
        z << geo_pose.pose.position.x, geo_pose.pose.position.y, vx, vy;
        x << geo_pose.pose.position.x, geo_pose.pose.position.y, vx, vy, 0, 0;
    } else {
        z << geo_pose.pose.position.x, geo_pose.pose.position.y, tf::getYaw(geo_pose.pose.orientation), std::sqrt(object.absolute_velocity.x * object.absolute_velocity.x + object.absolute_velocity.y * object.absolute_velocity.y);
        x << geo_pose.pose.position.x, geo_pose.pose.position.y, tf::getYaw(geo_pose.pose.orientation), std::sqrt(object.absolute_velocity.x * object.absolute_velocity.x + object.absolute_velocity.y * object.absolute_velocity.y), 0, 0;
    
    }
        
    if (g_obstacle_paths_estimated.find(object.id) == g_obstacle_paths_estimated.end()) {  
        // auto model = ModelGenerator::generateIMMModel(0.04, x);
        // auto model = ModelGenerator::generateIMMModel(geo_pose.header.stamp.toSec(), x);
        auto model = ModelGenerator::generateCTRAModel(geo_pose.header.stamp.toSec(), x);


        g_obstacle_models[object.id] = model;
        nav_msgs::Path path;
        path.poses.push_back(geo_pose);
        path.header = geo_pose.header;
        g_obstacle_paths_estimated[object.id] = path;
    } else {
        const geometry_msgs::PoseStamped& pre_pose = *(g_obstacle_paths_estimated[object.id].poses.end()-1);
        if (std::sqrt(pow(geo_pose.pose.position.x - pre_pose.pose.position.x, 2) + pow(geo_pose.pose.position.y - pre_pose.pose.position.y, 2)) < 5) {
            
            g_obstacle_models[object.id]->updateOnce(geo_pose.header.stamp.toSec(), &z);

            // g_obstacle_models[object.id]->predict(geo_pose.header.stamp.toSec());
            // g_obstacle_models[object.id]->update(z);

            geometry_msgs::PoseStamped ekf_pose = geo_pose;
            Eigen::VectorXd ekf_state = g_obstacle_models[object.id]->x();
            ekf_pose.pose.position.x = ekf_state(0);
            ekf_pose.pose.position.y = ekf_state(1);
            tf::Quaternion q;
            if (IMM_MODE == true) {
                q = tf::createQuaternionFromYaw(atan2(ekf_state(3),ekf_state(2)));
            } else {
                q = tf::createQuaternionFromYaw(ekf_state(2));
            }
            ekf_pose.pose.orientation.w = q.w();
            ekf_pose.pose.orientation.x = q.x();
            ekf_pose.pose.orientation.y = q.y();
            ekf_pose.pose.orientation.z = q.z();
  
            g_obstacle_paths_estimated[object.id].poses.push_back(ekf_pose);
            g_obstacle_paths_estimated[object.id].header = ekf_pose.header;
            if (g_obstacle_paths_estimated[object.id].poses.size() > 100) {
                g_obstacle_paths_estimated[object.id].poses.erase(g_obstacle_paths_estimated[object.id].poses.begin());
            }
        } 
 
    }
    // std::cout << g_obstacle_paths_estimated.size() << std::endl;
}

void updatePaths(const geometry_msgs::PoseStamped& geo_pose, const int& id) {
    if (g_obstacle_paths.find(id) == g_obstacle_paths.end()) {
        nav_msgs::Path path;
        path.poses.push_back(geo_pose);
        path.header = geo_pose.header;
        g_obstacle_paths[id] = path;
    } else {
        const geometry_msgs::PoseStamped& pre_pose = *(g_obstacle_paths[id].poses.end()-1);
        if (std::sqrt(pow(geo_pose.pose.position.x - pre_pose.pose.position.x, 2) + pow(geo_pose.pose.position.y - pre_pose.pose.position.y, 2)) < 5) {
            g_obstacle_paths[id].poses.push_back(geo_pose);
            g_obstacle_paths[id].header = geo_pose.header;

            if (g_obstacle_paths[id].poses.size() > 100) {
                g_obstacle_paths[id].poses.erase(g_obstacle_paths[id].poses.begin());
            }
        }

    }
    // std::cout << g_obstacle_paths.size() << std::endl;
}



void objCallback(ibeo_msgs::ObjectData2280ConstPtr msgs)
{
    tf::StampedTransform transformer;
    try
    {
        // g_tf_listener->waitForTransform("world", msgs->header.frame_id, ros::Time(0), ros::Duration(5));
        // g_tf_listener->lookupTransform("world", msgs->header.frame_id, ros::Time(0), transformer);
        g_tf_listener->waitForTransform("world", msgs->header.frame_id, msgs->header.stamp, ros::Duration(5));
        g_tf_listener->lookupTransform("world", msgs->header.frame_id, msgs->header.stamp, transformer);
    }
    catch (tf::TransformException &e)
    {
        ROS_ERROR("%s", e.what());
    }

    ros::Time t = ros::Time::now();
    for (auto object : msgs->objects)
    {
        
        // ros::Time t0 = ros::Time::now();
        if (object.tracking_model == ibeo_msgs::Object2280::STATIC_MODEL)
            continue;
        vec_map_cpp_msgs::GetPredictedTrajectory srv;

        tf::Pose pose;
        pose.setOrigin(tf::Vector3(object.object_box_center.x, object.object_box_center.y, 0));
        pose.setRotation(tf::createQuaternionFromYaw(atan2(object.absolute_velocity.y, object.absolute_velocity.x)));

        geometry_msgs::PoseStamped geo_pose;
        tf::poseTFToMsg(pose, geo_pose.pose);
        geo_pose.header = msgs->header;
        try
        {
            g_tf_listener->transformPose("world", geo_pose, geo_pose);
        }
        catch (tf::TransformException ex)
        {
            ROS_WARN("transfrom exception : %s", ex.what());
            return;
        }

        updatePathsEstimated(geo_pose, object);
        updatePaths(geo_pose, object.id);

        // ROS_INFO_STREAM("ekf cost: " << ros::Time::now() - t0);    
        srv.request.current_pose.header = msgs->header;
        srv.request.current_pose.header.frame_id = "world";
        
        srv.request.current_pose = *(g_obstacle_paths_estimated[object.id].poses.end()-1);
        Eigen::VectorXd x = g_obstacle_models[object.id]->x();
        generatePathFromState(g_obstacle_models[object.id], 4.0, &(srv.request.path));
        if (IMM_MODE == true) {
            srv.request.speed = sqrt(x(2) * x(2) + x(3) * x(3));
            srv.request.accelaration = sqrt(x(4) * x(4) + x(5) * x(5));
        } else {
            srv.request.speed = x(3);
            srv.request.accelaration = x(5);
        }
        srv.request.yaw_rate = 0;
        srv.request.point_margin = 0.5;
        srv.request.request_length = 80.0;                                     // request_length
        srv.request.speed_orientation = tf::getYaw(geo_pose.pose.orientation); // speed_orientation
        ros::Time t1 = ros::Time::now();
        if (raw_map_client.call(srv))
        {
            visualization_msgs::MarkerArray markers;
            std_msgs::ColorRGBA rgba;
            rgba.a = 1.0;
            if (srv.response.state == vec_map_cpp_msgs::GetPredictedTrajectoryResponse::HIGH_UNCERTAINTY)
            {
                rgba.r = 1.0;
                markers = getTrajVis(srv.response, rgba);
                // std::cout << "high uncertainty." << std::endl;
            }
            else if (srv.response.state == vec_map_cpp_msgs::GetPredictedTrajectoryResponse::NORMAL)
            {
                rgba.b = 1.0;
                markers = getTrajVis(srv.response, rgba);
                // std::cout << "normal." << std::endl;
            }
            else
            {
                // std::cout << "out map" << std::endl;
            }
            pub_trajectory_predicted.publish(markers);
        }
        else
        {
            ROS_ERROR("Failed to call service vec_map_server_node");
        }

        // ROS_INFO_STREAM("predict cost: " << ros::Time::now() - t1);
    }

    std::map<int, nav_msgs::Path>::iterator it = g_obstacle_paths.begin();
    while (it != g_obstacle_paths.end()) {
        if (msgs->header.stamp - it->second.header.stamp > ros::Duration(0.2)) {
            g_obstacle_paths.erase(it++);
            continue;
        }
        it++;
    }
    it = g_obstacle_paths_estimated.begin();
    while (it != g_obstacle_paths_estimated.end()) {
        if (msgs->header.stamp - it->second.header.stamp > ros::Duration(0.2)) {
            g_obstacle_paths_estimated.erase(it++);
            continue;
        }
        it++;
    }

    for (auto path : g_obstacle_paths) {
        pub_obstacle_paths.publish(path.second);
    }
    for (auto path : g_obstacle_paths_estimated) {
        pub_obstacle_paths_estimated.publish(path.second);
    }
    // ROS_INFO_STREAM("total cost: " << ros::Time::now() - t << "\n");

}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "prediction_viewer_node");
    ros::NodeHandle nh;
    g_tf_listener.reset(new tf::TransformListener());
    pub_trajectory_predicted = nh.advertise<visualization_msgs::MarkerArray>("/obstacle/path_vis", 10, true);
    pub_obstacle_paths = nh.advertise<nav_msgs::Path>("/obstacle/paths", 10, true);
    pub_obstacle_paths_estimated = nh.advertise<nav_msgs::Path>("/obstacle/paths_estimated", 10, true);
    raw_map_client = nh.serviceClient<vec_map_cpp_msgs::GetPredictedTrajectory>("get_predicted_trajectory");
    ros::Subscriber object_sub = nh.subscribe<ibeo_msgs::ObjectData2280>("ibeo/ObjectData2280", 1, objCallback);
    ros::spin();
    return 0;
}
