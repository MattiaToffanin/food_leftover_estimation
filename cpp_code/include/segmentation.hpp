#ifndef SEGMENTATION_HPP
#define SEGMENTATION_HPP



#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>



cv::Mat kmeansData(const cv::Mat img);																		//create feature vectors for each pixel
std::vector<cv::Vec3b> getHsvColors(const cv::Mat centers);													//get hsv color from kmeans cluster centers
cv::Mat clusterize(const cv::Mat img, const std::vector<int> labels, const std::vector<cv::Vec3b> colors);	//clusterize the image using kmeans cluster centers
std::vector<cv::Mat> kmeansClustering(const cv::Mat img, const int num_clusters);							//compute kmeans algorithm and clusterize an image 
cv::Vec3b findFood(const cv::Mat centers);																	//gives in output the color of food's cluster
cv::Mat segmentFood(const cv::Mat centers, const cv::Mat clust_img, cv::Mat img);							//segments the food, given plate clusterization	


#endif