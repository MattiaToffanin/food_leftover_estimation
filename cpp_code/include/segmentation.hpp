#ifndef SEGMENTATION_HPP
#define SEGMENTATION_HPP



#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>



cv::Mat kmeansData(const cv::Mat img, const unsigned char selection);																		//create feature vectors for each pixel
std::vector<cv::Vec3b> getHsvColors(const cv::Mat centers);													//get hsv color from kmeans cluster centers
cv::Mat clusterize(const cv::Mat img, const std::vector<int> labels, const std::vector<cv::Vec3b> colors);	//clusterize the image using kmeans cluster centers
std::vector<cv::Mat> kmeansClustering(const cv::Mat img, const int num_clusters, const unsigned char selection );							//compute kmeans algorithm and clusterize an image
cv::Vec3b getClusterFoodColor(const cv::Mat centers, const unsigned char selection);  											//gives in output the color of food's cluster
cv::Mat segmentFood(const cv::Mat img, const cv::Mat clust_img, const cv::Mat centers, const unsigned char selection);							//segments the food, given plate clusterization
cv::Mat getFoodMask(cv::Mat img, const unsigned char selection);


#endif