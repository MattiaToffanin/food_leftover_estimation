#ifndef FIND_FOOD_HPP
#define FIND_FOOD_HPP



#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>



std::vector<cv::Vec3f> findDishes(const cv::Mat img);												//find dishes and return the circles parameters: center x, center y and radius r
std::vector<cv::Rect> getFoodRect(const std::vector<cv::Vec3f> dishes_set);							//get Rectangle containing the dishes
std::vector<cv::Mat> getFoodMask(const std::vector<cv::Vec3f> dishes_set, const cv::Size size);		//get the mask of the dishes
std::vector<cv::Vec3f> findBread(cv::Mat masked_img);												//find possible bread locations
std::vector<cv::Vec3f> refineBreadSelection(std::vector<cv::Vec3f> breadSet, cv::Mat img);			//remove bread circles inside dishes circles
void expandBreadSelection(std::vector<cv::Vec3f>& breadSet);										//expande bread circles radius
cv::Mat removeDishes(cv::Mat img, std::vector<cv::Vec3f> dishesSet);


#endif