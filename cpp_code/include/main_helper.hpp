#ifndef MAIN_HEADER_HPP
#define MAIN_HEADER_HPP


#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


void estimateFoodLeftovers(std::string tray_image_dir, std::string leftover_image_dir);
std::vector<std::vector<int>> parseLine(std::string line);

#endif