#include "../include/find_food.hpp"


//define global parameters
const int MIN_RADIUS = 170;
const int MAX_RADIUS = 380;


//
// This function finds the dishes and returns a list of parameters for each found dish: center x and center y positions and radius r.
//
std::vector<cv::Vec3f> findDishes(const cv::Mat img) {



	std::vector<cv::Vec3f> circles;			//circles parameters
	cv::Mat gray, smoothed;




	/*--------define constant parameters--------*/



	//illumination and contras parameters
	const double contrast = 0.8, illumination = -10;

	//bilateral filter parameters init.
	const int d = 9; //Diameter of each pixel neighborhood 					
	const double sigmaCol = 10, sigmaSpace = 200;


	//HoughCircles parameters init.
	const double dp = 1.2;
	const double minDist = 450;
	const double param1 = 85, param2 = 90;




	/*--------- preprocess image--------*/



	//convert image in gray scale
	cvtColor(img, gray, cv::COLOR_BGR2GRAY);


	//image equalization
	equalizeHist(gray, gray);


	//image contrast(0.8) and illumination(-10) modification
	gray.convertTo(gray, -1, contrast, illumination);


	//smoothing and sharpening
	cv::bilateralFilter(gray, smoothed, d, sigmaCol, sigmaSpace);




	/*-----------find circles------------*/



	HoughCircles(smoothed, circles, cv::HOUGH_GRADIENT, dp, minDist, param1, param2, MIN_RADIUS, MAX_RADIUS);


	return circles;
}




std::vector<cv::Rect> getFoodRect(const std::vector<cv::Vec3f> dishes_set) {



	std::vector<cv::Rect> dishesRect;



	//convert circles in rectangles containing the circles
	for (cv::Vec3f dish : dishes_set) {


		//get the circle parameters
		float x = dish[0];
		float y = dish[1];
		float radius = dish[2];


		//define the rect 
		cv::Rect dishRect((int)x - radius, (int)y - radius, 2 * radius, 2 * radius);

		
		//save 
		dishesRect.push_back(dishRect);
	}


	return dishesRect;
}




std::vector<cv::Mat> getFoodMask(const std::vector<cv::Vec3f> dishes_set, const cv::Size size) {



	std::vector<cv::Mat> masks;



	//get the mask of each dish
	for (cv::Vec3f dish : dishes_set)
	{


		//Get dish parameters
		int x = round(dish[0]);
		int y = round(dish[1]);
		int r = round(dish[2]);


		//Get the dish mask
		cv::Mat mask(size, CV_8UC1, cv::Scalar(0));


		//draw circle the get the mask
		circle(mask, cv::Point(x, y), r, 1, -1);


		//save
		masks.push_back(mask);
	}


	return masks;
}




//remove bread circles inside dishes circles
std::vector<cv::Vec3f> refineBreadSelection(std::vector<cv::Vec3f> breadSet, cv::Mat img) {



	std::vector<cv::Vec3f> finalSet;



	//check if a given circle goes inside of a dish and neglet it
	for (cv::Vec3f parameters : breadSet) {
		if (img.at<cv::Vec3b>(parameters[1], parameters[0]) != cv::Vec3b(0, 0, 0)) {


			//save the parameters
			finalSet.push_back(parameters);
		}
	}

	return finalSet;

}




//expand the radius of the bread circles
void expandBreadSelection(std::vector<cv::Vec3f>& breadSet) {



	//expand the circles to get a largest area
	for (cv::Vec3f& circle : breadSet) {


		//expand the circle radius
		circle[2] = circle[2] + 50;
	}
}





//
// This function finds the possible bread circles and returns a list of parameters for each found circle: center x and center y positions and radius r.
//
std::vector<cv::Vec3f> findBread(cv::Mat masked_img) {




	/*------define constant parameters------*/



	//define parameters
	const int count = 40;
	const int distance = 120;
	const int max = 80;
	const int minrad = 99;
	const int maxrad = 157;




	/*-------define variables-------*/



	std::vector<cv::Vec3f> bread_set;				//set of circles parameters
	std::vector<cv::Vec3f> bred_set_refined;		//set of bread circles refined


	cv::Mat hsv;
	cv::Mat hsv_masked;
	cv::Mat h_channel, s_channel;
	cv::Mat h_channel_mask;




	/*-----image preprocessing-----*/



	//convert image to hsv
	cvtColor(masked_img, hsv, cv::COLOR_BGR2HSV);


	//consider the H channel of HSV image
	std::vector<cv::Mat> HSV;
	cv::split(hsv, HSV);
	h_channel = HSV[0];


	//treshold the H channel
	cv::threshold(h_channel, h_channel_mask, 0, 255, cv::THRESH_OTSU + cv::THRESH_BINARY_INV);


	//mask the hsv image
	hsv.copyTo(hsv_masked, h_channel_mask);


	//consider the S channel of masked HSV image 
	cv::split(hsv_masked, HSV);
	s_channel = HSV[1];


	//change illumination and contrast of the S channel
	s_channel.convertTo(s_channel, -1, 1.1, -20);


	//reduce noice and smooth
	cv::medianBlur(s_channel, s_channel, 5);




	/*------find circles------*/



	//find circles with hough transform					  
	HoughCircles(s_channel, bread_set, cv::HOUGH_GRADIENT, 1.5, distance, max, count, minrad, maxrad);


	//expand the radius of the circles
	expandBreadSelection(bread_set);


	//remove bread circles inside the dishes
	bred_set_refined = refineBreadSelection(bread_set, masked_img);


	return bred_set_refined;
}




cv::Mat removeDishes(cv::Mat img, std::vector<cv::Vec3f> dishesSet) {



	/*------define constant parameters-----*/


	const cv::Scalar COLOR_MASK(0, 0, 0);



	/*------initialize variables-------*/


	cv::Mat out = img.clone();

	

	/*------remove the dishes from the image------*/



	//mask out the dishes 
	for (cv::Vec3f dish : dishesSet) {


		//draw the dishes circles
		cv::circle(out, cv::Point(dish[0], dish[1]), dish[2], COLOR_MASK, -1);
	}


	return out;
}

