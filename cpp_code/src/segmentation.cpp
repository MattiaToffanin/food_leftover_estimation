#include "../include/segmentation.hpp"




//
// Create the dataset from given image and given number of feature
//
cv::Mat kmeansData(const cv::Mat img) {




	/*--------define constant paramenters--------*/



	//dimension of the feature vector
	const int num_features = 4;


	//define data type and dimension
	cv::Mat data(img.cols * img.rows, num_features, CV_32F);


	//priors 
	const cv::Vec3b DISH_COLOR(189, 34, 182); 	//average dish color in hsv
	const cv::Vec3b BACKGROUND_COL(0, 0, 0); 	//average background color in hsv




	/*--------create kmeans data matrix---------*/



	//iterate through all the image pixels to assign to each pixel its feature values
	int index = 0;
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {



			//get pixel color at current pixel position (i,j)
			cv::Vec3b color = img.at<cv::Vec3b>(i, j);




			//------Set the current data features--------



			// Color (H and S channels only)
			data.at<float>(index, 0) = color[0] / 180.0;
			data.at<float>(index, 1) = color[1] / 256.0;


			// Distance from reference colors normalized first by max values and then by optimal values
			data.at<float>(index, 2) = (float)norm(color, BACKGROUND_COL, cv::NORM_L2) / sqrt(pow(180, 2) + pow(256, 2) + pow(256, 2));
			data.at<float>(index, 3) = (float)norm(color, DISH_COLOR, cv::NORM_L2) / sqrt(pow(122, 2) + pow(236, 2) + pow(182, 2));


			//next line
			index++;
		}
	}

	return data;
}





//
// Get hsv colors from kmeans centers
//
std::vector<cv::Vec3b> getHsvColors(const cv::Mat centers) {



	/*-------define constant paramenter------*/


	const float V = 50; 	//define arbitrary V channel color



	/*-------define variables--------*/


	std::vector<cv::Vec3b> colors;
	


	/*-------get clusters color------*/


	for (int i = 0; i < centers.rows; i++) {


		//get colors and denormalize them
		float H = round(centers.at<float>(i, 0) * 180);
		float S = round(centers.at<float>(i, 1) * 256);


		//get the color
		cv::Vec3b color(H, S, V);


		//save
		colors.push_back(color);
	}


	return colors;
}





//
// clusterize the image based on kmeans clusters and labels
//
cv::Mat clusterize(const cv::Mat img, const std::vector<int> labels, const std::vector<cv::Vec3b> colors) {




	/*-------initialize variables------*/


	//define output image dimension and type
	cv::Mat out(img.size(), CV_8UC3, cv::Scalar(0, 0, 0));



	/*-------get clustered image-------*/


	//iterate through all the labels and assign each image pixel the corresponding cluster color
	for (int i = 0; i < labels.size(); i++) {



		//get the current pixel row and column
		int row = i / img.cols;
		int col = i % img.cols;



		//set the current pixel to its cluster color
		out.at<cv::Vec3b>(row, col) = colors[labels[i]];
	}


	return out;
}





//
// Clusterize an image using kmeans 
//
std::vector<cv::Mat> kmeansClustering(const cv::Mat img, const int num_clusters) {



	/*------define constant parameters-------*/


	//define kmeans parameters
	cv::TermCriteria tc(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1.0);
	const int attempts = 10;



	/*-------define variables------*/


	std::vector<int> labels;
	cv::Mat centers;




	/*-------clusterize image------*/



	//define data for kmeans clustering
	cv::Mat data = kmeansData(img);


	//compute kmeans
	kmeans(data, num_clusters, labels, tc, attempts, cv::KMEANS_PP_CENTERS, centers);


	//get centers color
	std::vector<cv::Vec3b> colors = getHsvColors(centers);


	//clusterize the image by cluster colors
	cv::Mat out = clusterize(img, labels, colors);


	//unify clusterized img and centers in one vector
	std::vector<cv::Mat> output = { out, centers };


	return output;
}





//
//gives in output the color of food's cluster
//
cv::Vec3b findFood(const cv::Mat centers)  //NOTES TO ERASE: this method is good with food in white dishes only
{



	/*-------define constant parameters--------*/



	const uchar V = 50;
	const cv::Vec3b DISH_COL(189, 34, 50);
	const cv::Vec3b BACK_COL(8, 11, 50);
	int max = 0;
	int argmax = 0;



	/*-------define variables--------*/



	std::vector<int> dish_distance;
	std::vector<int> backgr_distance;



	//get centers colors
	std::vector<cv::Vec3b> colors = getHsvColors(centers);



	//get cluster dintances
	for (int i = 0; i < centers.rows; i++)
	{


		//evaluate distances from reference colors for each centroid
		dish_distance.push_back(cv::norm(DISH_COL, colors[i], cv::NORM_L1));
		backgr_distance.push_back(cv::norm(BACK_COL, colors[i], cv::NORM_L1));



		//find color with max distance from dish and background
		if (max < dish_distance[i] + backgr_distance[i])
		{


			//get max
			max = dish_distance[i] + backgr_distance[i];
			argmax = i;
		}
	}



	//take the cluster with color with max distance from the reference colors
	return colors[argmax];
}





//
//segments the food, given plate clusterization
//
cv::Mat segmentFood(const cv::Mat img, const cv::Mat clust_img, const cv::Mat centers)
{



	/*-------define constant parameters-------*/


	
	//init kernel dimensions for morphological operations 
	const cv::Size close_kernel(22, 22);
	const cv::Size open_kernel(20, 20);
	const cv::Size dilate_kernel(2, 2);


	//find food centroid's H,S,V colors 
	const cv::Vec3b food_col(findFood(centers));




	/*--------initialize variables--------*/



	//init
	cv::Mat mask;
	cv::Mat food(img.size(), img.type(), cv::Scalar(0, 0, 0));




	/*-------get cluster segmentation--------*/



	//create a mask that takes only pixels with value = food_col
	inRange(clust_img, food_col, food_col, mask);



	//morphological operations to fill mask voids
	morphologyEx(mask, mask, cv::MORPH_CLOSE, getStructuringElement(cv::MORPH_ELLIPSE, close_kernel));
	morphologyEx(mask, mask, cv::MORPH_OPEN, getStructuringElement(cv::MORPH_ELLIPSE, open_kernel));
	morphologyEx(mask, mask, cv::MORPH_DILATE, getStructuringElement(cv::MORPH_ELLIPSE, dilate_kernel));



	//apply the mask to the img to segment the food
	img.copyTo(food, mask);

	return food;
}


