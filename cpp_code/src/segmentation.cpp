#include "../include/segmentation.hpp"




//
// Create the dataset from given image and given number of feature
//
cv::Mat kmeansData(const cv::Mat img, const unsigned char selection) {




    /*--------define constant paramenters--------*/



    //dimension of the feature vector
    const int num_features = 4;


    //define data type and dimension
    cv::Mat data(img.cols * img.rows, num_features, CV_32F);


    //priors
    cv::Vec3b DISH_COLOR; 	//average dish color in hsv
    const cv::Vec3b BACKGROUND_COL(0, 0, 0); 	//average background color in hsv



    //first meals segmentation data (all pasta's , riso , seafood)
    if (selection == 0) {
        DISH_COLOR = cv::Vec3b(120, 20, 182);
    }


        //second meals segmentation data
    else if (selection == 1) {
        DISH_COLOR = cv::Vec3b(189, 34, 182);
    }


        //salad segmentation data
    else if (selection == 2) {
        DISH_COLOR = cv::Vec3b(189, 34, 182);
    }



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
std::vector<cv::Mat> kmeansClustering(const cv::Mat img, const int num_clusters, const unsigned char selection ) {



    /*------define constant parameters-------*/


    //define kmeans parameters
    cv::TermCriteria tc(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1.0);
    const int attempts = 10;



    /*-------define variables------*/


    std::vector<int> labels;
    cv::Mat centers;




    /*-------clusterize image------*/



    //define data for kmeans clustering
    cv::Mat data;


//first meals segmentation data (all pasta's , riso , seafood)
    if (selection == 0) {
        data = kmeansData(img,selection);
    }


        //second meals segmentation data
    else if (selection == 1) {
        data = kmeansData(img,selection);
    }


        //salad segmentation data
    else if (selection == 2) {
        data = kmeansData(img,selection);
    }

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
cv::Vec3b getClusterFoodColor(const cv::Mat centers, const unsigned char selection)
{



    /*-------define constant parameters--------*/



    const unsigned char V = 50;
    cv::Vec3b DISH_COL;
    const cv::Vec3b BACK_COL(8, 11, 50);
    int max = 0;
    int argmax = 0;


//first meals segmentation data (all pasta's , riso , seafood)
    if (selection == 0) {
        DISH_COL = cv::Vec3b(120, 20, 50);
    }


        //second meals segmentation data
    else if (selection == 1) {
        DISH_COL = cv::Vec3b(189, 34, 50);
    }


        //salad segmentation data
    else if (selection == 2) {
        DISH_COL = cv::Vec3b(189, 34, 50);
    }



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
cv::Mat segmentFood(const cv::Mat img, const cv::Mat clust_img, const cv::Mat centers, const unsigned char selection)
{



    /*-------define constant parameters-------*/



    //init kernel dimensions for morphological operations
    cv::Size close_kernel;
    cv::Size open_kernel;
    cv::Size dilate_kernel;





    //find food centroid's H,S,V colors
    const cv::Vec3b food_col(getClusterFoodColor(centers,selection));




    /*--------initialize variables--------*/



    //init
    cv::Mat mask, dish, out_dish;
    cv::Mat food(img.size(), img.type(), cv::Scalar(0, 0, 0));




    /*-------get cluster segmentation--------*/



    //create a mask that takes only pixels with value = food_col
    inRange(clust_img, food_col, food_col, mask);





    //first meals segmentation data (all pasta's , riso , seafood)
    if (selection == 0) {

        close_kernel = cv::Size(75, 75);
        open_kernel = cv::Size(75, 75);
        dilate_kernel = cv::Size(4, 4);

        //save mask
        cv::Mat mask_original = mask.clone();

        //get smaller dish area
        circle(mask, cv::Point(img.cols / 2, img.rows / 2), img.cols/2 - 20, cv::Scalar(100),-1, 2);

        inRange(mask, 100, 100, dish);

        //get outside element of the original mask
        threshold(dish, out_dish, 254, 255, cv::THRESH_BINARY_INV);
        cv::Mat residual = out_dish & mask; //find the cluster outside the dish


        //check if outside the circle borders there is a cluster
        if(cv::countNonZero(residual) > 0){

            cv::Mat food_in_dish_mask = mask_original & dish ;

            //morphological operations to fill mask voids
            cv::morphologyEx(food_in_dish_mask,food_in_dish_mask,cv::MORPH_CLOSE, getStructuringElement(cv::MORPH_ELLIPSE, close_kernel));
            cv::morphologyEx(food_in_dish_mask,food_in_dish_mask,cv::MORPH_OPEN, getStructuringElement(cv::MORPH_ELLIPSE, open_kernel));
            //cv::morphologyEx(food_in_dish_mask,food_in_dish_mask,cv::MORPH_OPEN, getStructuringElement(cv::MORPH_ELLIPSE, open_kernel));
            //cv::morphologyEx(food_in_dish_mask,food_in_dish_mask,cv::MORPH_DILATE, getStructuringElement(cv::MORPH_ELLIPSE, dilate_kernel));




            img.copyTo(food, food_in_dish_mask);

            return food_in_dish_mask;
        }

        //morphological operations to fill mask voids
        cv::morphologyEx(mask_original, mask_original, cv::MORPH_CLOSE, getStructuringElement(cv::MORPH_ELLIPSE, close_kernel));
        cv::morphologyEx(mask_original, mask_original, cv::MORPH_OPEN, getStructuringElement(cv::MORPH_ELLIPSE, open_kernel));
        cv::morphologyEx(mask_original, mask_original, cv::MORPH_DILATE, getStructuringElement(cv::MORPH_ELLIPSE, dilate_kernel));

        //img.copyTo(food, mask_original);

        return mask_original;
    }


    //second meals segmentation data
    if (selection == 1) {

        close_kernel = cv::Size(22, 22);
        open_kernel = cv::Size(20, 20);
        dilate_kernel = cv::Size(2, 2);

        //save mask
        cv::Mat mask_original = mask.clone();

        //get smaller dish area
        circle(mask, cv::Point(img.cols / 2, img.rows / 2), img.cols/2 - 20, cv::Scalar(100),-1, 2);

        inRange(mask, 100, 100, dish);

        //get outside element of the original mask
        threshold(dish, out_dish, 254, 255, cv::THRESH_BINARY_INV);
        cv::Mat residual = out_dish & mask; //find the cluster outside the dish


        //check if outside the circle borders there is a cluster
        if(cv::countNonZero(residual) > 0){

            cv::Mat food_in_dish_mask = mask_original & dish ;

            //morphological operations to fill mask voids
            cv::morphologyEx(food_in_dish_mask,food_in_dish_mask,cv::MORPH_CLOSE, getStructuringElement(cv::MORPH_ELLIPSE, close_kernel));
            cv::morphologyEx(food_in_dish_mask,food_in_dish_mask,cv::MORPH_OPEN, getStructuringElement(cv::MORPH_ELLIPSE, open_kernel));
            //cv::morphologyEx(food_in_dish_mask,food_in_dish_mask,cv::MORPH_OPEN, getStructuringElement(cv::MORPH_ELLIPSE, open_kernel));
            //cv::morphologyEx(food_in_dish_mask,food_in_dish_mask,cv::MORPH_DILATE, getStructuringElement(cv::MORPH_ELLIPSE, dilate_kernel));

            img.copyTo(food, food_in_dish_mask);

            return food_in_dish_mask;
        }

        //morphological operations to fill mask voids
        cv::morphologyEx(mask_original, mask_original, cv::MORPH_CLOSE, getStructuringElement(cv::MORPH_ELLIPSE, close_kernel));
        cv::morphologyEx(mask_original, mask_original, cv::MORPH_OPEN, getStructuringElement(cv::MORPH_ELLIPSE, open_kernel));
        cv::morphologyEx(mask_original, mask_original, cv::MORPH_DILATE, getStructuringElement(cv::MORPH_ELLIPSE, dilate_kernel));

        //img.copyTo(food, mask_original);

        return mask_original;
    }


    //salad segmentation data
    if (selection == 2) {

        close_kernel = cv::Size(75, 75);
        open_kernel = cv::Size(75, 75);
        dilate_kernel = cv::Size(4, 4);

        //save mask
        cv::Mat mask_original = mask.clone();

        //get smaller dish area
        circle(mask, cv::Point(img.cols / 2, img.rows / 2), img.cols/2 - 20, cv::Scalar(100),-1, 2);

        inRange(mask, 100, 100, dish);

        //get outside element of the original mask
        threshold(dish, out_dish, 254, 255, cv::THRESH_BINARY_INV);
        cv::Mat residual = out_dish & mask; //find the cluster outside the dish


        //check if outside the circle borders there is a cluster
        if(cv::countNonZero(residual) > 0){

            cv::Mat food_in_dish_mask = mask_original & dish ;

            //morphological operations to fill mask voids
            cv::morphologyEx(food_in_dish_mask,food_in_dish_mask,cv::MORPH_CLOSE, getStructuringElement(cv::MORPH_ELLIPSE, close_kernel));
            cv::morphologyEx(food_in_dish_mask,food_in_dish_mask,cv::MORPH_OPEN, getStructuringElement(cv::MORPH_ELLIPSE, open_kernel));
            //cv::morphologyEx(food_in_dish_mask,food_in_dish_mask,cv::MORPH_OPEN, getStructuringElement(cv::MORPH_ELLIPSE, open_kernel));
            //cv::morphologyEx(food_in_dish_mask,food_in_dish_mask,cv::MORPH_DILATE, getStructuringElement(cv::MORPH_ELLIPSE, dilate_kernel));




            img.copyTo(food, food_in_dish_mask);

            return food_in_dish_mask;
        }

        //morphological operations to fill mask voids
        cv::morphologyEx(mask_original, mask_original, cv::MORPH_CLOSE, getStructuringElement(cv::MORPH_ELLIPSE, close_kernel));
        cv::morphologyEx(mask_original, mask_original, cv::MORPH_OPEN, getStructuringElement(cv::MORPH_ELLIPSE, open_kernel));
        cv::morphologyEx(mask_original, mask_original, cv::MORPH_DILATE, getStructuringElement(cv::MORPH_ELLIPSE, dilate_kernel));

        //img.copyTo(food, mask_original);

        return mask_original;
    }

    return food;
}
//get masked image
cv::Mat getFoodMask(cv::Mat img, const unsigned char selection) {



    /*-------define varibales--------*/


    cv::Mat img_corr;
    cv::Mat hsv;
    std::vector<cv::Mat> hsv_clustered;
    cv::Mat mask;
    cv::Mat clusters;
    cv::Mat centroids;
    cv::Mat hsv_correction;




    /*--------pre-process---------*/

    //first meals segmentation data (all pasta's , riso , seafood)
    if(selection == 0){

        //adjust illum and contrast
        img.convertTo(img_corr, -1, 0.25, -2);

        // convert image to hsv
        cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);


        //noise reduction
        cv::blur(hsv, hsv, cv::Size(9, 9));


        //change illumination and contrast
        hsv.convertTo(hsv_correction, -1, 1.36, -24);

        //noise reduction
        cv::blur(hsv_correction, hsv_correction, cv::Size(9, 9));
    }

    else if(selection == 1){

        //adjust illum and contrast
        img.convertTo(img_corr, -1, 0.25, -2);

        // convert image to hsv
        cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);


        //noise reduction
        cv::blur(hsv, hsv, cv::Size(9, 9));


        //change illumination and contrast
        hsv.convertTo(hsv_correction, -1, 1.36, -24);

        //noise reduction
        cv::blur(hsv_correction, hsv_correction, cv::Size(9, 9));
    }


    else if (selection == 2) {

        //adjust illum and contrast
        img.convertTo(img_corr, -1, 0.25, -2);

        // convert image to hsv
        cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);


        //noise reduction
        cv::blur(hsv, hsv, cv::Size(9, 9));


        //change illumination and contrast
        hsv.convertTo(hsv_correction, -1, 1.36, -24);

        //noise reduction
        cv::blur(hsv_correction, hsv_correction, cv::Size(9, 9));
    }

    /*--------segment by clusterization----------*/



    //clusterize image
    hsv_clustered = kmeansClustering(hsv_correction, 3, selection);


    //pick the clustered img and the centroids
    clusters = hsv_clustered[0];
    centroids = hsv_clustered[1];


    //get segmentation mask
    mask = segmentFood(img, clusters, centroids, selection);


    return mask;
}