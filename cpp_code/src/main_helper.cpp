#include "../include/main_helper.hpp"
#include "../include/find_food.hpp"
#include "../include/multi_classify.hpp"
#include "../include/single_classify.hpp"
#include "../include/segmentation.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>

/*void writeImage_dir(std::string img_dir){
    const std::string write_dir = "tmp/food_indo.txt";

    std::ofstream file(write_dir);
    if(file)
}*/

std::vector<std::vector<int>> parseLine(std::string line) {




    //--------definevariables-------//



    std::vector<std::vector<int>> rectAndIndexValues;

    std::vector<int> label;

    std::istringstream iss(line);
    std::string token;




    //--------parse the line--------/



    //stream the line untile the open square bracket
    while (std::getline(iss, token, '[')) {


        std::vector<int> temp;


        //stream untile the close square brackert
        if (std::getline(iss, token, ']')) {


            //take the current token
            std::istringstream valueStream(token);


            //get tokens and remove whitespaces
            while (std::getline(valueStream, token, ',')) {


                //remove whitespaces
                token.erase(std::remove_if(token.begin(), token.end(), ::isspace), token.end());


                //save
                int value = std::stoi(token);
                temp.push_back(value);
            }
        }

        rectAndIndexValues.push_back(temp);
    }


    return rectAndIndexValues;
}


void estimateFoodLeftovers(std::string tray_image_dir, std::string leftover_image_dir) {


    //check images

    /*-------load images-------*/



    cv::Mat tray_img = cv::imread(tray_image_dir);
    cv::Mat leftover_img = cv::imread(leftover_image_dir);
    bool bread = false;
    int rect_bread = 0;



    /*-------find dishes-------*/



    //find dishes in tray image
    std::vector<cv::Vec3f> tray_dishes = findDishes(tray_img);
    std::vector<cv::Mat> tray_dishesMasks = getRegionMask(tray_dishes, tray_img.size());
    std::vector<cv::Rect> tray_dishesRect = getFoodRect(tray_dishes, tray_img);


    //find dishes in leftover image
    std::vector<cv::Vec3f> leftover_dishes = findDishes(leftover_img);
    std::vector<cv::Mat> leftover_dishesMasks = getRegionMask(leftover_dishes, leftover_img.size());
    std::vector<cv::Rect> leftover_dishesRect = getFoodRect(leftover_dishes, leftover_img);




    /*-------find dishes-------*/



    //remove dishes from the image to find the bread outside
    cv::Mat tray_img_without_dishes = removeDishes(tray_img, tray_dishes);
    cv::Mat leftover_img_without_dishes = removeDishes(leftover_img, leftover_dishes);


    //find the bread in tray image
    std::vector<cv::Vec3f> tray_bread_regions = findBread(tray_img_without_dishes);
    std::vector<cv::Rect> tray_bread_regionsRect = getFoodRect(tray_bread_regions, tray_img);


    //find the bread in leftover image
    std::vector<cv::Vec3f> leftover_bread_regions = findBread(leftover_img_without_dishes);
    std::vector<cv::Rect> leftover_bread_regionsRect = getFoodRect(leftover_bread_regions, leftover_img);


    //write rects

    std::ofstream file1("tmp/food_info.txt");

    if (file1.is_open()) {
        file1 << tray_image_dir;
    } else {
        std::cout << "Error opening the file." << std::endl;
    }
    file1.close();

    std::ofstream file2("tmp/rectangles.txt");

    if (file2.is_open()) {

        for (cv::Rect rect: tray_dishesRect) {
            file2 << "[" << rect.x << ", " << rect.y << ", " << rect.width << ", " << rect.height << "]\n";
        }

        for (cv::Rect rect: tray_bread_regionsRect) {
            file2 << "[" << rect.x << ", " << rect.y << ", " << rect.width << ", " << rect.height << "]\n";
        }


    } else {
        std::cout << "Error opening the file." << std::endl;
    }
    file2.close();




    /*-------classify found regions-------*/



    multi_classify();


    //read labels

    // Replace "filename.txt" with the actual name of your generated file.
    std::ifstream inputFile("tmp/labeled_rectangles.txt");

    //cv::waitKey();
    // Check if the file was opened successfully.
    if (!inputFile.is_open()) {
        std::cout << "Error opening the file." << std::endl;
    }

    // Read the content from the file.
    std::string line;
    std::vector<cv::Rect> rects;
    std::vector<std::vector<int>> labels;
    while (std::getline(inputFile, line)) {
        // Process each line here.
        std::vector<std::vector<int>> parsed = parseLine(line);

        std::vector<int> rect_values = parsed[0];
        rects.push_back(cv::Rect(rect_values[0], rect_values[1], rect_values[2], rect_values[3]));

        labels.push_back(parsed[1]);

        std::cout << line << std::endl;

    }

    int i = 0;
    for (cv::Rect value: rects) {
        std::cout << "[RECTANGLE " << std::to_string(i) << " ]" << std::endl;
        std::cout << value << std::endl;
        i++;
    }

    i = 0;
    for (std::vector<int> line: labels) {
        std::cout << "[LABELS " << std::to_string(i) << " ]" << std::endl;
        for (int val: line) {
            std::cout << val << std::endl;
        }
        i++;
    }


    // Close the file after reading.
    inputFile.close();



    /*---------segmentation----------*/


    std::vector<std::vector<int>> foodGroup = {{1,  2, 3, 4,  5, 9},
                                               {6,  7, 8, 10, 11},
                                               {12, 13}};

    std::vector<cv::Rect> boundingRects;

    //segment each rectangle
    for (int i = 0; i < rects.size(); i++) {


        //select the type of segmentation based on the food type
        unsigned char selection = 5;


        std::vector<int> label = labels[i];


        //take the current labels
        for (int value: label) {

            if (value == 13) {

                bread = true;
                rect_bread = i;
            } else
                bread = false;


            //check the food type
            for (int j = 0; j < foodGroup.size(); j++) {


                auto it = std::find(foodGroup[j].begin(), foodGroup[j].end(), value);


                if (it != foodGroup[j].end()) {

                    selection = j;
                }

            }
        }


        //convert img in masked roi image
        cv::Mat roi(tray_img.size(), tray_img.type(), cv::Scalar(0, 0, 0));

        if (bread) {
            //roi = tray_img(rects[rect_bread]);
            cv::Mat mask_bread(tray_img.size(), CV_8UC1, cv::Scalar(0));
            rectangle(mask_bread, rects[rect_bread], cv::Scalar(255), -1);

            tray_img.copyTo(roi, mask_bread);
        } else {

            //masked dish img
            tray_img.copyTo(roi, tray_dishesMasks[i]);
        }


        cv::Mat mask = getFoodMask(roi, selection);
        //imshow("Rectangle" + std::to_string(i), mask);

        // Find the contours in the binary mask
        std::vector<std::vector<cv::Point>> contours;

        //find contours
        cv::findContours(mask, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

        // Calculate the bounding rectangles for each contour

        for (std::vector<cv::Point> contour: contours) {
            cv::Rect rect = cv::boundingRect(contour);
            rect = rect & cv::Rect(0, 0, tray_img.rows, tray_img.cols);
            boundingRects.push_back(rect);
            rectangle(tray_img, rect, cv::Scalar(0, 0, 255), 2);
        }
    }

    std::ofstream file3("tmp/segmentations_rectangle.txt");
    if (file3.is_open()) {

        for (cv::Rect rect: boundingRects) {
            file3 << "[" << rect.x << ", " << rect.y << ", " << rect.width << ", " << rect.height << "]\n";
        }

    } else {
        std::cout << "Error opening the file." << std::endl;
    }

    file3.close();


    //imshow("BOUNDING", tray_img);
    //cv::waitKey();


    /* if(label >= 1 && label <=5)
         kmeans()
 */
    single_classify();


}