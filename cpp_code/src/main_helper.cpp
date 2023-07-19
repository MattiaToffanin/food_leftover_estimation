#include "../include/main_helper.hpp"
#include "../include/find_food.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

void estimateFoodLeftovers(std::string tray_image_dir, std::string leftover_image_dir) {


    //check images

    /*-------load images-------*/



    cv::Mat tray_img = cv::imread(tray_image_dir);
    cv::Mat leftover_img = cv::imread(leftover_image_dir);




    /*-------find dishes-------*/



    //find dishes in tray image
    std::vector<cv::Vec3f> tray_dishes = findDishes(tray_img);
    std::vector<cv::Mat> tray_dishesMasks = getFoodMask(tray_dishes, tray_img.size());
    std::vector<cv::Rect> tray_dishesRect = getFoodRect(tray_dishes);


    //find dishes in leftover image
    std::vector<cv::Vec3f> leftover_dishes = findDishes(leftover_img);
    std::vector<cv::Mat> leftover_dishesMasks = getFoodMask(leftover_dishes, leftover_img.size());
    std::vector<cv::Rect> leftover_dishesRect = getFoodRect(leftover_dishes);




    /*-------find dishes-------*/



    //remove dishes from the image to find the bread outside
    cv::Mat tray_img_without_dishes = removeDishes(tray_img, tray_dishes);
    cv::Mat leftover_img_without_dishes = removeDishes(leftover_img, leftover_dishes);


    //find the bread in tray image
    std::vector<cv::Vec3f> tray_bread_regions = findBread(tray_img_without_dishes);
    std::vector<cv::Rect> tray_bread_regionsRect = getFoodRect(tray_bread_regions);


    //find the bread in leftover image
    std::vector<cv::Vec3f> leftover_bread_regions = findBread(leftover_img_without_dishes);
    std::vector<cv::Rect> leftover_bread_regionsRect = getFoodRect(leftover_bread_regions);


    //write rects

    for (int i = 0; i < tray_dishesMasks.size(); i++) {


        //Select current dish
        cv::Mat tray_dish_mask = tray_dishesMasks[i];
        cv::Rect kernel = tray_dishesRect[i];


        //Asjust kernel into the image frame to prevent overflow
        cv::Rect img_frame(0, 0, tray_img.cols, tray_img.rows);
        kernel = kernel & img_frame;


        //Get the current dish roi
        cv::Mat roi = tray_dish_mask(kernel);

        cv::imshow("tray dish " + std::to_string(i), roi);

    }


    for (int i = 0; i < leftover_dishesMasks.size(); i++) {


        //Select current dish
        cv::Mat leftover_dish_mask = leftover_dishesMasks[i];
        cv::Rect kernel = leftover_dishesRect[i];


        //Asjust kernel into the image frame to prevent overflow
        cv::Rect img_frame(0, 0, leftover_img.cols, leftover_img.rows);
        kernel = kernel & img_frame;


        //Get the current dish roi
        cv::Mat roi = leftover_dish_mask(kernel);

        cv::imshow("leftover dish " + std::to_string(i), roi);

    }


    for (int i = 0; i < tray_bread_regionsRect.size(); i++) {


        //Select current dish
        cv::Rect kernel = tray_bread_regionsRect[i];


        //Asjust kernel into the image frame to prevent overflow
        cv::Rect img_frame(0, 0, tray_img.cols, tray_img.rows);
        kernel = kernel & img_frame;


        //Get the current dish roi
        cv::Mat roi = tray_img(kernel);

        cv::imshow("tray bread " + std::to_string(i), roi);

    }


    for (int i = 0; i < leftover_bread_regionsRect.size(); i++) {


        //Select current dish
        cv::Rect kernel = leftover_bread_regionsRect[i];


        //Asjust kernel into the image frame to prevent overflow
        cv::Rect img_frame(0, 0, leftover_img.cols, leftover_img.rows);
        kernel = kernel & img_frame;


        //Get the current dish roi
        cv::Mat roi = leftover_img(kernel);

        cv::imshow("leftover bread " + std::to_string(i), roi);

    }

    cv::waitKey();


    //classification


    //segmentation




}