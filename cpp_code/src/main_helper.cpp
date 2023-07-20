#include "../include/main_helper.hpp"
#include "../include/find_food.hpp"
#include "../include/multi_classify.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>

/*void writeImage_dir(std::string img_dir){
    const std::string write_dir = "tmp/food_indo.txt";

    std::ofstream file(write_dir);
    if(file)
}*/

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

    std::ofstream file1("tmp/food_info.txt");

    if (file1.is_open()) {
        file1 << tray_image_dir;
    } else {
        std::cout << "Impossibile aprire il file." << std::endl;
    }
    file1.close();

    std::ofstream file2("tmp/rectangles.txt");

    if (file2.is_open()) {

        for(cv::Rect rect : tray_dishesRect){
            file2 << "["<<rect.x<<", "<<rect.y<<", "<<rect.width<<", "<<rect.height<<"]\n";
        }

        for(cv::Rect rect : tray_bread_regionsRect){
            file2 << "["<<rect.x<<", "<<rect.y<<", "<<rect.width<<", "<<rect.height<<"]\n";
        }


    } else {
        std::cout << "Impossibile aprire il file." << std::endl;
    }
    file2.close();

    //classification

    multi_classify();







    //segmentation




}