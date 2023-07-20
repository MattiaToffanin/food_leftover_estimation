#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include "iostream"
#include <unistd.h>
#include <iostream>
#include "include/main_helper.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "include/mean_average_precision_helper.h"


int main() {
    std::vector<std::string> image_names = {"food_image", "leftover1", "leftover2", "leftover3"};
    /*Py_Initialize();

    for (int i = 4; i < 9; ++i) {
        for (int j = 1; j < image_names.size(); ++j) {
            estimateFoodLeftovers("dataset/test_dataset/tray" + std::to_string(i) + "/" + image_names[0] + ".jpg",
                                  "dataset/test_dataset/tray" + std::to_string(i) + "/" + image_names[j] + ".jpg");
        }
    }

    Py_Finalize();*/

    for (int i = 1; i < 9; ++i) {
        for (const std::string& image_name : image_names) {
            calculate_map_over_all_dataset(i, image_name);
        }
    }

    return 0;
}