#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include "iostream"
#include <unistd.h>
#include <iostream>
#include "include/main_helper.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


int main() {
    Py_Initialize();

    for (int i = 4; i < 9; ++i) {
        std::vector<std::string> image_names = {"food_image", "leftover1", "leftover2", "leftover3"};
        for (int j = 1; j < image_names.size(); ++j) {
            estimateFoodLeftovers("dataset/test_dataset/tray" + std::to_string(i) + "/" + image_names[0] + ".jpg",
                                  "dataset/test_dataset/tray" + std::to_string(i) + "/" + image_names[j] + ".jpg");
        }
    }

    Py_Finalize();


    return 0;
}