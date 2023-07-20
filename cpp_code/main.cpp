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

    //for (int i = 1; i < 9; ++i) {
        estimateFoodLeftovers("dataset/test_dataset/tray" + std::to_string(3) + "/leftover1.jpg",
                              "dataset/test_dataset/tray1/leftover1.jpg");
    //}

    Py_Finalize();

    return 0;
}