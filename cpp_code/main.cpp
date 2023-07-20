#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include "iostream"
#include <unistd.h>
#include <iostream>
#include "include/main_helper.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


int main() {
    estimateFoodLeftovers("dataset/test_dataset/tray3/leftover1.jpg", "dataset/test_dataset/tray1/leftover1.jpg");

    return 0;
}