#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include "iostream"
#include <unistd.h>
#include <iostream>
#include "include/main_helper.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>



int my_main() {
    /*const char* version = Py_GetVersion();
    std::cout << "Python interpreter version: " << version << std::endl;*/
    Py_Initialize();

    /*// Codice Python per installare il pacchetto numpy
    const char *install_code = R"(
    try:
        import numpy as np
        import tensorflow as tf
        from tensorflow.keras import layers
        import json
        from PIL import Image
        import cv2
    except ImportError as e:
        import pip
        pip.main(['install', 'numpy', 'tensorflow', 'Pillow', 'opencv-python'])
    )";
    int result = PyRun_SimpleString(install_code);

    if (result != 0) {
        PyErr_Print();
    }*/



    //PyRun_SimpleString("print('Hello')");
    const char* python_code_path = "../python_code";
    int chdir_result = chdir(python_code_path);
    if (chdir_result != 0) {
        std::cerr << "Failed to change directory!" << std::endl;
        return 1;
    }
    PyObject * obj = Py_BuildValue("s", "../python_code/multilabel_classifier.py");
    FILE *fp = _Py_fopen_obj(obj, "r+");
    if (fp != NULL)
        PyRun_SimpleFile(fp, "../python_code/multilabel_classifier.py");
    Py_Finalize();

    return 0;
}

int main() {
    std::cout << "Hello, World!" << std::endl;

    estimateFoodLeftovers("../dataset/test_dataset/tray1/food_image.jpg", "../dataset/test_dataset/tray1/leftover1.jpg");

    return 0;
}