//
// Created by Mattia Toffanin
//

#include "../include/single_classify.hpp"
#include <Python.h>
#include "iostream"
#include <unistd.h>

int single_classify() {
    //Py_Initialize();

    // installing dependencies
    const char *install_code = R"(
try:
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras import layers
    import json
    import cv2
except ImportError as e:
    import pip
    pip.main(['install', 'numpy', 'tensorflow', 'opencv-python'])
    )";
    int result = PyRun_SimpleString(install_code);

    if (result != 0) {
        PyErr_Print();
    }

    // setting working directory
    const char *python_code_path = "python_code";
    int chdir_result = chdir(python_code_path);
    if (chdir_result != 0) {
        std::cerr << "Failed to change directory!" << std::endl;
        return 1;
    }
    PyObject * obj = Py_BuildValue("s", "singlelabel_classifier.py");
    FILE *fp = _Py_fopen_obj(obj, "r+");
    if (fp != NULL)
        PyRun_SimpleFile(fp, "singlelabel_classifier.py");
    //Py_Finalize();

    // re-setting working directory
    const char *root_path = "../";
    chdir_result = chdir(root_path);
    if (chdir_result != 0) {
        std::cerr << "Failed to change directory!" << std::endl;
        return 1;
    }

    return 0;
}

