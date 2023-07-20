//
// Created by Mattia Toffanin
//

#ifndef FOOD_LEFTOVER_ESTIMATION_MEAN_AVERAGE_PRECISION_HELPER_H
#define FOOD_LEFTOVER_ESTIMATION_MEAN_AVERAGE_PRECISION_HELPER_H

#include <iostream>


struct BoundingBox;

BoundingBox parseBoundingBox(const std::string &bboxString);

float calculateIoU(const BoundingBox &box1, const BoundingBox &box2);

float calculateAP(const std::vector<BoundingBox> &trueBoxes, const std::vector<BoundingBox> &predBoxes, float iouThreshold);

void calculate_map_over_all_dataset(int tray, const std::string &imgName);

#endif //FOOD_LEFTOVER_ESTIMATION_MEAN_AVERAGE_PRECISION_HELPER_H
