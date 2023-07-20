//
// Created by Mattia Toffanin
//

#include "../include/mean_average_precision_helper.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <sstream>
#include <filesystem>

struct BoundingBox {
    int id;
    float x, y, width, height;
};

// Function to parse bounding box string "[x, y, width, height]" and extract values
BoundingBox parseBoundingBox(const std::string &bboxString) {
    BoundingBox box;
    std::string coordinates = bboxString.substr(bboxString.find('[') + 1,
                                                bboxString.find(']') - bboxString.find('[') - 1);
    std::replace(coordinates.begin(), coordinates.end(), ',', ' ');
    std::stringstream ss(coordinates);
    ss >> box.x >> box.y >> box.width >> box.height;
    return box;
}

// Function to calculate Intersection over Union (IoU)
float calculateIoU(const BoundingBox &box1, const BoundingBox &box2) {
    // Calculate coordinates of intersection rectangle
    float x1 = std::max(box1.x, box2.x);
    float y1 = std::max(box1.y, box2.y);
    float x2 = std::min(box1.x + box1.width, box2.x + box2.width);
    float y2 = std::min(box1.y + box1.height, box2.y + box2.height);

    // Calculate area of intersection rectangle
    float intersectionArea = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    //std::cout << intersectionArea << std::endl;

    // Calculate areas of both bounding boxes
    float box1Area = box1.width * box1.height;
    float box2Area = box2.width * box2.height;

    // Calculate IoU
    return intersectionArea / (box1Area + box2Area - intersectionArea);
}

// Function to calculate Average Precision (AP)
float
calculateAP(const std::vector<BoundingBox> &trueBoxes, const std::vector<BoundingBox> &predBoxes, float iouThreshold) {
    // Sort the predicted boxes by confidence (if available)
    // In your case, you can skip this step if you don't have confidence scores

    // Initialize variables for precision-recall calculation
    int truePositive = 0;
    int falsePositive = 0;
    int totalTrue = trueBoxes.size();

    // Calculate precision and recall for each predicted box
    for (const auto &predBox: predBoxes) {
        bool foundMatch = false;
        for (const auto &trueBox: trueBoxes) {
            float iou = calculateIoU(predBox, trueBox);
            if (iou >= iouThreshold && predBox.id == trueBox.id) {
                foundMatch = true;
                break;
            }
        }
        if (foundMatch) {
            truePositive++;
        } else {
            falsePositive++;
        }
    }

    // Calculate precision and recall
    float precision = static_cast<float>(truePositive) / (truePositive + falsePositive);
    float recall = static_cast<float>(truePositive) / totalTrue;

    // Calculate Average Precision (AP)
    return precision * recall;
}


void calculate_map_over_all_dataset(int tray, const std::string &imgName) {
    // Load true bounding boxes from the file
    std::string trueFilePath = "dataset/test_dataset/tray" + std::to_string(tray) + "/bounding_boxes/" + imgName + "_bounding_box.txt";
    std::ifstream trueFile(trueFilePath);
    if (!trueFile.is_open()) {
        std::cout << "Error opening the file." << std::endl;
    }
    std::vector<BoundingBox> trueBoxes;
    int trueId;
    std::string line;
    while (std::getline(trueFile, line)) {
        std::istringstream ss(line);
        std::string idString;
        ss >> idString >> trueId;
        std::string bboxString;
        std::getline(ss, bboxString);
        BoundingBox box = parseBoundingBox(bboxString);
        box.id = trueId;
        trueBoxes.push_back(box);
    }
    trueFile.close();

    // Load predicted bounding boxes from the file
    std::string predFilePath = "outputs/tray" + std::to_string(tray) + "/bounding_boxes/" + imgName + ".txt";
    std::ifstream predFile(predFilePath);
    if (!predFile.is_open()) {
        std::cout << "Error opening the file." << std::endl;
    }
    std::vector<BoundingBox> predBoxes;
    int predId;
    while (std::getline(predFile, line)) {
        std::istringstream ss(line);
        std::string idString;
        ss >> idString >> predId;
        std::string bboxString;
        std::getline(ss, bboxString);
        BoundingBox box = parseBoundingBox(bboxString);
        box.id = predId;
        predBoxes.push_back(box);
    }
    predFile.close();

    // Calculate mAP for different IoU thresholds
    float iouThreshold = 0.5;
    float mAP = calculateAP(trueBoxes, predBoxes, iouThreshold);
    std::cout << " mAP: " << mAP << std::endl;

    std::string outputFilePath = "outputs/tray" + std::to_string(tray) + "/bounding_boxes/" + imgName + "_mAP.txt";
    std::ofstream outputFile(outputFilePath);
    outputFile << " mAP: " << mAP << std::endl;
}
