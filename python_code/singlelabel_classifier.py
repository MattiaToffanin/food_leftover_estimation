# Created by Mattia Toffanin

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import json
import cv2

input_shape = (224, 224, 3)


def get_tray_and_image(file_path):
    index = file_path.split("tray")[1].split("/")[0]
    image = file_path.split(".")[0].split("/")[-1]
    return index, image


# function to read on which image we are working
def read_food_info(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            return line


# function to read rectangles from dish detection
def read_rectangles_from_file(file_path):
    rectangles = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()[1:-1]
            values = line.split(',')
            rect = [int(val) for val in values]
            rectangles.append(rect)
    return rectangles


# function to store labels of rectangles
def store_final_bounding_box(file_path, output_lines):
    with open(file_path, 'w') as file:
        for line in output_lines:
            file.write(line + '\n')


# function to set up the model
def setup_model(num_classes):
    base_model = tf.keras.applications.EfficientNetB1(include_top=False)
    inputs = layers.Input(shape=input_shape, name="input_layer")
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D(name="pooling_layer")(x)
    x = layers.Dropout(.3)(x)
    x = layers.Dense(num_classes)(x)
    outputs = layers.Activation("softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.load_weights("weights/weights_final.h5")
    model.compile(loss="sparse_categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(0.001),
                  metrics=["accuracy"])
    return model


# detect all food in rectangles provided
def detect_one(path_img):
    # get class names from json file (stored during training)
    with open("weights/class_names.json", "r") as f:
        class_names = json.load(f)

    class_names = list(class_names.keys())  # get class names as list
    # print(class_names)

    class_names_labels = []  # get class label (id representing class)
    for e in class_names:
        lb, description = e.split(' ', 1)
        class_names_labels.append(int(lb))
    # print(class_names_labels)

    # set up model
    model = setup_model(len(class_names))

    # get rectangles
    rectangles_path = "../tmp/segmentations_rectangle.txt"
    rectangles = read_rectangles_from_file(rectangles_path)
    # print(rectangles)

    # get image
    image_path = "../" + path_img
    image = cv2.imread(image_path)
    # cv2.imshow("immagine", image) # uncomment to see image
    # cv2.waitKey(0)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # conversion used for classification

    output_lines = []  # list to store lines to print in output file

    print("Predicting final food label")
    for rect in rectangles:
        # get rect sub-image
        rect_image = image[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
        # cv2.imshow("sottoimmagine", rect_image)
        # cv2.waitKey(0)

        # preprocess image for prediction
        image_resized = cv2.resize(rect_image, (224, 224))
        image_array = np.array(image_resized)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = image_array / 255.0

        # do prediction
        predictions = model.predict(image_array)
        top_index = np.argmax(predictions)  # take only first
        current_label = class_names_labels[top_index]

        new_output_line = "ID: " + str(current_label) + "; [" + ', '.join([str(n) for n in rect]) + "]"
        #print(current_label)
        output_lines.append(new_output_line)

    tray, img_name = get_tray_and_image(path_img)
    store_final_bounding_box("../outputs/tray" + tray + "/bounding_boxes/" + img_name + ".txt", output_lines)


if __name__ == '__main__':
    path_img = read_food_info("../tmp/food_info.txt")
    detect_one(path_img)
