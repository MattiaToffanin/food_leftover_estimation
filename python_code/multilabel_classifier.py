import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import json
from PIL import Image
import cv2

input_shape = (224, 224, 3)


def read_food_info(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            return line.split(' ')


def read_rectangles_from_file(file_path):
    rectangles = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()[1:-1]
            values = line.split(',')
            rect = [int(val) for val in values]
            rectangles.append(rect)
    return rectangles


def store_rectangles_to_file(file_path, output_lines):
    with open(file_path, 'w') as file:
        for line in output_lines:
            file.write(line + '\n')


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
def detect_more(tray, img):
    # import os
    # current_directory = os.getcwd()
    # print("Current working directory:", current_directory)

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

    # set food distribution
    background = [0]  # background
    single_food = [13]  # bread always single
    first_food = [1, 2, 3, 4, 5, 12]  # food always single
    second_food = [6, 7, 8, 9, 10, 11]  # food probably together

    # set up model
    model = setup_model(len(class_names))

    # get rectangles
    rectangles_path = "../tmp/rectangles.txt"
    rectangles = read_rectangles_from_file(rectangles_path)

    # get image
    image_path = "../dataset/test_dataset/tray" + str(tray) + "/" + img + ".jpg"
    image = cv2.imread(image_path)
    # cv2.imshow("immagine", image)
    # cv2.waitKey(0)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # conversion used for classification

    output_lines = []  # list to store lines to print in output file

    print("Predicting food in dishes and out dishes")
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
        top_indices = np.argsort(predictions)[0, ::-1][:5]  # take only 5 most probable predictions

        # last classification based on type of food
        index = 0
        current_label = class_names_labels[top_indices[index]]
        current_prob = predictions[0, top_indices[index]]

        if current_label in background:
            new_output_line = "[" + ', '.join([str(n) for n in rect]) + "], [" + str(current_label) + "]"
            # print(new_output_line)
            output_lines.append(new_output_line)
            continue

        if current_label in first_food:
            new_output_line = "[" + ', '.join([str(n) for n in rect]) + "] [" + str(current_label) + "]"
            # print(new_output_line)
            output_lines.append(new_output_line)
            continue

        if current_label in single_food:
            if current_prob > 0.5:
                new_output_line = "[" + ', '.join([str(n) for n in rect]) + "] [" + str(current_label) + "]"
                # print(new_output_line)
                output_lines.append(new_output_line)
                continue
            index += 1

        varius_food_label = []
        for i in range(index, len(top_indices)):
            current_label = class_names_labels[top_indices[i]]
            current_prob = predictions[0, top_indices[i]]
            if current_label in second_food and current_prob > 0.1:
                varius_food_label.append(current_label)
        new_output_line = "[" + ', '.join([str(n) for n in rect]) + "] [" + ', '.join(
            [str(n) for n in varius_food_label]) + "]"
        # print(new_output_line)
        output_lines.append(new_output_line)

    store_rectangles_to_file("../tmp/labeled_rectangles.txt", output_lines)
    # print(output_lines)

    # print(output_lines)

    # print(class_names[top_indices[0]])
    # print("Prediction on entire image")
    # for j in top_indices:
    #     class_name = class_names[j]
    #     probability = predictions[0, j]
    #
    #     print("Classe:", class_name)
    #     print("Probabilità:", probability)
    #     print()


if __name__ == '__main__':
    tray, img = read_food_info("../tmp/food_info.txt")
    detect_more(tray, img)

