import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import json
from PIL import Image
import cv2

input_shape = (224, 224, 3)


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


# detect all food in rectangles provided
def detect_one(tray, img):
    # get class names from json file (stored during training)
    with open("littleaug_weights_final_sparse_olddataset_class_names.json", "r") as f:
        class_names = json.load(f)

    class_names = list(class_names.keys())  # get class names as list
    # print(class_names)

    class_names_labels = []  # get class label (id representing class)
    for e in class_names:
        lb, description = e.split(' ', 1)
        class_names_labels.append(int(lb))
    print(class_names_labels)

    # set up model
    base_model = tf.keras.applications.EfficientNetB1(include_top=False)
    inputs = layers.Input(shape=input_shape, name="input_layer")
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D(name="pooling_layer")(x)
    x = layers.Dropout(.3)(x)
    x = layers.Dense(len(class_names))(x)
    outputs = layers.Activation("softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.load_weights("littleaug_weights_final_sparse_olddataset.h5")
    model.compile(loss="sparse_categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(0.001),
                  metrics=["accuracy"])

    # get rectangles
    bounding_box_path = "benchmark_dataset/tray" + str(tray) + "/bounding_box/" + img + "_bounding_box.txt"
    rectangles = read_rectangles_from_file(bounding_box_path)

    # get image
    image_path = "benchmark_dataset/tray" + str(tray) + "/" + img + ".jpg"
    image = cv2.imread(image_path)
    # cv2.imshow("immagine", image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # conversion used for classification

    output_lines = []  # list to store lines to print in output file

    for rect in rectangles:
        # get rect sub-image
        rect_image = image[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
        #cv2.imshow("sottoimmagine", rect_image)
        #cv2.waitKey(0)

        # preprocess image for prediction
        image_resized = cv2.resize(rect_image, (224, 224))
        image_array = np.array(image_resized)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = image_array / 255.0

        # do prediction
        predictions = model.predict(image_array)
        top_index = np.argsort(predictions)[0, ::-1][0]  # take only first
        current_label = class_names_labels[top_index]

        new_output_line = "ID: " + str(current_label) + "; [" + ', '.join([str(n) for n in rect]) + "]"
        print(current_label)
        output_lines.append(new_output_line)

    out_path = 'benchmark_dataset/tray' + str(tray) + "/bounding_boxes/" + img + "_single_predicted.txt"
    store_rectangles_to_file(out_path, output_lines)

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
    imgs = ["food_image", "leftover1", "leftover2", "leftover3"]
    for i in range(1, 9):
        for img in imgs:
            detect_one(i, img)
