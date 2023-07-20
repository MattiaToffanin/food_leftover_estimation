import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json

weights_food101_filename = "weights_food101.h5"
weights_final = "weights_final.h5"

input_shape = (224, 224, 3)


# preprocess image
def preprocess_img(image, label, img_size=224):
    image = tf.image.resize(image, [img_size, img_size])
    image = tf.cast(image, tf.float16)
    return image, label


# function for training on final dataset
def train_final():
    print("Train on final dataset")
    dataset_directory = "../dataset/train_dataset"  # directory of the final dataset

    batch_size = 8

    # data augmentation
    datagen = ImageDataGenerator(rescale=1. / 255,
                                 validation_split=0.2,
                                 rotation_range=5,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 shear_range=0.1,
                                 zoom_range=(0.0, 0.2),
                                 horizontal_flip=True,
                                 fill_mode='nearest'
                                 )

    # getting training set
    train_data = datagen.flow_from_directory(
        dataset_directory,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='sparse',
        subset='training',
        shuffle=True
    )

    # getting test set
    test_data = datagen.flow_from_directory(
        dataset_directory,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='sparse',
        subset='validation',
        shuffle=True
    )

    num_classes = len(train_data.class_indices)  # getting number of classes
    # saving class names
    with open("weights/class_names.json", "w") as f:
        json.dump(train_data.class_indices, f)

    # setting early stopping
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(restore_best_weights=True, patience=15, verbose=1,
                                                               monitor="val_loss")
    # setting learning rate annealing
    lower_lr = tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, monitor='val_loss', min_lr=1e-7, patience=0,
                                                    verbose=1)
    # model definition
    base_model = tf.keras.applications.EfficientNetB1(include_top=False)
    inputs = layers.Input(shape=input_shape, name="input_layer")
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D(name="pooling_layer")(x)
    x = layers.Dropout(.3)(x)
    x = layers.Dense(101)(x)
    outputs = layers.Activation("softmax")(x)
    base_model = tf.keras.Model(inputs, outputs)
    base_model.load_weights("weights/" + weights_food101_filename)
    x = base_model.layers[-3].output
    x = layers.Dense(num_classes)(x)
    outputs = layers.Activation("softmax")(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
    model.summary()  # printing model structure

    # compiling model
    model.compile(loss="sparse_categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(0.001),
                  metrics=["accuracy"])

    # training model
    model.fit(train_data, epochs=100, steps_per_epoch=len(train_data), validation_data=test_data,
              validation_steps=int(0.5 * len(test_data)),
              callbacks=[early_stopping_callback, lower_lr])

    # evaluating model performance
    model.evaluate(test_data)

    # saving weights
    model.save_weights(weights_final)


def train_food101():
    print("Train on food101 dataset")
    # getting training and test set
    (train_data, test_data), ds_info = tfds.load(name='food101', split=['train', 'validation'], shuffle_files=False,
                                                 as_supervised=True, with_info=True)
    class_names = ds_info.features['label'].names

    batch_size = 32

    # preprocess training set
    train_data = train_data.map(preprocess_img, tf.data.AUTOTUNE)
    train_data = train_data.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # preprocess test set
    test_data = test_data.map(preprocess_img, tf.data.AUTOTUNE)
    test_data = test_data.batch(batch_size)

    # setting early stopping
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(restore_best_weights=True, patience=3, verbose=1,
                                                               monitor="val_accuracy")
    # setting learning rate annealing
    lower_lr = tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, monitor='val_accuracy', min_lr=1e-7, patience=0,
                                                    verbose=1)

    mixed_precision.set_global_policy(policy='mixed_float16')

    # model definition
    base_model = tf.keras.applications.EfficientNetB1(include_top=False)
    inputs = layers.Input(shape=input_shape, name="input_layer")
    x = base_model(inputs)
    x = layers.GlobalAveragePooling2D(name="pooling_layer")(x)
    x = layers.Dropout(.3)(x)
    x = layers.Dense(len(class_names))(x)
    outputs = layers.Activation("softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.summary()  # printing model structure

    # compiling model
    model.compile(loss="sparse_categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(0.001),
                  metrics=["accuracy"])

    # training model
    model.fit(train_data, epochs=50, steps_per_epoch=len(train_data), validation_data=test_data,
              validation_steps=int(0.15 * len(test_data)), callbacks=[early_stopping_callback, lower_lr])

    # evaluating model performance
    model.evaluate(test_data)

    # saving weights
    model.save_weights(weights_food101_filename)
