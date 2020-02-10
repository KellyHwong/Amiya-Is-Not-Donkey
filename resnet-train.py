#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Feb-03-20 23:44
# @Author  : Your Name (you@example.org)
# @Link    : https://www.kaggle.com/c/semi-conductor-image-classification-first

import os
import numpy as np
import pandas as pd
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator, load_img
from sklearn.model_selection import train_test_split
from resnet import model_depth, resnet_v2, lr_schedule
from model import auc

# Training parameters
IF_DATA_AUGMENTATION = False
NUM_CLASSES = 2
IMAGE_WIDTH = IMAGE_HEIGHT = 224
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 1
INPUT_SHAPE = [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS]
# data path
TRAIN_DATA_DIR = "./data/train/"
SAVES_DIR = "./models-resnetv2/"

# constants
IF_FAST_RUN = True
EPOCHS_OVER_NIGHT = 50
BATCH_SIZE = 15
# BATCH_SIZE = 32  # orig paper trained all networks with batch_size=128


def main():
    """ Prepare Model """
    n = 2
    version = 2

    depth = model_depth(n, version)
    model_type = 'ResNet%dv%d' % (depth, version)

    model = resnet_v2(input_shape=INPUT_SHAPE, depth=depth, num_classes=2)

    model.compile(loss='categorical_crossentropy',
                  #   optimizer=Adam(learning_rate=lr_schedule(0)),
                  optimizer='adam',
                  metrics=['accuracy', auc])
    model.summary()
    print(model_type)

    if os.path.isfile("model-" + model_type + ".h5"):
        model.load_weights("model-" + model_type + ".h5")

    # Prepare model model saving directory.
    model_name = "%s.{epoch:03d}-auc-{auc:.4f}.h5" % model_type
    if not os.path.isdir(SAVES_DIR):
        os.makedirs(SAVES_DIR)
    filepath = os.path.join(SAVES_DIR, model_name)

    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(filepath=filepath, monitor="auc", verbose=1)
    earlystop = EarlyStopping(patience=10)
    learning_rate_reduction = ReduceLROnPlateau(monitor="auc",
                                                patience=2,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.00001)
    callbacks = [earlystop, learning_rate_reduction, checkpoint]

    """Prepare Data Frame"""
    filenames = os.listdir(TRAIN_DATA_DIR)
    categories = []
    for filename in filenames:
        category = filename.split('.')[0]
        if category == 'bad':  # bad 1
            categories.append(1)
        else:  # good 0
            categories.append(0)
    df = pd.DataFrame({
        'filename': filenames,
        'category': categories
    })

    """Prepare data"""
    df["category"] = df["category"].replace({0: 'good', 1: 'bad'})

    """ 这里用来自动划分 train 集和 val 集 """
    train_df, validate_df = train_test_split(
        df, test_size=0.20, random_state=42)
    train_df = train_df.reset_index(drop=True)
    validate_df = validate_df.reset_index(drop=True)

    # train_df['category'].value_counts().plot.bar()

    # size of train
    total_train = train_df.shape[0]
    total_validate = validate_df.shape[0]

    print('Using real-time data augmentation.')

    """Traning Generator"""
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        TRAIN_DATA_DIR,
        x_col='filename',
        y_col='category',
        target_size=IMAGE_SIZE,
        color_mode="grayscale",
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    """Validation Generator"""
    validation_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = validation_datagen.flow_from_dataframe(
        validate_df,
        TRAIN_DATA_DIR,
        x_col='filename',
        y_col='category',
        target_size=IMAGE_SIZE,
        color_mode="grayscale",
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    """Example Generation"""
    example_df = train_df.sample(n=1).reset_index(drop=True)
    example_generator = train_datagen.flow_from_dataframe(
        example_df,
        TRAIN_DATA_DIR,
        x_col='filename',
        y_col='category',
        target_size=IMAGE_SIZE,
        color_mode="grayscale",
        class_mode='categorical'
    )

    """Fit Model"""
    epochs = 3 if IF_FAST_RUN else EPOCHS_OVER_NIGHT
    history = model.fit_generator(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=total_validate//BATCH_SIZE,
        steps_per_epoch=total_train//BATCH_SIZE,
        callbacks=callbacks
    )

    """Save Model"""
    model.save_weights("model-" + model_type + ".h5")


if __name__ == "__main__":
    main()
