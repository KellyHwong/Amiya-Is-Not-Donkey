#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Feb-03-20 23:44
# @Author  : Your Name (you@example.org)
# @Link    : https://www.kaggle.com/c/semi-conductor-image-classification-first

import os
import json
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras  # tf2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator, load_img

from sklearn.model_selection import train_test_split

from resnet import model_depth, resnet_v2, lr_schedule
# from model import auc # tf1
from keras.metrics import AUC  # 等价于 from tf.keras.metrics import AUC
from metrics import AUC0

# Training parameters we care
BATCH_SIZE = 16
# BATCH_SIZE = 32  # orig paper trained all networks with batch_size=128

# Training parameters
START_EPOCH = 0
IF_FAST_RUN = False
TRAINING_EPOCHS = 50

IF_DATA_AUGMENTATION = True
NUM_CLASSES = 2
IMAGE_WIDTH = IMAGE_HEIGHT = 224
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 1
INPUT_SHAPE = [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS]


METRICS = [
    keras.metrics.BinaryAccuracy(name='accuracy'),  # 整体的 accuracy
    AUC(name='auc_good_0'),  # 实际上是以 good 为 positive 的 AUC
    AUC0(name='auc_bad_1')  # 以 bad 为 positive 的 AUC
]


def config_env():
    print("If in eager mode: ", tf.executing_eagerly())
    assert tf.__version__[0] == "2"
    print("Use tensorflow version 2.")

    print("Load Config ...")
    with open('./config.json', 'r') as f:
        CONFIG = json.load(f)
    return CONFIG


def main():
    CONFIG = config_env()
    ROOT_PATH = CONFIG["ROOT_PATH"]
    TRAIN_DATA_DIR = os.path.join(ROOT_PATH, CONFIG["TRAIN_DATA_DIR"])

    print("Prepare Model")
    n = 2  # order of ResNetv2, 2 or 6
    version = 2
    depth = model_depth(n, version)
    MODEL_TYPE = 'ResNet%dv%d' % (depth, version)
    SAVES_DIR = os.path.join(ROOT_PATH, "models-%s/" % MODEL_TYPE)
    MODEL_CKPT_FILE = os.path.join(SAVES_DIR, CONFIG["MODEL_CKPT_FILE"])
    if not os.path.exists(SAVES_DIR):
        os.mkdir(SAVES_DIR)

    model = resnet_v2(input_shape=INPUT_SHAPE, depth=depth, num_classes=2)

    model.compile(loss='categorical_crossentropy',
                  #   optimizer=Adam(learning_rate=lr_schedule(0)),
                  optimizer='adam',
                  metrics=METRICS)
    model.summary()
    print(MODEL_TYPE)

    print("Resume Training...")
    model_ckpt_file = MODEL_CKPT_FILE
    if os.path.isfile(model_ckpt_file):
        print("Model ckpt found! Loading...:%s" % model_ckpt_file)
        model.load_weights(model_ckpt_file)

    # Prepare model model saving directory.
    model_name = "%s.start-%d-epoch-{epoch:03d}-auc_good_0-{auc_good_0:.4f}-auc_bad_1-{auc_bad_1:.4f}.h5" % (
        MODEL_TYPE, START_EPOCH)

    filepath = os.path.join(SAVES_DIR, model_name)
    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(
        filepath=filepath, monitor="auc_good_0", verbose=1)
    earlystop = EarlyStopping(patience=10)
    learning_rate_reduction = ReduceLROnPlateau(monitor="auc_good_0",
                                                patience=2,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.00001)
    callbacks = [learning_rate_reduction, checkpoint]  # 不要 earlystop

    print("Prepare data frame from train dir: ", TRAIN_DATA_DIR)
    # filenames = os.listdir(TRAIN_DATA_DIR)
    # categories = []
    # for filename in filenames:
    #     category = filename.split('.')[0]
    #     if category == 'donkey':  # donkey 1
    #         categories.append(1)
    #     else:  # rabbit 0
    #         categories.append(0)
    # df = pd.DataFrame({
    #     'filename': filenames,
    #     'category': categories
    # })

    # """ 这里用来自动划分 train 集和 val 集 """
    # train_df, validate_df = train_test_split(
    #     df, test_size=0.20, random_state=42)
    # train_df = train_df.reset_index(drop=True)
    # validate_df = validate_df.reset_index(drop=True)

    # train_df['category'].value_counts().plot.bar()

    # size of train
    # total_train = train_df.shape[0]
    # total_validate = validate_df.shape[0]

    print('Using real-time data augmentation.')
    print("Training Generator...")
    train_datagen = ImageDataGenerator(
        validation_split=0.2,
        rescale=1./255,
        rotation_range=15,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DATA_DIR,
        subset='training',
        target_size=IMAGE_SIZE,
        color_mode="grayscale",
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42
    )

    print("Validation Generator...")
    valid_datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)
    validation_generator = valid_datagen.flow_from_directory(
        TRAIN_DATA_DIR,
        subset='validation',
        target_size=IMAGE_SIZE,
        color_mode="grayscale",
        class_mode='categorical',
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42
    )

    # """Example Generation"""
    # example_df = train_df.sample(n=1).reset_index(drop=True)
    # example_generator = train_datagen.flow_from_dataframe(
    #     example_df,
    #     TRAIN_DATA_DIR,
    #     x_col='filename',
    #     y_col='category',
    #     target_size=IMAGE_SIZE,
    #     color_mode="grayscale",
    #     class_mode='categorical'
    # )

    images_per_epoch = 100
    print("Fit Model...")
    epochs = 3 if IF_FAST_RUN else TRAINING_EPOCHS
    history = model.fit_generator(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=images_per_epoch//BATCH_SIZE,
        steps_per_epoch=images_per_epoch//BATCH_SIZE,
        callbacks=callbacks
    )

    print("Save Model...")
    model.save_weights("model-" + MODEL_TYPE + ".h5")

    print("Save History...")
    with open('./history', 'wb') as pickle_file:
        pickle.dump(history.history, pickle_file)


if __name__ == "__main__":
    main()
