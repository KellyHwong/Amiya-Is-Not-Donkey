#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Feb-03-20 11:09
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org

import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from keras import backend as K
import tensorflow as tf


def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


def simple_CNN(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS):
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization

    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(
        IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    # 2 because we have cat and dog classes
    model.add(Dense(2, activation='softmax'))

    # model.compile(loss='categorical_crossentropy',
    #   optimizer='rmsprop', metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy', auc])

    return model


def main():
    pass


if __name__ == "__main__":
    main()
