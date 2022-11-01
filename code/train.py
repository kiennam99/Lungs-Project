import os
from venv import create
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer
from sklearn.metrics import classification_report, confusion_matrix as cm
import seaborn as sn
import argparse
from tool import *

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Activation, MaxPooling1D, Dropout
from tensorflow.keras.utils import plot_model, to_categorical


def create_model():
    model = Sequential()
    model.add(Conv1D(64, kernel_size=5, activation='relu', input_shape=(193, 1)))

    model.add(Conv1D(128, kernel_size=5, activation='relu'))
    model.add(MaxPooling1D(2)) 

    model.add(Conv1D(256, kernel_size=5, activation='relu'))

    model.add(Dropout(0.3))
    model.add(Flatten())

    model.add(Dense(512, activation='relu'))   
    model.add(Dense(8, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train(args):
    model = create_model()
    # if args.load_model == True:
    #     weight = tf.train.latest_checkpoint('./checkpoint')

    lb_path = './label.npy'
    img_path = './img.npy'
    root = './Dataset/Respiratory_Sound_Database/'
    
    if args.load_feature == True:

        labels = np.load(lb_path)
        images = np.load(img_path)
    else:
        start = timer()
        labels, images = data_points(root)
        np.save(lb_path,labels)
        np.save(img_path,images)
        print('Time taken: ', (timer() - start))

    X_train, X_test, y_train, y_test = preprocessing(labels, images)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='./checkpoint/cp.ckpt',
                                                        save_weight_only = True,
                                                        verbose = 1
                                                        )
    tf.debugging.set_log_device_placement(True)
    model.fit(tf.constant(X_train), tf.constant(y_train), validation_data=(X_test, y_test), epochs=70,
        batch_size=200, callbacks=[cp_callback],verbose=1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_model',action='store_true')
    parser.add_argument('--load_feature',action='store_true')

    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()
