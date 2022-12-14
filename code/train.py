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
import json

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Activation, MaxPooling1D, Dropout
from tensorflow.keras.utils import plot_model, to_categorical


def train(args):
    model = create_model()
    # if args.load_model == True:
    #     weight = tf.train.latest_checkpoint('./checkpoint')

    root = './Dataset/Respiratory_Sound_Database/'
    datadir = root + 'audio_and_txt_files/train/'
    json_path = './train.json'

    if os.path.exists(json_path):
        with open(json_path) as infile: 
            my_dict = json.load(infile)
            id, labels, images = np.array(my_dict['id']),np.array(my_dict['labels']),np.array(my_dict['images'])
    else:
        id, labels, images = data_points(root,datadir)
        mydict = {'id':id.tolist(),'labels':labels.tolist(),'images':images.tolist()}
        with open(json_path,'w') as output:
            json.dump(mydict, output)

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
