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

def eval(args):
    model = create_model()
    weight = tf.train.latest_checkpoint('./checkpoint')

    root = './Dataset/Respiratory_Sound_Database/'
    datadir = root + 'audio_and_txt_files/test/'
    json_path = './feature.json'

    if os.path.exists(json_path):
        with open(json_path) as infile: 
            my_dict = json.load(infile)
            id, labels, images = np.array(my_dict['id']),np.array(my_dict['labels']),np.array(my_dict['images'])
    else:
        id, labels, images = data_points(root,datadir)
        mydict = {'id':id.tolist(),'labels':labels.tolist(),'images':images.tolist()}
        with open(json_path,'w') as output:
            json.dump(mydict, output)

    X_test, y_test = preprocessing_test(labels, images)

    model.load_weights(weight).expect_partial()
    tf.debugging.set_log_device_placement(True)
    loss, acc = model.evaluate(tf.constant(X_test), tf.constant(y_test),
        batch_size=200,verbose=2)
    print(f"Loss: {loss}, Acc: {acc}")
    if (args.pred == True):
        pred = model.predict(X_test)
        prediction = np.array([np.argmax(i) for i in pred])
        pred_dict = zip(id, prediction)
        pd.DataFrame(pred_dict,columns=['ID','Prediction']).set_index('ID').to_csv('./prediction.csv')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred',action='store_true')
    args = parser.parse_args()
    eval(args)

if __name__ == '__main__':
    main()
