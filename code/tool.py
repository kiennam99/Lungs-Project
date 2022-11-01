import os
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer
from sklearn.metrics import classification_report, confusion_matrix as cm
import seaborn as sn

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Activation, MaxPooling1D, Dropout
from tensorflow.keras.utils import plot_model, to_categorical

def preprocessing(labels, images):    

  # Remove Asthma and LRTI
  # images = np.delete(images, np.where((labels == 7) | (labels == 6))[0], axis=0) 
  # labels = np.delete(labels, np.where((labels == 7) | (labels == 6))[0], axis=0)      

  # Split data
  X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=5)
  
  # Hot one encode the labels
  y_train = to_categorical(y_train)
  y_test = to_categorical(y_test,num_classes=8)  

  # Format new data
  y_train = np.reshape(y_train, (y_train.shape[0], 8))
  X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
  y_test = np.reshape(y_test, (y_test.shape[0], 8))
  X_test = np.reshape(X_test, (X_test.shape[0], X_train.shape[1],  1))

  return X_train, X_test, y_train, y_test

def preprocessing_test(labels, images):    
  
  # Hot one encode the labels
  y_test = to_categorical(labels,num_classes=8)  

  # Format new data
  y_test = np.reshape(y_test, (y_test.shape[0], 8))
  X_test = np.reshape(images, (images.shape[0], images.shape[1],  1))

  return X_test, y_test


def audio_features(filename): 
  sound, sample_rate = librosa.load(filename)
  stft = np.abs(librosa.stft(sound))  
 
  mfccs = np.mean(librosa.feature.mfcc(y=sound, sr=sample_rate, n_mfcc=40),axis=1)
  chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate),axis=1)
  mel = np.mean(librosa.feature.melspectrogram(y=sound, sr=sample_rate),axis=1)
  contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate),axis=1)
  tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(sound), sr=sample_rate),axis=1)
    
  concat = np.concatenate((mfccs,chroma,mel,contrast,tonnetz))
  return concat

def data_points(root,datadir):
  labels = []
  images = []
  id = []

  to_hot_one = {"COPD":0, "Healthy":1, "URTI":2, "Bronchiectasis":3, "Pneumonia":4, "Bronchiolitis":5, "Asthma":6, "LRTI":7}

  count = 0
  for f in diagnosis_data(root,datadir):
    print(count)
    id.append(f.id)
    labels.append(to_hot_one[f.diagnosis]) 
    images.append(audio_features(f.image_path))
    count+=1

  return np.array(id), np.array(labels), np.array(images)


class Diagnosis():
    def __init__(self, id, diagnosis, image_path):
        self.id = id
        self.diagnosis = diagnosis
        self.image_path = image_path 

def diagnosis_data(root,datadir): 
  diagnosis = pd.read_csv(root+'patient_diagnosis.csv')
  
  wav_files, audio_path = get_wav_files(datadir)
  diag_dict = { 101 : "URTI"}  
  diagnosis_list = []
  
  for index , row in diagnosis.iterrows():
    diag_dict[row[0]] = row[1]     

  for f in wav_files:
    diagnosis_list.append(Diagnosis(int(f[:3]), diag_dict[int(f[:3])], audio_path+f))  

  return diagnosis_list

def get_wav_files(datadir):
  audio_path = datadir
  files = [f for f in os.listdir(audio_path) if os.path.isfile(os.path.join(audio_path, f))]  #Gets all files in dir
  wav_files = [f for f in files if f.endswith('.wav')]  # Gets wav files 
  wav_files = sorted(wav_files)
  return wav_files, audio_path

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