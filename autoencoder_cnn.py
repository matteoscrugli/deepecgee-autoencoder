#%%
# import h5py
import pandas as pd
import tensorflow as tf
import numpy as np
import sklearn
import shutil
import argparse
import json
import re
import mne
import sys, os
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.utils import class_weight
from pathlib import Path
# from classical_ml_seizures.metrics_EEG import calculate_metrics
import matplotlib.pyplot as plt

import datetime
import time
import math



arguments = ''
arguments += ' -n test'
arguments += ' -e 20'
arguments += ' -o'

parser = argparse.ArgumentParser()

parser.add_argument('-n','--name', dest='name', required=True, help="session name")
# parser.add_argument('-S','--subjects', dest='subjects', required=True, type=int, nargs='*', help="subject") #, nargs='*'
parser.add_argument('-t','--test', dest='test', type=int, help="test file") #, nargs='*'
parser.add_argument('-o','--overwrite', dest='overwrite', action='store_true', help="overwrite the session if it already exists")
parser.add_argument('-e','--epochs', dest='epochs', default = 10, type = int,  help ='training epochs')
parser.add_argument('-s','--split', dest='split', default='0.7', help="choice of dataset splitting")
parser.add_argument('-r','--randomseed', dest='randomseed', type=int, default=0, help='random seed for dataset randomization')
parser.add_argument('-f','--frame_len_sec', type=float, default=8, help='frame length in seconds') #window length
parser.add_argument('-F','--frame_shift_sec', type=float, default=8, help='frame shift in seconds') #hopsize train/valid
parser.add_argument('-T','--test_nshift', type=int, default=4, help='test frame shift in seconds')  #hopsize test (4 -> quante repliche ci sono rispetto alla finestra di training - rapportato alla riga 32)
parser.add_argument('-m','--model', dest='model', type=str, help='external model')
parser.add_argument('-c','--cnn', dest='cnn', type=str, help='anomaly cnn')

if 'ipykernel_launcher.py' in sys.argv[0]:
    args = parser.parse_args(arguments.split())
else:
    args = parser.parse_args()

np.set_printoptions(threshold=sys.maxsize)


session_name = args.name
session_overwrite = args.overwrite
num_epochs = args.epochs
dataset_split = float(args.split)
random_seed = args.randomseed
# subjects = args.subjects
test_file = args.test
external_model = args.model
s_freq = 256

class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def calculate_metrics(y_pred,y_true):
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    for i in range(len(y_pred)):
        pred = y_pred[i]
        true = y_true[i]
        if pred == 1 and true == 1:
            TP = TP + 1
        elif pred == 1 and true == 0:
            FP = FP + 1
        elif pred == 0 and true == 0:
            TN = TN + 1
        else:
            FN = FN + 1
    # print("\nTP: " + str(TP) + " FN: " + str(FN) + " FP: " + str(FP) + " TN: " + str(TN)) # ORIGINAL
    if(TP + FN != 0):
        sensitivity = (TP / (TP + FN)) * 100
    else:
        sensitivity = 0
    if(TN + FP != 0):
        specificity = (TN / (TN + FP)) * 100
    else:
        specificity = 0
    accuracy = (TP + TN) / (TN + FN + TP + FP) * 100
    if(TP + FP != 0):
        precision = (TP / (TP + FP)) * 100
    else:
        precision = 0
 #   print("precision:" + str(precision) + " sensitivity: " + str(sensitivity))
    # print(str(precision>0)) # ORIGINAL
    # print(str(sensitivity>0.0)) # ORIGINAL
    if precision > 0 or sensitivity > 0.0:
        f1_score = (2 * ((precision*sensitivity)/(precision+sensitivity)))
    else:
        f1_score = 0
    return accuracy, sensitivity, specificity, precision, f1_score, TP, FN, FP, TN

def sliding_window_majority_backwards(y_pred):
    y_pred_new = y_pred.copy()
    for i in range(len(y_pred)):
        if(i == 0):
            sums = 0 + 0 + y_pred[i]
        elif(i == 1):
            sums = 0 + y_pred[i-1] + y_pred[i]
        else:
            sums = y_pred[i-2] + y_pred[i-1] + y_pred[i]
        if(sums > 1):
            y_pred_new[i] = 1
        else:
            y_pred_new[i] = 0
    return y_pred_new

def sliding_window_majority_middle(y_pred):
    y_pred_new = y_pred.copy()
    for i in range(len(y_pred)):
        if(i == 0):
            sums = 0 + y_pred[i] + y_pred[i+1]
        elif(i == len(y_pred)-1):
            sums = y_pred[i-1] + y_pred[i] + 0
        else:
            sums = y_pred[i-1] + y_pred[i] + y_pred[i+1]
        if(sums > 1):
            y_pred_new[i] = 1
        else:
            y_pred_new[i] = 0
    return y_pred_new

def sliding_window_majority_middle_extended(y_pred):
    y_pred_new = y_pred.copy()
    for i in range(len(y_pred)):
        if(i == 0):
            sums = 0 + 0 + y_pred[i] + y_pred[i+1] + y_pred[i+2]
        elif(i == 1):
            sums = 0 + y_pred[i-1] + y_pred[i] + y_pred[i+1] + y_pred[i+2]
        elif(i == len(y_pred)-2):
            sums = y_pred[i-2] + y_pred[i-1] + y_pred[i] + y_pred[i+1] + 0
        elif(i == len(y_pred)-1):
            sums = y_pred[i-2] + y_pred[i-1] + y_pred[i] + 0 + 0
        else:
            sums = y_pred[i-2] + y_pred[i-1] + y_pred[i] + y_pred[i+1] + y_pred[i+2]
        if(sums > 2):
            y_pred_new[i] = 1
        else:
            y_pred_new[i] = 0
    return y_pred_new


frame_len_sec = args.frame_len_sec
if test_file != None:
    frame_shift_sec = args.frame_shift_sec # FIX ME
else:
    frame_shift_sec = args.frame_len_sec # FIX ME
downscaling = 1
median = 0

test_nshift = args.test_nshift # int(args.test_nshift / (frame_len_sec / frame_shift_sec))

augmentation_rsize = [0] # range(args.augrsize[0], args.augrsize[1] + 1, args.augrsize[2])
augmentation_rotat = [0] # range(args.augrotat[0], args.augrotat[1] + 1, args.augrotat[2])
augmentation_shift_nontest = [0] # range(args.augshift[0], args.augshift[1] + 1, args.augshift[2])
augmentation_shift_nontest = [int(a * 256) * 256 for a in augmentation_shift_nontest]
if not test_nshift or test_nshift == 1:
    augmentation_shift_test = [0]
else:
    augmentation_shift_test = list(np.arange(-frame_shift_sec/2, frame_shift_sec/2, frame_len_sec/test_nshift))
    augmentation_shift_test = [int(a * 256) for a in augmentation_shift_test]

if test_file != None:
    th_seizure = 0
else:
    th_seizure = 0 # use 0 so normal data does not include any portion of seizure

th_seizure_test = 0.7 #0.75

dataset_split = 0.5
non_seiz_ratio = 55

kernel = 5

# patience = 40

if test_file != None:
    learning_rate = 0.00005   #subject specific training
else:
    learning_rate = 0.0005   #training multipaziente

print('learning_rate: ', learning_rate)


# temp = 's'
# for s in subjects:
#     temp += f'_{s}'

session_path = f"output/train_raw/"
session_path += f"{session_name}/"
# session_path += f"{temp}/"
# if test_file != None:
#     session_path += f"v0_t1/" # f"v{len(allset_summary['subjects'][subject]['valid_set'])}_t{len(allset_summary['subjects'][subject]['test_set'])}/"
#     session_path += f"t{test_file}/" # f"{temp}/"

if os.path.isdir(session_path):
    if args.overwrite:
        try:
            shutil.rmtree(session_path)
            Path(session_path).mkdir(parents=True, exist_ok=True)
        except OSError:
            print("Error in session creation ("+session_path+").")
            exit()
    else:
        print(f'Session path ({session_path}) already exists')
        exit()
else:
    try:
        Path(session_path).mkdir(parents=True, exist_ok=True)
    except OSError:
        print("Error in session creation ("+session_path+").")
        exit()

out_file = session_path + 'log.txt'

################################################################################################################################################################################################
# Define neural network
################################################################################################################################################################################################

class AnomalyDetector(Model):
  def __init__(self):
    super(AnomalyDetector, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Conv2D(filters=64, kernel_size=[kernel,1], strides=[kernel,1], activation ="relu", padding='valid'),
      layers.Conv2D(filters=64, kernel_size=[kernel,1], strides=[kernel,1], activation ="relu", padding='valid')])

    self.decoder = tf.keras.Sequential([
      layers.Conv2DTranspose(filters=64, kernel_size=[kernel,1], strides=[kernel,1], activation ="relu", padding='valid', output_padding = [4,0]),
      layers.Conv2DTranspose(filters=1, kernel_size=[kernel,1], strides=[kernel,1], activation ="sigmoid", padding='valid', output_padding = [4,0])])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded[:,:-1,:,:]


def print_stats(predictions, labels):
  print("Accuracy = {}".format(accuracy_score(labels, predictions)))
  print("Precision = {}".format(precision_score(labels, predictions)))
  print("Recall = {}".format(recall_score(labels, predictions)))



def one_subject_gather_temporally(sub, sampling_frequency): #to_take second arg
    dataset_path = '/media/paola/TOSHIBA EXT/dottorato/chb-mit-scalp-eeg-database-1.0.0/'#'dataset/chb-mit-scalp-eeg-database-1.0.0/'
    filename = f"{dataset_path}chb{sub:02d}/chb{sub:02d}-summary.txt"

    with open(filename, "r", encoding = 'utf-8') as f:
        lines = f.readlines()

    lines_proc = []
    for i, l in enumerate(lines):
        if not i or l == "\n":
            lines_proc.append([])
        lines_proc[-1].append(l.strip())

    record_files = []
    seizure_EDF = []
    non_seizure_EDF = []
    seizure_number = []
    seizure_start = []
    seizure_end = []

    for l1 in lines_proc:
        if any('File Name' in l for l in l1):
            r_files = l1[[idx for idx, s in enumerate(l1) if 'File Name' in s][0]].split(':')[-1].strip()
            s_number = int(l1[[idx for idx, s in enumerate(l1) if 'Number of Seizures in File' in s][0]].split(':')[-1].strip())
            record_files.append(r_files)
            if s_number:
                seizure_EDF.append(r_files)
                seizure_number.append(s_number)
                for l2 in l1:
                    if 'Seizure' in l2:
                        if 'Start Time' in l2:
                            seizure_start.append(int(int(re.sub('\D', '', l2.split(':')[-1].strip())) * sampling_frequency))
                        if 'End Time' in l2:
                            seizure_end.append(int(int(re.sub('\D', '', l2.split(':')[-1].strip())) * sampling_frequency))
            else:
                non_seizure_EDF.append(r_files)
    total_seizures = sum(seizure_number)

    return seizure_EDF, non_seizure_EDF, record_files, seizure_number, seizure_start, seizure_end, total_seizures





if __name__ == "__main__":

################################################################################################################################################################################################
# Load dataset
################################################################################################################################################################################################
#     max_data = 1
#     np.random.seed(random_seed)

#     separate_valid = False # bool(len(allset_summary['subjects'][subject]['valid_set']))

#     X_train_t = np.empty((0, 4 ,1 , int(frame_len_sec * (256 / downscaling))), int)
#     Y_train_t = np.empty((0, ), int)
#     X_valid_t = np.empty((0, 4 ,1 , int(frame_len_sec * (256 / downscaling))), int)
#     Y_valid_t = np.empty((0, ), int)
#     if len(subjects) <= 1:
#         X_test_t = np.empty((0, 4 ,1 , int(frame_len_sec * (256 / downscaling))), int)
#         Y_test_t = np.empty((0, ), int)

#     for subject in subjects:
#         if not separate_valid:

#             seizure_EDF, non_seizure_EDF, record_files, seizure_number, seizure_start, seizure_end, total_seizures = one_subject_gather_temporally(subject, 256) #  sampling_frequency)
#             print('seizure_EDF')
#             print(seizure_EDF)          
#             print('seizure_number')
#             print(seizure_number)
#             print('seizure_start')
#             print(seizure_start)
#             print('seizure_end')
#             print(seizure_end)
#             print('total_seizures')
#             print(total_seizures)

#             if test_file != None:
#                 if test_file >= len(seizure_EDF):
#                     shutil.rmtree(session_path)
#                     exit()
#                 else:
#                     test_file_name = seizure_EDF[test_file]


#             train_list_test = non_seizure_EDF[:6] + [test_file_name]
#             print(train_list_test)

#             seizure_episodes = []

#             cnt = 0
#             for i in range(len(seizure_EDF)):

#                 seizure_episodes.append([])
#                 for n in range(seizure_number[i]):
#                     seizure_episodes[-1].append([seizure_start[cnt], seizure_end[cnt]])
#                     cnt +=1
#             X = []
#             Y = []
#             X_test = []
#             Y_test = []
          
#             f_log = open(out_file, "a")
#             for i, file in enumerate(train_list_test):
#                 print(file)
#                 data = mne.io.read_raw_edf(f'/media/paola/TOSHIBA EXT/dottorato/chb-mit-scalp-eeg-database-1.0.0/chb{subject:02d}/' + file, verbose = 0)#'dataset/chb-mit-scalp-eeg-database-1.0.0/chb{subject:02d}/' + file, verbose = 0)
#                 raw_data = data.get_data() * 1000000
#                 raw_data = raw_data[[1,2,13,14],:] #train/test on temporal channels only
#                 raw_data_len, = raw_data[0].shape

#                 if file == test_file_name:
#                     augmentation_shift = augmentation_shift_test
#                 else:
#                     augmentation_shift = augmentation_shift_nontest

#                 labels = np.zeros(raw_data.shape[1])
#                 if file in seizure_EDF:
#                   print("seizure records in list ",file)
#                   for [start, end] in seizure_episodes[seizure_EDF.index(file)]:
#                       labels[start : end] = 1

#                 print('Z')

#                 for aug_rsize in augmentation_rsize:
#                     frame_len = int(frame_len_sec * (256 * (1 + (aug_rsize / downscaling))))   #aug_rsize is 0 and downscaling is 1
#                     frame_shift = int(frame_shift_sec * (256 * (1 + (aug_rsize / downscaling))))
#                     for aug_rotat in augmentation_rotat:
#                         for frame in range(0, raw_data_len - frame_len + 1, frame_shift):
#                             for aug_shift in augmentation_shift:
#                                 if frame + aug_shift >= 0 and frame + frame_shift + aug_shift <= raw_data_len: # and sym[i] in sub_labels:
#                                     temp = sum(labels[frame + aug_shift : frame + aug_shift + frame_len : downscaling + aug_rsize])/len(labels[frame + aug_shift : frame + aug_shift + frame_len : downscaling + aug_rsize])
#                                     if file == test_file_name and len(subjects) <= 1:
#                                         if temp >= th_seizure_test:
#                                             temp_Y = 1
#                                         else:
#                                             temp_Y = 0
#                                     else:
#                                         if temp:
#                                             if temp >= th_seizure:
#                                                 temp_Y = 1
#                                             else:
#                                                 continue
#                                         else:
#                                             temp_Y = 0
#                                     if aug_rotat:
#                                         pass

#                                     else:
#                                         temp_X = [
#                                             [list(raw_data[[0], frame + aug_shift : frame + aug_shift + frame_len : downscaling + aug_rsize][0])],
#                                             [list(raw_data[[1], frame + aug_shift : frame + aug_shift + frame_len : downscaling + aug_rsize][0])],
#                                             [list(raw_data[[2], frame + aug_shift : frame + aug_shift + frame_len : downscaling + aug_rsize][0])],
#                                             [list(raw_data[[3], frame + aug_shift : frame + aug_shift + frame_len : downscaling + aug_rsize][0])]
#                                         ]

#                                     if file == test_file_name and len(subjects) <= 1:
#                                         X_test.append(temp_X)
#                                         Y_test.append(temp_Y)

#                                     else:
#                                         if not aug_rsize and not aug_shift and not aug_rotat:
#                                             if temp_Y:
#                                                 X.append(temp_X)
#                                                 Y.append(temp_Y)
#                                             else:
#                                                 X.append(temp_X)
#                                                 Y.append(temp_Y)

  
#             print('D')

#             print("seizurelabels ",sum(Y))

#             X_train = X[:round(np.size(X,0)*dataset_split)]
#             Y_train = Y[:round(np.size(X,0)*dataset_split)]
      

#             X_valid = X[round(np.size(X,0)*dataset_split):]
#             Y_valid = Y[round(np.size(X,0)*dataset_split):]
   


#             if len(subjects) <= 1:
#                 print("Test file is: ",file)
#                 X_test = np.array(X_test)
#                 Y_test = np.array(Y_test)
#                 print("seizurelabels test",sum(Y_test))  

#         train_saiz = np.size(Y_train)
#         train_Seiz = np.sum(Y_train)
#         valid_Seiz = np.sum(Y_valid)

#         print('D')

#         print('C')

#         if len(subjects) <= 1:
#             X_train_t = X_train
#             Y_train_t = Y_train
#             X_valid_t = X_valid
#             Y_valid_t = Y_valid
#             X_test_t = X_test
#             Y_test_t =Y_test
#         else:
#             X_train_t = np.concatenate((X_train_t, X_train), axis=0)
#             Y_train_t = np.concatenate((Y_train_t, Y_train), axis=0)
#             X_valid_t = np.concatenate((X_valid_t, X_valid), axis=0)
#             Y_valid_t = np.concatenate((Y_valid_t, Y_valid), axis=0)


#     x_data_T = []
#     x_data_V = []

#     min_val = tf.reduce_min(X_train_t)
#     max_val = tf.reduce_max(X_train_t)

#     X_train_t = (X_train_t - min_val) / (max_val - min_val)
#     X_valid_t = (X_valid_t - min_val) / (max_val - min_val)

#     if len(subjects) <= 1:
#        print(len(X_test_t))
#        X_test_t = (X_test_t - min_val) / (max_val - min_val)

#     # train only on normal non-seizure data
#     for index in range(len(X_train_t)):
#        if Y_train_t[index] == 0:
#          x_data_T.append(X_train_t[index])
#     for index in range(len(X_valid_t)):
#          x_data_V.append(X_valid_t[index])

#     x_data_T = np.array(x_data_T)
#     x_data_V = np.array(x_data_V)
#     print(x_data_T.shape)
#     x_data_T = np.swapaxes(x_data_T,1,3)
#     x_data_V = np.swapaxes(x_data_V,1,3)
#     nonseiz_test=[]
#     seiz_test=[]
#     if len(subjects) <= 1:
#         x_data_t=np.array(X_test_t)
#        # print(x_data_t.shape)
#         x_data_test = np.swapaxes(x_data_t,1,3)
#         y_data_t = Y_test_t
#         for i in range(x_data_test.shape[0]):
#           if y_data_t[i] == 0:
#              nonseiz_test.append(x_data_test[i])
#           else:
#              seiz_test.append(x_data_test[i])  
#         nonseiz_test = np.array(nonseiz_test)
#         seiz_test = np.array(seiz_test)
#     print("train data",x_data_T.shape, len(X_train_t))
#     print("valid data",x_data_V.shape, len(X_valid_t))
#   #  print("Seizure in training dataset: " + str(int(train_Seiz)))
#   #  print("Seizure in validation dataset: " + str(valid_Seiz))

#     np.save(session_path + "train_data.npy",x_data_T)
#     np.save(session_path + "valid_data.npy",x_data_V)


    dataset_path = 'dataset/medici/data/holter_bioharness_parsed/'
    acc_filtered = True
    acc_filtered_sat = True
    acc_axes = True


    for root, dirs, files in os.walk(dataset_path):
        for file in sorted(files):
            if '.json' in file and '2022_09_29-10_07_06' in file:
                if 'template' in file:
                    continue
                elif '-proc' in file:
                    print('PROC', dataset_path + file)
                    with open(dataset_path + file, 'r') as json_file:
                        data = json.load(json_file)
                        RR_time = data['RR']['Time']
                        if acc_filtered:
                            acc_time = data['Interpolated_Filtered_Accel']['Time']
                            if acc_filtered_sat:
                                acc_data = [acc if acc < 40 else 40 for acc in data['Interpolated_Filtered_Accel']['Module']]
                            else:
                                acc_data = data['Interpolated_Filtered_Accel']['Module']
                        else:
                            acc_time = data['Interpolated_Accel']['Time']
                            if acc_axes:
                                acc_data_x = data['Interpolated_Accel']['X']
                                acc_data_y = data['Interpolated_Accel']['Y']
                                acc_data_z = data['Interpolated_Accel']['Z']
                            else:
                                acc_data = data['Interpolated_Accel']['Module']
                elif '-quiet' in file:
                    if '400_40' in file:
                        print('QUIET', dataset_path + file)
                        with open(dataset_path + file, 'r') as json_file:
                            data = json.load(json_file)
                            RR_quiet_time = data['RR_quiet_index']
                else:
                    print(dataset_path + file)
                    with open(dataset_path + file, 'r') as json_file:
                        data = json.load(json_file)
                        ecg_data = data['ECG']['EcgWaveform']
                        ecg_time = data['ECG']['Time']
                # with open('dataset_path + file', 'r') as json_file:
                #     config = json.load(json_file)
    
    # print(acc_data)
    # exit()


    # print(len(ecg_data), len(acc_data))
    # print(type(ecg_time[0]), type(acc_time[0]), type(ecg_data[0]), type(acc_data[0]))

    ecg_time_start = datetime.datetime.strptime(ecg_time[0], "%d/%m/%Y %H:%M:%S.%f")

    # index_test = 3
    # delta = round((datetime.datetime.strptime(RR_quiet_time[index_test], "%d/%m/%Y %H:%M:%S.%f") - ecg_time_start) / datetime.timedelta(seconds = (1 / 250)))

    half_win = 999
    dataset_split = 0.7

    X_out = []
    for rr in RR_time:
        index = round((datetime.datetime.strptime(rr, "%d/%m/%Y %H:%M:%S.%f") - ecg_time_start) / datetime.timedelta(seconds = (1 / 250)))
        if index - half_win >= 0 and index + half_win < len(ecg_data):
            X_out.append([[[float(d) for d in ecg_data[index - half_win : index + half_win]]]])

    X_in = []
    for rr in RR_quiet_time:
        index = round((datetime.datetime.strptime(rr, "%d/%m/%Y %H:%M:%S.%f") - ecg_time_start) / datetime.timedelta(seconds = (1 / 250)))
        if index - half_win >= 0 and index + half_win < len(ecg_data):
            if not acc_filtered and acc_axes:
                X_in.append(
                    [[[float(d) for d in ecg_data[index - half_win : index + half_win]]],
                    [[float(d) for d in acc_data_x[index - half_win : index + half_win]]],
                    [[float(d) for d in acc_data_y[index - half_win : index + half_win]]],
                    [[float(d) for d in acc_data_z[index - half_win : index + half_win]]]])
            else:
                X_in.append(
                    [[[float(d) for d in ecg_data[index - half_win : index + half_win]]],
                    [[float(d) for d in acc_data[index - half_win : index + half_win]]]])

    X_extra = random.choices(X_in, k = len(X_out) - len(X_in))  # or [random.choice(lst) for _ in range(n)]
    X_in.extend(X_extra)

    X_out = np.array(X_out)
    X_in = np.array(X_in)
    RR_time = np.array(RR_time)

    # print(X_out.shape)
    # print(X_in.shape)
    # print(RR_time.shape)
    # exit()

    # %%

    row_indices = np.random.permutation(X_out.shape[0])
    X_out = np.take(X_out, row_indices, axis=0)
    X_in = np.take(X_in, row_indices, axis=0)
    RR_time = np.take(RR_time, row_indices, axis=0)

    ecg_min_val = np.min(X_in[:,0,0,:])
    ecg_max_val = np.max(X_in[:,0,0,:])
    if not acc_filtered and acc_axes:
        acc_min_val = np.min([X_in[:,1,0,:], X_in[:,2,0,:], X_in[:,3,0,:]])
        acc_max_val = np.max([X_in[:,1,0,:], X_in[:,2,0,:], X_in[:,3,0,:]])
    else:
        acc_min_val = np.min(X_in[:,1,0,:])
        acc_max_val = np.max(X_in[:,1,0,:])

    X_in[:, 0, :, :] -= ecg_min_val
    X_in[:, 0, :, :] /= (ecg_max_val - ecg_min_val)

    if not acc_filtered and acc_axes:
        X_in[:, 1, :, :] -= acc_min_val
        X_in[:, 1, :, :] /= (acc_max_val - acc_min_val)
        X_in[:, 2, :, :] -= acc_min_val
        X_in[:, 2, :, :] /= (acc_max_val - acc_min_val)
        X_in[:, 3, :, :] -= acc_min_val
        X_in[:, 3, :, :] /= (acc_max_val - acc_min_val)
    else:
        X_in[:, 1, :, :] -= acc_min_val
        X_in[:, 1, :, :] /= (acc_max_val - acc_min_val)

    X_out[:, 0, :, :] -= ecg_min_val
    X_out[:, 0, :, :] /= (ecg_max_val - ecg_min_val)

    X_out_train = X_out[ : round(np.size(X_out,0) * dataset_split)]
    X_out_valid = X_out[round(np.size(X_out,0) * dataset_split) : round(np.size(X_out,0) * 0.9)]
    X_out_test = X_out[round(np.size(X_out,0) * 0.9) : ]

    X_in_train = X_in[ : round(np.size(X_in,0) * dataset_split)]
    X_in_valid = X_in[round(np.size(X_in,0) * dataset_split) : round(np.size(X_in,0) * 0.9)]
    X_in_test = X_in[round(np.size(X_in,0) * 0.9) : ]

    RR_time_train = RR_time[ : round(np.size(RR_time,0) * dataset_split)]
    RR_time_valid = RR_time[round(np.size(RR_time,0) * dataset_split) : round(np.size(RR_time,0) * 0.9)]
    RR_time_test = RR_time[round(np.size(RR_time,0) * 0.9) : ]

    with open('test_time.json', 'w') as json_file:
        json.dump({'test_time' : [str(rr) for rr in RR_time_test]}, json_file, indent=4)

    X_out_train = np.swapaxes(X_out_train, 1, 3)
    X_out_valid = np.swapaxes(X_out_valid, 1, 3)
    X_out_test = np.swapaxes(X_out_test, 1, 3)
    X_in_train = np.swapaxes(X_in_train, 1, 3)
    X_in_valid = np.swapaxes(X_in_valid, 1, 3)
    X_in_test = np.swapaxes(X_in_test, 1, 3)

    print(X_out_train.shape)
    print(X_out_valid.shape)
    print(X_out_test.shape)
    print(X_in_train.shape)
    print(X_in_valid.shape)
    print(X_in_test.shape)
    print(RR_time_train.shape)
    print(RR_time_valid.shape)
    print(RR_time_test.shape)
    
    #%%

################################################################################################################################################################################################
# Instantiate Autoencoder
################################################################################################################################################################################################

    autoencoder = AnomalyDetector()


    autoencoder.compile(optimizer='adam', loss="mae")

    filepath = session_path + "mymodel.ckpt"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8,
                              patience=5, min_lr=1e-5, verbose=1)
    history = autoencoder.fit(X_in_train, X_out_train, 
          epochs=num_epochs, 
          batch_size=16,
          validation_data=(X_in_valid, X_out_valid),
          shuffle=True,callbacks=[reduce_lr]) #checkpoint,

    # fig = plt.figure(figsize=(8,8))
    # out_file=session_path + "TrainHistory.png"
    # plt.plot(history.history["loss"], label="Training Loss")
    # plt.plot(history.history["val_loss"], label="Validation Loss")
    # plt.legend()
    # fig.savefig(out_file)


    # autoencoder.load_weights(filepath+"")
    weights=autoencoder.get_weights()



    #%%
    for i, n in enumerate(range(len(X_in_test))):

        encoded_data = autoencoder.encoder(np.expand_dims(X_in_test[n], axis=0)).numpy()
        decoded_data = autoencoder.decoder(encoded_data).numpy()

        fig, ecg_plt = plt.subplots(figsize=(10,7)) #(math.ceil(math.sqrt(len(n_test))), math.ceil(math.sqrt(len(n_test))), 2)
        ecg_plt.plot(X_in_test[n, : , 0, 0], 'b', alpha=0.5)

        ecg_plt = ecg_plt.twinx()
        ecg_plt.plot(X_out_test[n, : , 0, 0], 'b')

        if not acc_filtered and acc_axes:
            acc_plt = ecg_plt.twinx()
            acc_plt.plot(X_in_test[n, : , 0, 1], 'r', alpha=0.5)
            acc_plt.plot(X_in_test[n, : , 0, 2], 'r', alpha=0.5)
            acc_plt.plot(X_in_test[n, : , 0, 3], 'r', alpha=0.5)
        else:
            acc_plt = ecg_plt.twinx()
            acc_plt.plot(X_in_test[n, : , 0, 1], 'r')
        acc_plt.set_ylim([0, 1])

        dec_plt = ecg_plt.twinx()
        dec_plt.plot(decoded_data[0, : -1, 0, 0], 'g')

        plt.title(f'{RR_time_test[n]}')
        print(f'{RR_time_test[n]}')

        plt.show()



    #%%

    first_sample = np.swapaxes(nonseiz_test[0],0,2)
    first_rec_sample = np.swapaxes(decoded_data[0],0,2)
    first_sample = np.reshape(first_sample,(first_sample.shape[0],first_sample.shape[2]))
    first_rec_sample = np.reshape(first_rec_sample,(first_rec_sample.shape[0],first_rec_sample.shape[2]))
    print(first_sample.shape)

    for i in range(1):
        fig = plt.figure(figsize=(8,8))
    plt.plot(first_sample[i] + i, 'b')
    plt.plot(first_rec_sample[i] + i, 'r')
    plt.legend(labels=["Input", "Reconstruction"])


    exit()




    if len(subjects) <= 1:
        autoencoder.load_weights(filepath+"")
        weights=autoencoder.get_weights()

        encoded_data = autoencoder.encoder(nonseiz_test).numpy()
        decoded_data = autoencoder.decoder(encoded_data).numpy()

        first_sample = np.swapaxes(nonseiz_test[0],0,2)
        first_rec_sample = np.swapaxes(decoded_data[0],0,2)
        first_sample = np.reshape(first_sample,(first_sample.shape[0],first_sample.shape[2]))
        first_rec_sample = np.reshape(first_rec_sample,(first_rec_sample.shape[0],first_rec_sample.shape[2]))
        print(first_sample.shape)

        for i in range(1):
          fig = plt.figure(figsize=(8,8))
          plt.plot(first_sample[i] + i, 'b')
          plt.plot(first_rec_sample[i] + i, 'r')
        plt.legend(labels=["Input", "Reconstruction"])
        fig.savefig("nonseiztest.png")

        encoded_data = autoencoder.encoder(seiz_test).numpy()
        decoded_data = autoencoder.decoder(encoded_data).numpy()

        first_sample = np.swapaxes(seiz_test[0],0,2)
        first_rec_sample = np.swapaxes(decoded_data[0],0,2)
        first_sample = np.reshape(first_sample,(first_sample.shape[0],first_sample.shape[2]))
        first_rec_sample = np.reshape(first_rec_sample,(first_rec_sample.shape[0],first_rec_sample.shape[2]))
        print(first_sample.shape)

        for i in range(1):
          fig = plt.figure(figsize=(8,8))
          plt.plot(first_sample[i] + i, 'b')
          plt.plot(first_rec_sample[i] + i, 'r')
        plt.legend(labels=["Input", "Reconstruction"])
        fig.savefig("seiztest.png")

        reconstructions = autoencoder.predict(x_data_V)
        train_loss = tf.keras.losses.mae(np.reshape(reconstructions,(reconstructions.shape[0],reconstructions.shape[1]*reconstructions.shape[2]*reconstructions.shape[3])),
                                         np.reshape(x_data_V,(x_data_V.shape[0],x_data_V.shape[1]*x_data_V.shape[2]*x_data_V.shape[3])))
        print(train_loss.shape)

        fig = plt.figure(figsize=(8,8))
        out_file=session_path + "TrainLoss.png"
        plt.hist(train_loss[None,:], bins=50)
        plt.xlabel("Train loss")
        plt.ylabel("No of examples")
        fig.savefig(out_file)

######################################################################################################################
#Threshold definition
######################################################################################################################

        per_ch_data = np.transpose(x_data_V,axes=[3,0,1,2])
        per_ch_recon = np.transpose(reconstructions,axes=[3,0,1,2])
        loss_ch = []

        for ch in range(per_ch_data.shape[0]):
           train_loss = tf.keras.losses.mae(np.reshape(per_ch_recon[ch],(per_ch_recon.shape[1],per_ch_recon.shape[2]*per_ch_recon.shape[3])),
                                         np.reshape(per_ch_data[ch],(per_ch_data.shape[1],per_ch_data.shape[2]*per_ch_data.shape[3])))
           loss_ch.append(train_loss)

        train_loss = np.array(loss_ch)#.swapaxes(0,1)
        print(train_loss.shape)

        mean_loss = np.zeros(per_ch_data.shape[0])
        std_loss = np.zeros(per_ch_data.shape[0])
        max_loss = np.zeros(per_ch_data.shape[0])
        threshold = np.zeros(per_ch_data.shape[0])

        for ch in range(per_ch_data.shape[0]):

           mean_loss[ch] = np.mean(train_loss[ch])
           std_loss[ch] = np.std(train_loss[ch])
           max_loss[ch] = np.max(train_loss[ch]) 

           threshold[ch] = mean_loss[ch] + std_loss[ch]

        reconstructions = autoencoder.predict(x_data_test)

        test_loss = tf.keras.losses.mae(np.reshape(reconstructions,(reconstructions.shape[0],reconstructions.shape[1]*reconstructions.shape[2]*reconstructions.shape[3])),
                                         np.reshape(x_data_test,(x_data_test.shape[0],x_data_test.shape[1]*x_data_test.shape[2]*x_data_test.shape[3])))

        fig = plt.figure(figsize=(8,8))
        out_file=session_path + "TestLoss.png"
        plt.hist(test_loss[None, :], bins=50)
        plt.xlabel("Test loss")
        plt.ylabel("No of examples")
        fig.savefig(out_file)

        f_log.close

        print("Min value is ", min_val)
        print("Max value is ", max_val)
        n_records = len(seizure_EDF)
        training_summary = {
     
            'train_path' : session_path,
            'pretrained_model' : external_model,
            'subjects' : subject,
            'preprocessing' : False,
            'frame_len_sec' : frame_len_sec,
            'frame_shift_sec' : frame_shift_sec,
            'median' : median,
            'th_seizure' : th_seizure,
            'th_seizure_test' : th_seizure_test,
            'epochs' : num_epochs,
            'learning_rate' : learning_rate,
            'batch_size' : 16,
            'threshold' : threshold.tolist(),
            'mean_loss' : mean_loss.tolist(),
            'std_loss' : std_loss.tolist(),
            'max_loss' : max_loss.tolist(),
            'min_val' : tf.keras.backend.get_value(min_val),
            'max_val' : tf.keras.backend.get_value(max_val),
            'test_file' : test_file,
            'test_file_name' : test_file_name,
            'test_frame_shift_sec' : frame_len_sec / test_nshift,
            'n_records': n_records,
        }

        with open(session_path+'training_summary.json', 'w') as json_file:
            json.dump(training_summary, json_file, indent=4)

# %%
