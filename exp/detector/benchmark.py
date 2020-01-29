import os
import sys
import csv
import glob

#from collections import Counter

import numpy as np
import utils

from StatHistoryBatchGenerator import StatHistoryBatchGenerator as HistoryBatchGenerator
from sklearn.ensemble import IsolationForest


if __name__=="__main__":

    # Loaddata
    # Sequential data in the form of (Timeframe, Features)
    # Training only leverages normal data. Abnormaly data only for testing.
    parser.add_argument('--normal_data_dir', type = str, default = "data/", help='The directory of normal data')
    parser.add_argument('--normal_data_name_train', type = str, default = "baseline_train.npy", help='The file name of training normal data')
    parser.add_argument('--normal_data_name_test', type = str, default = "baseline_test.npy", help='The file name of testing normal data')

    parser.add_argument('--abnormal_data_dir', type = str, default = "data/", help='The directory of abnormal data')
    parser.add_argument('--abnormal_data_name', type = str, default = "attack1_test.npy", help='The file name of abnormal data')

    # Window size
    parser.add_argument('--window_size', type = int, default = 10, help='Window size of vectorization')

    args = parser.parse_args()

    training_normal_data, ref_normal_data, testing_normal_data = load_normal_data(
        data_dir = args.normal_data_dir, file_name = args.normal_data_name_train, split = (0.8, 0.1, 0.1))

    testing_normal_data, testing_normal_data, testing_normal_data = load_normal_data(
        data_dir = normal_data_dir, file_name = normal_data_name_test, split=(0.0, 0.0, 1.0))

    testing_abnormal_data = load_abnormal_data(data_dir = abnormal_data_dir,
        file_name = abnormal_data_name, split=(0.0, 0.0, 1.0))

    print "training_normal_data.shape", training_normal_data.shape
    print "ref_normal_data.shape", ref_normal_data.shape
    print "testing_normal_data.shape", testing_normal_data.shape



    training_wrapper = HistoryBatchGenerator(training_data, training_label, windowSize)

    testing_data_normal = testing_data[testing_label[:,0] == 0, :]
    testing_label_normal = testing_label[testing_label[:,0] == 0]
    testing_data_anor = testing_data[testing_label[:,0] == 1, :]
    testing_label_anor = testing_label[testing_label[:,0] == 1]

    testing_normal_wrapper = HistoryBatchGenerator(testing_data_normal, testing_label_normal, windowSize)
    testing_anor_wrapper = HistoryBatchGenerator(testing_data_anor, testing_label_anor, windowSize)
    #val_wrapper = BatchGenerator(val_data, val_label, unroll_num)


    training_data_run, training_label_run = training_wrapper.next(training_sample_size)

    testing_data_normal_run, testing_label_normal_run = testing_normal_wrapper.next(testing_sample_size)
    testing_data_anor_run, testing_label_anor_run = testing_anor_wrapper.next(testing_sample_size)
    testing_data_run = np.concatenate((testing_data_normal_run, testing_data_anor_run), axis=0)
    testing_label_run = np.concatenate((testing_label_normal_run, testing_label_anor_run), axis=0)

    # Debug
    #testing_data_run, testing_label_run = testing_normal_wrapper.next(testing_sample_size)


    assert training_data_run.shape[0] == training_label_run.shape[0]
    assert testing_data_run.shape[0] == testing_label_run.shape[0]

    print "training_data_run.shape", training_data_run.shape
    print "testing_data_run.shape", testing_data_run.shape


    cls = IsolationForest(n_estimators=1000, contamination = 0.1)
    cls.fit(training_data_run)
