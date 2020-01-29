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
    parser.add_argument('--window_size', type = int, default = 10, help='Window size for vectorization')

    args = parser.parse_args()

    training_normal_data, ref_normal_data, val_normal_data = load_data_split(
        data_dir = args.normal_data_dir,
        file_name = args.normal_data_name_train,
        split = (0.9, 0.0, 0.1)
    )

    testing_normal_data = load_data_all(
        data_dir = args.normal_data_dir,
        file_name = args.normal_data_name_test
    )

    testing_abnormal_data = load_data_all(
        data_dir = abnormal_data_dir,
        file_name = abnormal_data_name
    )

    print "training_normal_data.shape", training_normal_data.shape
    print "ref_normal_data.shape", ref_normal_data.shape
    print "testing_normal_data.shape", testing_normal_data.shape


    training_normal_data = seq_win_vectorize(
        seq = training_normal_data,
        window_size = args.window_size
    )
    testing_normal_data = seq_win_vectorize(
        seq = testing_normal_data,
        window_size = args.window_size
    )
    testing_abnormal_data = seq_win_vectorize(
        seq = testing_abnormal_data,
        window_size = args.window_size
    )

    training_data_run = training_normal_data
    testing_data_run = np.concatinate(
        (
            testing_normal_data,
            testing_abnormal_data
        ),
        axis=0
    )

    cls = IsolationForest(n_estimators=1000, contamination = 0.1)
    cls.fit(training_data_run)
    pred = cls.predict(testing_data_run)
    pred_score = cls.score_samples(testing_data_run)

    eval_metrics(
        truth = true_label,
        pred = pred,
        pred_score = pred_score
    )