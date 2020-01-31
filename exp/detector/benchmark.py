import os
import sys
import csv
import glob

#from collections import Counter

import argparse
import numpy as np
from loaddata import *
from utils import *

from sklearn.ensemble import IsolationForest
from pyod.models.ocsvm import OCSVM
from sklearn.neighbors import LocalOutlierFactor
from pyod.models.abod import ABOD


def main(
        model,
        normal_data_dir,
        normal_data_name_train,
        normal_data_name_test,
        abnormal_data_dir,
        abnormal_data_name,
        window_size
    ):

    training_normal_data, ref_normal_data, val_normal_data = load_data_split(
        data_dir = normal_data_dir,
        file_name = normal_data_name_train,
        split = (0.9, 0.0, 0.1)
    )

    testing_normal_data = load_data_all(
        data_dir = normal_data_dir,
        file_name = normal_data_name_test
    )

    testing_abnormal_data = load_data_all(
        data_dir = abnormal_data_dir,
        file_name = abnormal_data_name
    )

    # Normalize training data
    training_normal_data_mean = get_mean(training_normal_data)
    training_normal_data_std = get_std(training_normal_data)

    training_normal_data = normalize(
        training_normal_data, training_normal_data_mean, training_normal_data_std
    )
    ref_normal_data = normalize(
        ref_normal_data, training_normal_data_mean, training_normal_data_std
    )
    val_normal_data = normalize(
        val_normal_data, training_normal_data_mean, training_normal_data_std
    )
    testing_normal_data = normalize(
        testing_normal_data, training_normal_data_mean, training_normal_data_std
    )
    testing_abnormal_data = normalize(
        testing_abnormal_data, training_normal_data_mean, training_normal_data_std
    )

    print "training_normal_data.shape", training_normal_data.shape
    print "ref_normal_data.shape", ref_normal_data.shape
    print "testing_normal_data.shape", testing_normal_data.shape


    training_normal_data = seq_win_vectorize(
        seq = training_normal_data,
        window_size = window_size
    )
    testing_normal_data = seq_win_vectorize(
        seq = testing_normal_data,
        window_size = window_size
    )
    testing_abnormal_data = seq_win_vectorize(
        seq = testing_abnormal_data,
        window_size = window_size
    )

    print "Vectorized training_normal_data.shape", training_normal_data.shape
    print "Vectorized ref_normal_data.shape", ref_normal_data.shape
    print "Vectorized testing_normal_data.shape", testing_normal_data.shape

    # +1 is normal, -1 is abnormal
    true_label_normal = np.ones(len(testing_normal_data))
    true_label_abnormal = -np.ones(len(testing_abnormal_data))
    true_label = np.concatenate(
        (
            true_label_normal,
            true_label_abnormal
        )
    )

    training_data_run = training_normal_data
    testing_data_run = np.concatenate(
        (
            testing_normal_data,
            testing_abnormal_data
        ),
        axis=0
    )

    assert len(testing_data_run) == len(true_label)

    reverse = False
    if model == 'IF':
        cls = IsolationForest(n_estimators=1000, contamination = 0.1, behaviour='new')
    elif model == 'OCSVM':
        cls = OCSVM()
        reverse = True
    elif model == 'LOF':
        # Bigger is better, i.e. large values correspond to inliers.
        # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html
        cls = LocalOutlierFactor(novelty=True, contamination=0.01)
        reverse = True
    elif model == 'ABOD':
        # Outliers have higher outlier scores
        cls = ABOD(contamination=1e-4)
    else:
        print "Model not support"
        exit(1)


    cls.fit(training_data_run)
    pred = cls.predict(testing_data_run)
    pred_score = cls.decision_function(testing_data_run)

    # Pay special attention here the score is the anomaly score
    tp, fp, fn, tn, acc, prec, rec, f1, fpr, tpr, thresholds = \
    eval_metrics(
        truth = true_label,
        pred = pred,
        pred_score = pred_score if not reverse else 1-pred_score
    )

    return fpr, tpr, thresholds

if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type = str, default = "OCSVM", help='Anomaly detection models')

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

    main(
        model = args.model,
        normal_data_dir = args.normal_data_dir,
        normal_data_name_train = args.normal_data_name_train,
        normal_data_name_test = args.normal_data_name_test,
        abnormal_data_dir = args.abnormal_data_dir,
        abnormal_data_name = args.abnormal_data_name,
        window_size = args.window_size
    )
