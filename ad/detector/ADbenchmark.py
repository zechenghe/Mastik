import os
import sys
import time

#from collections import Counter
import argparse
import numpy as np
import loaddata
import utils

from sklearn.ensemble import IsolationForest
from pyod.models.ocsvm import OCSVM
from pyod.models.lof import LOF
from pyod.models.abod import ABOD
from pyod.models.pca import PCA


def run_benchmark(
        model,
        training_normal_data,
        testing_normal_data,
        testing_abnormal_data,
        window_size,
        n_samples_train = None,
        verbose = True
    ):

    # Normalize training data
    training_normal_data_mean = utils.get_mean(training_normal_data)
    training_normal_data_std = utils.get_std(training_normal_data)

    training_normal_data = utils.normalize(
        training_normal_data, training_normal_data_mean, training_normal_data_std
    )

    testing_normal_data = utils.normalize(
        testing_normal_data, training_normal_data_mean, training_normal_data_std
    )
    testing_abnormal_data = utils.normalize(
        testing_abnormal_data, training_normal_data_mean, training_normal_data_std
    )
    if verbose:
        print("training_normal_data.shape", training_normal_data.shape)
        print("testing_normal_data.shape", testing_normal_data.shape)
        print("testing_abnormal_data.shape", testing_abnormal_data.shape)


    training_normal_data = utils.seq_win_vectorize(
        seq = training_normal_data,
        window_size = window_size,
        n_samples = n_samples_train,
    )
    testing_normal_data = utils.seq_win_vectorize(
        seq = testing_normal_data,
        window_size = window_size
    )
    testing_abnormal_data = utils.seq_win_vectorize(
        seq = testing_abnormal_data,
        window_size = window_size
    )

    if verbose:
        print("Vectorized training_normal_data.shape", training_normal_data.shape)
        print("Vectorized testing_normal_data.shape", testing_normal_data.shape)
        print("Vectorized testing_abnormal_data.shape", testing_abnormal_data.shape)

    # +1 is normal, -1 is abnormal
    true_label_normal = np.ones(len(testing_normal_data))
    true_label_abnormal = -np.ones(len(testing_abnormal_data))
    true_label = np.concatenate(
        (
            true_label_normal,
            true_label_abnormal
        ),
        axis=0
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

    if model == 'IF':
        cls = IsolationForest(n_estimators=1000, contamination = 0.1)

    elif model == 'OCSVM':
        cls = OCSVM(kernel='linear', nu=0.1, contamination=0.1)

    elif model == 'LOF':
        cls = LOF(contamination=0.1)
    elif model == 'ABOD':
        # Outliers have higher outlier scores
        cls = ABOD(contamination=1e-4)
    elif model == 'PCA':
        cls = PCA()
    else:
        print("Model not support")
        exit(1)

    time_start = time.time()
    cls.fit(training_data_run)

    time_train_finish = time.time()
    print("Training takes {time} seconds".format(
        time=time_train_finish-time_start
        ))

    pred = cls.predict(testing_data_run)

    time_eval_finish = time.time()
    print("Evaluation takes {time} seconds".format(
        time=time_eval_finish-time_train_finish
        ))

    pred_score = cls.decision_function(testing_data_run)
    if need_convert:
        anomaly_score = 1 - pred_score
    else:
        anomaly_score = pred_score

    # Pay special attention here the score is the anomaly score
    tp, fp, fn, tn, acc, prec, rec, f1, fpr, tpr, thresholds, roc_auc = \
    utils.eval_metrics(
        truth = true_label,
        pred = pred,
        anomaly_score = pred_score if not reverse else 1-pred_score,
        verbose = verbose
    )

    return fpr, tpr, thresholds, roc_auc

if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type = str, default = "all", help='Anomaly detection models')

    # Loaddata
    # Sequential data in the form of (Timeframe, Features)
    # Training only leverages normal data. Abnormaly data only for testing.
    parser.add_argument('--data_dir', type = str, default = "../perf/data/core0/100us/", help='The directory of data')
    parser.add_argument('--train_normal', type = str, default = "train_normal.npy", help='The file name of training normal data')
    parser.add_argument('--test_normal', type = str, default = "test_normal.npy", help='The file name of testing normal data')
    parser.add_argument('--test_abnormal', type = str, default = "test_abnormal.npy", help='The file name of testing abnormal data')

    # Window size
    parser.add_argument('--window_size', type = int, default = 10, help='Window size for vectorization')
    parser.add_argument('--verbose', type = boolean, default = False)
    args = parser.parse_args()

    train_normal = np.load(args.data_dir + args.train_normal)
    test_normal = np.load(args.data_dir + args.test_normal)
    test_abnormal = np.load(args.data_dir + args.test_abnormal)

    if args.model == 'all':
        for model in ['LOF', 'OCSVM', 'IF', 'PCA']:
            print("Model: ", model)
            fpr, tpr, thresholds, roc_auc = run_benchmark(
                model = model,
                training_normal_data=train_normal,
                testing_normal_data=test_normal,
                testing_abnormal_data=test_abnormal,
                window_size = 200,
                n_samples_train = 100,   # Randomly sample 20,000 samples for training
                verbose = args.verbose
            )
            print ('model', model, 'ROC_AUC:', roc_auc)
            results_dir = 'roc/'
            np.save(results_dir + model + '_fpr', fpr)
            np.save(results_dir + model + '_tpr', tpr)
    else:
        model = args.model
        fpr, tpr, thresholds, roc_auc = run_benchmark(
            model = model,
            training_normal_data=train_normal,
            testing_normal_data=test_normal,
            testing_abnormal_data=test_abnormal,
            window_size = 200,
            n_samples_train = 100,   # Randomly sample 20,000 samples for training
            verbose = False
        )

    print ('model', model, 'ROC_AUC:', roc_auc)
    results_dir = 'roc/'
    np.save(results_dir + model + '_fpr', fpr)
    np.save(results_dir + model + '_tpr', tpr)
