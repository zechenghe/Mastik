import time
import math
import os
import numpy as np
import pickle

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

from SeqGenerator import *
from detector import *
from loaddata import *
from utils import *
from sklearn.metrics import roc_auc_score


def train(args):

    debug = args.debug
    gpu = args.gpu

    Nhidden = args.Nhidden                      # LSTM hidden nodes

    Nbatches = args.Nbatches                    # Training batches
    BatchSize = args.BatchSize                  # Training batch size
    ChunkSize = args.ChunkSize                  # The length for accumulating loss in training
    SubseqLen = args.SubseqLen                  # Split the training sequence into subsequences
    LearningRate = args.LearningRate            # Learning rate
    Eps = args.Eps                              # Eps used in Adam optimizer
    AMSGrad = args.AMSGrad                      # Use AMSGrad in Adam
    LRdecrease = args.LRdecrease                # Decrease learning rate

    save_model_dir = args.save_model_dir
    save_model_name = args.save_model_name

    normal_data_dir = args.normal_data_dir
    normal_data_name_train = args.normal_data_name_train

    RED_collection_len = args.RED_collection_len
    RED_points = args.RED_points
    Pvalue_th = args.Pvalue_th

    if args.dummydata:
        training_normal_data, ref_normal_data, testing_normal_data = load_normal_dummydata()
    else:
        training_normal_data, ref_normal_data, testing_normal_data = load_data_split(
            data_dir = normal_data_dir,
            file_name = normal_data_name_train,
            split = (0.8, 0.1, 0.1)
        )

    training_normal_data_mean = get_mean(training_normal_data)
    training_normal_data_std = get_std(training_normal_data)

    Nfeatures = training_normal_data.shape[1]
    AnomalyDetector = Detector(
        input_size = Nfeatures,
        hidden_size = Nhidden,
        th = Pvalue_th
    )
    AnomalyDetector.set_mean(training_normal_data_mean)
    AnomalyDetector.set_std(training_normal_data_std)

    training_normal_data = AnomalyDetector.normalize(training_normal_data)
    ref_normal_data = torch.tensor(AnomalyDetector.normalize(ref_normal_data))
    testing_normal_data = torch.tensor(AnomalyDetector.normalize(testing_normal_data))

    training_normal_wrapper = SeqGenerator(training_normal_data)
    training_normal_len = len(training_normal_data)

    MSELossLayer = torch.nn.MSELoss()
    optimizer = optim.Adam(
        params = AnomalyDetector.parameters(),
        lr = LearningRate,
        eps = Eps,
        amsgrad = True
    )

    if gpu:
        ref_normal_data = ref_normal_data.cuda()
        MSELossLayer = MSELossLayer.cuda()
        AnomalyDetector = AnomalyDetector.cuda()

    if debug:
        for name, para in AnomalyDetector.named_parameters():
            print name, para.size()

    for batch in range(Nbatches):
        t = 0
        init_state = (torch.zeros(1, BatchSize, Nhidden),
                    torch.zeros(1, BatchSize, Nhidden))
        training_batch = torch.tensor(training_normal_wrapper.next(BatchSize, SubseqLen))

        if gpu:
            init_state = (init_state[0].cuda(), init_state[1].cuda())
            training_batch = training_batch.cuda()

        state = init_state
        loss_hist = []
        while t + ChunkSize + 1 < SubseqLen:

            AnomalyDetector.zero_grad()

            pred, state = AnomalyDetector.forward(training_batch[t:t+ChunkSize, :, :], state)
            truth =  training_batch[t+1 : t+ChunkSize+1, :, :]

            loss = MSELossLayer(pred, truth)

            if debug:
                print "pred.size ", pred.size(), "truth.size ", truth.size()

            loss.backward()
            optimizer.step()

            if gpu:
                loss_hist.append(loss.detach().cpu().numpy())
            else:
                loss_hist.append(loss.detach().numpy())

            if debug:
                print "t =", t, 'Training loss: ', loss

            state = (state[0].detach(), state[1].detach())
            t += ChunkSize

        #plotsignal([truth.detach().numpy(), pred.detach().numpy()])

        print "Batch", batch, "Training loss", np.mean(loss_hist)

        if (batch + 1) % LRdecrease == 0:
            LearningRate = LearningRate / 2.0
            setLearningRate(optimizer, LearningRate)

    print "Training Done"
    print "Getting RED"

    AnomalyDetector.set_RED_config(RED_collection_len = RED_collection_len, RED_points = RED_points)
    AnomalyDetector.collect_ref_RED(ref_normal_data, gpu)

    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    torch.save(AnomalyDetector, save_model_dir + save_model_name)
    print "Model saved"

def eval_detector(args):

    load_model_dir = args.load_model_dir
    load_model_name = args.load_model_name
    normal_data_dir = args.normal_data_dir
    normal_data_name_test = args.normal_data_name_test
    abnormal_data_dir = args.abnormal_data_dir
    abnormal_data_name = args.abnormal_data_name
    Pvalue_th = args.Pvalue_th

    gpu = args.gpu

    AnomalyDetector = torch.load(load_model_dir + load_model_name)
    AnomalyDetector.eval()
    AnomalyDetector.th = Pvalue_th

    if args.dummydata:
        training_normal_data, ref_normal_data, testing_normal_data = load_normal_dummydata()
    else:
        training_normal_data, ref_normal_data, testing_normal_data = load_data_all(
            data_dir = normal_data_dir,
            file_name = normal_data_name_test
        )

    testing_normal_data = torch.tensor(AnomalyDetector.normalize(testing_normal_data))


    if args.dummydata:
        testing_abnormal_data = load_abnormal_dummydata()
    else:
        testing_abnormal_data = load_data_all(
            data_dir = abnormal_data_dir,
            file_name = abnormal_data_name
        )

    testing_abnormal_data = torch.tensor(AnomalyDetector.normalize(testing_abnormal_data))
    print "testing_abnormal_data.shape ", testing_abnormal_data.shape

    if gpu:
        AnomalyDetector = AnomalyDetector.cuda()
        testing_normal_data = testing_normal_data.cuda()
        testing_abnormal_data = testing_abnormal_data.cuda()


    true_label_normal = np.zeros(len(testing_normal_data) - AnomalyDetector.RED_collection_len * AnomalyDetector.RED_points - 1)
    true_label_abnormal = np.ones(len(testing_abnormal_data) - AnomalyDetector.RED_collection_len * AnomalyDetector.RED_points - 1)
    true_label = np.concatenate((true_label_normal, true_label_abnormal), axis=0)

    pred_normal, p_values_normal = AnomalyDetector.predict(testing_normal_data, gpu)
    #print "p_values_normal", p_values_normal[:100]
    #print "p_values_normal", p_values_normal[100:200]

    #print "p_values_normal", p_values_normal[-100:]
    #print "p_values_normal", p_values_normal[-200:-100]

    print "p_values_normal.shape ", len(p_values_normal)
    print "p_values_normal.mean ", np.mean(p_values_normal)

    pred_abnormal, p_values_abnormal = AnomalyDetector.predict(testing_abnormal_data, gpu)
    print "p_values_abnormal.shape ", len(p_values_abnormal)
    print "p_values_abnormal.mean ", np.mean(p_values_abnormal)

    pred = np.concatenate((pred_normal, pred_abnormal), axis=0)
    pred_scores = np.concatenate((p_values_normal, p_values_abnormal), axis=0)
    print "true_label.shape", true_label.shape, "pred.shape", pred.shape
    eval_metrics(true_label, pred)
    roc_auc = roc_auc_score(true_label, 1-pred_scores)
    print "ROC AUC = ", roc_auc


if __name__ == '__main__':
    import argparse
    import sys
    import traceback

    try:
        parser = argparse.ArgumentParser()

        # Training or Testing?
        parser.add_argument('--training', dest='training', action='store_true', help='Flag for training')
        parser.add_argument('--testing', dest='training', action='store_false', help='Flag for testing')
        parser.set_defaults(training=True)

        # Real data (private) or dummy data?
        parser.add_argument('--dummy', dest='dummydata', action='store_true', help='If dummy data is used instead of an input file')
        parser.set_defaults(dummydata=False)

        # LSTM Network config, can use other sequential models
        parser.add_argument('--Nhidden', type = int, default = 64, help='Number of hidden nodes in a LSTM cell')

        # Training parameters
        parser.add_argument('--Nbatches', type = int, default = 100, help='Number of batches in training')
        parser.add_argument('--BatchSize', type = int, default = 32, help='Size of a batch in training')
        parser.add_argument('--ChunkSize', type = int, default = 50, help='The length of a chunk in training')
        parser.add_argument('--SubseqLen', type = int, default = 5000, help='The length of the randomly selected sequence for training')
        parser.add_argument('--LearningRate', type = float, default = 1e-2, help='The initial learning rate of the Adam optimizer')
        parser.add_argument('--AMSGrad', type = bool, default = True, help='Whether the AMSGrad variant is used')
        parser.add_argument('--Eps', type = float, default = 1e-3, help='The term added to the denominator to improve numerical stability')
        parser.add_argument('--LRdecrease', type = int, default = 10, help='The number of batches that are processed each time before the learning rate is divided by 2')

        # Statistic test config
        parser.add_argument('--RED_collection_len', type = int, default = 20, help='The number of readings whose prediction errors are added as a data point')
        parser.add_argument('--RED_points', type = int, default = 20, help='The number of data points that are collected at a time on the testing data to form a testing RED')
        parser.add_argument('--Pvalue_th', type = float, default = 0.0001, help='The threshold of p-value in KS test')

        # Loaddata
        # Sequential data in the form of (Timeframe, Features)
        # Training only leverages normal data. Abnormaly data only for testing.
        parser.add_argument('--normal_data_dir', type = str, default = "data/", help='The directory of normal data')
        parser.add_argument('--normal_data_name_train', type = str, default = "baseline_train.npy", help='The file name of training normal data')
        parser.add_argument('--normal_data_name_test', type = str, default = "baseline_test.npy", help='The file name of testing normal data')

        parser.add_argument('--abnormal_data_dir', type = str, default = "data/", help='The directory of abnormal data')
        parser.add_argument('--abnormal_data_name', type = str, default = "attack1_test.npy", help='The file name of abnormal data')

        # Save and load model. Save after training. Load before testing.
        parser.add_argument('--save_model_dir', type = str, default = "checkpoints/", help='The directory to save the model')
        parser.add_argument('--save_model_name', type = str, default = "AnomalyDetector.ckpt", help='The file name of the saved model')

        parser.add_argument('--load_model_dir', type = str, default = "checkpoints/", help='The directory to load the model')
        parser.add_argument('--load_model_name', type = str, default = "AnomalyDetector.ckpt", help='The file name of the model to be loaded')

        # Debug and GPU config
        parser.add_argument('--debug', dest='debug', action='store_true', help='Whether debug information will be printed')
        parser.set_defaults(debug=False)
        parser.add_argument('--gpu', dest='gpu', action='store_true', help='Whether GPU acceleration is enabled')
        parser.set_defaults(gpu=False)

        args = parser.parse_args()

        if args.debug:
            print args

        if args.training:
            train(args)
        else:
            eval_detector(args)

    except SystemExit:
        sys.exit(0)

    except:
        print "Unexpected error:", sys.exc_info()[0]
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
