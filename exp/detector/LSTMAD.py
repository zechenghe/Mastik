import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

import time
import math
import os
import numpy as np
import pickle

from SeqGenerator import *
from detector import *
from loaddata import *
from utils import *


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
        training_normal_data, ref_normal_data, val_normal_data = load_normal_dummydata()
    else:
        training_normal_data, ref_normal_data, val_normal_data = load_data_split(
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
    val_normal_data = torch.tensor(AnomalyDetector.normalize(val_normal_data))

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
            print(name, para.size())

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
                print("pred.size ", pred.size(), "truth.size ", truth.size())

            loss.backward()
            optimizer.step()

            if gpu:
                loss_hist.append(loss.detach().cpu().numpy())
            else:
                loss_hist.append(loss.detach().numpy())

            if debug:
                print("t =", t, 'Training loss: ', loss)

            state = (state[0].detach(), state[1].detach())
            t += ChunkSize

        print("Batch", batch, "Training loss", np.mean(loss_hist))

        if (batch + 1) % LRdecrease == 0:
            LearningRate = LearningRate / 2.0
            setLearningRate(optimizer, LearningRate)

    print("Training Done")
    print("Getting RED")

    AnomalyDetector.set_RED_config(RED_collection_len = RED_collection_len, RED_points = RED_points)
    AnomalyDetector.collect_ref_RED(ref_normal_data, gpu)

    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    torch.save(AnomalyDetector, save_model_dir + save_model_name)
    print("Model saved")

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
        testing_normal_data = load_data_all(
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
    print("testing_abnormal_data.shape ", testing_abnormal_data.shape)

    if gpu:
        AnomalyDetector = AnomalyDetector.cuda()
        testing_normal_data = testing_normal_data.cuda()
        testing_abnormal_data = testing_abnormal_data.cuda()


    true_label_normal = np.zeros(len(testing_normal_data) - AnomalyDetector.RED_collection_len * AnomalyDetector.RED_points - 1)
    true_label_abnormal = np.ones(len(testing_abnormal_data) - AnomalyDetector.RED_collection_len * AnomalyDetector.RED_points - 1)
    true_label = np.concatenate((true_label_normal, true_label_abnormal), axis=0)

    pred_normal, p_values_normal = AnomalyDetector.predict(testing_normal_data, gpu)
    #print("p_values_normal", p_values_normal[:100])
    #print("p_values_normal", p_values_normal[100:200])

    #print("p_values_normal", p_values_normal[-100:])
    #print("p_values_normal", p_values_normal[-200:-100])

    print("p_values_normal.shape ", len(p_values_normal))
    print("p_values_normal.mean ", np.mean(p_values_normal))

    pred_abnormal, p_values_abnormal = AnomalyDetector.predict(testing_abnormal_data, gpu)
    print("p_values_abnormal.shape ", len(p_values_abnormal))
    print("p_values_abnormal.mean ", np.mean(p_values_abnormal))

    pred = np.concatenate((pred_normal, pred_abnormal), axis=0)
    pred_score = np.concatenate((p_values_normal, p_values_abnormal), axis=0)
    print("true_label.shape", true_label.shape, "pred.shape", pred.shape)

    tp, fp, fn, tn, acc, prec, rec, f1, fpr, tpr, thresholds, roc_auc = \
    eval_metrics(
        truth = true_label,
        pred = pred,
        pred_score = 1-pred_score
    )

    results_dir = 'results/'
    np.save(results_dir + 'LSTM-KS' + '_fpr', fpr)
    np.save(results_dir + 'LSTM-KS' + '_tpr', tpr)
    return roc_auc

if __name__ == '__main__':
    import argparse
    import sys
    import traceback

    try:
        args = create_parser()

        if args.debug:
            print(args)

        if args.training:
            train(args)
        else:
            eval_detector(args)

    except SystemExit:
        sys.exit(0)

    except:
        print("Unexpected error:", sys.exc_info()[0])
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
