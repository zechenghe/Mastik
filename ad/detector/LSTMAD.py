import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

import warnings
# Ignore warnings due to pytorch save models
# https://github.com/pytorch/pytorch/issues/27972
warnings.filterwarnings("ignore", "Couldn't retrieve source code")

import time
import math
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

import utils
import SeqGenerator
import detector
import loaddata

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
    val_and_ref_name = args.normal_data_name_val_and_ref

    RED_collection_len = args.RED_collection_len
    RED_points = args.RED_points
    Pvalue_th = args.Pvalue_th

    if args.dummydata:
        training_normal_data, val_normal_data, ref_normal_data = (
            loaddata.load_normal_dummydata()
        )
    else:
        _, training_normal_data, _ = (
            loaddata.load_data_split(
                data_dir = normal_data_dir,
                file_name = normal_data_name_train,
                # The first few readings could be unstable, remove it.
                split = (0.1, 0.8, 0.1)
                )
        )

        _, ref_normal_data, val_normal_data = loaddata.load_data_split(
            data_dir = normal_data_dir,
            file_name = val_and_ref_name,
            # The first few readings could be unstable, remove it.
            split = (0.1, 0.45, 0.45)
        )

    training_normal_data_mean = utils.get_mean(training_normal_data)
    training_normal_data_std = utils.get_std(training_normal_data)

    Nfeatures = training_normal_data.shape[1]
    AnomalyDetector = detector.Detector(
        input_size = Nfeatures,
        hidden_size = Nhidden,
        th = Pvalue_th
    )
    AnomalyDetector.set_mean(training_normal_data_mean)
    AnomalyDetector.set_std(training_normal_data_std)

    training_normal_data = AnomalyDetector.normalize(training_normal_data)
    val_normal_data = AnomalyDetector.normalize(val_normal_data)
    ref_normal_data = torch.tensor(AnomalyDetector.normalize(ref_normal_data))

    training_normal_wrapper = SeqGenerator.SeqGenerator(training_normal_data)
    val_normal_wrapper = SeqGenerator.SeqGenerator(val_normal_data)
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

        def step_fn(data_batch, is_train=True):
            t = 0
            init_state = (torch.zeros(1, BatchSize, Nhidden),
                        torch.zeros(1, BatchSize, Nhidden))

            if gpu:
                init_state = (init_state[0].cuda(), init_state[1].cuda())
                training_batch = training_batch.cuda()

            state = init_state
            loss_list = []
            while t + ChunkSize + 1 < SubseqLen:
                if is_train:
                    AnomalyDetector.zero_grad()

                pred, state = AnomalyDetector.forward(
                    data_batch[t:t+ChunkSize, :, :], state)
                truth = data_batch[t+1 : t+ChunkSize+1, :, :]

                loss = MSELossLayer(pred, truth)

                if debug:
                    print("pred.size ", pred.size(), "truth.size ", truth.size())

                if is_train:
                    loss.backward()
                    optimizer.step()

                if gpu:
                    loss_list.append(loss.detach().cpu().numpy())
                else:
                    loss_list.append(loss.detach().numpy())

                state = (state[0].detach(), state[1].detach())
                t += ChunkSize
            return loss_list

        training_batch = torch.tensor(
                    training_normal_wrapper.next(BatchSize, SubseqLen))
        train_loss_list = step_fn(training_batch, is_train=True)
        val_batch = torch.tensor(
                    val_normal_wrapper.next(BatchSize, SubseqLen))
        val_loss_list = step_fn(val_batch, is_train=False)
        print("Batch", batch, "Training loss", np.mean(train_loss_list), "Val loss", np.mean(val_loss_list))

        if (batch + 1) % LRdecrease == 0:
            LearningRate = LearningRate / 2.0
            utils.setLearningRate(optimizer, LearningRate)

    print("Training Done")
    print("Getting RED")

    AnomalyDetector.set_RED_config(
        RED_collection_len=RED_collection_len,
        RED_points=RED_points
        )
    AnomalyDetector.collect_ref_RED(ref_normal_data, gpu)

    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    torch.save(
        AnomalyDetector,
        save_model_dir + save_model_name
        )
    print("Model saved")

def eval_detector(args):

    load_model_dir = args.load_model_dir
    load_model_name = args.load_model_name
    normal_data_dir = args.normal_data_dir
    normal_data_name_train = args.normal_data_name_train
    normal_data_name_test = args.normal_data_name_test
    abnormal_data_dir = args.abnormal_data_dir
    abnormal_data_name = args.abnormal_data_name
    Pvalue_th = args.Pvalue_th
    val_and_ref_name = args.normal_data_name_val_and_ref

    gpu = args.gpu

    AnomalyDetector = torch.load(load_model_dir + load_model_name)
    AnomalyDetector.eval()
    AnomalyDetector.th = Pvalue_th

    if args.dummydata:
        _, _, testing_normal_data = loaddata.load_normal_dummydata()
    else:
        if args.debug:
            _, training_normal_data, _, = (
                loaddata.load_data_split(
                    data_dir = normal_data_dir,
                    file_name = normal_data_name_train,
                    split = (0.1, 0.8, 0.1)
                    )
                    )

            _, ref_normal_data, val_normal_data = loaddata.load_data_split(
                data_dir = normal_data_dir,
                file_name = val_and_ref_name,
                split = (0.1, 0.45, 0.45)
                )

            training_normal_data = torch.tensor(
                AnomalyDetector.normalize(training_normal_data))
            val_normal_data = torch.tensor(
                AnomalyDetector.normalize(val_normal_data))
            ref_normal_data = torch.tensor(
                AnomalyDetector.normalize(ref_normal_data))


        testing_normal_data = loaddata.load_data_all(
            data_dir = normal_data_dir,
            file_name = normal_data_name_test
        )

    testing_normal_data = torch.tensor(
        AnomalyDetector.normalize(testing_normal_data))

    if args.dummydata:
        testing_abnormal_data = loaddata.load_abnormal_dummydata()
    else:
        testing_abnormal_data = loaddata.load_data_all(
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

    pred_normal, p_values_normal = AnomalyDetector.predict(
        testing_normal_data,
        gpu,
        debug=args.debug
        )

    if args.debug:
        feature_idx = 0

        # debug_pred_normal is of size [seq_len-1, batch(=1), features]
        RE_normal, debug_pred_normal = AnomalyDetector._get_reconstruction_error(
            testing_normal_data,
            gpu=gpu)

        seq_dict = {
            "truth": testing_normal_data[1:,feature_idx].detach().numpy(),
            "pred": debug_pred_normal[:,0, feature_idx].detach().numpy(),
        }
        seq_dict["diff"] = (seq_dict["pred"] - seq_dict["truth"])**2
        utils.plot_seq(seq_dict, title="Testing normal prediction")

        # debug_pred_normal is of size [seq_len-1, batch(=1), features]
        RE_abnormal, debug_pred_abnormal = AnomalyDetector._get_reconstruction_error(
            testing_abnormal_data,
            gpu=gpu
            )

        seq_dict = {
            "truth": testing_abnormal_data[1:,feature_idx].detach().numpy(),
            "pred": debug_pred_abnormal[:,0, feature_idx].detach().numpy(),
        }
        seq_dict["diff"] = (seq_dict["pred"] - seq_dict["truth"])**2
        utils.plot_seq(seq_dict, title="Testing abnormal prediction")

        # debug_ref is of size [seq_len-1, batch(=1), features]
        RE_ref, debug_ref = AnomalyDetector._get_reconstruction_error(
            ref_normal_data,
            gpu=gpu)

        seq_dict = {
            "truth": ref_normal_data[1:,feature_idx].detach().numpy(),
            "pred": debug_ref[:,0, feature_idx].detach().numpy(),
            }
        seq_dict["diff"] = (seq_dict["pred"] - seq_dict["truth"])**2
        utils.plot_seq(seq_dict, title="Train normal ref prediction")

        RE_seq_dict = {
            "RE_reference": RE_ref,
            "RE_normal": RE_normal,
            "RE_abnormal": RE_abnormal
        }
        utils.plot_seq(RE_seq_dict, title="Reconstruction errors")
        utils.plot_cdf(RE_seq_dict, title="RED cdf")


    print("p_values_normal.shape ", len(p_values_normal))
    print("p_values_normal.mean ", np.mean(p_values_normal))

    pred_abnormal, p_values_abnormal = AnomalyDetector.predict(
        testing_abnormal_data,
        gpu,
        debug=args.debug
        )
    print("p_values_abnormal.shape ", len(p_values_abnormal))
    print("p_values_abnormal.mean ", np.mean(p_values_abnormal))

    pred = np.concatenate((pred_normal, pred_abnormal), axis=0)
    pred_score = np.concatenate((p_values_normal, p_values_abnormal), axis=0)
    print("true_label.shape", true_label.shape, "pred.shape", pred.shape)

    tp, fp, fn, tn, acc, prec, rec, f1, fpr, tpr, thresholds, roc_auc = (
        utils.eval_metrics(
            truth = true_label,
            pred = pred,
            anomaly_score = -np.log10(pred_score+1e-50)  # Anomaly score=-log(p_value)
            )
    )

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC of LSTM anomaly detector')
    plt.legend(loc="lower right")
    plt.show()

    return roc_auc

if __name__ == '__main__':
    import argparse
    import sys
    import traceback

    try:
        args = utils.create_parser()

        np.random.seed(0)
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

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
