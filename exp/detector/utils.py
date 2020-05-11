import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

def read_npy_data_single_flle(filename):
    print("Reading Data: " + filename)
    data = np.load(filename)
    return data

def write_npy_data_single_file(filename, data):
    print("Writing Data: " + filename)
    np.save(filename, data)
    return

def get_mean(data):
    return np.mean(data, axis = 0, keepdims = True)

def get_std(data):
    return np.std(data, axis = 0, keepdims = True)

def normalize(data, mean, std):

    assert data.shape[1] == mean.shape[1], "Feature size should match mean size"
    assert data.shape[1] == std.shape[1], "Feature size should match std size"

    eps = 1e-8
    return (data - mean) / (std + eps)

def eval_metrics( truth, pred, pred_score=None, verbose=True ):
    tp = np.sum( np.multiply((pred == 1) , (truth == 1)), axis=0 , dtype=np.float32)
    fp = np.sum( np.multiply((pred == 1) , (truth == 0)), axis=0 , dtype=np.float32)
    fn = np.sum( np.multiply((pred == 0) , (truth == 1)), axis=0 , dtype=np.float32)
    tn = np.sum( np.multiply((pred == 0) , (truth == 0)), axis=0 , dtype=np.float32)
    acc = np.sum( pred == truth , axis=0 )/ (1. * truth.shape[0])

    fpr = fp / (fp + tn) if fp + tn != 0 else None
    fnr = fn / (fn + tp) if fn + tp != 0 else None
    prec = tp / ( ( tp + fp ) * 1. ) if tp + fp != 0 else None
    rec =  tp / ( ( tp + fn ) * 1. ) if tp + fn != 0 else None
    f1 = 2.*prec*rec/(prec+rec) if prec != None and rec != None and prec+rec != 0 else None

    if verbose:
        print('----------------Detection Results------------------')
        print('False positives: ', fp)
        print('False negatives: ', fn)
        print('True positives: ', tp)
        print('True negatives: ', tn)
        print('False Positive Rate: ', fpr)
        print('False Negative Rate: ', fnr)
        print('Accuracy: ', acc)
        print('Precision: ', prec)
        print('Recall: ', rec)
        print('F1: ', f1)
        print('---------------------------------------------------')

    roc, roc_auc = None, None
    if pred_score is not None:
        fpr, tpr, thresholds = roc_curve(truth, pred_score)
        roc_auc = roc_auc_score(truth, pred_score)
        if verbose:
            print("ROC AUC = ", roc_auc)

    return tp, fp, fn, tn, acc, prec, rec, f1, fpr, tpr, thresholds, roc_auc


def setLearningRate(optimizer, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def plotsignal(sigs):

    T = len(sigs[0])
    t = np.linspace(0, T, num = T)
    signal0 = sigs[0][:,0,:]
    print("signal0.shape", signal0.shape)

    signal1 = sigs[1][:,0,:]
    print("signal1.shape", signal1.shape)

    plt.plot(t, signal0)
    plt.plot(t, signal1)
    plt.ylim(-2, 2)
    plt.show()

def seq_win_vectorize(seq, window_size):

    res = []
    for i in range(len(seq)-window_size+1):
        res.append(seq[i: i+window_size,:].reshape((-1)))
    return np.array(res)


def create_parser():
    import argparse
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

    args, unknown = parser.parse_known_args()
    return args
