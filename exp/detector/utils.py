import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

def read_npy_data_single_flle(filename):
    print "Reading Data: " + filename
    data = np.load(filename)
    return data

def write_npy_data_single_file(filename, data):
    print "Writing Data: " + filename
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
    return (data - self.mean) / (self.std + self.eps)

def eval_metrics( truth, pred, pred_score=None ):
    tp = np.sum( np.multiply((pred == 1) , (truth == 1)), axis=0 , dtype=np.float32)
    fp = np.sum( np.multiply((pred == 1) , (truth == 0)), axis=0 , dtype=np.float32)
    fn = np.sum( np.multiply((pred == 0) , (truth == 1)), axis=0 , dtype=np.float32)
    tn = np.sum( np.multiply((pred == 0) , (truth == 0)), axis=0 , dtype=np.float32)
    acc = np.sum( pred == truth , axis=0 )/ (1. * truth.shape[0])

    print '----------------Detection Results------------------'
    print 'False positives: ', fp
    print 'False negatives: ', fn
    print 'True positives: ', tp
    print 'True negatives: ', tn

    fpr = fp / (fp + tn) if fp + tn != 0 else None
    fnr = fn / (fn + tp) if fn + tp != 0 else None
    prec = tp / ( ( tp + fp ) * 1. ) if tp + fp != 0 else None
    rec =  tp / ( ( tp + fn ) * 1. ) if tp + fn != 0 else None
    f1 = 2.*prec*rec/(prec+rec) if prec != None and rec != None and prec+rec != 0 else None

    print 'False Positive Rate: ', fpr
    print 'False Negative Rate: ', fnr

    print 'Accuracy: ', acc

    print 'Precision: ', prec
    print 'Recall: ', rec
    print 'F1: ', f1
    print '---------------------------------------------------'

    roc, roc_auc = None, None
    if pred_score is not None:
        fpr, tpr, thresholds = roc_curve(truth, pred_score)
        roc_auc = roc_auc_score(truth, pred_score)
        print "ROC AUC = ", roc_auc

    return tp, fp, fn, tn, acc, prec, rec, f1, fpr, tpr, thresholds


def setLearningRate(optimizer, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def plotsignal(sigs):

    T = len(sigs[0])
    t = np.linspace(0, T, num = T)
    signal0 = sigs[0][:,0,:]
    print "signal0.shape", signal0.shape

    signal1 = sigs[1][:,0,:]
    print "signal1.shape", signal1.shape

    plt.plot(t, signal0)
    plt.plot(t, signal1)
    plt.ylim(-2, 2)
    plt.show()

def seq_win_vectorize(seq, window_size):

    res = []
    for i in range(len(seq)-window_size+1):
        res.append(seq[i: i+window_size,:].reshape((-1)))
    return np.array(res)
