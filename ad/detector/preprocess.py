'''
Covert csv data to npy arrays of shape [TimeFrame, Features].
'''

import argparse
import numpy as np
import os

import utils

def remove_outlier(data):
    mu = np.mean(data, axis=0)
    std = np.mean(data, axis=0)

    th = 3 * std
    data = (np.abs(data-mu) > th) * mu + (np.abs(data-mu) <= th) * data
    return data

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type = str, default = "../perf/data/core0/100us/", help='The directory of data')
parser.add_argument('--file_name', type = str, default = None, help='The directory of data')
args = parser.parse_args()

data_dir = args.data_dir
file_name = args.file_name
if file_name != None:
    data = utils.read_csv_file(data_dir+file_name)
    data = remove_outlier(data)
    np.save(data_dir + "".join(file_name.split('.')[:-1]) + '.npy', data)
else:
    for f in os.listdir(data_dir):
        extension = f.split('.')[-1]
        if extension == 'csv':
            data = utils.read_csv_file(data_dir+f)
            data = remove_outlier(data)
            n_ins_average = np.mean(data[:, 0])
            data = data / n_ins_average
            np.save(data_dir + "".join(f.split('.')[:-1]) + '.npy', data)
