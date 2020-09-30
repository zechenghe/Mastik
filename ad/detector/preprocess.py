'''
Covert csv data to npy arrays of shape [TimeFrame, Features].
'''

import argparse
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type = str, default = "../perf/data/core0/100us/", help='The directory of data')
args = parser.parse_args()

for f in os.listdir(data_dir):
    extension = f.split('.')[-1]
    if extension == 'csv':
        data = utils.read_csv_file(data_dir+f)
        np.save(data_dir + "".join(f.split('.')[:-1]) + '.npy', data)
