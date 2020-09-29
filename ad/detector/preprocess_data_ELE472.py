'''
Covert csv data to npy arrays of shape [TimeFrame, Features]
'''

import numpy as np
import os

data_dir = 'data/'
for f in os.listdir(data_dir):
    file_name_split = f.split('.')
    if file_name_split[-1] == 'csv':
        with open(data_dir+f, 'r') as file_handler:
            data = []
            for linenum, line in enumerate(file_handler):
                # Remove sensor names header
                if linenum == 0:
                    continue
                data.append(line.split(',')[1:])
            # Convert data to np array shape [TimeFrame, Features]
            # Use Accelerometer, Gyroscope and Magnetometer
            data_np = np.array(data, dtype=np.float32)[:, :6]

            # Compute norm
            acc_norm = np.linalg.norm(data_np[:, 0:3], axis=-1)
            speed_norm = np.linalg.norm(data_np[:, 3:6], axis=-1)
            data_np = np.stack([acc_norm, acc_norm], axis=-1)
        np.save(data_dir + file_name_split[0] + '.npy', data_np)
        data_load = np.load(data_dir + file_name_split[0] + '.npy')
        print file_name_split[0], data_load.shape
