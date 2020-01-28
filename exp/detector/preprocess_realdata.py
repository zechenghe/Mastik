import numpy as np

from utils import *

def main():

    # Remove all zeros columns

    data_dir = 'data/'
    HPC = {
        "baseline" : read_npy_data_single_flle(data_dir + "HPC_baseline.npy"),
        "attack1" : read_npy_data_single_flle(data_dir + "HPC_attack1.npy"),
        "attack2" : read_npy_data_single_flle(data_dir + "HPC_attack2.npy"),
        "attack3" : read_npy_data_single_flle(data_dir + "HPC_attack3.npy"),
        "attack4" : read_npy_data_single_flle(data_dir + "HPC_attack4.npy"),
        "attack5" : read_npy_data_single_flle(data_dir + "HPC_attack5.npy"),
        "attack6" : read_npy_data_single_flle(data_dir + "HPC_attack6.npy")
    }

    for key in HPC:
        print key, HPC[key].shape

    s = np.sum(HPC["baseline"], axis = 0)
    idx = np.where(s != 0)[0]

    for key in HPC:
        data_select = HPC[key][:, idx]
        print data_select.shape
        write_npy_data_single_file(data_dir + key + '.npy', data_select)


if __name__ == '__main__':
    main()
