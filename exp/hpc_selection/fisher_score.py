import numpy as np

def read_file(path):
    f = open(path, 'r')
    return np.array(list(f))

data_dir = 'results/'
SP = 'sensitive5'

for hpc in ['BR_CN', 'BR_INS', 'BR_MSP', 'BR_NTK', 'BR_PRC', 'FP_INS', 'L1_DCM', 'L1_ICM',
            'L1_LDM', 'L1_STM', 'L1_TCM', 'L2_TCA', 'L2_TCM', 'L3_TCA', 'L3_TCM', 'LD_INS',
            'SR_INS', 'STL_ICY', 'TLB_DM', 'TLB_IM', 'TOT_CYC', 'TOT_INS']:

    neg1_path = data_dir + '{SP}_{hpc}_none1'.format(
        SP = SP,
        hpc = hpc
    )
    neg2_path = data_dir + '{SP}_{hpc}_none2'.format(
        SP = SP,
        hpc = hpc
    )
    pos1_path = data_dir + '{SP}_{hpc}_sim_flush'.format(
        SP = SP,
        hpc = hpc
    )
    pos2_path = data_dir + '{SP}_{hpc}_sim_l3prime'.format(
        SP = SP,
        hpc = hpc
    )

    neg1 = read_file(neg1_path)
    neg2 = read_file(neg2_path)
    pos1 = read_file(pos1_path)
    pos2 = read_file(pos2_path)

    print neg1.shape, neg2.shape, pos1.shape, pos2.shape
