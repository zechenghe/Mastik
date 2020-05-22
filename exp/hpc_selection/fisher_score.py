import numpy as np

def read_file(path):
    return np.loadtxt(path)

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

    neg1 = read_file(neg1_path)[5000:6000]
    neg2 = read_file(neg2_path)[5000:6000]
    pos1 = read_file(pos1_path)[5000:6000]
    pos2 = read_file(pos2_path)[5000:6000]

    neg = neg1 #np.concatenate((neg1, neg2), axis=0)
    pos = pos1 #np.concatenate((pos1, pos2), axis=0)

    print neg.shape
    print pos.shape

    neg_mean = np.mean(neg)
    neg_var = np.var(neg)
    pos_mean = np.mean(pos)
    pos_var = np.var(pos)

    fisher_score = ((pos_mean-neg_mean) ** 2) / (neg_var+pos_var)
    print hpc, fisher_score
