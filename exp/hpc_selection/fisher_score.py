import numpy as np
import os
import collections

def read_file(path):
    return np.loadtxt(path)

def f_score(pos, neg):
    neg_mean = np.mean(neg)
    neg_var = np.var(neg)
    pos_mean = np.mean(pos)
    pos_var = np.var(pos)

    return ((pos_mean-neg_mean) ** 2) / (neg_var+pos_var)


def collect_f_score(collection, data_dir='results/', SP='sensitive5'):

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

        print neg1_path

        neg1 = read_file(neg1_path)
        neg2 = read_file(neg2_path)
        pos1 = read_file(pos1_path)
        pos2 = read_file(pos2_path)

        f1 = f_score(neg1, pos1)
        f2 = f_score(neg2, pos2)
        fisher_score = (f1+f2) / 2

        #print hpc, f1, f2, fisher_score
        collection[hpc] += fisher_score


data_dirs = os.listdir('archive/')
data_dir_filtered = []
for d in data_dirs:
    if d >= '20200523_020440':
        data_dir_filtered.append(d)


print 'Total', len(data_dir_filtered), 'runs'
D = collections.defaultdict(lambda:0)
for d in data_dir_filtered:
    collect_f_score(
        collection = D,
        data_dir = 'archive/' + d + '/results/'
    )

for k in D.keys():
    print k, D[k] / len(data_dir_filtered)
