
filename = 'all_attacks/{date}/{log_name}'.format(
    date = '0524',
    log_name = 'log_selected_1',
)

attacks = ['l3pp']
hpcs = ['SELECTED']
methods = ['LSTM+KS', 'OCSVM', 'IF', 'PCA']


for attack in attacks:
    for hpc in hpcs:
        for method in methods:
            print attack, hpc, method
            f = open(filename, "r")
            for line in f:
                words = line.split(' ')
                if len(words) > 1 and words[3] == attack and words[7] == hpc and words[9] == method:
                    print words[-1][:-1]
