
filename = '{attack}/{method}/log_{prog}_{hpc}'.format(
    attack = 'fr',
    method = 'bench_train_bench_test',
    prog = 'sensitive5',
    hpc = 'old_l3'
)

print filename

f = open(filename, "r")
for line in f:
  words = line.split(' ')
  if words[7] == 'LOF':
      print words[-1][:-1]
