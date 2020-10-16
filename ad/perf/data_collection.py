import os
import subprocess
import time
import argparse
import functools

def monitor_cmd(
    core,
    interval_cycles,
    n_readings,
    save_data_dir,
    save_data_name
    ):

    # Collect normal data
    cmd = 'sudo ./event_open_user {core} {interval_cycles} {n_readings} {save_data_dir}{save_data_name}'.format(
        core=core,
        interval_cycles=interval_cycles,
        n_readings=n_readings,
        save_data_dir=save_data_dir,
        save_data_name=save_data_name,
    )
    return cmd

parser = argparse.ArgumentParser()

parser.add_argument('--core', type = int, default = 0, help='core index')
parser.add_argument('--us', type = int, default = 100, help='interval in us')
parser.add_argument('--n_readings', type = int, default = 300000, help='number of HPC readings')
parser.add_argument('--bg_program', type = str, default = 'webserver', help='background program')

args = parser.parse_args()

interval_cycles = int(args.us * 1.2)

attacks = {
    'l1pp': 'taskset 0x1 /home/zechengh/Mastik/exp/test_workspace/spy_l1pp 1000 &',
    'l3pp': 'taskset 0x1 /home/zechengh/Mastik/exp/test_workspace/spy_l3pp 1000 &',
    'fr': 'taskset 0x1 /home/zechengh/Mastik/exp/test_workspace/spy_fr /home/zechengh/Mastik/gnupg-1.4.13/g10/gpg &',
    'ff': 'taskset 0x1 /home/zechengh/Mastik/exp/test_workspace/spy_ff /home/zechengh/Mastik/gnupg-1.4.13/g10/gpg &'
}

gpg_command = 'taskset 0x1 /home/zechengh/Mastik/ad/bg_program/run_gpg.sh'

save_data_dir = 'data/{bg_program}/{us}us/'.format(
    bg_program=args.bg_program,
    us=args.us
)

os.system('mkdir -p {save_data_dir}'.format(save_data_dir=save_data_dir))

monitor_cmd_fn=functools.partial(
    monitor_cmd,
    core=args.core,
    interval_cycles=interval_cycles,
    n_readings=args.n_readings,
    save_data_dir=save_data_dir,
)

# Normal data collection
cmd = monitor_cmd_fn(save_data_name='train_normal.csv')
print(cmd)
monitor_process = subprocess.Popen(cmd.split())
monitor_status = monitor_process.wait()

# Ref and val data collection
cmd = monitor_cmd_fn(save_data_name='ref_and_val.csv')
print(cmd)
monitor_process = subprocess.Popen(cmd.split())
monitor_status = monitor_process.wait()

# Ref and val data collection
cmd = monitor_cmd_fn(save_data_name='test_normal.csv')
print(cmd)
monitor_process = subprocess.Popen(cmd.split())
monitor_status = monitor_process.wait()

# Test normal with gpg running
gpg_process = subprocess.Popen(gpg_command.split())
time.sleep(10)
cmd = monitor_cmd_fn(save_data_name='test_normal_with_gpg.csv')
monitor_status = monitor_process.wait()
gpg_process.terminate()

for k in attacks.keys():
    attack_process = subprocess.Popen(attacks[k].split())
    # To make the attack actually run
    time.sleep(10)
    cmd = monitor_cmd_fn(save_data_name='test_abnormal_{attack}.csv'.format(
                attack=k
            )
        )
    print(cmd)
    monitor_process = subprocess.Popen(cmd.split())
    monitor_status = monitor_process.wait()

    # Test abnormal with gpg running
    gpg_process = subprocess.Popen(gpg_command.split())
    time.sleep(10)
    cmd = monitor_cmd_fn(save_data_name='test_abnormal_{attack}_with_gpg.csv'.format(
            attack=k
        )
    )
    print(cmd)
    monitor_status = monitor_process.wait()
    gpg_process.terminate()

    attack_process.terminate()


# Clean up
cmd = 'sudo chown zechengh ../ -R'
print(cmd)
monitor_process = subprocess.Popen(cmd.split())
monitor_status = monitor_process.wait()

cmd = 'python2 ../detector/preprocess.py --data_dir {save_data_dir}'.format(
    save_data_dir=save_data_dir
    )
print(cmd)
monitor_process = subprocess.Popen(cmd.split())
monitor_status = monitor_process.wait()
