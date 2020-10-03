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
    cmd = 'sudo ./event_open_user {core} {interval_cycles} {n_readings} {save_data_dir}{save_data_name}.csv'.format(
        core=core,
        interval_cycles=interval_cycles,
        n_readings=n_readings,
        save_data_dir=save_data_dir,
        save_data_name=save_data_name,
    )
    return cmd

parser = argparse.ArgumentParser()

parser.add_argument('--core', type = int, default = 0, help='core index')
parser.add_argument('--us', type = int, default = 1000, help='interval in us')
parser.add_argument('--n_readings', type = int, default = 300000, help='number of HPC readings')
parser.add_argument('--bg_program', type = str, default = 'webserver', help='background program')

args = parser.parse_args()

interval_cycles = int(args.us * 1.2)

attacks = {
    'l1pp': 'taskset 0x1 /home/zechengh/Mastik/exp/test_workspace/spy_l1pp 100000000000000 &',
    'l3pp': 'taskset 0x1 /home/zechengh/Mastik/exp/test_workspace/spy_l3pp 100000000000000 &',
    'fr': 'taskset 0x1 /home/zechengh/Mastik/exp/test_workspace/spy_fr /home/zechengh/Mastik/gnupg-1.4.13/g10/gpg &',
    'ff': 'taskset 0x1 /home/zechengh/Mastik/exp/test_workspace/spy_ff /home/zechengh/Mastik/gnupg-1.4.13/g10/gpg &'
}

save_data_dir = 'data/{bg_program}/{us}us/'.format(
    bg_program=args.bg_program,
    us=args.us
)

os.system('mkdir -p {save_data_dir}'.format(save_data_dir=save_data_dir))

monitor_cmd_fn=functools.partial(
    monitor_cmd,
    core=args.core,
    n_readings=args.n_readings,
    save_data_dir=save_data_dir,
)

cmd = monitor_cmd_fn(save_data_name='train_normal.csv')

print(cmd)
monitor_process = subprocess.Popen(cmd, shell=True)
print("Monitor PID: {pid}".format(pid=monitor_pid))

monitor_status = monitor_process.wait()
