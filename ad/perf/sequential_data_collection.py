import os
import subprocess
import time
import argparse
import functools
import utils
import random


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--core', type = int, default = 0, help='core index')
    parser.add_argument('--us', type = int, default = 100, help='interval in us')
    parser.add_argument('--n_readings', type = int, default = 300000, help='number of HPC readings')
    parser.add_argument('--bg_program', type = str, default = 'webserver', help='background program')

    args = parser.parse_args()

    interval_cycles = int(args.us / 3)
    schedule = []

    attacks = {
        'l1pp': 'taskset 0x1 /home/zechengh/Mastik/exp/test_workspace/spy_l1pp 1000 &',
        'l3pp': 'taskset 0x1 /home/zechengh/Mastik/exp/test_workspace/spy_l3pp 1000 &',
        'fr': 'taskset 0x1 /home/zechengh/Mastik/exp/test_workspace/spy_fr /home/zechengh/Mastik/gnupg-1.4.13/g10/gpg &',
        'ff': 'taskset 0x1 /home/zechengh/Mastik/exp/test_workspace/spy_ff /home/zechengh/Mastik/gnupg-1.4.13/g10/gpg &',
        'spectrev1': 'taskset 0x1 /home/zechengh/Mastik/ad/attack/spectre-v1/spectrev1 &',
        'spectrev2': 'taskset 0x1 /home/zechengh/Mastik/ad/attack/spectre-v2/spectrev2 &',
        'spectrev3': 'taskset 0x1 /home/zechengh/Mastik/ad/attack/meltdown/memdump &',
        'spectrev4': 'taskset 0x1 /home/zechengh/Mastik/ad/attack/spectre-ssb/spectrev4 &',
        }

    gpg_command = 'taskset 0x1 /home/zechengh/Mastik/ad/bg_program/run_gpg.sh'
    spec_command = '/home/zechengh/Mastik/ad/bg_program/run_fixed_spec.py'

    save_data_dir = 'data/{bg_program}/{us}us/'.format(
        bg_program=args.bg_program,
        us=args.us
    )

    os.system('mkdir -p {save_data_dir}'.format(save_data_dir=save_data_dir))

    monitor_cmd_fn=functools.partial(
        utils.monitor_cmd,
        core=args.core,
        interval_cycles=interval_cycles,
        n_readings=args.n_readings,
        save_data_dir=save_data_dir,
    )

    cmd = monitor_cmd_fn(save_data_name='eval_sequece.csv')
    monitor_process = subprocess.Popen(cmd.split())

    for k in attacks.keys():
        attack_process = subprocess.Popen(attacks[k].split())
        schedule.append("{attack} starts at {t}".format(
            attack=k,
            t=int(time.time()*1000000)
        ))
        # To make the attack actually run
        time.sleep(2)
        attack_process.terminate()

        time.sleep(random.randint(1,10))

    monitor_process.terminate()

    # Clean up
    cmd = 'sudo chown zechengh ../ -R'
    print(cmd)
    p = subprocess.Popen(cmd.split())
    p_status = p.wait()

    cmd = 'python2 ../detector/preprocess.py --data_dir {save_data_dir}'.format(
        save_data_dir=save_data_dir
    )
    print(cmd)
    p = subprocess.Popen(cmd.split())
    p_status = p.wait()

    print(schedule)
