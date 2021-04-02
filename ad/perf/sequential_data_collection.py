import os
import signal
import subprocess
import time
import argparse
import functools
import utils
import random
import json
import collections


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--core', type = int, default = 3, help='core index')
    parser.add_argument('--us', type = int, default = 100, help='interval in us')
    parser.add_argument('--n_readings', type = int, default = 300000, help='number of HPC readings')
    parser.add_argument('--bg_program', type = str, default = 'webserver', help='background program')

    args = parser.parse_args()

    interval_cycles = int(args.us / 3)
    schedule = collections.defaultdict(lambda: collections.defaultdict(lambda: list()))

    attacks = {
        'l1pp': 'taskset 0x8 /home/zechengh/Mastik/exp/test_workspace/spy_l1pp 1000 &',
        'l3pp': 'taskset 0x8 /home/zechengh/Mastik/exp/test_workspace/spy_l3pp 1000 &',
        'fr': 'taskset 0x8 /home/zechengh/Mastik/exp/test_workspace/spy_fr /home/zechengh/Mastik/gnupg-1.4.13/g10/gpg &',
        'ff': 'taskset 0x8 /home/zechengh/Mastik/exp/test_workspace/spy_ff /home/zechengh/Mastik/gnupg-1.4.13/g10/gpg &',
        'spectrev1': 'taskset 0x8 /home/zechengh/Mastik/ad/attack/spectre-v1/spectrev1 &',
        'spectrev2': 'taskset 0x8 /home/zechengh/Mastik/ad/attack/spectre-v2/spectrev2 &',
        'spectrev3': 'taskset 0x8 /home/zechengh/Mastik/ad/attack/meltdown/memdump &',
        'spectrev4': 'taskset 0x8 /home/zechengh/Mastik/ad/attack/spectre-ssb/spectrev4 &',
        'bufferoverflow': 'taskset 0x8 /home/zechengh/Mastik/ad/bg_program/run_bufferoverflow.sh &',
        }
    gpg_command = 'taskset 0x8 /home/zechengh/Mastik/ad/bg_program/run_gpg.sh'

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

    attack_processes = {}

    for k in attacks.keys():
        attack_processes[k] = subprocess.Popen(attacks[k].split())
        os.kill(attack_processes[k].pid, signal.SIGSTOP)

    cmd = monitor_cmd_fn(save_data_name='eval_sequence.csv')
    monitor_process = subprocess.Popen(cmd.split())

    # Start of HPC collection
    time.sleep(20)

    """
    # Run flush-reload attack 2 times, sleep 1s
    for _ in range(2):
        os.kill(attack_processes['fr'].pid, signal.SIGCONT)
        schedule['fr']['start'].append(utils.get_time())
        time.sleep(5)
        os.kill(attack_processes['fr'].pid, signal.SIGSTOP)
        schedule['fr']['end'].append(utils.get_time())
        time.sleep(1)

    time.sleep(30)

    # Run Spectre attack 5 times, sleep random time between 1-10s
    for _ in range(5):
        os.kill(attack_processes['spectrev1'].pid, signal.SIGCONT)
        schedule['spectrev1']['start'].append(utils.get_time())
        time.sleep(2)
        os.kill(attack_processes['spectrev1'].pid, signal.SIGSTOP)
        schedule['spectrev1']['end'].append(utils.get_time())
        time.sleep(random.randint(1,10))

    time.sleep(30)
    """

    # Randomly run multiple attacks
    for k in ['l1pp', 'spectrev4', 'spectrev1', 'ff', 'l3pp', 'bufferoverflow', 'spectrev2', 'spectrev3', 'fr']:
        os.kill(attack_processes[k].pid, signal.SIGCONT)
        schedule[k]['start'].append(utils.get_time())
        time.sleep(10)
        os.kill(attack_processes[k].pid, signal.SIGSTOP)
        schedule[k]['end'].append(utils.get_time())
        time.sleep(random.randint(10,30))

    for k, p in attack_processes.items():
        p.terminate()

    time.sleep(30)

    # GCC running
    cmd = utils.spec_cmd('gcc')
    print(cmd)
    spec_process = subprocess.Popen(cmd.split())

    schedule[k]['start'].append(utils.get_time())
    time.sleep(30)
    schedule[k]['end'].append(utils.get_time())
    spec_process.terminate()

    monitor_process.wait()

    # Clean up
    cmd = 'sudo chown zechengh ../ -R'
    print(cmd)
    p = subprocess.Popen(cmd.split())
    p_status = p.wait()

    cmd = 'python2 ../detector/preprocess.py --data_dir {save_data_dir} --window_size 100'.format(
        save_data_dir=save_data_dir
    )
    print(cmd)
    p = subprocess.Popen(cmd.split())
    p_status = p.wait()

    with open(os.path.join(save_data_dir, 'schedule'), 'w+') as f:
        json.dump(dict(schedule), f)
