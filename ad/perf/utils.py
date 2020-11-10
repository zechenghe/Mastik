import subprocess
import time

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

def clean_spec():

    spec_clean_cmd="/home/zechengh/Mastik/ad/bg_program/clean_spec.sh"
    print(spec_clean_cmd)
    spec_clean_process=subprocess.Popen(spec_clean_cmd.split())
    spec_clean_process.wait()
    return

def get_time():
    return int(time.time()*1000000)
