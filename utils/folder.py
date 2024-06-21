import os

def write_log(log_file, str, mode='a'):
    with open(log_file, mode) as f:
        f.write(str)
