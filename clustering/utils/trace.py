import math
import os
import subprocess
import time
from contextlib import contextmanager

import numpy as np
import psutil
import torch


def get_gpu_memory(
    cmd_path="nvidia-smi", target_properties=("memory.total", "memory.used")
):
    """
    ref: https://www.12-technology.com/2022/01/pythongpu.html
    Returns
    -------
    gpu_total : ndarray,  "memory.total"
    gpu_used: ndarray, "memory.used"
    """

    # format option
    format_option = "--format=csv,noheader,nounits"

    cmd = "%s --query-gpu=%s %s" % (
        cmd_path,
        ",".join(target_properties),
        format_option,
    )

    # Command execution in sub-processes
    cmd_res = subprocess.check_output(cmd, shell=True)

    gpu_lines = cmd_res.decode().split("\n")[0].split(", ")

    gpu_total = int(gpu_lines[0]) / 1024
    gpu_used = int(gpu_lines[1]) / 1024

    gpu_total = np.round(gpu_used, 1)
    gpu_used = np.round(gpu_used, 1)
    return gpu_total, gpu_used


class Trace:
    """utility class to trace memory, time and gpu."""

    cuda = torch.cuda.is_available()

    @contextmanager
    def timer(self, title, log):
        t0 = time.time()
        p = psutil.Process(os.getpid())
        cpu_m0 = p.memory_info().rss / 2.0**30
        if self.cuda:
            gpu_m0 = get_gpu_memory()[0]
        yield
        cpu_m1 = p.memory_info().rss / 2.0**30
        if self.cuda:
            gpu_m1 = get_gpu_memory()[0]

        cpu_delta = cpu_m1 - cpu_m0
        if self.cuda:
            gpu_delta = gpu_m1 - gpu_m0

        cpu_sign = "+" if cpu_delta >= 0 else "-"
        cpu_delta = math.fabs(cpu_delta)

        if self.cuda:
            gpu_sign = "+" if gpu_delta >= 0 else "-"
        if self.cuda:
            gpu_delta = math.fabs(gpu_delta)

        cpu_message = f"{cpu_m1:.1f}GB({cpu_sign}{cpu_delta:.1f}GB)"
        if self.cuda:
            gpu_message = f"{gpu_m1:.1f}GB({gpu_sign}{gpu_delta:.1f}GB)"

        if self.cuda:
            message = f"Total time and resource utilization = [cpu: {cpu_message}, gpu: {gpu_message}: {time.time() - t0:.1f}sec] Module = [{title}] "
        else:
            message = f"Total time and resource utilization = [cpu: {cpu_message}: {time.time() - t0:.1f}sec] Module = [{title}]"

        print(message)
        log.info(message)


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            pass

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df