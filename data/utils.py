import psutil
import subprocess
import os


def get_gpu_usage_smi():
    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
                            stdout=subprocess.PIPE, text=True)
    lines = result.stdout.strip().split('\n')
    usage_info = []
    for i, line in enumerate(lines):
        gpu_util, mem_used, mem_total = line.split(', ')
        usage_info.append({
            'gpu_index': i,
            'gpu_utilization_percent': int(gpu_util),
            'memory_used_MB': int(mem_used),
            'memory_total_MB': int(mem_total),
        })
    return usage_info


def get_memory_usage_gb():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    memory_usage_gb = mem_info.rss / (1024 ** 3)
    return memory_usage_gb
