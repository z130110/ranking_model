import numpy as np
import gc
import os
import psutil

class np_iter(object):
    def __init__(self, input_np):
        self.input_np = input_np

current_meomory_used = psutil.Process(os.getpid()).memory_info().rss / (2 ** 30)
print("memory used before create arr", current_meomory_used)
arr = np.random.rand(10000000, 20)
current_meomory_used = psutil.Process(os.getpid()).memory_info().rss / (2 ** 30)
print("memory used after create arr", current_meomory_used)

iter_ins = np_iter(arr)
current_meomory_used = psutil.Process(os.getpid()).memory_info().rss / (2 ** 30)
print("memory used after create iter_ins", current_meomory_used)

del arr, iter_ins.input_np
#gc.collect()
current_meomory_used = psutil.Process(os.getpid()).memory_info().rss / (2 ** 30)
print("memory used after delete iter_ins", current_meomory_used)
