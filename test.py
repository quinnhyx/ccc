import os
import time
import numpy as np
from ccc.coef.impl import ccc
import torch

def main():
    size = int(os.environ.get("SIZE", 10000000))
    n_features = int(os.environ.get("FEATURES", 16))

    print(f"{torch.cuda.device_count()} GPUs available: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
    print(f"Starting CCC with features={n_features}, size={size}")

    np.random.seed(123)
    data = np.random.normal(size=(n_features, size))

    start = time.time()
    result = ccc(data, gpu=True)  # internally uses multiprocessing/gpu_compute_coef
    end = time.time()

    print(f"CCC result: {result}")
    print(f"Time taken: {end - start:.2f} s")

if __name__ == '__main__':
    main()
