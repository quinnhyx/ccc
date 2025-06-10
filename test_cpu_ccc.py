import numpy as np
from ccc.coef import ccc
import time

data_sizes = [i for i in range(100000, 1000001, 100000)] 
feature_counts = list(range(2, 21, 2))  # 2, 4, ..., 20

print("Running CCC timing test...\n")

for n_features in feature_counts:
    for data_size in data_sizes:
        print(f"Testing with {n_features} features and {data_size} samples...")
        x = np.random.normal(size=(n_features, data_size))

        start = time.time()
        result = ccc(x)
        end = time.time()
        elapsed_ms = (end - start) * 1000

        print(f"CCC result: {result}")
        print(f"Time taken: {elapsed_ms:.2f} ms\n")
