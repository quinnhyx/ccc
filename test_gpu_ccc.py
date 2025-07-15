import numpy as np

import pandas as pd

from ccc.coef import ccc

import time



data_sizes = [10000000]  # 100 to 10,000,000

feature_counts = [16]  # 2, 4, 6, ..., 20



print("Running CCC with GPU support...\n")



for n_features in feature_counts:

    for data_size in data_sizes:

        print(f"Testing with {n_features} features and {data_size} samples...")

        x = np.random.normal(size=(n_features, data_size))



        start = time.time()

        result = ccc(x,gpu=True)

        end = time.time()

        elapsed_s = end - start



        # print(f"CCC result: {result}")

        print(f"Time taken: {elapsed_s:.2f} s\n")

