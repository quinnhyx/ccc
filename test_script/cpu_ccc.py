#import numpy as np
#from ccc.coef import ccc
#import time

#data_sizes = [10000,100000,1000000,10000000] 
#feature_counts = list(range(2, 21, 2))  # 2, 4, ..., 20

#print("Running CCC timing test...\n")

#for n_features in feature_counts:
 #   for data_size in data_sizes:
  #      print(f"Testing with {n_features} features and {data_size} samples...")
   #     x = np.random.normal(size=(n_features, data_size))

    #    start = time.time()
     #   result = ccc(x)
      #  end = time.time()
       # elapsed_ms = (end - start) * 1000

       # print(f"CCC result: {result}")
       # print(f"Time taken: {elapsed_ms:.2f} ms\n")

import os

import time

import numpy as np

from ccc.coef import ccc  # Your CPU-only CCC implementation



# Optional thread control for NumPy, if CCC uses multithreading

THREADS = int(os.environ.get("THREADS", 1))

os.environ["OMP_NUM_THREADS"] = str(THREADS)

os.environ["MKL_NUM_THREADS"] = str(THREADS)



# Parameters

SIZE = int(os.environ.get("SIZE", 1000000))

FEATURES = int(os.environ.get("FEATURES", 10))

NODES = int(os.environ.get("NODES", 1))



# Run CCC

x = np.random.normal(size=(FEATURES, SIZE))

start = time.time()

ccc(x)

end = time.time()



# Output log format

print(f"{NODES} {THREADS} {SIZE} {FEATURES} {end - start:.4f}")

