import os



# Set number of GPUs to use

GPUS_USED = int(os.environ.get("GPUS_USED", 1))

if "CUDA_VISIBLE_DEVICES" not in os.environ:

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(GPUS_USED)))



import numpy as np

import time

from ccc.coef import ccc



SIZE = int(os.environ.get("SIZE", 1000000))

FEATURES = int(os.environ.get("FEATURES", 10))

NODES = int(os.environ.get("NODES", 1))



x = np.random.normal(size=(FEATURES, SIZE))



start = time.time()

ccc(x)

end = time.time()



print(f"{NODES} {GPUS_USED} {SIZE} {FEATURES} {end - start:.4f}")

