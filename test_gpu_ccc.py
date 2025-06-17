import numpy as np

import os

import time

from ccc.coef import ccc



SIZE = int(os.environ.get("SIZE", 1000000))

FEATURES = int(os.environ.get("FEATURES", 10))

NODES = int(os.environ.get("NODES", 1))

GPUS_USED = int(os.environ.get("GPUS_USED", 1))



x = np.random.normal(size=(FEATURES, SIZE))



start = time.time()

ccc(x)

end = time.time()



print(f"{NODES} {GPUS_USED} {SIZE} {FEATURES} {end - start:.4f}")


