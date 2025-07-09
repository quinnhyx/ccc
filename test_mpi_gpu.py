import os
import time
import numpy as np
from mpi4py import MPI
from ccc.coef.impl import ccc

size = int(os.environ.get("SIZE", 10000))
n_features = int(os.environ.get("FEATURES", 20))

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

print(f"[Rank {rank}] Starting CCC with features={n_features}, size={size}")

np.random.seed(123)
data = np.random.normal(size=(n_features, size))

comm.Barrier()
if rank == 0:
    print(f"[rank 0] Calling CCC...")
    start = MPI.Wtime()

print(f"[rank {rank}] running on node")
result = ccc(data, gpu=True, node=True)

comm.Barrier()
if rank == 0:
    end = MPI.Wtime()
    print(f"[rank 0] Time elapsed: {end - start:.2f} seconds")

    print(f"result: {result}")
