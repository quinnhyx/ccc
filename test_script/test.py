from mpi4py import MPI
import numpy as np
from ccc.coef import ccc
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
np.random.seed(123)
# node = socket.gethostname()

size = int(os.environ.get("SIZE", 10000000))
n_features = int(os.environ.get("FEATURES", 16))
# n_jobs = int(os.environ.get("N_JOBS", 2))

x = np.random.normal(size=(n_features, size))

comm.Barrier()
if rank == 0:
    print(f"[rank 0] Calling CCC...")
    start = MPI.Wtime()

# print(f"[rank {rank}] running on node {node}")
ccc(x, gpu=True, mpi=True)
    
comm.Barrier()  
if rank == 0:
    end = MPI.Wtime()
    print(f"[rank 0] Time elapsed: {end - start:.4f} seconds")
