if __name__ == "__main__":
    import numpy as np
    from ccc.coef.impl import ccc
    import time

    n_features = 16
    n_objects = 10000000
    x = np.random.rand(n_features, n_objects)

    print("Running CCC with GPU support...")

    start = time.time()
    result= ccc(x, mpi=True)
    end = time.time()

    print(f"Time taken: {end - start:.2f} s")
    print(f"Result: {result}")
