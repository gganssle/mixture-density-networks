import numba
import numpy as np

one = np.ones((10000,10000))
two = np.ones((10000,10000))

def mully(a,b):
    np.matmul(a,b)

%%timeit
mully(one, two)

@numba.jit
def mully(a,b):
    np.matmul(a,b)

%%timeit
mully(one,two)
