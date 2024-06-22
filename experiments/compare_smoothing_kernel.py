import os
import sys
import time
import numpy as np
import numba.cuda
from numba import cuda

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from physics import smoothing_kernel, smoothing_kernel_numpy


def main_comparison():
    N = 1000 # Number of particles
    h = 1.0    # Smoothing length
    positions = np.random.rand(N, 3).astype(np.float32)
    masses = np.random.rand(N).astype(np.float32) + 1  # Avoid zero masses
    densities = np.random.rand(N).astype(np.float32) + 1  # Avoid zero densities
    field_values = np.random.rand(N).astype(np.float32)
    outputs_cuda = np.zeros(N, dtype=np.float32)
    outputs_numpy = np.zeros(N, dtype=np.float32)

    # CUDA setup
    threads_per_block = 16 
    blocks_per_grid = (N + (threads_per_block - 1)) // threads_per_block

    # Warm up CUDA
    smoothing_kernel[blocks_per_grid, threads_per_block](positions, masses, densities, field_values, outputs_cuda, h)
    cuda.synchronize()  # Ensure CUDA finishes before timing

    # Timing CUDA implementation
    start_time = time.time()
    for _ in range(100):
        smoothing_kernel[blocks_per_grid, threads_per_block](positions, masses, densities, field_values, outputs_cuda, h)
        cuda.synchronize()  # Ensure CUDA finishes
    cuda_time = time.time() - start_time

    # Timing NumPy implementation
    start_time = time.time()
    for _ in range(100):
        outputs_numpy = smoothing_kernel_numpy(positions, masses, densities, field_values, h)
    numpy_time = time.time() - start_time

    print(f"CUDA Execution Time: {cuda_time:.6f} seconds")
    print(f"NumPy Execution Time: {numpy_time:.6f} seconds")

if __name__ == '__main__':
    main_comparison()
