import numpy as np
from numba import cuda
import math

@cuda.jit(device=True)
def hash_function(x, y, grid_size):
    return int(x) + int(y) * grid_size

@cuda.jit(device=True)
def calculate_cell_index(x, y, cell_size, grid_width):
    cell_x = int((x + 1.0) / cell_size)  # Assuming x is in [-1, 1]
    cell_y = int((y + 1.0) / cell_size)  # Assuming y is in [-1, 1]
    return hash_function(cell_x, cell_y, grid_width)

@cuda.jit
def calculate_cell_indices(particles, cell_size, grid_width):
    i = cuda.grid(1)
    if i < particles.shape[0]:
        x, y = particles[i]['position']
        cell_index = calculate_cell_index(x, y, cell_size, grid_width)
        particles[i]['cell_index'] = cell_index

@cuda.jit
def count_sort_step(keys, values, output_keys, output_values, counts, bit_shift):
    i = cuda.grid(1)
    if i < keys.shape[0]:
        key = keys[i]
        digit = (key >> bit_shift) & 0xFF
        index = cuda.atomic.add(counts, digit, 1)
        output_keys[index] = key
        output_values[index] = values[i]

@cuda.jit
def compute_offsets(counts, offsets):
    i = cuda.grid(1)
    if i < counts.shape[0]:
        if i == 0:
            offsets[i] = 0
        else:
            offsets[i] = offsets[i-1] + counts[i-1]

@cuda.jit
def fill_array(arr, value):
    i = cuda.grid(1)
    if i < arr.shape[0]:
        arr[i] = value

class Grid:
    def __init__(self, width, height, cell_size):
        self.width = width
        self.height = height
        self.cell_size = cell_size

        self.grid_width = math.ceil(width / cell_size)
        self.grid_height = math.ceil(height / cell_size)
        self.total_cells = self.grid_width * self.grid_height

        # Initialize grid structure
        self.grid_start = cuda.to_device(np.zeros(self.total_cells, dtype=np.int32))
        self.grid_end = cuda.to_device(np.zeros(self.total_cells, dtype=np.int32))

        # Temporary arrays for radix sort
        self.temp_keys = None
        self.temp_values = None

    def radix_sort(self, keys, values):
        if self.temp_keys is None or self.temp_keys.shape[0] != keys.shape[0]:
            self.temp_keys = cuda.device_array_like(keys)
            self.temp_values = cuda.device_array_like(values)

        counts = cuda.device_array(256, dtype=np.int32)
        offsets = cuda.device_array(256, dtype=np.int32)

        threads_per_block = 256
        blocks_per_grid = (keys.shape[0] + threads_per_block - 1) // threads_per_block

        for shift in range(0, 32, 8):
            fill_array[1, 256](counts, 0)  # Fill counts with zeros
            count_sort_step[blocks_per_grid, threads_per_block](keys, values, self.temp_keys, self.temp_values, counts, shift)
            compute_offsets[1, 256](counts, offsets)
            keys, self.temp_keys = self.temp_keys, keys
            values, self.temp_values = self.temp_values, values

        return keys, values

    def update_grid(self, particles):
        threads_per_block = 256
        blocks_per_grid = (particles.shape[0] + threads_per_block - 1) // threads_per_block

        # Calculate cell indices
        calculate_cell_indices[blocks_per_grid, threads_per_block](particles, self.cell_size, self.grid_width)

        # Extract cell indices and particle indices
        cell_indices = cuda.device_array(particles.shape[0], dtype=np.int32)
        particle_indices = cuda.to_device(np.arange(particles.shape[0], dtype=np.int32))

        @cuda.jit
        def extract_cell_indices(particles, cell_indices):
            i = cuda.grid(1)
            if i < particles.shape[0]:
                cell_indices[i] = particles[i]['cell_index']

        extract_cell_indices[blocks_per_grid, threads_per_block](particles, cell_indices)

        # Sort particles based on cell indices
        sorted_cell_indices, sorted_particle_indices = self.radix_sort(cell_indices, particle_indices)

        # Reset grid
        grid_blocks = (self.total_cells + threads_per_block - 1) // threads_per_block
        fill_array[grid_blocks, threads_per_block](self.grid_start, 0)
        fill_array[grid_blocks, threads_per_block](self.grid_end, 0)

        # Update grid structure
        @cuda.jit
        def update_grid_structure(sorted_cell_indices, sorted_particle_indices, grid_start, grid_end):
            i = cuda.grid(1)
            if i < sorted_cell_indices.shape[0]:
                cell_index = sorted_cell_indices[i]
                if i == 0 or cell_index != sorted_cell_indices[i-1]:
                    grid_start[cell_index] = i
                if i == sorted_cell_indices.shape[0] - 1 or cell_index != sorted_cell_indices[i+1]:
                    grid_end[cell_index] = i + 1

        update_grid_structure[blocks_per_grid, threads_per_block](sorted_cell_indices, sorted_particle_indices, self.grid_start, self.grid_end)

        return sorted_particle_indices

@cuda.jit(device=True)
def get_cell_particles(grid_start, grid_end, sorted_particle_indices, cell_index):
    start = grid_start[cell_index]
    end = grid_end[cell_index]
    return sorted_particle_indices[start:end]

@cuda.jit(device=True)
def get_neighboring_cells(cell_index, grid_width, grid_height):
    cell_x = cell_index % grid_width
    cell_y = cell_index // grid_width
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            nx = cell_x + dx
            ny = cell_y + dy
            if 0 <= nx < grid_width and 0 <= ny < grid_height:
                neighbors.append(hash_function(nx, ny, grid_width))
    return neighbors
