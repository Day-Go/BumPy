import math
import numpy as np
import numba.cuda
from numba import cuda, float32

TARGET_DENSITY = 8
PRESSURE_MULTIPLIER = 150
EPSILON = 1e-5

@cuda.jit(device=True)
def convert_density_to_pressure(density, target_density, pressure_multiplier):
    density_error = density - target_density
    pressure = density_error * pressure_multiplier
    return max(pressure, 0)

@cuda.jit(device=True)
def euclidean_distance(x_i, x_j, y_i, y_j):
    return math.sqrt((x_i - x_j) ** 2 + (y_i - y_j) ** 2)

@cuda.jit(device=True)
def direction_to(x_i, y_i, x_j, y_j):
    dx = x_j - x_i
    dy = y_j - y_i
    distance = euclidean_distance(x_i, x_j, y_i, y_j)
    if distance > 0:
        dx /= distance
        dy /= distance
    return dx, dy

@cuda.jit(device=True)
def cubic_spline(q):
    if q >= 0.5 and q <= 1:
        base = 1 - q
        weight = 2 * math.pow(base, 3)
    elif q < 0.5:
        weight = 6 * (math.pow(q, 3) - math.pow(q, 2)) + 1
    else:
        weight = 0
    volume = 4 / (3 * math.pi)
    return weight / volume

@cuda.jit(device=True)
def cubic_spline_dq(q, h):
    if q >= 0.5 and q <= 1:
        base = 1 - q
        weight = -6 * math.pow(base, 2)
    elif q < 0.5:
        weight = 6 * q * (3 * q - 2) 
    else:
        weight = 0
    volume = 40 / (7 * math.pi * math.pow(h, 2))
    return weight / volume

@cuda.jit(device=True)
def poly6_kernel(r, h):
    if r > h: return 0
    influence = 315 / (64 * math.pi * h**9)
    return influence * (h**2 - r**2)**3

@cuda.jit(device=True)
def poly6_gradient_kernel(r, h):
    if r > h: return 0
    influence = -2835 / (64 * math.pi * math.pow(h, 3))
    return influence * (h**2 - r**2)**3

@cuda.jit(device=True)
def spiky_kernel(r, h):
    influence = 15.0 / (math.pi * h**6)
    return influence * (h - r)**3

@cuda.jit(device=True)
def spiky_kernel_gradient(r, h):
    influence = -90 / (math.pi * h**7)
    return influence * (h - r)**2

@cuda.jit(device=True)
def calc_shared_pressure(d1, d2, target_density, pressure_multiplier):
    p1 = convert_density_to_pressure(d1, target_density, pressure_multiplier)
    p2 = convert_density_to_pressure(d2, target_density, pressure_multiplier)
    return (p1 + p2) / 2

@cuda.jit(fastmath=True)
def calc_densities(particle_array, sorted_particle_indices, grid_start, grid_end, h):
    idx = cuda.grid(1)
    
    # Shared memory for particle positions and masses
    shared_x = cuda.shared.array(shape=512, dtype=float32)
    shared_y = cuda.shared.array(shape=512, dtype=float32)
    shared_masses = cuda.shared.array(shape=512, dtype=float32)
    
    grid_width = int(math.sqrt(grid_start.shape[0]))
    
    if idx < particle_array.shape[0]:
        particle_idx = sorted_particle_indices[idx]
        particle = particle_array[particle_idx]
        particle['density'] = 0
        x_i, y_i = particle['position']
        cell_index = int(particle['cell_index'])
        
        for cell_offset_y in range(-1, 2):
            for cell_offset_x in range(-1, 2):
                neighbor_cell = cell_index + cell_offset_x + cell_offset_y * grid_width
                if 0 <= neighbor_cell < grid_start.shape[0]:
                    start = int(grid_start[neighbor_cell])
                    end = int(grid_end[neighbor_cell])
                    
                    for j in range(start, end, cuda.blockDim.x):
                        # Load a block of particles into shared memory
                        shared_idx = cuda.threadIdx.x
                        if j + shared_idx < end:
                            neighbor_idx = sorted_particle_indices[j + shared_idx]
                            shared_x[shared_idx] = particle_array[neighbor_idx]['position'][0]
                            shared_y[shared_idx] = particle_array[neighbor_idx]['position'][1]
                            shared_masses[shared_idx] = particle_array[neighbor_idx]['mass']
                        
                        cuda.syncthreads()
                        
                        # Process particles in shared memory
                        for k in range(min(cuda.blockDim.x, end - j)):
                            x_j = shared_x[k]
                            y_j = shared_y[k]
                            r = euclidean_distance(x_i, x_j, y_i, y_j)
                            if r < h:
                                particle['density'] += poly6_kernel(r, h) * shared_masses[k]
                        
                        #cuda.syncthreads()

@cuda.jit(fastmath=True)
def calc_density_gradients(particle_array, sorted_particle_indices, grid_start, grid_end, h):
    idx = cuda.grid(1)
    
    # Shared memory for particle positions and masses
    shared_x = cuda.shared.array(shape=512, dtype=float32)
    shared_y = cuda.shared.array(shape=512, dtype=float32)
    shared_masses = cuda.shared.array(shape=512, dtype=float32)
    
    grid_width = int(math.sqrt(grid_start.shape[0]))
    
    if idx < particle_array.shape[0]:
        particle_idx = sorted_particle_indices[idx]
        particle = particle_array[particle_idx]
        particle['density_gradient'][0] = 0
        particle['density_gradient'][1] = 0
        x_i, y_i = particle['position']
        cell_index = int(particle['cell_index'])
        
        for cell_offset_y in range(-1, 2):
            for cell_offset_x in range(-1, 2):
                neighbor_cell = cell_index + cell_offset_x + cell_offset_y * grid_width
                if 0 <= neighbor_cell < grid_start.shape[0]:
                    start = int(grid_start[neighbor_cell])
                    end = int(grid_end[neighbor_cell])
                    
                    for j in range(start, end, cuda.blockDim.x):
                        # Load a block of particles into shared memory
                        shared_idx = cuda.threadIdx.x
                        if j + shared_idx < end:
                            neighbor_idx = sorted_particle_indices[j + shared_idx]
                            shared_x[shared_idx] = particle_array[neighbor_idx]['position'][0]
                            shared_y[shared_idx] = particle_array[neighbor_idx]['position'][1]
                            shared_masses[shared_idx] = particle_array[neighbor_idx]['mass']
                        
                        cuda.syncthreads()
                        
                        # Process particles in shared memory
                        for k in range(min(cuda.blockDim.x, end - j)):
                            if j + k != idx:
                                x_j = shared_x[k]
                                y_j = shared_y[k]
                                r = euclidean_distance(x_i, x_j, y_i, y_j)
                                if r < h:
                                    dx, dy = direction_to(x_i, y_i, x_j, y_j)
                                    slope = poly6_gradient_kernel(r, h)
                                    grad_x = shared_masses[k] * slope * dx / ((h * r) + EPSILON)
                                    grad_y = shared_masses[k] * slope * dy / ((h * r) + EPSILON)
                                    particle['density_gradient'][0] -= grad_x
                                    particle['density_gradient'][1] -= grad_y
                        
                        cuda.syncthreads()


@cuda.jit(fastmath=True)
def calc_pressure_force(particle_array, sorted_particle_indices, grid_start, grid_end, h, target_density, pressure_multiplier):
    idx = cuda.grid(1)
    
    # Shared memory for particle positions, densities, and masses
    shared_x = cuda.shared.array(shape=512, dtype=float32)
    shared_y = cuda.shared.array(shape=512, dtype=float32)
    shared_densities = cuda.shared.array(shape=512, dtype=float32)
    shared_masses = cuda.shared.array(shape=512, dtype=float32)
    
    grid_width = int(math.sqrt(grid_start.shape[0]))
    
    if idx < particle_array.shape[0]:
        particle_idx = sorted_particle_indices[idx]
        particle = particle_array[particle_idx]
        particle['pressure_force'][0] = 0
        particle['pressure_force'][1] = 0
        x_i, y_i = particle['position']
        cell_index = int(particle['cell_index'])
        
        for cell_offset_y in range(-1, 2):
            for cell_offset_x in range(-1, 2):
                neighbor_cell = cell_index + cell_offset_x + cell_offset_y * grid_width
                if 0 <= neighbor_cell < grid_start.shape[0]:
                    start = int(grid_start[neighbor_cell])
                    end = int(grid_end[neighbor_cell])
                    
                    for j in range(start, end, cuda.blockDim.x):
                        # Load a block of particles into shared memory
                        shared_idx = cuda.threadIdx.x
                        if j + shared_idx < end:
                            neighbor_idx = sorted_particle_indices[j + shared_idx]
                            shared_x[shared_idx] = particle_array[neighbor_idx]['position'][0]
                            shared_y[shared_idx] = particle_array[neighbor_idx]['position'][1]
                            shared_densities[shared_idx] = particle_array[neighbor_idx]['density']
                            shared_masses[shared_idx] = particle_array[neighbor_idx]['mass']
                        
                        cuda.syncthreads()
                        
                        # Process particles in shared memory
                        for k in range(min(cuda.blockDim.x, end - j)):
                            if j + k != idx:
                                x_j = shared_x[k]
                                y_j = shared_y[k]
                                r = euclidean_distance(x_i, x_j, y_i, y_j)
                                if r < h:
                                    q = r / h
                                    dx, dy = direction_to(x_i, y_i, x_j, y_j)
                                    slope = cubic_spline_dq(q, h)
                                    shared_pressure = calc_shared_pressure(particle['density'], shared_densities[k], target_density, pressure_multiplier)
                                    force_x = shared_pressure * dx * slope * particle['mass'] / (particle['density'] + EPSILON)
                                    force_y = shared_pressure * dy * slope * particle['mass'] / (particle['density'] + EPSILON)
                                    particle['pressure_force'][0] += force_x 
                                    particle['pressure_force'][1] += force_y 
                        
                        cuda.syncthreads()

@cuda.jit(device=True)
def update_velocity(velocity, pressure_force, density, dt):
    if density != 0:
        pressure_accel_x = pressure_force[0] / density
        pressure_accel_y = pressure_force[1] / density
    else:
        pressure_accel_x = 0
        pressure_accel_y = 0

    velocity[0] += pressure_accel_x * dt
    velocity[1] += pressure_accel_y * dt
    return velocity

@cuda.jit
def update_positions(particle_array, dt, screen_width, screen_height):
    idx = cuda.grid(1)
    stride = cuda.gridsize(1)
    for i in range(idx, particle_array.shape[0], stride):
        particle = particle_array[i]
        
        EPSILON = 1e-5
        
        update_velocity(
            particle['velocity'], particle['pressure_force'], max(particle['density'], EPSILON), dt
        )

        new_vx = particle['velocity'][0] + 0.5 * particle['pressure_force'][0] / max(particle['density'], EPSILON) * dt
        new_vy = particle['velocity'][1] + 0.5 * particle['pressure_force'][1] / max(particle['density'], EPSILON) * dt

        new_x = particle['position'][0] + new_vx * dt
        new_y = particle['position'][1] + new_vy * dt

        damping = 0.99
        if new_x <= -screen_width:
            new_x = -screen_width
            new_vx *= -damping
        elif new_x >= screen_width:
            new_x = screen_width
            new_vx *= -damping

        if new_y <= -screen_height:
            new_y = -screen_height
            new_vy *= -damping
        elif new_y >= screen_height:
            new_y = screen_height
            new_vy *= -damping

        particle['velocity'][0] = new_vx + 0.5 * particle['pressure_force'][0] / max(particle['density'], EPSILON) * dt
        particle['velocity'][1] = new_vy + 0.5 * particle['pressure_force'][1] / max(particle['density'], EPSILON) * dt

        particle['position'][0] = new_x
        particle['position'][1] = new_y

@cuda.jit
def extract_positions(particle_array, positions):
    i = cuda.grid(1)
    if i < particle_array.shape[0]:
        positions[i, 0] = particle_array[i]['position'][0]
        positions[i, 1] = particle_array[i]['position'][1]

@cuda.jit(device=True)
def rotate_point(x, y, angle):
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    return x * cos_angle - y * sin_angle, x * sin_angle + y * cos_angle

@cuda.jit(device=True)
def point_in_rotated_rectangle(px, py, rect_x, rect_y, rect_width, rect_height, angle):
    # Translate point to origin
    translated_x = px - rect_x
    translated_y = py - rect_y

    # Rotate point
    rotated_x, rotated_y = rotate_point(translated_x, translated_y, -angle)

    # Check if point is inside rectangle
    half_width = rect_width / 2
    half_height = rect_height / 2
    return -half_width <= rotated_x <= half_width and -half_height <= rotated_y <= half_height
