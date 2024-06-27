import math
import numpy as np
import numba.cuda
from numba import cuda, float32

TARGET_DENSITY = 8
PRESSURE_MULTIPLIER = 0.01

@cuda.jit(device=True)
def convert_density_to_pressure(density):
    density_error = density - TARGET_DENSITY
    pressure = density_error * PRESSURE_MULTIPLIER
    return pressure

@cuda.jit(device=True)
def gaussian_kernel(r, h):
    return math.exp(-(r**2) / (2 * h**2))


@cuda.jit(device=True)
def euclidean_distance(x_i, x_j, y_i, y_j):
    return (x_i - x_j) ** 2 + (y_i - y_j) ** 2

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
def cubic_spline_dq(q):
    if q >= 0.5 and q <= 1:
        base = 1 - q
        weight = -6 * math.pow(base, 2)
    elif q < 0.5:
        weight = 6 * q * (3 * q - 2) 
    else:
        weight = 0
    return weight


@cuda.jit
def calc_densities(particles_array, h):
    i = cuda.grid(1)
    if i >= particles_array.shape[0]:
        return

    particle = particles_array[i]
    particle['density'] = 0
    x_i, y_i = particle['position']
    # Calculate the influence of all particles on particle i
    for j in range(particles_array.shape[0]):
        x_j, y_j = particles_array[j]['position'] 
        r = euclidean_distance(x_i, x_j, y_i, y_j)
        q = 1 / h * r
        particle['density'] += cubic_spline(q) * particles_array[j]['mass']

@cuda.jit
def calc_density_gradients(particles_array, h):
    i = cuda.grid(1)
    if i >= particles_array.shape[0]:
        return

    particle = particles_array[i]
    particle['density_gradient'][0] = 0
    particle['density_gradient'][1] = 0

    x_i, y_i = particle['position']
    # Calculate the influence of all particles on particle i
    for j in range(particles_array.shape[0]):
        x_j, y_j = particles_array[j]['position']
        r = euclidean_distance(x_i, x_j, y_i, y_j)
        q = 1 / h * r
        dx, dy = direction_to(x_i, y_i, x_j, y_j)
        slope = cubic_spline_dq(q)

        grad_x = -particle['density'] * particles_array[j]['mass'] * slope * dx / particle['density']
        grad_y = -particle['density'] * particles_array[j]['mass'] * slope * dy / particle['density']
        particle['density_gradient'][0] += grad_x
        particle['density_gradient'][1] += grad_y

@cuda.jit(device=True)
def calc_shared_pressure(d1, d2):
    p1 = convert_density_to_pressure(d1)
    p2 = convert_density_to_pressure(d2)
    return (p1 + p2) / 2

@cuda.jit
def calc_pressure_force(particles_array, h):
    i = cuda.grid(1)
    if i >= particles_array.shape[0]:
        return

    particle = particles_array[i]
    particle['pressure_force'][0] = 0
    particle['pressure_force'][1] = 0
    x_i, y_i = particle['position']
    for j in range(particles_array.shape[0]):
        if i == j: continue

        x_j, y_j = particles_array[j]['position']
        r = euclidean_distance(x_i, x_j, y_i, y_j)
        q = 1 / h * r
        dx, dy = direction_to(x_i, y_i, x_j, y_j)
        slope = cubic_spline_dq(q)

        shared_pressure = calc_shared_pressure(particle['density'], particles_array[j]['density'])
        force_x = shared_pressure * dx * slope * particle['mass'] / particle['density']
        force_y = shared_pressure * dy * slope * particle['mass'] / particle['density']
        particle['pressure_force'][0] += force_x 
        particle['pressure_force'][1] += force_y 

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
def update_positions(particles_array, dt, screen_width, screen_height):
    i = cuda.grid(1)
    if i >= particles_array.shape[0]:
        return

    particle = particles_array[i]

    # Update velocity first (assuming this updates both components)
    update_velocity(
        particle['velocity'], particle['pressure_force'], particle['density'], dt
    )

    # Tentative new position calculation
    new_x = particle['position'][0] + particle['velocity'][0] * dt
    new_y = particle['position'][1] + particle['velocity'][1] * dt

    # Check for collisions with the horizontal walls
    if new_x <= -screen_width or new_x >= screen_width:
        particle['velocity'][0] *= -1  # Reverse the horizontal component of the velocity
        new_x = particle['position'][0]  # Reset position to current to prevent sticking to walls

    # Check for collisions with the vertical walls
    if new_y <= -screen_height or new_y >= screen_height:
        particle['velocity'][1] *= -1  # Reverse the vertical component of the velocity
        new_y = particle['position'][1]  # Reset position to current to prevent sticking to walls

    # Update position with potentially modified velocity
    particle['position'][0] = new_x
    particle['position'][1] = new_y

def gaussian_kernel_np(r, h):
    return np.exp(-(r**2) / (2 * h**2))


def smoothing_kernel_np(positions, masses, densities, field_values, h):
    n = positions.shape[0]
    outputs = np.zeros(n)

    # Compute all pairwise distances
    for i in range(n):
        r = np.linalg.norm(positions[i] - positions, axis=1)
        weights = gaussian_kernel_np(r, h) * masses / densities
        total_weight = np.sum(weights)
        weighted_sum = np.sum(weights * field_values)

        if total_weight > 0:
            outputs[i] = weighted_sum / total_weight
        else:
            outputs[i] = 0

    return outputs
