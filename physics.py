import math
import numpy as np
import numba.cuda
from numba import cuda, float32


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
    return weight

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
def calc_field_props(positions, masses, field_values, h):
    i = cuda.grid(1)
    if i >= positions.shape[0]:
        return

    field_values[i] = 0
    x_i, y_i = positions[i]
    # Calculate the influence of all particles on particle i
    for j in range(positions.shape[0]):
        x_j, y_j = positions[j]
        r = euclidean_distance(x_i, x_j, y_i, y_j)
        q = 1 / h * r
        field_values[i] += cubic_spline(q) * masses[j]

@cuda.jit
def calc_field_gradients(positions, masses, densities, h):
    i = cuda.grid(1)
    if i >= positions.shape[0]:
        return

    x_i, y_i = positions[i]
    # Calculate the influence of all particles on particle i
    for j in range(positions.shape[0]):
        x_j, y_j = positions[j]
        r = euclidean_distance(x_i, x_j, y_i, y_j)
        q = 1 / h * r
        dx, dy = direction_to(x_i, y_i, x_j, y_j)  
        field_values[i] += cubic_spline(q) * masses[j]

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
