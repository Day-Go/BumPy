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
    return min(pressure, 0)

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
def spiky_kernel(r, h):
    factor = 15.0 / (math.pi * h**6)
    return factor * (h - r)**3

@cuda.jit(device=True)
def spiky_kernel_gradient(r, h):
    factor = -45.0 / (math.pi * h**6)
    return factor * (h - r)**2

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
        r = math.sqrt(euclidean_distance(x_i, x_j, y_i, y_j))
        particle['density'] += spiky_kernel(r, h) * particles_array[j]['mass']

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
        r = math.sqrt(euclidean_distance(x_i, x_j, y_i, y_j))
        if r < h:
            dx, dy = direction_to(x_i, y_i, x_j, y_j)
            slope = spiky_kernel_gradient(r, h)

            grad_x = particles_array[j]['mass'] * slope * dx / (h * r)
            grad_y = particles_array[j]['mass'] * slope * dy / (h * r)
            particle['density_gradient'][0] += grad_x
            particle['density_gradient'][1] += grad_y

@cuda.jit(device=True)
def calc_shared_pressure(d1, d2, target_density, pressure_multiplier):
    p1 = convert_density_to_pressure(d1, target_density, pressure_multiplier)
    p2 = convert_density_to_pressure(d2, target_density, pressure_multiplier)
    return (p1 + p2) / 2

@cuda.jit
def calc_pressure_force(particles_array, h, target_density, pressure_multiplier):
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
        r = math.sqrt(euclidean_distance(x_i, x_j, y_i, y_j))
        if r < h:
            q = r / h
            dx, dy = direction_to(x_i, y_i, x_j, y_j)
            slope = cubic_spline_dq(q, h)

            shared_pressure = calc_shared_pressure(particle['density'], particles_array[j]['density'], target_density, pressure_multiplier)
            force_x = shared_pressure * dx * slope * particle['mass'] / (particle['density'] + EPSILON)
            force_y = shared_pressure * dy * slope * particle['mass'] / (particle['density'] + EPSILON)
            particle['pressure_force'][0] -= force_x 
            particle['pressure_force'][1] -= force_y 

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
def update_positions(particles_array, dt, screen_width, screen_height, rect_width, rect_height, rect_angle):
    #def update_positions(particles_array, dt, screen_width, screen_height):
    i = cuda.grid(1)
    if i >= particles_array.shape[0]:
        return

    particle = particles_array[i]

    # Update velocity first (assuming this updates both components)
    update_velocity(
        particle['velocity'], particle['pressure_force'], particle['density'], dt
    )

    # Leapfrog integration
    new_vx = particle['velocity'][0] + 0.5 * particle['pressure_force'][0] / particle['density'] * dt
    new_vy = particle['velocity'][1] + 0.5 * particle['pressure_force'][1] / particle['density'] * dt

    # Tentative new position calculation
    new_x = particle['position'][0] + new_vx * dt
    new_y = particle['position'][1] + new_vy * dt

    # Boundary conditions
    damping = 0.99  # Velocity damping factor
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

    if point_in_rotated_rectangle(new_x, new_y, 0, 0, rect_width, rect_height, rect_angle):
        # Calculate normal vector of the rectangle side
        normal_x, normal_y = rotate_point(0, 1, rect_angle)
        
        # Project velocity onto normal
        dot_product = new_vx * normal_x + new_vy * normal_y
        
        # Reflect velocity
        new_vx -= 2 * dot_product * normal_x
        new_vy -= 2 * dot_product * normal_y
        
        # Apply damping
        new_vx *= damping
        new_vy *= damping
        
        # Move particle outside of rectangle
        while point_in_rotated_rectangle(new_x, new_y, 0, 0, rect_width, rect_height, rect_angle):
            new_x += normal_x * 0.001
            new_y += normal_y * 0.001

    # Update velocity again (leapfrog)
    particle['velocity'][0] = new_vx + 0.5 * particle['pressure_force'][0] / particle['density'] * dt
    particle['velocity'][1] = new_vy + 0.5 * particle['pressure_force'][1] / particle['density'] * dt
    
    # Update position with potentially modified velocity
    particle['position'][0] = new_x
    particle['position'][1] = new_y

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
