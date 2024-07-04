import random
import numpy as np

particle_dtype = np.dtype([
    ('position', np.float32, (2,)),
    ('force', np.float32, (2,)),
    ('velocity', np.float32, (2,)),
    ('acceleration', np.float32, (2,)),
    ('density', np.float32),
    ('density_gradient', np.float32, (2,)),
    ('pressure', np.float32),
    ('pressure_force', np.float32, (2,)),
    ('mass', np.float32)
])


def generate_particles(n: int, SCREEN_RESOLUTION: tuple, spacing: float, stochasticity: float):
    particle_array = np.zeros(n, dtype=particle_dtype)
    dim = int(np.sqrt(particle_array.shape[0]))

    # Calculate the total width and height of the particle grid
    total_width = spacing * (dim - 1)
    total_height = spacing * (dim - 1)

    # Calculate offsets to center the grid
    offset_x = (SCREEN_RESOLUTION[0] - total_width) / 2
    offset_y = (SCREEN_RESOLUTION[1] - total_height) / 2

    for i in range(dim):
        for j in range(dim):
            # Calculate base position
            x = offset_x + spacing * i
            y = offset_y + spacing * j

            # Add stochasticity
            x += random.uniform(-stochasticity, stochasticity)
            y += random.uniform(-stochasticity, stochasticity)

            # Normalize to [-1, 1] range
            x_norm = (x / SCREEN_RESOLUTION[0]) * 2 - 1
            y_norm = (y / SCREEN_RESOLUTION[1]) * 2 - 1

            particle_array[i*dim+j]['position'] = [x_norm, y_norm]
            particle_array[i*dim+j]['force'] = [0, 0]
            particle_array[i*dim+j]['velocity'] = [0, 0]
            particle_array[i*dim+j]['acceleration'] = [0, 0]
            particle_array[i*dim+j]['density'] = 0
            particle_array[i*dim+j]['density_gradient'] = [0, 0]
            particle_array[i*dim+j]['pressure'] = 0
            particle_array[i*dim+j]['pressure_force'] = [0, 0]
            particle_array[i*dim+j]['mass'] = 1

    return particle_array
