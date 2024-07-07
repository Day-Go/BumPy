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
    ('mass', np.float32),
    ('cell_index', np.int32)
])

def generate_particles(n: int, SCREEN_RESOLUTION: tuple, spacing: float, stochasticity: float):
    particle_array = np.zeros(n, dtype=particle_dtype)
    rows = int(np.sqrt(n))
    cols = (n + rows - 1) // rows  # Ceiling division to ensure all particles are placed

    total_width = spacing * (cols - 1)
    total_height = spacing * (rows - 1)
    offset_x = (SCREEN_RESOLUTION[0] - total_width) / 2
    offset_y = (SCREEN_RESOLUTION[1] - total_height) / 2

    for i in range(n):
        row = i // cols
        col = i % cols
        x = offset_x + spacing * col
        y = offset_y + spacing * row
        x += random.uniform(-stochasticity, stochasticity)
        y += random.uniform(-stochasticity, stochasticity)
        x_norm = (x / SCREEN_RESOLUTION[0]) * 2 - 1
        y_norm = (y / SCREEN_RESOLUTION[1]) * 2 - 1
        
        particle_array[i]['position'] = [x_norm, y_norm]
        particle_array[i]['force'] = [0, 0]
        particle_array[i]['velocity'] = [0, 0]
        particle_array[i]['acceleration'] = [0, 0]
        particle_array[i]['density'] = 0
        particle_array[i]['density_gradient'] = [0, 0]
        particle_array[i]['pressure'] = 0
        particle_array[i]['pressure_force'] = [0, 0]
        particle_array[i]['mass'] = 1

    return particle_array
