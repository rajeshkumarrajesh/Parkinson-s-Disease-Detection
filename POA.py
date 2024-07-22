import time
import numpy as np


# Peafowl Optimization Algorithm (EPOA)
def POA(population, fobj, VRmin, VRmax, max_iter):
    population_size, num_dimensions = population.shape[0], population.shape[1]
    lb = VRmin[0, :]
    ub = VRmax[0, :]

    # Initialize velocities randomly
    velocities = np.random.uniform(low=-0.1, high=0.1, size=(population_size, num_dimensions))

    # Initialize personal best positions and fitness
    personal_best_positions = population.copy()
    personal_best_fitness = np.apply_along_axis(fobj, 1, population)

    # Initialize global best position and fitness
    global_best_index = np.argmin(personal_best_fitness)
    global_best_position = personal_best_positions[global_best_index].copy()
    global_best_fitness = personal_best_fitness[global_best_index]

    Convergence_curve = np.zeros((max_iter, 1))

    t = 0
    ct = time.time()
    # EPOA optimization loop
    for t in range(max_iter):
        for i in range(population_size):
            # Update velocity
            velocities[i] = velocities[i] + 0.1 * np.random.rand() * (
                    personal_best_positions[i] - population[i]) + 0.1 * np.random.rand() * (
                                    global_best_position - population[i])

            # Update position
            population[i] = population[i] + velocities[i]

            # Clip positions to the search space bounds
            population[i] = np.clip(population[i], 0, 1)

            # Evaluate fitness of the new position
            fitness = fobj(population[i])

            # Update personal best if needed
            if fitness < personal_best_fitness[i]:
                personal_best_positions[i] = population[i].copy()
                personal_best_fitness[i] = fitness

                # Update global best if needed
                if fitness < global_best_fitness:
                    global_best_position = personal_best_positions[i].copy()
                    global_best_fitness = fitness

        Convergence_curve[t] = global_best_fitness
        t = t + 1
    global_best_fitness = Convergence_curve[max_iter - 1][0]
    ct = time.time() - ct
    return global_best_fitness, Convergence_curve, global_best_position, ct
