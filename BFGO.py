import time
import numpy as np


# BFGO algorithm(Bamboo Forest Growth Optimization Algorithm )
def BFGO(population, fobj, VRmin, VRmax, Max_iter):
    N, dim = population.shape[0], population.shape[1]
    lb = VRmin[0, :]
    ub = VRmax[0, :]

    alpha = 0.5
    beta = 0.5

    best_solution = None
    best_fitness = float('inf')

    Convergence_curve = np.zeros((Max_iter, 1))

    t = 0
    ct = time.time()
    for t in range(Max_iter):
        for i in range(N):
            # Update each individual's position based on the bamboo forest growth behavior
            delta = np.random.uniform(-1, 1, dim)
            population[i] += alpha * delta * (population[i] - population.mean(axis=0))
            fitness = fobj(population[i, :])

            # Update the best solution found so far
            if fitness < best_fitness:
                best_solution = population[i]
                best_fitness = fitness

        # Update alpha and beta for the next iteration
        alpha *= beta

        Convergence_curve[t] = best_solution
        t = t + 1
    best_solution = Convergence_curve[Max_iter - 1][0]
    ct = time.time() - ct

    return best_solution, Convergence_curve, best_fitness, ct
