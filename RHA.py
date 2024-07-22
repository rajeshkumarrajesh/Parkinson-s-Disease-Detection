import time
import numpy as np


# Red‑tailed Hawk Algorithm (RHA)
def RHA(agents_position, fobj, VRmin, VRmax, Max_iter):
    num_agents, num_variables = agents_position.shape[0], agents_position.shape[1]
    lb = VRmin[0, :]
    ub = VRmax[0, :]

    agents_fitness = np.array([fobj(agent) for agent in agents_position])
    best_agent_index = np.argmin(agents_fitness)
    best_position = agents_position[best_agent_index]
    best_fitness = agents_fitness[best_agent_index]

    Convergence_curve = np.zeros((Max_iter, 1))

    t = 0
    ct = time.time()
    for iteration in range(Max_iter):
        for i in range(num_agents):
            for j in range(num_variables):
                rand = np.random.rand()
                if rand > 0.5:
                    agents_position[i, j] = best_position[j] + np.random.uniform() * (
                            best_position[j] - agents_position[i, j])
                else:
                    agents_position[i, j] = best_position[j] - np.random.uniform() * (
                            best_position[j] - agents_position[i, j])

                # Check boundary constraints
                agents_position[i, j] = max(min(agents_position[i, j], ub), lb)

            fitness = fobj(agents_position[i])
            if fitness < agents_fitness[i]:
                agents_fitness[i] = fitness
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_position = agents_position[i]

        Convergence_curve[t] = best_position
        t = t + 1
    best_position = Convergence_curve[Max_iter - 1][0]
    ct = time.time() - ct

    return best_position, Convergence_curve, best_fitness, ct
