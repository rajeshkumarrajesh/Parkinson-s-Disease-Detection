import random
import time
from math import exp
import numpy as np
from numpy import mean
from numpy.matlib import repmat


#  Dwarf Mongoose Optimization (DMO)
def DMO(Positions, fobj, VRmin, VRmax, Max_iter):
    global P, Position, Cost, phi
    N, dim = Positions.shape[0], Positions.shape[1]
    lb = VRmin[0, :]
    ub = VRmax[0, :]

    Convergence_curve = np.zeros((Max_iter, 1))
    VarSize = [N]
    nBabysitter = 3
    nAlphaGroup = dim - nBabysitter
    nScout = nAlphaGroup
    L = round(0.6 * dim * nBabysitter)
    peep = 2

    BEP = np.zeros((dim, 1))
    BEF = float('inf')
    BestSol = float('inf')
    tau = float('inf')
    Iter = 1
    sm = float('inf')

    pop = repmat(nAlphaGroup, 1, 1)

    for i in range(nAlphaGroup):
        Position = random.uniform(lb, ub)
        Cost = fobj(Position[i, :])
        if Cost <= BestSol.Cost:
            BestSol = pop[i]

    C = np.zeros((nAlphaGroup, 1))
    CF = (1 - Iter / Max_iter) ^ (2 * Iter / Max_iter)
    BestCost = np.zeros((Max_iter, 1))

    it = 0
    ct = time.time()
    for it in range(Max_iter):
        F = np.zeros((nAlphaGroup, 1))
        MeanCost = mean(Cost)
        for i in range(nAlphaGroup):
            F[i] = exp(-Cost / MeanCost)
            P = F / sum(F)

    for m in range(nAlphaGroup):
        i = P
        K = [1, i + 1, nAlphaGroup]
        k = K
        phi = (peep / 2) * random.uniform(-1, +1)

        newpop_Position = pop[i].Position + phi * (pop[i].Position - pop[k].Position)
        newpop_Cost = fobj(Position[i, :])

        if newpop_Cost <= Cost:
            pop[i] = Position
        else:
            C[i] = C[i] + 1

    for i in range(nScout):
        K = [1, i - 1, i + 1, (nAlphaGroup)]
        k = K

        phi = (peep / 2) * random.uniform(-1, +1)
        newpop_Position = pop[i].Position + phi * (pop[i].Position - pop[k])

        newpop_Cost = fobj(newpop_Position)

        sm = (newpop_Cost - pop[i].Cost) / max(newpop_Cost, pop[i].Cost)

        if newpop_Cost <= pop[i].Cost:
            pop[i] = Cost
        else:
            C[i] = C[i] + 1

    for i in range(nBabysitter):
        if C[i] >= L:
            Position = random.uniform(lb, ub)
            Cost = fobj(Position)
            C[i] = 0

    for i in range(nAlphaGroup):
        if Cost <= BestSol.Cost:
            BestSol = pop[i]

    newtau = mean(sm)
    for i in range(nScout):
        M = (pop[i].Position * sm) / pop[i].Position
        if newtau > tau:
            newpop_Position = pop[i].Position - CF * phi * np.random.rand() * (pop[i].Position - M)
        else:
            newpop_Position = pop[i].Position + CF * phi * np.random.rand() * (pop[i].Position - M)
        tau = newtau

    for i in range(nAlphaGroup):
        if Cost <= BestSol.Cost:
            BestSol = pop[i]

        BestCost[it] = BestSol.Cost
        BEF = BestSol.Cost
        BEP = BestSol.Position

        Convergence_curve[it] = BEF
        it = it + 1
    BEF = Convergence_curve[Max_iter - 1][0]
    ct = time.time() - ct

    return BEF, Convergence_curve, BEP, ct
