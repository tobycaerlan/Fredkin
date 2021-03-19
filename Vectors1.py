import numpy as np
# from qutip import *
from Operators import Utest, split, power_set, D

Phi = [3 / 5, -4 / 5]
StatePhi = np.outer(np.conj(Phi), Phi)
Statespace = {0, 1}

State0 = [StatePhi,
          StatePhi]

Idem = 0.5 * np.eye(2)


def Repertoire(P, M):
    # print(P)
    vec = []
    for i in range(2):
        if i in M:
            vec.append(State0[i])
        else:
            vec.append(Idem)
    Subject = np.kron(vec[0], vec[1])
    Subject2 = Utest(Subject)
    if P == set():
        return np.trace(Subject2)
    elif P == {1}:
        step1 = Subject2.reshape([2, 2, 2, 2])
        step2 = np.einsum('ijik->jk', step1)
        return step2
    elif P == {0}:
        step1 = Subject2.reshape([2, 2, 2, 2])
        step2 = np.einsum('jiki->jk', step1)
        return step2
    elif P == {0,1}:
        return Subject2



M = {0, 1, 2}
P = {1, 2}


def smallphi(P, M):
    Smallphi = 100
    minpurv = {1, 2, 3}
    minmech = {0, 1, 2}
    rho = Repertoire(P, M)
    for i in power_set(M):
        for j in power_set(P):

            left = Repertoire(j, i)
            right = Repertoire(P - j, M - i)
            sigma = np.kron(left, right)
            if (Smallphi > D(rho, sigma)) and ([i, j] != [set(), set()]) and ([M - i, P - j] != [set(), set()]):
                Smallphi = D(rho, sigma)
                minpurv = j
                minmech = i
    return round(Smallphi, 5), minpurv, minmech, P


def corecause(M):
    placeholder = (0, set(), set(), set())
    for j in power_set({0,1}):

        if j != set():
            trial = smallphi(j, M)
            if trial[0] > placeholder[0]:
                placeholder = trial
    return placeholder


for i in power_set({0, 1}):
    if i != set():
        print([corecause(i), i])
# Repertoire({1,3},{1})
print(smallphi({0},{1}))