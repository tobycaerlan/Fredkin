import numpy as np
# from qutip import *
from Operators import Ue, Uc, split, power_set, D, PartTrace, PartTrace2

Phi = [((np.e) ** 3j) * 3 / 5,((np.e) ** 3j) *  -4 / 5j]
Psi = [ ((np.e) ** 3j) * 12 / 13,((np.e) ** 3j) *  5 / 13j]
StatePhi = np.outer(np.conj(Phi), Phi)
StatePsi = np.outer(np.conj(Psi), Psi)

Statespace = {0, 1, 2}

State0 = [[[1, 0], [0, 0]],
          StatePhi,
          StatePsi]

State1 = [[[0, 0], [0, 1]],
          StatePsi,
          StatePhi]
Idem = 0.5 * np.eye(2)
M = {0, 1, 2}
P = {1, 2}

lambda1 = {1, 2}
lambda2 = Statespace - lambda1


def Repertoire(P, M, state, function, Global=False):
    if state == 'o':
        CurrentState = State1
    else:
        CurrentState = State0
    # print(P)
    vec = []
    for i in range(3):
        if i in M:
            vec.append(CurrentState[i])
        else:
            vec.append(Idem)
    Subject = np.kron(np.kron(vec[0], vec[1]), vec[2])

    if function == 'c':
        Subject2 = Uc(Subject)
    else:
        Subject2 = Ue(Subject)

    if Global:
        return PartTrace2(Subject2, P)
    else:
        return PartTrace(Subject2, P)


def Repertoire2(P, M, state, function, lambda1, Global=False):
    lambda2 = Statespace - lambda1
    if state == 'o':
        CurrentState = State1
    else:
        CurrentState = State0
    # print(P)
    vec = []
    veclambda = []
    for i in range(3):
        if (i in M) and (i in lambda1):
            veclambda.append(CurrentState[i])
        else:
            veclambda.append(Idem)
    for i in range(3):
        if i in M:
            vec.append(CurrentState[i])
        else:
            vec.append(Idem)
    Subject = np.kron(np.kron(vec[0], vec[1]), vec[2])
    Subjectlambda = np.kron(np.kron(veclambda[0], veclambda[1]), veclambda[2])

    if function == 'c':
        U1 = PartTrace(Uc(Subjectlambda), lambda1)
        U2 = PartTrace(Uc(Subject), lambda2)
    else:
        U1 = PartTrace(Ue(Subjectlambda), lambda1)
        U2 = PartTrace(Ue(Subject), lambda2)
    Subject2 = np.kron(U1, U2)
    if Global:
        return PartTrace2(Subject2, P)
    else:
        return PartTrace(Subject2, P)


def smallphi2(P, M, state, function, lambda1):
    Smallphi = 1
    minpurv = {0, 1, 2}
    minmech = {0, 1, 2}
    rho = Repertoire2(P, M, state, function, lambda1)
    for i in power_set(M):
        for j in power_set(P):

            left = Repertoire2(j, i, state, function, lambda1)
            right = Repertoire2(P - j, M - i, state, function, lambda1)
            sigma = np.kron(left, right)
            if (Smallphi > D(rho, sigma)) and ([i, j] != [set(), set()]) and ([M - i, P - j] != [set(), set()]):
                Smallphi = D(rho, sigma)
                minpurv = j
                minmech = i
    return round(Smallphi, 15), minpurv, minmech, P


def smallphi(P, M, state, function):
    Smallphi = 1
    minpurv = {0, 1, 2}
    minmech = {0, 1, 2}
    rho = Repertoire(P, M, state, function)
    for i in power_set(M):
        for j in power_set(P):

            left = Repertoire(j, i, state, function)
            right = Repertoire(P - j, M - i, state, function)
            sigma = np.kron(left, right)
            if (Smallphi > D(rho, sigma)) and ([i, j] != [set(), set()]) and ([M - i, P - j] != [set(), set()]):
                Smallphi = D(rho, sigma)
                minpurv = j
                minmech = i
    return round(Smallphi, 15), minpurv, minmech, P


def corecause(M, state='o', function='c'):
    placeholder = (0, 'no', 'concepts', 'here')
    for j in power_set({0, 1, 2}):

        if j != set():
            trial = smallphi(j, M, state, function)
            if trial[0] > placeholder[0]:
                placeholder = trial
    return placeholder


def corecause2(M, lambda1, state='o', function='c'):
    placeholder = (0, 'no', 'concepts', 'here')
    for j in power_set({0, 1, 2}):

        if j != set():
            trial = smallphi2(j, M, state, function, lambda1)
            if trial[0] > placeholder[0]:
                placeholder = trial
    return placeholder


# for state in ['one', 'zero']:
#     for function in ['cause', 'effect']:
#         for i in power_set({0, 1, 2}):
#             if i != set():
#                 print([corecause(i, state[0], function[0]), i, state, function])


# Repertoire({1,3},{1})
def ConceptStructure(lambda1, state):
    concept1 = []
    concept2 = []

    for i in power_set({0, 1, 2}):
        if i != set():
            Core = []
            Core2 = []
            for function in ['cause', 'effect']:
                Core.append(corecause(i, state[0], function[0]))
                Core2.append(corecause2(i, lambda1, state[0], function[0]))
            concept1.append([min(Core[0][0], Core[1][0]), Core[0][3], Core[1][3], i, state[0]])
            concept2.append([min(Core2[0][0], Core2[1][0]), Core2[0][3], Core2[1][3], i, state[0]])
            # these take the form phi, core cause, core effect, mechanism, state
    Grep1 = 0
    for j in range(len(concept1)):
        a = concept1[j][0] * Repertoire(concept1[j][1], concept1[j][3],
                                        concept1[j][4], 'c', True)
        b = concept2[j][0] * Repertoire2(concept2[j][1], concept2[j][3],
                                         concept2[j][4], 'c', lambda1, True)
        c = concept1[j][0] * Repertoire(concept1[j][2], concept1[j][3],
                                        concept1[j][4], 'e', True)
        d = concept2[j][0] * Repertoire2(concept2[j][2], concept2[j][3],
                                         concept2[j][4], 'e', lambda1, True)
        Grep1 += 0.5 * (D(a, b) + D(c, d))
        e=1
    return Grep1


# for i in power_set({0,1,2}):
#     print([ConceptStructure(i, 'o'), i])

concept1 = []
concept2 = []
for state in ['zero']:
    for function in ['cause', 'effect']:
        for i in power_set({0, 1, 2}):
            if i != set():
                Core = corecause(i, state[0], function[0])
                Core2 = corecause(i, 'o', function[0])
                concept1.append([Core[0], Core[3], i, state[0], function[0]])
                concept2.append([Core2[0], Core2[3], i, 'o', function[0]])
                # these take the form phi, core cause, mechanism, state, function
Grep1 = 0
for j in range(len(concept1)):
    a = concept1[j][0] * Repertoire(concept1[j][1], concept1[j][2],
                                    concept1[j][3], concept1[j][4], True)
    b = concept2[j][0] * Repertoire(concept2[j][1], concept2[j][2],
                                    concept2[j][3], concept2[j][4], True)
    Grep1 += 0.5 * D(a,b)
    e=1
print(Grep1)