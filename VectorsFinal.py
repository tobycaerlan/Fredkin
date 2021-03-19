import numpy as np
# from qutip import *
from Operators import Ue, Uc, split, power_set, D, PartTrace, PartTrace2, BasisShift

listostuff = []
valuees = []
# the wavefunctions then density matrices of the two qubits in the second register



Statespace = {0, 1, 2}
# the two states that are superposed in the normal fredkin experiment


Idem = 0.5 * np.eye(2)
M = {0, 1, 2}
P = {1, 2}

lambda1 = {1, 2}
lambda2 = Statespace - lambda1


def Repertoire2(P, M, state, function, lambda1=set(), Global=False):
    #     # this function calculates the repertoire of a specific P and M.
    #     # P - purview
    #     # M - mechanism
    #     # state - selects whether we are performing this function on State0 or State1
    #     # function - selects whether we are calculating cause or effect repertoire
    #     # Global - if true this calculates the global repertoire, which is only used for evaluating the C-space
    lambda2 = Statespace - lambda1
    CurrentState = state
    # vec is set to be the current state with appropriate noise added as according to the mechanism
    # veclambda is the unipartitioned  state - ie it is vec but with the parts not in lambda removed
    # - this is used when calculating U_p
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
    # Subject is the noised current state - 8x8 density matrix
    # Similarly Subjectlambda is the noise from M and lambda1 (lambda1 is the unipartition)
    Subject = np.kron(np.kron(vec[0], vec[1]), vec[2])
    Subjectlambda = np.kron(np.kron(veclambda[0], veclambda[1]), veclambda[2])
    # Performs the unitary evolution of the system- forward or back
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
    # calculates \phi for a specific system and returns: [\phi, minimum partition purview, minimum partition mechanism,
    # purview]
    # I have come across the following issue- the literature does not specify the order in which the two subfactorised
    # matrices (here as left and right) should be produced to form the sigma comparable to rho
    # for example if M={1,2,3} -> {1,3} U {2}, P = {1,2} -> {1} U {2} the optimal tensor product is far from obvious
    # (to me)
    Smallphi = 100
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
            # this i think fixes the issue mentioned above
            if P == {0, 1, 2} and j == {0, 2}:
                sigma2 = BasisShift @ sigma @ BasisShift
                if (Smallphi > D(rho, sigma2)):
                    Smallphi = D(rho, sigma2)
                    minpurv = j
                    minmech = i

    return round(Smallphi, 10), minpurv, minmech, P


def corecause2(M, state, function='c', lambda1=set()):
    # this loops through all the possible purviews of a mechanism to find its core cause
    # returns [\phi, minimum partition purview, minimum partition mechanism, purview]
    placeholder = (0, 'no', 'concepts', 'here')
    for j in power_set({0, 1, 2}):

        if j != set():
            trial = smallphi2(j, M, state, function, lambda1)
            if trial[0] > placeholder[0]:
                placeholder = trial
    return placeholder


def ConceptStructure(lambda1, state):
    # to avoid creating classes i never actually create the Concept structure for a particular set up
    # the first loop calculates the triple of [\phi, core cause, core effect] and attaches necessary
    # information for the calculation of \Phi
    concept1 = []
    concept2 = []

    for i in power_set({0, 1, 2}):
        if i != set():
            Core = []
            Core2 = []
            for function in ['cause', 'effect']:
                Core.append(corecause2(i, state, function[0]))
                Core2.append(corecause2(i, state, function[0], lambda1))
            concept1.append([min(Core[0][0], Core[1][0]), Core[0][3], Core[1][3], i, state])
            concept2.append([min(Core2[0][0], Core2[1][0]), Core2[0][3], Core2[1][3], i, state])
            # these take the form \phi, core cause, core effect, mechanism, state
    # this returns Grep1- the value of D(C(U),C(U_p)) for the p being lambda given as the input
    # if speed is required in future we could halve computation time by calculating C(U) just once but for now its fast
    # enough not to warrant bothering
    Grep1 = 0
    for j in range(len(concept1)):
        a = concept1[j][0] * Repertoire2(concept1[j][1], concept1[j][3],
                                         concept1[j][4], 'c', set(), True)
        b = concept2[j][0] * Repertoire2(concept2[j][1], concept2[j][3],
                                         concept2[j][4], 'c', lambda1, True)
        c = concept1[j][0] * Repertoire2(concept1[j][2], concept1[j][3],
                                         concept1[j][4], 'e', set(), True)
        d = concept2[j][0] * Repertoire2(concept2[j][2], concept2[j][3],
                                         concept2[j][4], 'e', lambda1, True)
        Grep1 += 0.5 * (D(a, b) + D(c, d))
    return Grep1


def Phi(state):
    CapitalPhi = [10, set()]
    for i in power_set({0, 1, 2}):
        if i != set() and i != {0, 1, 2}:
            Distance = [ConceptStructure(i, state), f"The MIP for state is {i}"]
            if Distance[0] < CapitalPhi[0]:
                CapitalPhi = Distance
    return CapitalPhi


def Comparison(state1, state2):
    concept1 = []
    concept2 = []

    for i in power_set({0, 1, 2}):
        if i != set():
            Core = []
            Core2 = []
            for function in ['cause', 'effect']:
                Core.append(corecause2(i, state1, function[0]))
                Core2.append(corecause2(i, state2, function[0]))
            concept1.append([min(Core[0][0], Core[1][0]), Core[0][3], Core[1][3], i, state1])
            concept2.append([min(Core2[0][0], Core2[1][0]), Core2[0][3], Core2[1][3], i, state2])
            # these take the form phi, core cause, core effect, mechanism, state
    Grep1 = 0
    for j in range(len(concept1)):
        a = concept1[j][0] * Repertoire2(concept1[j][1], concept1[j][3],
                                         concept1[j][4], 'c', set(), True)
        b = concept2[j][0] * Repertoire2(concept2[j][1], concept2[j][3],
                                         concept2[j][4], 'c', set(), True)
        c = concept1[j][0] * Repertoire2(concept1[j][2], concept1[j][3],
                                         concept1[j][4], 'e', set(), True)
        d = concept2[j][0] * Repertoire2(concept2[j][2], concept2[j][3],
                                         concept2[j][4], 'e', set(), True)
        Grep1 += 0.5 * (D(a, b) + D(c, d))
        e = 1
    return Grep1

#this is a value for the comparison between the Conceptual structures of state0 and state1
#this is different form of metric from the one used by chalmers et al
#print(Comparison('one', 'z'))

# these are the individual \Phi values of states zero and one.
#print(Phi('zero'))
#print(Phi('one'))

def ConceptStructure3(lambda1, state):
    # to avoid creating classes i never actually create the Concept structure for a particular set up
    # the first loop calculates the triple of [\phi, core cause, core effect] and attaches necessary
    # information for the calculation of \Phi
    concept1 = []
    concept2 = []

    for i in power_set({0, 1, 2}):
        if i != set():
            Core = []
            Core2 = []
            for function in ['cause', 'effect']:
                Core.append(corecause2(i, state, function[0]))
                Core2.append(corecause2(i, state, function[0], lambda1))
            concept1.append([min(Core[0][0], Core[1][0]), Core[0][3], Core[1][3], i])
            concept2.append([min(Core2[0][0], Core2[1][0]), Core2[0][3], Core2[1][3], i])
    return concept1


for alpha in np.linspace(0.64,0.67, num=2):
    Phi = [1,0]
    Psi = [np.cos(alpha),np.sin(alpha)]
    StatePhi = np.outer(np.conj(Phi), Phi)
    StatePsi = np.outer(np.conj(Psi), Psi)
    State0 = [[[1, 0], [0, 0]],
              StatePhi,
              StatePsi]

    State1 = [[[0, 0], [0, 1]],
              StatePsi,
              StatePhi]
    listostuff.append(ConceptStructure3(set(),State0))
    valuees.append(ConceptStructure3(set(),State1))

print(listostuff)
print(valuees)

# import matplotlib.pyplot as plt
# fig2 = plt.figure()
# ax2 = plt.axes()
# ax2.plot(valuees,listostuff)
# plt.show()