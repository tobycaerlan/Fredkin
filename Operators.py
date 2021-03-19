from numpy import *
from numpy import linalg as LA
import numpy as np

Hadamard1 = 2 ** -0.5 * array([[1, 1], [1, -1]])
Hadamard = kron(Hadamard1, eye(4))
Swap1 = array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
op0 = array([[1, 0], [0, 0]])
op1 = array([[0, 0], [0, 1]])
Swap = kron(op0, eye(4)) + kron(op1, Swap1)
BasisShift = kron(eye(2), Swap1)


# print(Hadamard @ Swap - Swap @ Hadamard)

def Utest(X):
    return Swap1 @ X @ Swap1


def Ue(X):
    return Swap @ Hadamard @ X @ Hadamard @ Swap


def Uc(X):
    return Hadamard @ Swap @ X @ Swap @ Hadamard


def split(array, nrows, ncols):
    """Split a matrix into sub-matrices."""

    r, h = array.shape
    return (array.reshape(h // nrows, nrows, -1, ncols)
            .swapaxes(1, 2)
            .reshape(-1, nrows, ncols))


def power_set(A):
    """A is an iterable (list, tuple, set, str, etc)
    returns a set which is the power set of A."""
    length = len(A)
    l = [a for a in A]
    ps = []

    for i in range(2 ** length):
        selector = f'{i:0{length}b}'
        subset = {l[j] for j, bit in enumerate(selector) if bit == '1'}
        ps.append(set(subset))

    return ps


def D(rho, sigma):
    return 0.5 * LA.norm(rho - sigma, 'nuc')


def PartTrace(Subject2, P):
    # takes a purview over which we find the new density matrix of the system
    # Subject2 - the density matrix of the system
    # P - the purview over which we find this matrix - ie the only bit of statespace we care about
    if P == set():
        return np.trace(Subject2)
    elif P == {0}:
        step1 = Subject2.reshape([4, 2, 4, 2])
        step2 = np.einsum('jiki->jk', step1)
        step3 = step2.reshape([2, 2, 2, 2])
        step4 = np.einsum('jiki->jk', step3)
        return step4
    elif P == {1}:
        step1 = Subject2.reshape([4, 2, 4, 2])
        step2 = np.einsum('jiki->jk', step1)
        step3 = step2.reshape([2, 2, 2, 2])
        step4 = np.einsum('ijik->jk', step3)
        return step4
    elif P == {2}:
        step1 = Subject2.reshape([2, 4, 2, 4])
        step2 = np.einsum('ijik->jk', step1)
        step3 = step2.reshape([2, 2, 2, 2])
        step4 = np.einsum('ijik->jk', step3)
        return step4
    elif P == {0, 1}:
        step1 = Subject2.reshape([4, 2, 4, 2])
        step2 = np.einsum('jiki->jk', step1)
        return step2
    elif P == {1, 2}:
        step1 = Subject2.reshape([2, 4, 2, 4])
        step2 = np.einsum('ijik->jk', step1)
        return step2
    elif P == {0, 1, 2}:
        return Subject2
    else:
        # this is the only non-obvious one - this swaps the basis- swapping the third qubit and the second,
        # this is because taking a partial trace over the middle of three tensor products is really hard -
        # there is a method for this but is much less easily generalisable
        step1 = BasisShift @ Subject2 @ BasisShift
        step2 = step1.reshape([4, 2, 4, 2])
        step3 = np.einsum('jiki->jk', step2)
        return step3


def PartTrace2(Subject2, P):
    # essentially the same as PartTrace but this time calculates the global repertoire, this essentially
    # involves adding noise to where information was taken from
    if P == set():
        return np.trace(Subject2) * np.eye(8) * 0.125
    elif P == {0}:
        step1 = Subject2.reshape([4, 2, 4, 2])
        step2 = np.einsum('jiki->jk', step1)
        step3 = step2.reshape([2, 2, 2, 2])
        step4 = np.einsum('jiki->jk', step3)
        step5 = np.kron(step4, 0.25 * eye(4))
        return step5
    elif P == {1}:
        step1 = Subject2.reshape([4, 2, 4, 2])
        step2 = np.einsum('jiki->jk', step1)
        step3 = step2.reshape([2, 2, 2, 2])
        step4 = np.einsum('ijik->jk', step3)
        step5 = 0.25 * np.kron(np.kron(eye(2), step4), eye(2))
        return step5
    elif P == {2}:
        step1 = Subject2.reshape([2, 4, 2, 4])
        step2 = np.einsum('ijik->jk', step1)
        step3 = step2.reshape([2, 2, 2, 2])
        step4 = np.einsum('ijik->jk', step3)
        step5 = 0.25 * np.kron(eye(4), step4)
        return step5
    elif P == {0, 1}:
        step1 = Subject2.reshape([4, 2, 4, 2])
        step2 = np.einsum('jiki->jk', step1)
        step3 = np.kron(step2, 0.5 * eye(2))
        return step3
    elif P == {1, 2}:
        step1 = Subject2.reshape([2, 4, 2, 4])
        step2 = np.einsum('ijik->jk', step1)
        step3 = np.kron(0.5 * eye(2), step2)
        return step3
    elif P == {0, 1, 2}:
        return Subject2
    else:
        step1 = BasisShift @ Subject2 @ BasisShift
        step2 = step1.reshape([4, 2, 4, 2])
        step3 = np.einsum('jiki->jk', step2)
        step4 = np.kron(step3, 0.5 * eye(2))
        step5 = BasisShift @ step4 @ BasisShift
        return step5
