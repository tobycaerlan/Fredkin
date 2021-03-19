## this is the original reertoire function in case anythin goes wrong

def Repertoire(P, M, state='o', function = 'c'):

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

    if P == set():
        return np.trace(Subject2)
    elif P == {1}:
        step1 = Subject2.reshape([4, 2, 4, 2])
        step2 = np.einsum('jiki->jk', step1)
        step3 = step2.reshape([2, 2, 2, 2])
        step4 = np.einsum('jiki->jk', step3)
        return step4
    elif P == {2}:
        step1 = Subject2.reshape([4, 2, 4, 2])
        step2 = np.einsum('jiki->jk', step1)
        step3 = step2.reshape([2, 2, 2, 2])
        step4 = np.einsum('ijik->jk', step3)
        return step4
    elif P == {3}:
        step1 = Subject2.reshape([2, 4, 2, 4])
        step2 = np.einsum('ijik->jk', step1)
        step3 = step2.reshape([2, 2, 2, 2])
        step4 = np.einsum('ijik->jk', step3)
        return step4
    elif P == {1, 2}:
        step1 = Subject2.reshape([4, 2, 4, 2])
        step2 = np.einsum('jiki->jk', step1)
        return step2
    elif P == {2, 3}:
        step1 = Subject2.reshape([2, 4, 2, 4])
        step2 = np.einsum('ijik->jk', step1)
        return step2
    elif P == {1, 2, 3}:
        return Subject2
    else:
        step1 = split(Subject2, 2, 2)
        step2 = np.concatenate((step1[0] + step1[5], step1[2] + step1[7]), axis=1)
        step3 = np.concatenate((step1[8] + step1[13], step1[10] + step1[15]), axis=1)
        step4 = np.concatenate((step2, step3), axis=0)
        return step4

[(0.25, {2}, set(), {2}), {2}, 'one', 'cause']
[(0.25, {2}, set(), {2}), {1}, 'one', 'cause']
[(0.5, {0, 1}, {1, 2}, {0, 1, 2}), {1, 2}, 'one', 'cause']
[(0.5, {0}, set(), {0}), {0}, 'one', 'cause']
[(0.25, {1}, {2}, {1}), {0, 2}, 'one', 'cause']
[(0.25, {2}, {1}, {2}), {0, 1}, 'one', 'cause']
[(0.25, {1}, {2}, {1, 2}), {0, 1, 2}, 'one', 'cause']
[(0.25, {2}, set(), {2}), {2}, 'one', 'effect']
[(0.25, {2}, set(), {2}), {1}, 'one', 'effect']
[(0.5, {0}, set(), {0, 1, 2}), {1, 2}, 'one', 'effect']
[(0.375, {0}, {0}, {0, 1, 2}), {0}, 'one', 'effect']
[(0.5, set(), {2}, {0, 1, 2}), {0, 2}, 'one', 'effect']
[(0.5, set(), {1}, {0, 1, 2}), {0, 1}, 'one', 'effect']
[(0.5, set(), {2}, {0, 1, 2}), {0, 1, 2}, 'one', 'effect']
[(0.25, {2}, set(), {2}), {2}, 'zero', 'cause']
[(0.25, {2}, set(), {2}), {1}, 'zero', 'cause']
[(0.5, {1, 2}, {1, 2}, {0, 1, 2}), {1, 2}, 'zero', 'cause']
[(0.5, {0}, set(), {0}), {0}, 'zero', 'cause']
[(0.25, {2}, {2}, {2}), {0, 2}, 'zero', 'cause']
[(0.25, {1}, {1}, {1}), {0, 1}, 'zero', 'cause']
[(0.25, {1}, {1}, {1, 2}), {0, 1, 2}, 'zero', 'cause']
[(0.25, {2}, set(), {2}), {2}, 'zero', 'effect']
[(0.25, {2}, set(), {2}), {1}, 'zero', 'effect']
[(0.5, {0}, set(), {0, 1, 2}), {1, 2}, 'zero', 'effect']
[(0.375, {0}, {0}, {0, 1, 2}), {0}, 'zero', 'effect']
[(0.5, set(), {2}, {0, 1, 2}), {0, 2}, 'zero', 'effect']
[(0.5, set(), {1}, {0, 1, 2}), {0, 1}, 'zero', 'effect']
[(0.5, set(), {2}, {0, 1, 2}), {0, 1, 2}, 'zero', 'effect']

#this is state one
# in the form \Phi , lambda1
[3.1543990452064904e-16, set()]
[1.1926383224488546, {2}]
[1.094026953422233, {1}]
[1.3075802463343162, {1, 2}]
[0.7022170278254924, {0}]
[0.9622346520859897, {0, 2}]
[0.8766029268799829, {0, 1}]
[4.028640527888495e-16, {0, 1, 2}]
#this is state zero
[3.1543990452064904e-16, set()]
[1.156526953422233, {2}]
[1.1301383224488546, {1}]
[1.260027464611864, {1, 2}]
[0.7022170278254922, {0}]
[1.0245049813385712, {0, 2}]
[0.7516029268799829, {0, 1}]
[4.028640527888495e-16, {0, 1, 2}]

# This is the standard difference between 0 and 1
0.125

# this is when you take the proper definition of \phi for zero
[2.5423831801460944e-16, set()]
[0.8954129014154955, {2}]
[0.861635335288613, {1}]
[1.035596938587316, {1, 2}]
[0.7334670278254922, {0}]
[0.8389152632714445, {0, 2}]
[0.6922990587727085, {0, 1}]
[3.305169738479937e-16, {0, 1, 2}]

# one
[2.5423831801460944e-16, set()]
[0.9244549574619056, {2}]
[0.8320701945757536, {1}]
[1.083149720309768, {1, 2}]
[0.7334670278254922, {0}]
[0.7824397285112539, {0, 2}]
[0.7049154361022465, {0, 1}]
[3.305169738479937e-16, {0, 1, 2}]

#idek
[7.465545911082038e-16, set()]
[0.9244549574619052, {2}]
[0.8320701945757529, {1}]
[1.0831497203097675, {1, 2}]
[0.7334670278254916, {0}]
[0.7824397285112534, {0, 2}]
[0.7049154361022458, {0, 1}]
[6.971281169057181e-16, {0, 1, 2}]