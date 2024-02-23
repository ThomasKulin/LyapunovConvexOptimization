import numpy as np


def magic(n):
  n = int(n)
  if n < 3:
    raise ValueError("Size must be at least 3")
  if n % 2 == 1:
    p = np.arange(1, n+1)
    return n*np.mod(p[:, None] + p - (n+3)//2, n) + np.mod(p[:, None] + 2*p-2, n) + 1
  elif n % 4 == 0:
    J = np.mod(np.arange(1, n+1), 4) // 2
    K = J[:, None] == J
    M = np.arange(1, n*n+1, n)[:, None] + np.arange(n)
    M[K] = n*n + 1 - M[K]
  else:
    p = n//2
    M = magic(p)
    M = np.block([[M, M+2*p*p], [M+3*p*p, M+p*p]])
    i = np.arange(p)
    k = (n-2)//4
    j = np.concatenate((np.arange(k), np.arange(n-k+1, n)))
    M[np.ix_(np.concatenate((i, i+p)), j)] = M[np.ix_(np.concatenate((i+p, i)), j)]
    M[np.ix_([k, k+p], [0, k])] = M[np.ix_([k+p, k], [0, k])]
  return M 

def balanceQuadraticForm(S, P):
    # copied from the old drake, with only syntax swap
    #  Quadratic Form "Balancing"
    #
    #    T = balqf(S,P)
    #
    #  Input:
    #    S -- n-by-n symmetric positive definite.
    #    P -- n-by-n symmetric, full rank.
    #
    #  Finds a T such that:
    #    T'*S*T = D
    #    T'*P*T = D^(-1)

    if np.linalg.norm(S - S.T,1) > 1e-8:
        raise Error('S must be symmetric')
    if np.linalg.norm(P - P.T,1) > 1e-8:
        raise Error('P must be symmetric')
    if np.linalg.cond(P) > 1e10:
        raise Error('P must be full rank')

    # Tests if S positive def. for us.
    V = np.linalg.inv(np.linalg.cholesky(S).T)
    [U, l, N] = np.linalg.svd((V.T.dot(P)).dot(V))
    T = (V.dot(U)).dot(np.diag(np.power(l, -.25, dtype=float)))
    D = np.diag(np.power(l, -.5, dtype=float))
    return T, D

S=magic(5)@magic(5).T
P=np.eye(5)
print(S)
T,D = balanceQuadraticForm(P,S)
print (T)
print(D)
