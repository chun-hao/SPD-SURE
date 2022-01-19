from pymanopt.manifolds import Product, Euclidean, PositiveDefinite
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent
from numpy.random import uniform, normal, multivariate_normal
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import logm


def vec(X):
    # X.shape = (n, d, d)
    if X.ndim == 2:
        X = np.array([X])
    n, d = X.shape[0:2]
    SPD = PositiveDefinite(d)
    tri_ind = np.triu_indices(d, 1)
    I = np.array([np.eye(d) for i in range(n)])
    tmp = SPD.log(I, X)
    logX = np.array([np.concatenate((np.diag(t), np.sqrt(2) * t[tri_ind])) for t in tmp])
    return logX

def vec_inv(x, d):
    # x.shape = (n, p)
    if x.ndim == 1:
        x = np.array([x])
    n = x.shape[0]
    SPD = PositiveDefinite(d, n)
    tri_ind = np.triu_indices(d, 1)
    logX = np.zeros((n, d, d))
    for i in range(n):
        logX[i] = np.diag(x[i, 0:d]/2)
 
        logX[i][tri_ind] = x[i, d:]/np.sqrt(2)

    logX = (logX + logX.transpose((0, 2, 1)))
    if n == 1:
        logX = logX[0]
        I = np.eye(d)
    else:
        I = np.array([np.eye(d) for i in range(n)])
    return SPD.exp(I, logX)

def SPD_normal(n, M, S):
    m = vec(M)[0]
    d = M.shape[0]
    x = multivariate_normal(m, S, n)
    X = vec_inv(x, d)
    return X

def dist_LE(X, Y):
    return norm(logm(X) - logm(Y))

def dist_GL(X, Y):
    d = X.shape[0]
    Xinv = inv(X)
    Xinvsqrt = sqrtm(Xinv)
    return np.linalg.norm(logm(np.matmul(np.matmul(Xinvsqrt, Y), Xinvsqrt)))

def FM_logE(X):
    n, d = X.shape[0:2]
    s = np.zeros((d, d))
    SPD = PositiveDefinite(d)
    I = np.array([np.eye(d) for i in range(n)])
    logX = SPD.log(I, X)
    return SPD.exp(I[0], np.mean(logX, axis = 0))

def var_logE(X):
    n, d = X.shape[0:2]
    q = int(d*(d+1)/2)
    logX = vec(X)
    v = np.mean(np.var(logX, axis = 0))
    return v

def cov_logE(X):
    n, d = X.shape[0:2]
    q = int(d*(d+1)/2)
    logX = vec(X)
    v = np.cov(logX.T)
    return v

def FM_GL_rec(X):
    n, d = X.shape[0:2]
    S = X[0]
    SPD = PositiveDefinite(d)
    for i in range(1, n):
        S = SPD.exp(S, (1/(i+1))*SPD.log(S, X[i]))
    return S

def FM_GL(X):
    n, d = X.shape[0:2]
    man = PositiveDefinite(d)
    def objective(y):  # weighted Frechet variance
        acc = 0
        for i in range(n):
            acc += dist_GL(y, X[i]) ** 2
        return acc 
    def gradient(y):
        g = man.zerovec(y)
        for i in range(n):
            g -= man.log(y, X[i])
        return g
    solver = SteepestDescent(maxiter=15)
    problem = Problem(man, cost=objective, grad = gradient, verbosity=0)
    return solver.solve(problem)