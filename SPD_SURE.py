"""
SURE estimates of SPD under log-euclidean framework
"""
import sys
from pymanopt.manifolds import Product, Euclidean, PositiveDefinite
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent, ParticleSwarm
import autograd.numpy as np
from autograd.numpy.linalg import inv, det, norm
from autograd.scipy.linalg import logm, expm, sqrtm 
from autograd import grad, jacobian, hessian
from numpy.random import uniform, normal, multivariate_normal
from scipy.optimize import minimize
from scipy.stats import wishart

def vec(X):
    # X.shape = (n, d, d)
    if X.ndim == 2:
        X = np.array([X])
    n, d = X.shape[0:2]
    SPD = SymmetricPositiveDefinite(d)
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
    SPD = SymmetricPositiveDefinite(d, n)
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
    return norm(logm(np.matmul(np.matmul(Xinvsqrt, Y), Xinvsqrt)))

def FM_logE(X):
    n, d = X.shape[0:2]
    s = np.zeros((d, d))
    SPD = SymmetricPositiveDefinite(d)
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
    SPD = SymmetricPositiveDefinite(d)
    for i in range(1, n):
        S = SPD.exp(S, (1/(i+1))*SPD.log(S, X[i]))
    return S

def FM_GL(X):
    n, d = X.shape[0:2]
    man = SymmetricPositiveDefinite(d)
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

def loss(X, theta):
    p = X.shape[0]
    logX = vec(X)
    logtheta = vec(theta)
    l = norm(logX - logtheta)**2
    return l/p

def loss_GL(X, theta):
    n = X.shape[0]
    l = 0
    for i in range(n):
        l = l + dist_GL(X[i], theta[i])**2
    
    return l/n

"""
2-stage model
"""

def JS(X, A):
    n, d = X.shape[0:2]
    
    q = int(d*(d+1)/2)
    logX = vec(X)
    
    a = (q-2)*A/(np.sum(norm(logX, axis = 1)**2)) 
    logtheta = max(0, 1-a) * logX
    theta = vec_inv(logtheta, d)
    return theta

def SURE_or(X, A, theta):
    # compute the oracle "estimates"
    n, d = X.shape[0:2]
    assert (n == A.shape[0]), "The lengths of X and A should be the same."
    
    p = int(d*(d+1)/2)
    logX = vec(X)
    logtheta = vec(theta)
    
    """
    special case 1: Lambda = tau*I
    """
    def cost(x):
        tau = x[0]
        mu = x[1:]
        r = 0
        for i in range(n):
            r = r + norm(tau/(tau+A[i]) * logX[i] + A[i]/(tau+A[i]) * mu - logtheta[i])**2

        return r/n

    x0 = np.concatenate(([1], np.zeros(p)))
    bnds = tuple([(0, None)] + [(None, None)]*p)
    res = minimize(cost, x0, method="SLSQP", bounds=bnds)
    tau = res.x[0]
    mu = res.x[1:]
    logtheta = np.zeros(logX.shape)
    for i in range(n):
        logtheta[i] = tau/(tau+A[i]) * logX[i] + A[i]/(tau+A[i]) * mu
    theta = vec_inv(logtheta, d)
    return tau, vec_inv(mu, d), theta

def SURE_const(X, A):
    p, N = X.shape[0:2]
    assert (p == A.shape[0]), "The lengths of X and A should be the same."
    
    q = int(N*(N+1)/2)
    logX = vec(X)
    
    """
    special case 1: Lambda = lam*I
    """
    def cost(x):
        lam = x[0]
        mu = x[1:]
        r = 0
        for i in range(p):
            r = r + (A[i]*q*(lam**2-A[i]**2))/(lam+A[i])**2 + (A[i]/(lam+A[i]))**2*norm(logX[i] - mu)**2

        return r/p

    x0 = np.concatenate(([1], np.zeros(q)))
    bnds = tuple([(0, None)] + [(None, None)]*q)
    res = minimize(cost, x0, method="SLSQP", bounds=bnds)
    lam = res.x[0]
    mu = res.x[1:]
    logtheta = np.zeros(logX.shape)
    for i in range(p):
        logtheta[i] = lam/(lam+A[i]) * logX[i] + A[i]/(lam+A[i]) * mu
    theta = vec_inv(logtheta, N)
    return lam, vec_inv(mu, N), theta

def SURE_diag(X, A):
    n, d = X.shape[0:2]
    assert (n == A.shape[0]), "The lengths of X and A should be the same."
    
    p = int(d*(d+1)/2)
    logX = vec(X) 
    
    """
    special case 2: Lambda = diagonal matrix
    """
    def cost(x):
        di = x[0:p]
        mu = x[p:]
        r = 0
        for i in range(n):
            tmp = np.diag(1/(di + A[i]))
            tmp2 = np.matmul(tmp, logX[i] - mu)
            r = r + A[i]*p - 2*A[i]**2*np.trace(tmp) + A[i]**2*norm(tmp2)**2

        return r/n

    x0 = np.concatenate((np.ones(p), np.zeros(p)))
    bnds = tuple([(0, None)]*p + [(None, None)]*p)
    res = minimize(cost, x0, method="SLSQP", bounds=bnds)
    di = res.x[0:p]
    mu = res.x[p:]
    logtheta = np.zeros(logX.shape)
    for i in range(n):
        logtheta[i] = np.matmul(np.diag(di/(di+A[i])), logX[i]) + np.matmul(np.diag(A[i]/(di+A[i])), mu)
    theta = vec_inv(logtheta, d)
    return di, vec_inv(mu, d), theta

def SURE(X, A):
    n, d = X.shape[0:2]
    assert (n == A.shape[0]), "The lengths of X and A should be the same."
    
    p = int(d*(d+1)/2)
    logX = vec(X) 

    """
    General case: Lambda is SPD
    """
    manifold = Product([PositiveDefinite(p), Euclidean(p)])    
    I = np.eye(p)
    def cost(theta):
        Lam = theta[0]
        mu = theta[1]
        r = 0
        for i in range(n):
            tmp = inv(Lam + A[i]*I)
            tmp2 = np.matmul(tmp, logX[i] - mu)
            r = r + A[i]*p - 2*A[i]**2*np.trace(tmp) + A[i]**2*norm(tmp2)**2
        return r/n
    
    problem = Problem(manifold=manifold, cost=cost, verbosity=1)
    solver = SteepestDescent()
    # solver = ParticleSwarm()
    Xopt = solver.solve(problem)
    Lambda = Xopt[0]
    mu = Xopt[1]
    logtheta = np.zeros(logX.shape) 
    for i in range(n):
        tmp = inv(Lambda + A[i] * I)
        logtheta[i] = np.matmul(np.matmul(Lambda, tmp), logX[i]) + A[i] * np.matmul(tmp, mu)
    theta = vec_inv(logtheta, d)
    return Lambda, vec_inv(mu, d), theta

def SURE_full(X, S, n, verbose=False):
    # estimate both the means and the covariance matrices for the Log-Normal
    # distribution
    # X: (p, N, N) array; for each i, X[i] is in SPD(N)
    # S: (p, q, q); q = N(N+1)/2; for each i, S[i] is in SPD(q)
    # n: Si ~ W(n-1, Sigma_i)
    p, N = X.shape[0:2]
    assert (p == S.shape[0]), "The lengths of X and S should be the same."
    
    q = int(N*(N+1)/2)
    logX = vec(X)
    S_eigval = np.linalg.eigh(S)[0]
    trS = np.sum(S_eigval, axis=1)
    tr_S2 = np.sum(S_eigval**2, axis = 1)
    trS_2 = trS**2
    logX_norm2 = norm(logX)**2
    logX_sum = np.sum(logX, axis = 0)
    trS_sum = np.sum(trS)
    tr_S2_sum = np.sum(tr_S2)
    trS_2_sum = np.sum(trS_2)
    
    # hyperparameters: (\lambda > 0, \mu, \nu > q-1, \Psi > 0)
    def cost(x):
        lam = x[0]
        mu = x[1:q+1]
        nu = x[q+1]
        Psi = vec_inv(x[q+2:], q)
        trPsi2 = np.trace(np.matmul(Psi,Psi))
        
        S1 = (lam/(lam + n))**2*(logX_norm - 2 * np.sum(logX_sum*mu) + p*norm(mu)**2) + (n-lam**2/n)/((n-1)*(lam+n)**2) * trS_sum
        S2 = ( (n-3+(nu-q-1)**2)/((n+1)*(n-2))* tr_S2_sum \
                + ((n-1)**2-(nu-q-1)**2)/((n-2)*(n-1)*(n+1)) * trS_2_sum \
                - 2*(nu-q-1)/(n-1)*np.trace(np.matmul(Psi, S[i])) \
                + trPsi2)/(nu+n-q-2)**2
        r = S1 + S2
        #r = 0
        #for i in range(p):
        #    S1 = (lam/(lam + n))**2*norm(logX[i] - mu)**2 + \
        #    (n-lam**2/n)/((n-1)*(lam+n)**2) * trS[i]
        #    S2 = ( (n-3+(nu-q-1)**2)/((n+1)*(n-2))* tr_S2[i] \
        #            + ((n-1)**2-(nu-q-1)**2)/((n-2)*(n-1)*(n+1)) * trS_2[i] \
        #            - 2*(nu-q-1)/(n-1)*np.trace(np.matmul(Psi, S[i])) \
        #            + trPsi2)/(nu+n-q-2)**2

        #    r = r + S1 + S2

        return r/p
    
    jac = jacobian(cost)

    mu_0 = np.mean(logX, axis = 0)
    lam_0 = 1/(np.trace(np.cov(logX.T))/np.mean(np.sum(S_eigval, axis = 1)/(n-1))-1/n)
    nu_0 = (q+1)/((n-q-2)/(q*(n-1))*np.trace(np.matmul(np.mean(S, axis = 0), \
                                           np.mean(np.linalg.inv(S), axis = 0))) - 1)+ q + 1
    nu_0 = np.maximum(nu_0, q + 2)
    Psi_0 = vec(np.mean(S/(n-1)*(nu_0-q-1), axis = 0))[0]
    x0 = np.concatenate(([lam_0], mu_0, [nu_0], Psi_0))
    print(jac(x0))
    bnds = tuple([(0, None)] + [(None, None)]*q + [(q-1, None)] + [(None,
        None)]*int(q*(q+1)/2))
    res = minimize(cost, x0, method="L-BFGS-B", bounds=bnds)
    lam = res.x[0]
    mu = res.x[1:q+1]
    nu = res.x[q+1]
    Psi = vec_inv(res.x[q+2:], q)
    logtheta = np.zeros(logX.shape)
    Sig_SURE = np.zeros(S.shape)
    for i in range(p):
        logtheta[i] = n/(lam+n) * logX[i] + lam/(lam+n) * mu
        Sig_SURE[i] = (Psi + S[i])/(nu+n-q-2) 
    theta = vec_inv(logtheta, N)
    if verbose:
        print(res['message'])
        print("number of iteration: ", res['nit'])
    return lam, vec_inv(mu, N), nu, Psi, theta, Sig_SURE


"""
experiments
"""



if __name__ == "__main__":
    N = 3
    p = 50
    q = int(N*(N+1)/2)
    SPD = SymmetricPositiveDefinite(N)
    I = np.eye(N)
    Iq = np.eye(q)

    tau = 0.5
    A = uniform(0.1, 1, p)
    # Lambda = 0.5 * SPD.rand()
    # Lambda = 0.1*np.ones((p, p)) + 0.4 * np.eye(p)
    Lambda = tau * np.eye(q)
    theta = SPD_normal(n, I, Lambda)
    X = np.array([SPD_normal(1, theta[i], A[i]*Iq) for i in range(p)])

    tau_hat, mu_const, theta_SURE_const = SURE_const(X, A)
    di_hat, mu_diag, theta_SURE_diag = SURE_diag(X, A)
    Lam_hat, mu, theta_SURE = SURE(X, A)
    theta_MLE = X

    l_SURE_const = loss(theta_SURE_const, theta)
    l_SURE_diag = loss(theta_SURE_diag, theta)
    l_SURE = loss(theta_SURE, theta)
    l_MLE = loss(theta_MLE, theta)

    #########################################################

    



