# synthetic exps for estimating the means and the covariances for the Log-Normal distribution
#import numpy as np
#from numpy.random import uniform, normal, multivariate_normal
#from pymanopt.manifolds import Product, Euclidean, SymmetricPositiveDefinite
import pandas as pd
from timeit import default_timer as timer
from scipy.stats import invwishart
import multiprocessing
from joblib import Parallel, delayed
import pickle
from plotnine import *
from plotnine.data import mpg
import scipy.linalg as sla
import sys
sys.path.insert(1, '../')
from SPD_SURE_pytorch import *

def exp_lognormal(n, p, lam, mu, nu, Psi, ran_seed=0, verbose = False):
    N = mu.shape[0]
    q = Psi.shape[0]
    I = np.eye(N)
    Iq = np.eye(q)

    res = {'risk_M': pd.Series([0, 0, 0, 0], 
                index = ['FM_logE', 'FM_GL', 'SURE', 'SURE_full']),
           'risk_Sig': pd.Series([0, 0, 0, 0], 
                index = ['FM_logE', 'FM_GL', 'SURE', 'SURE_full']),
            't': pd.Series([0, 0, 0, 0], 
                index = ['FM_logE', 'FM_GL', 'SURE', 'SURE_full'])} 
    res = pd.DataFrame(res)

    np.random.seed(ran_seed)
    Sigma = invwishart.rvs(nu, Psi, size = p)
    M = np.zeros((p, N, N))
    M = np.array([SPD_normal(1, mu, Sigma[i]/lam) for i in range(p)])
    X = np.array([SPD_normal(n, M[i], Sigma[i]) for i in range(p)])

    ## log-Euclideam mean
    start = timer()
    M_logE = np.array([FM_logE(X[i]) for i in range(p)])
    end = timer()
    t_logE = end - start
    
    S_logE = (n-1)*np.array([cov_logE(X[i]) for i in range(p)])
    S_eigval = np.linalg.eigh(S_logE)[0]
    
    ## GL_rec mean
    start = timer()
    M_GL = np.array([FM_GL_rec(X[i]) for i in range(p)])
    end = timer()
    t_GL = end - start

    ## SURE (mean only)
    start = timer()
    lam_hat, mu_hat, M_SURE = SURE_const(M_logE, np.mean(S_eigval, axis = 1)/(n*(n-1)))
    end = timer()
    t_SURE = end - start
    
    ## SURE (mean and covariance)
    start = timer()
    lam_hat, mu_hat, nu_hat, Psi_hat, M_SURE_full, Sig_SURE_full = SURE_full(M_logE, S_logE, n, verbose = verbose)
    end = timer()
    t_SURE_full = end - start    

    ## risk
    res.loc['FM_logE', 'risk_M'] = loss(M, M_logE)
    res.loc['FM_GL', 'risk_M'] = loss(M, M_GL)
    res.loc['SURE', 'risk_M'] = loss(M, M_SURE)
    res.loc['SURE_full', 'risk_M'] = loss(M, M_SURE_full)
    res.loc['FM_logE', 'risk_Sig'] = np.sum((S_logE/(n-1)-Sigma)**2)/p
    res.loc['FM_GL', 'risk_Sig'] = np.sum((S_logE/(n-1)-Sigma)**2)/p
    res.loc['SURE', 'risk_Sig'] = np.sum((S_logE/(n-1)-Sigma)**2)/p
    res.loc['SURE_full', 'risk_Sig'] = np.sum((Sig_SURE_full-Sigma)**2)/p
    res.loc['FM_logE', 't'] = t_logE
    res.loc['FM_GL', 't'] = t_GL
    res.loc['SURE', 't'] = t_SURE
    res.loc['SURE_full', 't'] = t_SURE_full

    return res.values

def exp(n, p_vec, lam_vec, nu_vec, mu, Psi, out_file, m, ran_seed = 12345):
    num_cores = -1
    risk_M = pd.DataFrame(np.zeros((len(lam_vec)*len(nu_vec)*len(p_vec), 8)))
    risk_M.columns = ['p', 'n', 'lambda', 'nu', 'FM_LogE', 'FM_GL', 'SURE', 'SURE_full']
    risk_M_sd = risk_M.copy()
    risk_Sig = risk_M.copy()
    risk_Sig_sd = risk_M.copy()
    time = risk_M.copy()
    time_sd = risk_M.copy()
    r_ind = 0
    for p in p_vec:
        for nu in nu_vec:
            for lam in lam_vec:
                print('p =', p, ', nu =', nu, ', lam =', lam)
                results = Parallel(n_jobs=num_cores)(delayed(exp_lognormal)(n, p, lam, mu, nu, Psi, ran_seed + i) \
                                                     for i in range(m))
                res = np.mean(np.array(results), axis = 0)
                res_sd = np.std(np.array(results), axis = 0)/np.sqrt(m)
                res = pd.DataFrame(res, index = ['FM_logE', 
                    'FM_GL', 'SURE', 'SURE_full'])
                res_sd = pd.DataFrame(res_sd, index = ['FM_logE', 
                    'FM_GL', 'SURE', 'SURE_full'])
                res.columns = ['risk_M', 'risk_Sig', 't']
                res_sd.columns = ['risk_M', 'risk_Sig', 't']
                risk_M.values[r_ind] = np.array([p, n, lam, nu, 
                    res.loc['FM_logE', 'risk_M'], 
                    res.loc['FM_GL', 'risk_M'], 
                    res.loc['SURE', 'risk_M'],
                    res.loc['SURE_full', 'risk_M']])                
                risk_Sig.values[r_ind] = np.array([p, n, lam, nu,
                    res.loc['FM_logE', 'risk_Sig'], 
                    res.loc['FM_GL', 'risk_Sig'], 
                    res.loc['SURE', 'risk_Sig'],
                    res.loc['SURE_full', 'risk_Sig']])                
                time.values[r_ind] = np.array([p, n, lam, nu,
                    res.loc['FM_logE', 't'], 
                    res.loc['FM_GL', 't'], 
                    res.loc['SURE', 't'],
                    res.loc['SURE_full', 't']])                
                risk_M_sd.values[r_ind] = np.array([p, n, lam, nu,
                    res_sd.loc['FM_logE', 'risk_M'], 
                    res_sd.loc['FM_GL', 'risk_M'], 
                    res_sd.loc['SURE', 'risk_M'],
                    res_sd.loc['SURE_full', 'risk_M']])                
                risk_Sig_sd.values[r_ind] = np.array([p, n, lam, nu,
                    res_sd.loc['FM_logE', 'risk_Sig'], 
                    res_sd.loc['FM_GL', 'risk_Sig'], 
                    res_sd.loc['SURE', 'risk_Sig'],
                    res_sd.loc['SURE_full', 'risk_Sig']])                
                time_sd.values[r_ind] = np.array([p, n, lam, nu,
                    res_sd.loc['FM_logE', 't'], 
                    res_sd.loc['FM_GL', 't'], 
                    res_sd.loc['SURE', 't'],
                    res_sd.loc['SURE_full', 't']])                
                r_ind += 1
                print('Success!')
                
    pickle.dump({'mu':mu, 'Psi':Psi, 'risk_M':risk_M, 'risk_Sig':risk_Sig, 'time':time,
        'risk_M_sd':risk_M_sd, 'risk_Sig_sd':risk_Sig_sd, 
        'time_sd':time_sd}, open(out_file, 'wb'))
    
def OR_loss(n, lam, nu, ran_seed = 0, verbose = False):
    p = 10000
    N = 3
    q = int(N*(N+1)/2)
    I = np.eye(N)
    Iq = np.eye(q)
    
    Psi = np.eye(q)
    mu = np.eye(N)

    np.random.seed(ran_seed)
    Sigma = invwishart.rvs(nu, Psi, size = p)
    M = np.zeros((p, N, N))
    M = np.array([SPD_normal(1, mu, Sigma[i]/lam) for i in range(p)])
    X = np.array([SPD_normal(n, M[i], Sigma[i]) for i in range(p)])
    M_logE = np.array([FM_logE(X[i]) for i in range(p)])
    S_logE = (n-1)*np.array([cov_logE(X[i]) for i in range(p)])
    S_eigval = np.linalg.eigh(S_logE)[0]
    logX = vec(M_logE)
    logM = vec(M)
    # hyperparameters: (\lambda > 0, \mu, \nu > q-1, \Psi > 0)
    def cost(x):
        lam = x[0]
        mu = x[1:q+1]
        nu = x[q+1]
        Psi = vec_inv(x[q+2:], q)
        logM_est = np.zeros(logX.shape)
        Sig_est = np.zeros(S_logE.shape)
        for i in range(p):
            logM_est[i] = n/(lam+n) * logX[i] + lam/(lam+n) * mu
            Sig_est[i] = (Psi + S_logE[i])/(nu+n-q-2) 
        r = (norm(logM_est - logM)**2 + np.sum((Sig_est - Sigma)**2))/p

        return r
    
    mu_0 = np.mean(logX, axis = 0)
    lam_0 = 1/(np.trace(np.cov(logX.T))/np.mean(np.sum(S_eigval, axis = 1)/(n-1))-1/n)
    nu_0 = (q+1)/((n-q-2)/(q*(n-1))*np.trace(np.matmul(np.mean(S_logE, axis = 0), \
                                           np.mean(np.linalg.inv(S_logE), axis = 0))) - 1)+ q + 1
    nu_0 = np.maximum(nu_0, q + 2)
    Psi_0 = vec(np.mean(S_logE/(n-1)*(nu_0-q-1), axis = 0))[0]
    x0 = np.concatenate(([lam_0], mu_0, [nu_0], Psi_0))
    bnds = tuple([(0, None)] + [(None, None)]*q + [(q-1, None)] + [(None,
        None)]*int(q*(q+1)/2))
    res = minimize(cost, x0, method="L-BFGS-B", bounds=bnds)
    if verbose:
        print(res['message'])
        print("number of iteration: ", res['nit'])
    lam = res.x[0]
    mu = res.x[1:q+1]
    nu = res.x[q+1]
    Psi = vec_inv(res.x[q+2:], q)
    logtheta = np.zeros(logX.shape)
    Sig_SURE = np.zeros(S_logE.shape)
    for i in range(p):
        logtheta[i] = n/(lam+n) * logX[i] + lam/(lam+n) * mu
        Sig_SURE[i] = (Psi + S_logE[i])/(nu+n-q-2) 
    r_M = norm(logtheta - logM)**2/p
    r_Sig = np.sum((Sig_SURE - Sigma)**2)/p
    return r_M, r_Sig