from SPD_SURE_pytorch import *
import numpy as np
from numpy.random import uniform, normal, multivariate_normal
from pymanopt.manifolds import Product, Euclidean, SymmetricPositiveDefinite
import pandas as pd
from timeit import default_timer as timer
from scipy.stats import invwishart
import multiprocessing
from joblib import Parallel, delayed
import pickle
from plotnine import *
from plotnine.data import mpg
import scipy.linalg as sla
from datetime import datetime

def check_SPD(X):
    # X be a n x N x N array
    # check if X[i]'s are SPD
    n = X.shape[0]
    N = X.shape[1]
    I = np.eye(N)
    res = np.zeros(X.shape)
    for i in range(n):
        min_eigval = np.min(np.linalg.eigvalsh(X[i]))
        if min_eigval < 0:
            res[i] = X[i] + (abs(min_eigval) + 1e-3)*I
        else:
            res[i] = X[i]
            
    return res

def exp_rs_fMRI(n, N, mat, M, Sigma, ran_seed = 0, verbose = False):
    names = ['TD', 'ADHD_C', 'ADHD_I', 'H', 'P', 'CON', 'PSP']
    #N = 10 # number of regions/nodes 
    p = len(names)
    
    res = {'risk_M': pd.Series([0, 0, 0], 
                index = ['FM_logE', 'SURE', 'SURE_full']),
           'risk_Sig': pd.Series([0, 0, 0], 
                index = ['FM_logE', 'SURE', 'SURE_full'])} 
    res = pd.DataFrame(res)
    
    X = np.zeros((p, n, N, N))
    
    for i, name in enumerate(names):
        tmp = mat['con_mat_' + name]
        region = mat[name + '_region'][0:N]
        ind = np.random.choice(tmp.shape[0], n, replace = True)
        X[i] = check_SPD(tmp[ind][:, :, region][:, region])
        
        
    ## log-Euclideam mean
    M_logE = np.array([FM_logE(X[i]) for i in range(p)])
    #logX = np.array([vec(X[i]) for i in range(p)])

    
    S_logE = (n-1)*np.array([cov_logE(X[i]) for i in range(p)])
    S_eigval = np.linalg.eigh(S_logE)[0]

    ## SURE (mean only)
    lam_hat, mu_hat, M_SURE = SURE_const(M_logE, np.mean(S_eigval, axis = 1)/(n*(n-1)))
    
    ## SURE (mean and covariance)
    lam_hat, mu_hat, nu_hat, Psi_hat, M_SURE_full, Sig_SURE_full = SURE_full(M_logE, S_logE, n, verbose = verbose)

    ## risk
    res.loc['FM_logE', 'risk_M'] = loss(M, M_logE)
    res.loc['SURE', 'risk_M'] = loss(M, M_SURE)
    res.loc['SURE_full', 'risk_M'] = loss(M, M_SURE_full)
    res.loc['FM_logE', 'risk_Sig'] = np.sum((S_logE/(n-1)-Sigma)**2)/p
    res.loc['SURE', 'risk_Sig'] = np.sum((S_logE/(n-1)-Sigma)**2)/p
    res.loc['SURE_full', 'risk_Sig'] = np.sum((Sig_SURE_full-Sigma)**2)/p

    return res.values




if __name__ == '__main__':
    mat = np.load('connectivity_matrix.npz')    
    names = np.array(['TD', 'ADHD_C', 'ADHD_I', 'H', 'P', 'CON', 'PSP'])
    
    n = 5
    m = 1000 # repetition
    N_vec = np.array([3, 5, 7, 10])
    ran_seed = 12345
    
    #tmp = np.load('LE_mean_cov.npz')
    #M = tmp['M']
    #Sigma = tmp['Sigma']
    
    out_file = 'rs-fMRI_exp.p'
    
    risk_M = pd.DataFrame(np.zeros((len(N_vec), 4)))
    risk_M.columns = ['N', 'FM_LogE', 'SURE', 'SURE_full']
    risk_M_sd = risk_M.copy()
    risk_Sig = risk_M.copy()
    risk_Sig_sd = risk_M.copy()
    r_ind = 0
    
    for N in N_vec:
        #Compute the papulation mean/cov
        p = len(names)
        q = int(N*(N + 1)/2)


        M = np.zeros((p, N, N))
        Sigma = np.zeros((p, q, q))


        for i, name in enumerate(names):
            tmp = mat['con_mat_' + name]
            region = mat[name + '_region'][0:N]
            tmp1 = check_SPD(tmp[:, :, region][:, region])
            #print(tmp[:, :, region][:, region].shape)
            M[i] = FM_logE(tmp1)
            #print(M[i].shape)
            Sigma[i] = cov_logE(tmp1)

        Sigma = check_SPD(Sigma)
        
        
        
        ###############################################################
        print('N = ', N)
        results = np.zeros((m, 3, 2))
        for i in range(m):
            results[i] = exp_rs_fMRI(n, N, mat, M, Sigma, ran_seed + i)
            
        res = np.mean(np.array(results), axis = 0)
        res_sd = np.std(np.array(results), axis = 0)/np.sqrt(m)
        res = pd.DataFrame(res, index = ['FM_logE', 'SURE', 'SURE_full'])
        res_sd = pd.DataFrame(res_sd, index = ['FM_logE', 'SURE', 'SURE_full'])
        res.columns = ['risk_M', 'risk_Sig']
        res_sd.columns = ['risk_M', 'risk_Sig']
        risk_M.values[r_ind] = np.array([N,
            res.loc['FM_logE', 'risk_M'], 
            res.loc['SURE', 'risk_M'],
            res.loc['SURE_full', 'risk_M']])                
        risk_Sig.values[r_ind] = np.array([N,
            res.loc['FM_logE', 'risk_Sig'], 
            res.loc['SURE', 'risk_Sig'],
            res.loc['SURE_full', 'risk_Sig']])                             
        risk_M_sd.values[r_ind] = np.array([N,
            res_sd.loc['FM_logE', 'risk_M'], 
            res_sd.loc['SURE', 'risk_M'],
            res_sd.loc['SURE_full', 'risk_M']])                
        risk_Sig_sd.values[r_ind] = np.array([N,
            res_sd.loc['FM_logE', 'risk_Sig'], 
            res_sd.loc['SURE', 'risk_Sig'],
            res_sd.loc['SURE_full', 'risk_Sig']])                            
        r_ind += 1
        print('Success!')
    
    #results = np.zeros((m, 3, 2))
    #for i in range(m):
    #    results[i] = exp_rs_fMRI(n, N, M, Sigma, ran_seed + i)
    #print(exp_rs_fMRI(n, mat, M, Sigma, ran_seed))
    
    #num_cores = -1
    #results = Parallel(n_jobs=num_cores)(delayed(exp_rs_fMRI)(n, N_vec[0], M, Sigma, ran_seed + i) for i in range(m))
    #pickle.dump({'N':N_vec, 'risk_M':risk_M, 'risk_Sig':risk_Sig, 
    #    'risk_M_sd':risk_M_sd, 'risk_Sig_sd':risk_Sig_sd}, open(out_file, 'wb'))    
    
    #print('res: ', res)
    #print('res_sd: ', res_sd)

    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print("Done @ ", dt_string)
    
    print('risk_M: \n', risk_M)
    print('\n')
    print('risk_Sig: \n', risk_Sig)
    print('\n')
    print('risk_M_sd: \n', risk_M_sd)
    print('\n')
    print('risk_Sig_sd: \n', risk_Sig_sd)
    print('\n')
