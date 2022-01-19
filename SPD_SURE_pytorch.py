"""
SURE estimates of SPD under log-euclidean framework with PyTorch
"""
import sys
from spd_utils import *
import torch
from torch.autograd import Variable

def loss(X, theta):
    p = X.shape[0]
    logX = vec(X)
    logtheta = vec(theta)
    l = np.linalg.norm(logX - logtheta)**2
    return l/p

def loss_GL(X, theta):
    n = X.shape[0]
    l = 0
    for i in range(n):
        l = l + dist_GL(X[i], theta[i])**2
    
    return l/n

def SURE_const(X, A, verbose = True):
    p, N = X.shape[0:2]
    assert (p == A.shape[0]), "The lengths of X and A should be the same."
    
    q = int(N*(N+1)/2)
    logX = vec(X)
    logX_ = Variable(torch.from_numpy(logX)).type(torch.FloatTensor)
    A_ = Variable(torch.from_numpy(A)).type(torch.FloatTensor)
    logX_norm_ = Variable(torch.norm(logX_, dim = 1)**2).type(torch.FloatTensor)
    
    """
    special case 1: Lambda = lam*I
    """
    def cost(x):
        lam = torch.exp(x[0]) # ensure that lam is positive
        mu = x[1:]
        r = (A_ * q * (lam**2 - A_**2))/(lam + A_)**2 + \
            (A_/(lam + A_))**2 * (logX_norm_ - 2 * torch.matmul(logX_, mu) + torch.norm(mu)**2)
        return torch.mean(r)

    x0 = Variable(torch.rand(1 + q), requires_grad=True)
    
    # optimize the cost function
    optimizer = torch.optim.Adam([x0], lr = 1)   
    tol = 1e-3
    
    for t in range(500):
        optimizer.zero_grad()
        current_loss = cost(x0)
        current_loss.backward()
        optimizer.step()
        if verbose:
            print(f"t = {t}, loss = {current_loss}, x0 = {x0.detach().numpy()}, grad = {optimizer.state_dict()['state'][0]['exp_avg']}")
        if max(abs(optimizer.state_dict()['state'][0]['exp_avg'])) < tol:
            break
    
    lam = np.exp(x0.data.numpy()[0])
    mu = x0.data.numpy()[1:]
    logtheta = np.zeros(logX.shape)
    for i in range(p):
        logtheta[i] = lam/(lam+A[i]) * logX[i] + A[i]/(lam+A[i]) * mu
    theta = vec_inv(logtheta, N)
    return lam, vec_inv(mu, N), theta

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
    logX_norm = np.linalg.norm(logX)**2
    logX_sum = torch.from_numpy(np.sum(logX, axis = 0))
    trS_sum = np.sum(trS)
    tr_S2_sum = np.sum(tr_S2)
    trS_2_sum = np.sum(trS_2)
    S_sum = torch.from_numpy(np.sum(S, axis = 0))
    
    tri_ind = np.triu_indices(q, 1)
    
    # hyperparameters: (\lambda > 0, \mu, \nu > q+1, \Psi > 0)
    def cost(x):
        lam = torch.exp(x[0])
        mu = x[1:q+1]
        nu = x[q+1] + q + 1
        
        tmp = x[q+2:]
        logPsi = torch.diag(tmp[0:q]/2)
        logPsi[tri_ind] = tmp[q:]/np.sqrt(2)
        logPsi = logPsi + torch.transpose(logPsi, 0, 1)
        Psi = torch.matrix_exp(logPsi)
        
        trPsi2 = torch.trace(torch.matmul(Psi,Psi))
        
        S1 = (lam/(lam + n))**2*(logX_norm - 2 * torch.sum(logX_sum*mu) + p*torch.norm(mu)**2) + (n-lam**2/n)/((n-1)*(lam+n)**2) * trS_sum
        S2 = ( (n-3+(nu-q-1)**2)/((n+1)*(n-2))* tr_S2_sum \
                + ((n-1)**2-(nu-q-1)**2)/((n-2)*(n-1)*(n+1)) * trS_2_sum \
                - 2*(nu-q-1)/(n-1)*torch.trace(torch.matmul(Psi, S_sum)) \
                + trPsi2)/(nu+n-q-2)**2
        r = S1 + S2
        return r/p
    
    mu_0 = np.mean(logX, axis = 0)
    #lam_0 = np.log(1/(np.trace(np.cov(logX.T))/np.mean(np.sum(S_eigval, axis = 1)/(n-1))-1/n))
    lam_0 = np.log(np.maximum(1/(np.trace(np.cov(logX.T))/np.mean(np.sum(S_eigval, axis = 1)/(n-1))-1/n), 1e-3))
    #nu_0 = (q+1)/((n-q-2)/(q*(n-1))*np.trace(np.matmul(np.mean(S, axis = 0), \
    #                                       np.mean(np.linalg.inv(S), axis = 0))) - 1)+ q - 1
    #nu_0 = np.maximum(nu_0, 0)
    nu_0 = 0
    Psi_0 = vec(np.mean(S/(n-1), axis = 0))[0]
    if any(np.isnan(Psi_0)):
        Psi_0 = np.zeros(int(q*(q+1)/2))
    x0 = Variable(torch.from_numpy(np.concatenate(([lam_0], mu_0, [nu_0], Psi_0))), requires_grad=True)
    
    #x0 = Variable(torch.zeros(1 + q + 1 + int(q*(q+1)/2), dtype = torch.float64), requires_grad=True)
    #print(x0.dtype)
    
    # optimize the cost function
    optimizer = torch.optim.Adam([x0])   
    tol = 1e-3
    
    for t in range(500):
        optimizer.zero_grad()
        current_loss = cost(x0)
        current_loss.backward()
        optimizer.step()
        if verbose:
            print(f"t = {t}, loss = {current_loss}, x0 = {x0.detach().numpy()}, grad = {optimizer.state_dict()['state'][0]['exp_avg']}")
        if max(abs(optimizer.state_dict()['state'][0]['exp_avg'])) < tol:
            break

    
    x = x0.data.numpy()
    x = np.nan_to_num(x)
    lam = np.exp(x[0])
    mu = x[1:q+1]
    nu = x[q+1] + q + 1
    #Psi = vec_inv(x[q+2:], q)
    Psi = np.mean(S, axis = 0)/(n-1)*(nu-q-1)
    logtheta = np.zeros(logX.shape)
    Sig_SURE = np.zeros(S.shape)
    for i in range(p):
        logtheta[i] = n/(lam+n) * logX[i] + lam/(lam+n) * mu
        Sig_SURE[i] = (Psi + S[i])/(nu+n-q-2) 
    theta = vec_inv(logtheta, N)
    return lam, vec_inv(mu, N), nu, Psi, theta, Sig_SURE



def SURE_const_log(X):
    # same as SURE_const(X), but take logX as input and log(M_SURE) as output, which is more memory efficient
    p, n, q = X.shape
    X_mean = np.mean(X, axis = 1)
    A = np.mean(np.var(X, axis = 1), axis = 1)
    
    logX_ = Variable(torch.from_numpy(X_mean)).type(torch.FloatTensor)
    A_ = Variable(torch.from_numpy(A)).type(torch.FloatTensor)
    logX_norm_ = Variable(torch.norm(logX_, dim = 1)**2).type(torch.FloatTensor)
    
    """
    special case 1: Lambda = lam*I
    """
    def cost(x):
        lam = torch.exp(x[0]) # ensure that lam is positive
        mu = x[1:]
        r = (A_ * q * (lam**2 - A_**2))/(lam + A_)**2 + \
            (A_/(lam + A_))**2 * (logX_norm_ - 2 * torch.matmul(logX_, mu) + torch.norm(mu)**2)
        return torch.mean(r)

    x0 = Variable(torch.rand(1 + q), requires_grad=True)
    
    # optimize the cost function
    optimizer = torch.optim.Adam([x0])   
    optimizer.zero_grad()
    l = cost(x0)
    l.backward()
    optimizer.step()

    
    lam = np.exp(x0.data.numpy()[0])
    log_mu = x0.data.numpy()[1:]
    logtheta = np.zeros((p, q))
    for i in range(p):
        logtheta[i] = lam/(lam+A[i]) * X_mean[i] + A[i]/(lam+A[i]) * log_mu
    return lam, log_mu, logtheta
    
    
def SURE_full_log(X, verbose=False):
    # same as SURE_full(X, S, n), but take log-transformed data as input
    p, n, q = X.shape
    
    logX = np.mean(X, axis = 1)
    S_eigval = np.array([np.linalg.eigh((n-1)*np.cov(X[i].T))[0] for i in range(p)])
    trS = np.array([np.sum(S_eigval, axis=1) for i in range(p)])
    tr_S2 = np.sum(S_eigval**2, axis = 1)
    trS_2 = trS**2
    logX_norm = np.linalg.norm(logX)**2
    logX_sum = torch.from_numpy(np.sum(logX, axis = 0))
    trS_sum = np.sum(trS)
    tr_S2_sum = np.sum(tr_S2)
    trS_2_sum = np.sum(trS_2)
    S_sum = np.zeros((q, q))
    for i in range(p):
        S_sum = S_sum + (n-1)*np.cov(X[i].T)
    S_sum = torch.from_numpy(S_sum)
    
    tri_ind = np.triu_indices(q, 1)
    
    # hyperparameters: (\lambda > 0, \mu, \nu > q+1, \Psi > 0)
    def cost(x):
        lam = torch.exp(x[0])
        mu = x[1:q+1]
        nu = x[q+1] + q + 1
        
        tmp = x[q+2:]
        logPsi = torch.diag(tmp[0:q]/2)
        logPsi[tri_ind] = tmp[q:]/np.sqrt(2)
        logPsi = logPsi + torch.transpose(logPsi, 0, 1)
        Psi = torch.matrix_exp(logPsi)
        
        trPsi2 = torch.trace(torch.matmul(Psi,Psi))
        
        S1 = (lam/(lam + n))**2*(logX_norm - 2 * torch.sum(logX_sum*mu) + p*torch.norm(mu)**2) + (n-lam**2/n)/((n-1)*(lam+n)**2) * trS_sum
        S2 = ( (n-3+(nu-q-1)**2)/((n+1)*(n-2))* tr_S2_sum \
                + ((n-1)**2-(nu-q-1)**2)/((n-2)*(n-1)*(n+1)) * trS_2_sum \
                - 2*(nu-q-1)/(n-1)*torch.trace(torch.matmul(Psi, S_sum)) \
                + trPsi2)/(nu+n-q-2)**2
        r = S1 + S2
        return r/p
    
    mu_0 = np.mean(logX, axis = 0)
    lam_0 = np.log(1/(np.trace(np.cov(logX.T))/np.mean(np.sum(S_eigval, axis = 1)/(n-1))-1/n))
    #nu_0 = (q+1)/((n-q-2)/(q*(n-1))*np.trace(np.matmul(np.mean(S, axis = 0), \
    #                                       np.mean(np.linalg.inv(S), axis = 0))) - 1)+ q - 1
    #nu_0 = np.maximum(nu_0, 0)
    nu_0 = 0
    #Psi_0 = vec(S_sum.numpy()/(p*(n-1))*(nu_0-q-1))[0]
    Psi_0 = vec(S_sum.numpy()/(p*(n-1)))[0]
    #Psi_0 = np.zeros(int(q*(q+1)/2))
    x0 = Variable(torch.from_numpy(np.concatenate(([lam_0], mu_0, [nu_0], Psi_0))), requires_grad=True)
    
    #x0 = Variable(torch.rand(1 + q + 1 + int(q*(q+1)/2)), requires_grad=True)
    #print(x0.dtype)
    
    # optimize the cost function
    optimizer = torch.optim.Adam([x0])   
    optimizer.zero_grad()
    l = cost(x0)
    l.backward()
    optimizer.step()

    
    x = x0.data.numpy()
    x = np.nan_to_num(x)
    lam = np.exp(x[0])
    log_mu = x[1:q+1]
    nu = x[q+1] + q + 1
    Psi = vec_inv(x[q+2:], q)
    logtheta = np.zeros(logX.shape)
    #Sig_SURE = np.zeros(S.shape)
    for i in range(p):
        logtheta[i] = n/(lam+n) * logX[i] + lam/(lam+n) * log_mu
        #Sig_SURE[i] = (Psi + S[i])/(nu+n-q-2) 
    return lam, log_mu, nu, Psi, logtheta