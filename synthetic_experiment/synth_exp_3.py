from exp_setup import *

# synth exp 2 (Section 4.2.2)


def exp_lognormal(n, p, N, lam, log_mu, nu, Psi, ran_seed=0, verbose = False):
    q = int(N*(N+1)/2)

    res = {'risk_M': pd.Series([0, 0, 0], 
                index = ['FM_logE', 'SURE', 'SURE_full']),
           'risk_Sig': pd.Series([0, 0, 0], 
                index = ['FM_logE', 'SURE', 'SURE_full'])} 
    res = pd.DataFrame(res)
    np.random.seed(ran_seed)

    #log_mu = vec(mu)[0]
    #print("Generating data!")
    Sigma = invwishart.rvs(nu, Psi, size = 1)
    a = np.random.rand(p)
    
    Sigma_sqrt = sla.sqrtm(Sigma)
    
    tmp = np.random.randn(p, n+1, q)

    #log_M = np.array([multivariate_normal(log_mu, Sigma[i]/lam, 1)[0] for i in range(p)])
    log_M = np.array([log_mu + np.matmul(Sigma_sqrt*np.sqrt(a[i]/lam), tmp[i, 0]) for i in range(p)])
    #logX = np.array([multivariate_normal(log_M[i], Sigma[i], n) for i in range(p)])
    logX = np.array([[log_M[i] + np.matmul(Sigma_sqrt*np.sqrt(a[i]), tmp[i,j]) for j in range(1,n+1)] for i in range(p)])

    Sigma = None
    Sigma_sqrt = None
    
    log_M_logE = np.mean(logX, axis = 1)
    S_logE = np.array([(n-1)*np.cov(logX[i].T) for i in range(p)])
    #print('Optimizing!')
    ## SURE (mean only)
    _, _, log_M_SURE = SURE_const_log(logX)
    
    ## SURE (mean and covariance)
    _, _, nu_hat, Psi_hat, log_M_SURE_full = SURE_full_log(logX, verbose = verbose)
    

    ## risk
    res.loc['FM_logE', 'risk_M'] = np.linalg.norm(log_M - log_M_logE)**2
    res.loc['SURE', 'risk_M'] = np.linalg.norm(log_M - log_M_SURE)**2
    res.loc['SURE_full', 'risk_M'] = np.linalg.norm(log_M - log_M_SURE_full)**2
    #tmp = 0
    #tmp1 = 0
    #for i in range(p):
    #    tmp = tmp + np.sum((np.cov(logX[i].T)-Sigma[i])**2)/p
    #    tmp1 = tmp1 + np.sum(((Psi_hat + np.cov(logX[i].T))/(nu_hat + n - q - 2) -Sigma[i])**2)/p
        
    #res.loc['FM_logE', 'risk_Sig'] = tmp
    #res.loc['SURE', 'risk_Sig'] = tmp
    #res.loc['SURE_full', 'risk_Sig'] = tmp1

    return res.values
 
    
if __name__ == '__main__':
    n = 5
    lam_vec = np.array([10, 50])
    nu_vec = np.array([10, 25])
    p = 10
    N_vec = np.array([5, 10, 20, 30, 50, 100])
    #N_vec = np.array([100])
    ran_seed = 12345
    m = 100 # number of replications


    out_file = 'exp_full_size.p'

    num_cores = -1
    risk_M = pd.DataFrame(np.zeros((len(lam_vec)*len(nu_vec)*len(N_vec), 8)))
    risk_M.columns = ['p', 'n', 'lambda', 'nu', 'N', 'FM_LogE', 'SURE', 'SURE_full']
    risk_M_sd = risk_M.copy()
    risk_Sig = risk_M.copy()
    risk_Sig_sd = risk_M.copy()
    r_ind = 0
    
    

    for N in N_vec:
        q = int(N*(N+1)/2)
        mu = np.eye(N)
        log_mu = vec(mu)[0]
        Psi = np.eye(q)
        for nu in nu_vec:
            for lam in lam_vec:                
                
                print('p =', p, ', nu =', nu, ', lam =', lam, ', N = ', N)
                #results = Parallel(n_jobs=num_cores)(delayed(exp_lognormal)(n, p, N, lam, log_mu, nu+q, Psi, ran_seed + i) for i in range(m))
                
                results = np.zeros((m, 3, 2))
                for i in range(m):
                    results[i] = exp_lognormal(n, p, N, lam, log_mu, nu+q, Psi, ran_seed + i) 
                
                
                res = np.mean(np.array(results), axis = 0)
                res_sd = np.std(np.array(results), axis = 0)/np.sqrt(m)
                res = pd.DataFrame(res, index = ['FM_logE', 
                    'SURE', 'SURE_full'])
                res_sd = pd.DataFrame(res_sd, index = ['FM_logE', 
                    'SURE', 'SURE_full'])
                res.columns = ['risk_M', 'risk_Sig']
                res_sd.columns = ['risk_M', 'risk_Sig']
                risk_M.values[r_ind] = np.array([p, n, lam, nu, N,
                    res.loc['FM_logE', 'risk_M'], 
                    res.loc['SURE', 'risk_M'],
                    res.loc['SURE_full', 'risk_M']])                
                #risk_Sig.values[r_ind] = np.array([p, n, lam, nu, N,
                #    res.loc['FM_logE', 'risk_Sig'], 
                #    res.loc['SURE', 'risk_Sig'],
                #    res.loc['SURE_full', 'risk_Sig']])                           
                risk_M_sd.values[r_ind] = np.array([p, n, lam, nu, N,
                    res_sd.loc['FM_logE', 'risk_M'], 
                    res_sd.loc['SURE', 'risk_M'],
                    res_sd.loc['SURE_full', 'risk_M']])                
                #risk_Sig_sd.values[r_ind] = np.array([p, n, lam, nu, N,
                #    res_sd.loc['FM_logE', 'risk_Sig'], 
                #    res_sd.loc['SURE', 'risk_Sig'],
                #    res_sd.loc['SURE_full', 'risk_Sig']])                              
                r_ind += 1
                print('Success!')
                
    pickle.dump({'risk_M':risk_M}, open(out_file, 'wb'))
    
    print(risk_M)
    ##################################################################
    ## plots
    
    risk_M = pd.melt(risk_M, ['p','n','lambda', 'nu', 'N'], var_name='Estimator', value_name='risk')
    #risk_Sig = pd.melt(risk_Sig, ['p','n','lambda', 'nu', 'N'], var_name='Estimator', value_name='risk')

    def col_label(s):
        return '$\lambda = ' + str(s) + '$'

    def row_label(s):
        return '$\\nu = ' + str(s) + '$'

    risk_M['risk'] = risk_M['risk']/risk_M['N']**2

    #risk_M = risk_M[risk_M['Estimator'] != 'FM_GL']
    p = (ggplot(risk_M)
        + aes(x='N', y='risk', color='Estimator', linetype='Estimator')
        + geom_line(size = 0.8)
        #+ geom_errorbar(aes(ymin = 'lower', ymax = 'upper'), width = 30)
        + facet_grid(['nu', 'lambda'], scales = 'free', labeller = labeller(rows = row_label, cols=col_label))
        + xlab('Size of SPD matrices (N)')
        + ylab('Average Loss')
        + scale_color_manual(labels = ['FM.LE (MLE)', 'SURE-FM', 'SURE.Full-FM'], 
                           values = ['red', 'blue', 'green'])
        + scale_linetype_manual(labels = ['FM.LE (MLE)', 'SURE-FM', 'SURE.Full-FM'], 
                              values = ['dashdot','dashed','solid'])
        + theme(axis_title=element_text(size=8), 
              axis_text=element_text(size=8),
              legend_title=element_text(size=8),
              legend_text=element_text(size=8),
              strip_text=element_text(size=8))
        #+ ggtitle(r'$\mu=diag(2,0.5,0.5)$, $\Psi = I_6$, $\lambda = 10$, $\nu=15$')
        )
    
    p.save('risk_M_size_normalized.pdf', dpi = 320, width = 6, height = 4, unit="in")



