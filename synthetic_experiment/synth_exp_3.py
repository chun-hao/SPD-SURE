from exp_setup import *

# synth exp 2 (Section 4.2.2)

n = 5
p = 10
lam_vec = np.array([10, 50])
nu_vec = np.array([10, 25])
N_vec = np.array([10, 20, 30, 50, 75, 100])
ran_seed = 12345
m = 1000 # number of replications

out_file = 'exp_size.p'

num_cores = -1
risk_M = pd.DataFrame(np.zeros((len(lam_vec)*len(nu_vec)*len(N_vec), 8)))
risk_M.columns = ['p', 'N', 'n', 'lambda', 'nu', 'FM_LogE', 'SURE', 'SURE_full']
r_ind = 0
for N in N_vec:
    q = int(N*(N+1)/2)
    mu = np.eye(N)
    Psi = np.eye(q)

    for nu in nu_vec:
        for lam in lam_vec:
            print('N =', N, ', nu =', nu, ', lam =', lam)
            results = Parallel(n_jobs=num_cores)(delayed(exp_lognormal)(n + q, p, lam, mu, q + nu, Psi, ran_seed + i) \
                                                 for i in range(m))
            res = np.mean(np.array(results), axis = 0)
            res = pd.DataFrame(res, index = ['FM_logE', 
                'FM_GL', 'SURE', 'SURE_full'])
            res.columns = ['risk_M', 'risk_Sig', 't']
            risk_M.values[r_ind] = np.array([p, N, n, lam, nu, 
                res.loc['FM_logE', 'risk_M'], 
                res.loc['SURE', 'risk_M'],
                res.loc['SURE_full', 'risk_M']])                
            r_ind += 1
            print('Success!')

pickle.dump({'risk_M':risk_M}, open(out_file, 'wb'))


# plots for synth exp 2 (Fig. 3)

f = open('exp_size.p', 'rb')
result = pickle.load(f)
risk_M = result['risk_M']

risk_M = pd.melt(risk_M, ['p','n','lambda', 'nu', 'N'], var_name='Estimator', value_name='risk')

def col_label(s):
    return '$\lambda = ' + str(s) + '$'

def row_label(s):
    return '$\\nu = ' + str(s) + '$'

risk_M['risk'] = risk_M['risk']/risk_M['N']**2

p = (ggplot(risk_M)
    + aes(x='N', y='risk', color='Estimator', linetype='Estimator')
    + geom_line(size = 0.8)
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
    )

p.save('risk_M_size.pdf', dpi = 320, width = 6, height = 4, unit="in")
