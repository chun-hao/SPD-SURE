from exp_setup import *

# synth exp 1-1 (Section 4.2.1)

n = 10
lam_vec = np.array([10, 50])
nu_vec = np.array([15, 30])
p_vec = np.array([5, 10, 15, 20, 30, 40, 50, 70, 100, 120, 150, 170, 200])
ran_seed = 12345
m = 1000 # number of replications
N = 3
q = int(N*(N+1)/2)
mu = np.eye(N)
Psi = np.eye(q)
out_file = 'exp_full_lognormal.p'

exp(n, p_vec, lam_vec, nu_vec, mu, Psi, out_file, m, ran_seed = 12345)

# plots for synth exp 1-1 (Fig. 1)

f = open('exp_full_lognormal.p', 'rb')
result = pickle.load(f)
risk_M = result['risk_M']
risk_M_sd = result['risk_M_sd']
risk_M = pd.melt(risk_M, ['p','n','lambda', 'nu'], var_name='Estimator', value_name='risk')
risk_Sig = result['risk_Sig']
risk_Sig = pd.melt(risk_Sig, ['p','n','lambda', 'nu'], var_name='Estimator', value_name='risk')


risk_M_sd = pd.melt(risk_M_sd, ['p','n','lambda', 'nu'], var_name='Estimator', value_name='sd')
risk_M['sd'] = risk_M_sd['sd']
risk_M['lower'] = risk_M['risk'] - risk_M['sd']
risk_M['upper'] = risk_M['risk'] + risk_M['sd']


def col_label(s):
    return '$\lambda = ' + str(s) + '$'

def row_label(s):
    return '$\\nu = ' + str(s) + '$'

risk_M = risk_M[risk_M['Estimator'] != 'FM_GL']

p = (ggplot(risk_M)
    + aes(x='p', y='risk', color='Estimator', linetype='Estimator')
    + geom_line(size = 0.8)
    #+ geom_errorbar(aes(ymin = 'lower', ymax = 'upper'), width = 30)
    + facet_grid(['nu', 'lambda'], scales = 'free', labeller = labeller(rows = row_label, cols=col_label))
    + xlab('Spatial Dimension (p)')
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
#p
p.save('risk_M.pdf', dpi = 320, width = 6, height = 4, unit="in")