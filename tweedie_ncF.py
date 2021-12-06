import numpy as np
from scipy.stats import ncx2, ncf, norm, bernoulli
import statsmodels.api as sm
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial


def tweedie_ncF(z, df1, df2, maxit = 1000, tol = 1e-3, **kwargs):
    # compute tweedie-type estimates for the non-centrality parameters for the non-central F distribution with known dfs df1 and df2
    

    lam_MOM = np.maximum(df1*(df2 - 2)/df2 * z - df1, 0) 
    lam = lam_MOM # use the MOM estimates as the initial values
    for i in range(maxit):
        U = ncf.cdf(z, dfn = df1, dfd = df2, nc = lam)
        U = np.minimum(np.maximum(U, 1e-7), 1-1e-7) # make sure U is in [0,1]
        U = np.nan_to_num(U, 0.5)
        Y = ncx2.ppf(U, df = df1, nc = lam)
        
        pr_model = density_pr(Y, **kwargs)
        dl1 = deriv_density_pr(pr_model, 1)
        dl2 = deriv_density_pr(pr_model, 2)
        l1 = dl1(Y)
        l2 = dl2(Y)
        
        lam_new = np.maximum((Y - df1 + 4)*(1+2*l1) + 2*Y*(2*l2 + l1*(1+2*l1)), 0)
        

        lam_new[np.isnan(lam_new)] = lam_MOM[np.isnan(lam_new)]
        
        diff = np.max(np.abs(lam_new - lam))
        if diff < tol:
            lam = lam_new.copy()
            print('reached tolerance level in ', i, ' iterations')
            break
        else:
            lam = lam_new.copy()

    return lam    


def tweedie_x2(z, df, **kwargs):
    pr_model = density_pr(z, **kwargs)
    l1 = deriv_density_pr(pr_model, 1)(z)
    l2 = deriv_density_pr(pr_model, 2)(z)

    lam = (z - df + 4)*(1+2*l1) + 2*z*(2*l2 + l1*(1+2*l1))
    return lam


def density_pr(x, K = 5, nbin = 100, plot = False, model_summary = False, trim_lo = 0.01, trim_up = 0.01):
    # density estimation with poisson regression with polynomial of degree K
    
    # # remove missing and infinite values
    x = x[~np.isnan(x)]
    x = x[~np.isinf(x)]
    
    # remove outliers that may lead to convergence problems
    v = np.quantile(x, q = [trim_lo, 1 - trim_up])
    lo = v[0]
    up = v[1]
    #x = x[(x > lo) * (x < up)]
    
    #nbin = 20
    count, bins = np.histogram(x, bins=nbin)
    yy = count
    xx = (bins[0:nbin] + bins[1:nbin+1])/2
    
    # scale factor (= area under the histogram bars)
    # this is used turn f below into a density
    scale = np.sum(np.ediff1d(bins)*count)
    
    df = pd.DataFrame({'x':xx,'y':yy})

    xp = PolynomialFeatures(degree=K).fit_transform(df[['x']])
    #xp = np.linalg.qr(xp)[0][:,1:]

    model = sm.GLM(df['y'],sm.add_constant(xp),family=sm.families.Poisson()).fit()
    
    if model_summary:
        print(model.summary())
    
    if plot:
        plt.clf()
        n, bins, patches = plt.hist(x=x, bins=nbin, color='#0504aa',
                            stacked=True, density = True, alpha=0.7, rwidth=0.85)
        yy_pred = model.predict(sm.add_constant(xp))/scale
        plt.plot(xx, yy_pred, color = 'red')
        
    return model

def deriv_density_pr(pr_model, d):
    # return the dth derivative of the log-density
    if d < 1 or not isinstance(d, int):
        raise ValueError('d must be a positive integer!')
    
    beta = pr_model.params
    l = Polynomial(beta)
    return l.deriv(d)