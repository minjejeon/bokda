def dfuller(*args, **kwargs):
    '''Augmented Dicker Fuller test'''
    from statsmodels.tsa.stattools import adfuller
    verbose = kwargs.pop('verbose', False)
    ans = adfuller(*args, **kwargs)
    if verbose:
        regresults = kwargs.get('regresults', False)
        if regresults:
            stat,pval,cv,regout = ans
            nobs = regout.nobs
            lags = regout.usedlag
        else:
            stat,pval,lags,nobs,cv,_ = ans
        print('Augmented Dickery-Fuller test for unit root')
        print('')
        print('Number of obs  = %5d' % nobs)
        print('Number of lags = %5d' % lags)
        print('')
        style = kwargs.get('regression', 'c')
        if style=='n':
            desc = 'no constant, no trend'
        elif style=='c':
            desc = 'constant only'
        elif style=='ct':
            desc = 'constant and trend'
        elif style=='ctt':
            desc = 'constant, and linear and quadratic trend'
        print(f'H0: Random walk ({desc})')
        print('')
        print('     Test      -------- critical value ---------')
        print('statistic          10%           5%           1%')
        print('------------------------------------------------')
        print('%9.3f    %9.3f    %9.3f    %9.3f' % \
              (stat, cv['10%'], cv['5%'], cv['1%']))
        print('------------------------------------------------')
        print('p-value = %.4f' % pval)
    return ans

def kpss(*args, **kwargs):
    '''KPSS test (with statsmodels branched)'''
    import warnings
    from statsmodels.tsa.stattools import kpss
    verbose = kwargs.pop('verbose', False)
    if verbose:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ans = kpss(*args, **kwargs)
    else:
        ans = kpss(*args, **kwargs)
    if verbose:
        regresults = kwargs.get('store', False)
        if regresults:
            stat,pval,lags,cv,regout = ans
            nobs = regout.nobs
        else:
            stat,pval,lags,nobs,cv = ans
        print('KPSS test for stationarity')
        print('')
        print('Number of obs  = %5d' % nobs)
        print('Number of lags = %5d' % lags)
        print('')
        style = kwargs.get('regression', 'c')
        if style=='c':
            desc = 'constant'
        elif style=='ct':
            desc = 'trend'
        print(f'H0: Stationary around a {desc}')
        print('')
        print('     Test      --------------- critical value ---------------')
        print('statistic          10%           5%         2.5%           1%')
        print('-------------------------------------------------------------')
        print('%9.3f    %9.3f    %9.3f    %9.3f    %9.3f' % \
              (stat, cv['10%'], cv['5%'], cv['2.5%'], cv['1%']))
        print('-------------------------------------------------------------')
        if stat < cv['10%']:
            print('p-value > 0.1')
        elif stat > cv['1%']:
            print('p-value < 0.01')
        else:
            print('p-value = %.4f' % pval)
    return ans

def var(*args, **kwargs):
    '''Vector autoregression'''
    from statsmodels.tsa.api import VAR
    maxlags = kwargs.pop('maxlags',1)
    model = VAR(*args, **kwargs)
    return model.fit(maxlags=maxlags)

def vecrank(*args, **kwargs):
    '''Johansen cointegration test'''
    from statsmodels.tsa.vector_ar.vecm import (
        coint_johansen,
        select_coint_rank
    )
    verbose = kwargs.pop('verbose', False)
    #ans = coint_johansen(*args, **kwargs)
    ans = select_coint_rank(*args, **kwargs)
    if verbose: print(ans)
    return ans

def vecm(*args, **kwargs):
    '''Vector Error Correction'''
    from statsmodels.tsa.vector_ar.vecm import VECM
    model = VECM(*args, **kwargs)
    ans = model.fit()
