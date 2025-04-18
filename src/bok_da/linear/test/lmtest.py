def _format_stat_pval(stat, pval, dist, *args, **kwargs):
    '''Format statistic and pvalue'''
    if not len(args) and dist=='N':
        s1 = 'N(0,1)'
    else:
        s1 = '%s(%d' % (dist, args[0])
        if dist in ['F']:
            s1 += ', %d' % args[1]
        s1 += ')'
    s2 = f'Prob > {dist}'
    n = max(len(s1),len(s2))
    fmt = f'%{n}s = %6s'
    app = kwargs.get('append', '')
    if isinstance(app,int): app = ' '*app
    line1 = (fmt % (s1, '%.2f' % stat) + app)
    line2 = (fmt % (s2, '%.4f' % pval) + app)
    return [line1, line2]

def het_test(ols, method = 'bp', verbose=False): # "bp" | "w" | "ws"
    r"""
    Heteroskedasticity test (Breusch-Pagan, or White, or White Simplified).

    Parameters
    ----------
    ols: OLS RegressionResults
        Possibly outcome of -regress- or
        statsmodels.regression.linear_model.OLS.fit()
    method: str
        "bp" = Breusch-Pagan (default)
        "w" = White
        "ws" = Simplified white (yhat and yhat**2)

    Details
    -------
    The BP test calls statsmodels.stats.diagnostic.het_breuschpagan,
    and the White tests (including the simplified) call
    statsmodels.stats.api.het_white.

    Returns
    -------
    Tuple of LM, LM p-value, F, F p-value.

    Examples
    --------
    import bok
    import pandas

    df = bok.read.csv('Death.csv')
    fm = 'deathrate~smoke+drink+I(smoke-aged)+aged+I(smoke+aged)+C(year)'
    # note collinearity above
    ols = bok.regress(fm, data=df, weights='regpop', vce='cl', cluster='region')
    bok.het_test(ols)
    # (10.1249752272411, 0.07177059126247581, 2.058693698248833, 0.07113504408834326)
    """
    import statsmodels.stats.api as sms
    import statsmodels.api as sm
    if method in ['bp', 'breuschpagan', 'breusch-pagan']:
        ans = sms.het_breuschpagan(ols.resid, ols.model.exog)
        if verbose:
            print("Breusch-Pagan test for heteroskedasticity")
            print("Variables: All independent variables")
            print("")
            print("H0: Constant variance")
            print("")
            s1 = _format_stat_pval(
                ans[0], ans[1], 'chi2', ols.df_model, append=4
            )
            s2 = _format_stat_pval(
                ans[2], ans[3], 'F', ols.df_model, ols.df_resid
            )
            s = "\n".join([x+y for (x,y) in zip(s1,s2)])
            print(s)
    elif method in ['w', 'white']:
        ans = sms.het_white(ols.resid, ols.model.exog)
        if verbose:
            print("White test for heteroskedasticity")
            print("Variables: All independent variables and quadratic")
            print("")
            print("H0: Constant variance")
            print("")
            s1 = _format_stat_pval(ans[0], ans[1], 'chi2', ans[4], append=4)
            s2 = _format_stat_pval(ans[2], ans[3], 'F', ans[4], ols.df_resid)
            s = "\n".join([x+y for (x,y) in zip(s1,s2)])
            print(s)
    elif method in ['ws', 'white-simplified']:
        ans = sms.het_white(ols.resid, sm.add_constant(ols.fittedvalues))
        if verbose:
            print("White test for heteroskedasticity")
            print("Variables: Fitted values and square")
            print("")
            print("H0: Constant variance")
            print("")
            s1 = _format_stat_pval(ans[0], ans[1], 'chi2', ans[4], append=4)
            s2 = _format_stat_pval(ans[2], ans[3], 'F', ans[4], ols.df_resid)
            s = "\n".join([x+y for (x,y) in zip(s1,s2)])
            print(s)
    else:
        warnings.warn('Use one of "bp", "w", and "ws".')
        return None
    return ans

def ac_test(ols, type='bg', nlags=None, verbose=False):
    r"""
    Autocorrelation test (Breusch-Godfrey, or Durbin-Watson test).

    Parameters
    ----------
    ols: OLS RegressionResults
        Possibly outcome of -regress- or
        statsmodels.regression.linear_model.OLS.fit()
    type: str
        "bg" = Breusch-Godfrey (default)
        "dw" = Durbin-Watson (test statistic only)

    Details
    -------
    The BP test calls statsmodels.stats.diagnostic.acorr_breusch_godfrey,
    and the DW test calls statsmodels.stats.diagnostic.durbin_watson.
    For p-value for DW test, use the 'dwtest' module.

    $ pip install dwtest

    import dwtest
    dwtest.dwtest("consump~wagegovt", data=df)

    Returns
    -------
    Tuple of LM, LM p-value, F, F p-value for Breusch-Godfrey, and
    DW stat for Durbin-Watson.

    Examples
    --------
    import bok

    df = bok.read.stata('klein.dta')
    ols = bok.regress('consump~wagegovt', data=df)
    bok.ac_test(ols, nlags=3)
    # (16.426789702977413,
    #  0.0009269291529265771,
    #  16.702248199737763,
    #  2.576570772931991e-05)
    """
    from statsmodels.stats.diagnostic import acorr_breusch_godfrey
    from statsmodels.stats.stattools import durbin_watson
    import dwtest
    if type=='bg' or type=='breusch-godfrey':
        ans = acorr_breusch_godfrey(ols, nlags=nlags, store=True)
        if verbose:
            print("Breusch-Godfrey test for autocorrelation")
            print("")
            print("H0: No serial correlation")
            print("")
            s1 = _format_stat_pval(
                ans[0], ans[1], 'chi2', ans[4].usedlag, append=4
            )
            s2 = _format_stat_pval(
                ans[2], ans[3], 'F', ans[4].usedlag, ols.df_resid
            )
            s = "\n".join([x+y for (x,y) in zip(s1,s2)])
            print(s)
    elif type=='dw' or type=='durbin-watson':
        dw = durbin_watson(ols.resid, axis=0)
        print('* Use dwtest module for p-value.')
        ans = dw
    else:
        warnings.warn('Use type="bg".')
        return None
    return ans

def reset_test(*args, **kwargs):
    '''RESET test'''
    from statsmodels.stats.diagnostic import linear_reset
    verbose = kwargs.pop('verbose', False)
    kwargs.setdefault('use_f', True)
    if 'vce' in kwargs:
        if kwargs.pop('vce') in ['r', 'robust']:
            kwargs['cov_type'] = 'HC1'
    ans = linear_reset(*args, **kwargs)
    if verbose:
        print("Ramsey RESET test")
        print("")
        print("H0: Model is correctly linearly specified")
        print("")
        if hasattr(ans, 'fvalue'):
            s = _format_stat_pval(
                ans.fvalue, ans.pvalue, 'F', ans.df_num, ans.df_denom
            )
        else:
            s = _format_stat_pval(
                ans.statistic, ans.pvalue, 'chi2', ans.df_denom
            )
        print("\n".join(s))
    return ans
