def _check_formula_y(formula,y):
    if y is not None and formula is not None:
        msg = ("Should not use both formula and (X,y). "
               "Ignore formula and use (X,y).")
        warnings.warn(msg)
        formula = None
    return formula

def disp_res(res):
    '''Display regression results'''

def regress(
        formula=None, data={}, y=None, X=None, weights=None,
        vce = 'o', cluster = None, use_t = True,
        ret_y = False, ret_X = False, check_collinearity=True,
        collinear_eps=1e-12, collinear_pivoting = False, verbose = True,
        **kwargs
):
    r"""
    OLS and WLS regression of linear regression model.
    Implements Stata's -regress-.

    Parameters
    ----------
    formula: str
        The regression equation.
    data: pandas.DataFrame
        The data set.
    X: pandas.DataFrame (optional)
        The regressor matrix (should include constant if desired).
        This parameter, if exists, overrides formula.
    y: a numerical vector (optional)
        The regressand vector.
        This parameter, if exists, overrides formula.
    weights: str | numpy.ndarray | pandas.Series
        Weights for WLS.
        If str, it's the variable name in the 'data' parameter.
        Otherwise, it's the data.
    vce: str ('o' or 'r' or 'HC1', etc.) = "o"
        Variance-covariance estimator type.
        "o" or "ordinary" = ordinary
        "r" or "robust" = heteroskedasticity-robust (HC1)
        "c" or "cl" or "cluster" = cluster-robust (CC1)
            Requires 'cluster' parameter (variable name or data).
    cluster: str | numpy.ndarray | pandas.Series
        Extra parameter for cluster covariance estimation.
    use_t: bool = True
        Used as parameter for fit()
    ret_y: bool = False
        Return the y vector as model.y if True else None
    ret_X: bool = False
        Return the X matrix as model.X (collinearity checked) if True else None
    title: str | None
        Title to be displayed by summary3().
    collinear_check: bool = True
        Check collinearity if True
    collinear_eps: float = 1e-12
        The threshold value for collinearity check
    collinear_pivoting: bool = False
        Whether to use pivoting for QR decomposition.
    verbose: bool = True

    Examples
    --------
    import bok
    df = bok.read.csv('Death.csv')
    fm = 'deathrate~smoke+drink+I(smoke-aged)+aged+I(smoke+aged)+C(year)'
    # note collinearity above
    ols = bok.regress(fm, data=df, weights='regpop', vce='cl', cluster='region')
    print(ols.summary3())
    """

    import bok_da as bok
    import statsmodels.api as sm
    import numpy as np

    formula = _check_formula_y(formula,y)

    if weights is not None and isinstance(weights, str) and weights!="_auto":
        weights = data[weights]

    if formula is None:
        if weights is None:
            model = sm.OLS(y,X)
        else:
            model = sm.WLS(y,X,weights=weights)
    else:
        if weights is None:
            model = sm.OLS.from_formula(formula, data)
        else:
            model = sm.WLS.from_formula(formula, data, weights = weights)

    # fit model with given vce
    if vce=='o':
        ans = model.fit(
            check_collinearity = check_collinearity,
            use_t = use_t
        )
    elif vce=='r' or vce=='robust':
        ans = model.fit(
            check_collinearity = check_collinearity,
            cov_type='HC1', use_t = use_t)
    elif vce=='cl' or vce=='cluster' or vce=='clustered':
        grpvar = None
        if isinstance(cluster,str):
            grpvar = cluster
            cluster = data[cluster]
        elif isinstance(cluster,pd.Series):
            ans.set_extra_attributes(grpvar=cluster.name)
        ans = model.fit(
            check_collinearity = check_collinearity,
            cov_type='cluster',
            cov_kwds={'groups':cluster}
        )
        ngrp = len(np.unique(cluster))
        ans.set_extra_attributes(grpvar=grpvar)
        ans.set_extra_attributes(ngroups=ngrp)
        ans.set_extra_attributes(df_resid_inference=ngrp-1)
        ans.use_t = True
    else:
        ans = model.fit(
            check_collinearity = check_collinearity,
            cov_type=vce, use_t = use_t
        )

    ans.model.y = y if ret_y else None
    ans.model.X = X if ret_X else None
    ans.set_extra_attributes(cmd='regress')
    return ans

def prais(formula=None, 
          data={}, 
          y=None, 
          X=None,
          twostep=False, 
          corc=False,
          rhotype = 'regress', 
          maxiter=100,
          vce='o', 
          cluster=None,
          **kwargs,):
    """
    Prais-Winsten and Cochrane-Orcutt regression
    """
    import statsmodels.api as sm
    formula = _check_formula_y(formula,y)
    if formula is None:
        model = sm.Prais(y, X,
                         corc=corc, 
                         twostep=twostep,
                         rhotype=rhotype, 
                         maxiter=maxiter)
    else:
        model = sm.Prais.from_formula(formula, data,
                                      corc=corc, 
                                      twostep=twostep,
                                      rhotype=rhotype, 
                                      maxiter=maxiter)

    if vce is None or vce in ('o', '', 'ordinary', 'ord'):
        ans = model.iterative_fit()
    elif vce in ('r', 'robust'):
        ans = model.iterative_fit(cov_type = 'HC1')
    elif vce in ('cl', 'cluster', 'clustered'):
        if isinstance(cluster, str): cluster = data.loc[:,cluster]
        ans = model.fit(cov_type = 'cluster',
                        cov_kwds = {'groups':cluster})
    else:
        ans = model.iterative_fit(cov_type = vce)
    ans.set_extra_attributes(cmd='prais')
    return ans

def ivregress(
        formula=None, data={}, y=None, X=None, Y=None, Z=None,
        weights=None, method="2sls",
        vce = 'o', cluster = None, use_t = True,
        ret_y = False, ret_X = False, check_collinearity=True,
        collinear_eps=1e-12, collinear_pivoting = False, verbose = True,
        **kwargs
):
    """
    IV regression
    """
    import linearmodels as lm

    formula = _check_formula_y(formula,y)

    if weights is not None and isinstance(weights, str):
        weights = data[weights]

    if method=="2sls":
        if formula is None:
            model = lm.IV2SLS(y, X, Y, Z, weights = weights)
        else:
            model = lm.IV2SLS.from_formula(formula, data, weights = weights)
            
    elif method=="liml":
        if formula is None:
            model = lm.IVLIML(y, X, Y, Z, weights = weights)
        else:
            model = lm.IVLIML.from_formula(formula, data, weights = weights)
            
    else:
        raise ValueError('method must be "2sls" or "liml"')

        
    if vce is None or vce in ('o', '', 'ordinary', 'ord'):
        ans = model.fit(cov_type = 'unadjusted')
    elif vce in ('r', 'robust'):
        ans = model.fit(cov_type = 'robust',
                        debiased = kwargs.get('debiased', False))
    elif vce in ('cl', 'cluster', 'clustered'):
        if isinstance(cluster, str): cluster = data.loc[:,cluster]
        ans = model.fit(
            cov_type = 'clustered',
            debiased = kwargs.get('debiased', False),
            clusters = cluster,
        )
    else:
        print(f'* vce={vce} argument not allowed. use ordinary.')
        ans = model.fit(cov_type = 'unadjusted')

    ans.set_extra_attributes(cmd='ivregress')
    return ans
