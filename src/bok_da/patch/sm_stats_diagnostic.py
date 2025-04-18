import numpy as np
from statsmodels.tools.validation import array_like
from statsmodels.stats.diagnostic import _check_het_test
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.compat.pandas import deprecate_kwarg
from statsmodels.tsa.tsatools import lagmat
from statsmodels.stats.diagnostic import ResultsStore

def new_het_white(resid, exog):
    """
    White's Lagrange Multiplier Test for Heteroscedasticity.

    Parameters
    ----------
    resid : array_like
        The residuals. The squared residuals are used as the endogenous
        variable.
    exog : array_like
        The explanatory variables for the variance. Squares and interaction
        terms are automatically included in the auxiliary regression.

    Returns
    -------
    lm : float
        The lagrange multiplier statistic.
    lm_pvalue :float
        The p-value of lagrange multiplier test.
    fvalue : float
        The f-statistic of the hypothesis that the error variance does not
        depend on x. This is an alternative test variant not the original
        LM test.
    f_pvalue : float
        The p-value for the f-statistic.

    Notes
    -----
    Assumes x contains constant (for counting dof).

    question: does f-statistic make sense? constant ?

    References
    ----------
    Greene section 11.4.1 5th edition p. 222. Test statistic reproduces
    Greene 5th, example 11.3.
    """
    x = array_like(exog, "exog", ndim=2)
    y = array_like(resid, "resid", ndim=2, shape=(x.shape[0], 1))
    _check_het_test(x, "White's heteroskedasticity")
    nobs, nvars0 = x.shape
    i0, i1 = np.triu_indices(nvars0)
    exog = x[:, i0] * x[:, i1]
    nobs, nvars = exog.shape
    assert nvars == nvars0 * (nvars0 - 1) / 2. + nvars0
    resols = OLS(y ** 2, exog).fit()
    fval = resols.fvalue
    fpval = resols.f_pvalue
    lm = nobs * resols.rsquared
    # Note: degrees of freedom for LM test is nvars minus constant
    # degrees of freedom take possible reduced rank in exog into account
    # df_model checks the rank to determine df
    # extra calculation that can be removed:
    assert resols.df_model == np.linalg.matrix_rank(exog) - 1
    lmpval = stats.chi2.sf(lm, resols.df_model)
    return lm, lmpval, fval, fpval, resols.df_model


@deprecate_kwarg("results", "res")
def new_acorr_breusch_godfrey(res, nlags=None, store=False):
    """
    Breusch-Godfrey Lagrange Multiplier tests for residual autocorrelation.

    Parameters
    ----------
    res : RegressionResults
        Estimation results for which the residuals are tested for serial
        correlation.
    nlags : int, optional
        Number of lags to include in the auxiliary regression. (nlags is
        highest lag).
    store : bool, default False
        If store is true, then an additional class instance that contains
        intermediate results is returned.

    Returns
    -------
    lm : float
        Lagrange multiplier test statistic.
    lmpval : float
        The p-value for Lagrange multiplier test.
    fval : float
        The value of the f statistic for F test, alternative version of the
        same test based on F test for the parameter restriction.
    fpval : float
        The pvalue for F test.
    res_store : ResultsStore
        A class instance that holds intermediate results. Only returned if
        store=True.

    Notes
    -----
    BG adds lags of residual to exog in the design matrix for the auxiliary
    regression with residuals as endog. See [1]_, section 12.7.1.

    References
    ----------
    .. [1] Greene, W. H. Econometric Analysis. New Jersey. Prentice Hall;
      5th edition. (2002).
    """

    x = np.asarray(res.resid).squeeze()
    if x.ndim != 1:
        raise ValueError("Model resid must be a 1d array. Cannot be used on"
                         " multivariate models.")
    exog_old = res.model.exog
    nobs = x.shape[0]
    if nlags is None:
        nlags = min(10, nobs // 5)

    x = np.concatenate((np.zeros(nlags), x))

    xdall = lagmat(x[:, None], nlags, trim="both")
    nobs = xdall.shape[0]
    xdall = np.c_[np.ones((nobs, 1)), xdall]
    xshort = x[-nobs:]
    if exog_old is None:
        exog = xdall
    else:
        exog = np.column_stack((exog_old, xdall))
    k_vars = exog.shape[1]

    resols = OLS(xshort, exog).fit(check_collinearity=False, method='pinv')
    ft = resols.f_test(np.eye(nlags, k_vars, k_vars - nlags))
    fval = ft.fvalue
    fpval = ft.pvalue
    fval = float(np.squeeze(fval))
    fpval = float(np.squeeze(fpval))
    lm = nobs * resols.rsquared
    lmpval = stats.chi2.sf(lm, nlags)
    # Note: degrees of freedom for LM test is nvars minus constant = usedlags

    if store:
        res_store = ResultsStore()
        res_store.resols = resols
        res_store.usedlag = nlags
        return lm, lmpval, fval, fpval, res_store
    else:
        return lm, lmpval, fval, fpval
