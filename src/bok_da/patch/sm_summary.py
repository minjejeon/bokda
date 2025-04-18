import numpy as np
from statsmodels.compat.python import lrange, lzip
from statsmodels.iolib.summary import _getnames
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.tableformatting import fmt_params


# global function in statsmodels.iolib.summary module
def forg(x, prec=3):
    x = np.squeeze(x)
    if prec == 3:
        # for 3 decimals
        if (abs(x) >= 1e4) or (abs(x) < 1e-4):
            return '%9.3g' % x
        else:
            return '%9.3f' % x
    elif prec == 4:
        if (abs(x) >= 1e4) or (abs(x) < 1e-4):
            return '%10.4g' % x
        else:
            return '%10.4f' % x
    elif prec > 4:
        if (abs(x) >= 1e4) or (abs(x) < 1e-4):
            return f'%10.{prec}g' % x
        else:
            return f'%10.{prec}f' % x
    elif prec < 3:
        if (abs(x) >= 1e4) or (abs(x) < 1e-4):
            return f'%9.{prec}g' % x
        else:
            return f'%9.{prec}f' % x
    else:
        raise ValueError("`prec` argument must be either 3 or 4, not {prec}"
                         .format(prec=prec))


# global function in statsmodels.iolib.summary module
def summary_params(results, yname=None, xname=None, alpha=.05, use_t=True,
                   skip_header=False, title=None):
    '''create a summary table for the parameters

    Parameters
    ----------
    res : results instance
        some required information is directly taken from the result
        instance
    yname : {str, None}
        optional name for the endogenous variable, default is "y"
    xname : {list[str], None}
        optional names for the exogenous variables, default is "var_xx"
    alpha : float
        significance level for the confidence intervals
    use_t : bool
        indicator whether the p-values are based on the Student-t
        distribution (if True) or on the normal distribution (if False)
    skip_headers : bool
        If false (default), then the header row is added. If true, then no
        header row is added.

    Returns
    -------
    params_table : SimpleTable instance
    '''

    # Parameters part of the summary table
    # ------------------------------------
    # Note: this is not necessary since we standardized names,
    #   only t versus normal

    if isinstance(results, tuple):
        # for multivariate endog
        # TODO: check whether I do not want to refactor this
        #we need to give parameter alpha to conf_int
        results, params, std_err, tvalues, pvalues, conf_int = results
    else:
        params = np.asarray(results.params)
        std_err = np.asarray(results.bse)
        tvalues = np.asarray(results.tvalues)  # is this sometimes called zvalues
        pvalues = np.asarray(results.pvalues)
        conf_int = np.asarray(results.conf_int(alpha))
    if params.size == 0:
        return SimpleTable([['No Model Parameters']])
    # Dictionary to store the header names for the parameter part of the
    # summary table. look up by modeltype
    if use_t:
        param_header = ['coef', 'std err', 't', 'P>|t|',
                        '[' + str(alpha/2), str(1-alpha/2) + ']']
    else:
        param_header = ['coef', 'std err', 'z', 'P>|z|',
                        '[' + str(alpha/2), str(1-alpha/2) + ']']

    if skip_header:
        param_header = None

    _, xname = _getnames(results, yname=yname, xname=xname)

    if len(xname) != len(params):
        raise ValueError('xnames and params do not have the same length')

    params_stubs = xname

    exog_idx = lrange(len(xname))
    params = np.asarray(params)
    std_err = np.asarray(std_err)
    tvalues = np.asarray(tvalues)
    pvalues = np.asarray(pvalues)
    conf_int = np.asarray(conf_int)
    params_data = lzip([forg(params[i], prec=5) for i in exog_idx],
                       [forg(std_err[i], prec=5) for i in exog_idx],
                       [forg(tvalues[i], prec=2) for i in exog_idx],
                       ["%#6.3f" % (pvalues[i]) for i in exog_idx],
                       [forg(conf_int[i,0], prec=5) for i in exog_idx],
                       [forg(conf_int[i,1], prec=5) for i in exog_idx])
    parameter_table = SimpleTable(params_data,
                                  param_header,
                                  params_stubs,
                                  title=title,
                                  txt_fmt=fmt_params
                                  )

    return parameter_table

