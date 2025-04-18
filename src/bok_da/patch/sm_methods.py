import numpy as np
from typing import Literal, Optional
from collections.abc import Sequence
from statsmodels.tools.tools import pinv_extended
from statsmodels.regression.linear_model import OLS, WLS, OLSResults, RegressionResults, RegressionResultsWrapper
from statsmodels.tools.validation import bool_like, float_like, string_like

from .sm_fast_lm import _fast_ols
from .sm_prais import Prais

def new_lm_fit(
        self,
        method: Literal["pinv", "qr"] = "qr",
        cov_type: Literal[
            "nonrobust",
            "fixed scale",
            "HC0",
            "HC1",
            "HC2",
            "HC3",
            "HAC",
            "hac-panel",
            "hac-groupsum",
            "cluster",
        ] = "nonrobust",
        cov_kwds=None,
        # use_t: bool | None = None,
        use_t: Optional[bool] = None,
        **kwargs
):
    """
    Full fit of the model.

    The results include an estimate of covariance matrix, (whitened)
    residuals and an estimate of scale.

    Parameters
    ----------
    method : str, optional
        Can be "pinv", "qr".  "pinv" uses the Moore-Penrose pseudoinverse
        to solve the least squares problem. "qr" uses the QR
        factorization.
    cov_type : str, optional
        See `regression.linear_model.RegressionResults` for a description
        of the available covariance estimators.
    cov_kwds : list or None, optional
        See `linear_model.RegressionResults.get_robustcov_results` for a
        description required keywords for alternative covariance
        estimators.
    use_t : bool, optional
        Flag indicating to use the Student's t distribution when computing
        p-values.  Default behavior depends on cov_type. See
        `linear_model.RegressionResults.get_robustcov_results` for
        implementation details.
    **kwargs
        Additional keyword arguments that contain information used when
        constructing a model using the formula interface.

    Returns
    -------
    RegressionResults
        The model estimation results.

    See Also
    --------
    RegressionResults
        The results container.
    RegressionResults.get_robustcov_results
        A method to change the covariance estimator used when fitting the
        model.

    Notes
    -----
    The fit method uses the pseudoinverse of the design/exogenous variables
    to solve the least squares minimization.
    """
    check_col = kwargs.get('check_collinearity', True)
    if check_col:
        self.check_collinearity(
            eps = kwargs.get('collinearity_eps', 1e-10),
            quiet = kwargs.get('quiet', False)
        )
    if method == "pinv":
        if not (hasattr(self, 'pinv_wexog') and
                hasattr(self, 'normalized_cov_params') and
                hasattr(self, 'rank')):

            self.pinv_wexog, singular_values = pinv_extended(self.wexog)
            self.normalized_cov_params = np.dot(
                self.pinv_wexog, np.transpose(self.pinv_wexog))

            # Cache these singular values for use later.
            self.wexog_singular_values = singular_values
            self.rank = np.linalg.matrix_rank(np.diag(singular_values))

        beta = np.dot(self.pinv_wexog, self.wendog)

    elif method == "qr":
        if not (hasattr(self, 'exog_Q') and
                hasattr(self, 'exog_R') and
                hasattr(self, 'normalized_cov_params') and
                hasattr(self, 'rank')):
            Q, R = np.linalg.qr(self.wexog)
            self.exog_Q, self.exog_R = Q, R
            self.normalized_cov_params = np.linalg.inv(np.dot(R.T, R))

            # Cache singular values from R.
            self.wexog_singular_values = np.linalg.svd(R, 0, 0)
            self.rank = np.linalg.matrix_rank(R)
        else:
            Q, R = self.exog_Q, self.exog_R
        # Needed for some covariance estimators, see GH #8157
        self.pinv_wexog = np.linalg.pinv(self.wexog)
        # used in ANOVA
        self.effects = effects = np.dot(Q.T, self.wendog)
        beta = np.linalg.solve(R, effects)
    else:
        raise ValueError('method has to be "pinv" or "qr"')

    if self._df_model is None:
        self._df_model = float(self.rank - self.k_constant)
    if self._df_resid is None:
        self.df_resid = self.nobs - self.rank

    if isinstance(self, OLS):
        lfit = OLSResults(
            self, beta,
            normalized_cov_params=self.normalized_cov_params,
            cov_type=cov_type, cov_kwds=cov_kwds, use_t=use_t)
    else:
        lfit = RegressionResults(
            self, beta,
            normalized_cov_params=self.normalized_cov_params,
            cov_type=cov_type, cov_kwds=cov_kwds, use_t=use_t,
            **kwargs)
    return RegressionResultsWrapper(lfit)

def new_WLS_init(self, endog, exog, weights=1., missing='none', hasconst=None,
             **kwargs):
    if type(self) is WLS:
        self._check_kwargs(kwargs)

    is_auto_wgt = isinstance(weights, str) and weights=="_auto"
    if isinstance(weights, str): weights=1.

    weights = np.array(weights)
    if weights.shape == ():
        if (missing == 'drop' and 'missing_idx' in kwargs and
                kwargs['missing_idx'] is not None):
            # patsy may have truncated endog
            weights = np.repeat(weights, len(kwargs['missing_idx']))
        else:
            weights = np.repeat(weights, len(endog))
    # handle case that endog might be of len == 1
    if len(weights) == 1:
        weights = np.array([weights.squeeze()])
    else:
        weights = weights.squeeze()

    super(WLS, self).__init__(
        endog, exog, missing=missing,
        weights=weights, hasconst=hasconst, **kwargs
    )
    if is_auto_wgt:
        self.check_collinearity(quiet = True)
        bhat = _fast_ols(self.exog, self.endog)
        uhat = self.endog - np.dot(self.exog, bhat)
        lusq = np.log(uhat**2)
        bhat = _fast_ols(self.exog, lusq)
        weights = 1.0/np.exp(np.dot(self.exog, bhat))
        super(WLS, self).__init__(
            endog, exog, missing=missing,
            weights=weights, hasconst=hasconst, **kwargs
        )

    nobs = self.exog.shape[0]
    weights = self.weights
    if weights.size != nobs and weights.shape[0] != nobs:
        raise ValueError('Weights must be scalar or same length as design')


def new_RegressionResults_summary(
        self,
        yname: Optional[str] = None,
        xname: Optional[Sequence[str]] = None,
        title: Optional[str] = None,
        alpha: float = 0.05,
        slim: bool = False,
):
    """
    Summarize the Regression Results.

    Parameters
    ----------
    yname : str, optional
        Name of endogenous (response) variable. The Default is `y`.
    xname : list[str], optional
        Names for the exogenous variables. Default is `var_##` for ## in
        the number of regressors. Must match the number of parameters
        in the model.
    title : str, optional
        Title for the top table. If not None, then this replaces the
        default title.
    alpha : float, optional
        The significance level for the confidence intervals.
    slim : bool, optional
        Flag indicating to produce reduced set or diagnostic information.
        Default is False.

    Returns
    -------
    Summary
        Instance holding the summary tables and text, which can be printed
        or converted to various output formats.

    See Also
    --------
    statsmodels.iolib.summary.Summary : A class that holds summary results.
    """
    from statsmodels.stats.stattools import (
        durbin_watson,
        jarque_bera,
        omni_normtest,
    )
    alpha = float_like(alpha, "alpha", optional=False)
    slim = bool_like(slim, "slim", optional=False, strict=True)

    jb, jbpv, skew, kurtosis = jarque_bera(self.wresid)
    omni, omnipv = omni_normtest(self.wresid)

    eigvals = self.eigenvals
    condno = self.condition_number

    # TODO: Avoid adding attributes in non-__init__
    self.diagn = dict(jb=jb, jbpv=jbpv, skew=skew, kurtosis=kurtosis,
                      omni=omni, omnipv=omnipv, condno=condno,
                      mineigval=eigvals[-1])

    # TODO not used yet
    # diagn_left_header = ['Models stats']
    # diagn_right_header = ['Residual stats']

    # TODO: requiring list/iterable is a bit annoying
    #   need more control over formatting
    # TODO: default do not work if it's not identically spelled

    top_left = [('Dep. Variable:', None),
                ('Model:', None),
                ('Method:', ['Least Squares']),
                ('Date:', None),
                ('Time:', None),
                ('No. Observations:', None),
                ('Df Residuals:', None),
                ('Df Model:', None),
                ]

    if hasattr(self, 'cov_type'):
        top_left.append(('Covariance Type:', [self.cov_type]))

    rsquared_type = '' if self.k_constant else ' (uncentered)'
    top_right = [('R-squared' + rsquared_type + ':',
                  ["%#8.4f" % self.rsquared]),
                 ('Adj. R-squared' + rsquared_type + ':',
                  ["%#8.4f" % self.rsquared_adj]),
                 ('F-statistic:', ["%#8.4g" % self.fvalue]),
                 ('Prob (F-statistic):', ["%#6.4f" % self.f_pvalue]),
                 ('Log-Likelihood:', None),
                 ('AIC:', ["%#8.4g" % self.aic]),
                 ('BIC:', ["%#8.4g" % self.bic])
                 ]

    if slim:
        slimlist = ['Dep. Variable:', 'Model:', 'No. Observations:',
                    'Covariance Type:', 'R-squared:', 'Adj. R-squared:',
                    'F-statistic:', 'Prob (F-statistic):']
        diagn_left = diagn_right = []
        top_left = [elem for elem in top_left if elem[0] in slimlist]
        top_right = [elem for elem in top_right if elem[0] in slimlist]
        top_right = top_right + \
            [("", [])] * (len(top_left) - len(top_right))
    else:
        diagn_left = [('Omnibus:', ["%#6.3f" % omni]),
                      ('Prob(Omnibus):', ["%#6.3f" % omnipv]),
                      ('Skew:', ["%#6.3f" % skew]),
                      ('Kurtosis:', ["%#6.3f" % kurtosis])
                      ]

        diagn_right = [('Durbin-Watson:',
                        ["%#8.3f" % durbin_watson(self.wresid)]
                        ),
                       ('Jarque-Bera (JB):', ["%#8.3f" % jb]),
                       ('Prob(JB):', ["%#8.3g" % jbpv]),
                       ('Cond. No.', ["%#8.3g" % condno])
                       ]

    if title is None and hasattr(self.model, 'title'):
        title = self.model.title + ' Results'
    if title is None:
        title = self.model.__class__.__name__ + ' ' + "Regression Results"

    # create summary table instance
    from statsmodels.iolib.summary import Summary
    smry = Summary()
    smry.add_table_2cols(self, gleft=top_left, gright=top_right,
                         yname=yname, xname=xname, title=title)
    smry.add_table_params(self, yname=yname, xname=xname, alpha=alpha,
                          use_t=self.use_t)
    if not slim:
        smry.add_table_2cols(self, gleft=diagn_left, gright=diagn_right,
                             yname=yname, xname=xname,
                             title="")

    # Prais class
    from statsmodels.iolib.table import SimpleTable
    from statsmodels.iolib.summary import forg
    if isinstance(self.model, Prais) and hasattr(self,'rho'):
        x = [['%.6f' % self.rho], ['%.6f' % self.dw_orig], ['%.6f' % durbin_watson(self.wresid)]]
        mystubs = ['rho', 'Durbin-Watson statistic (original) ', 'Durbin-Watson statistic (transformed) ']
        tbl = SimpleTable(x, stubs = mystubs)
        smry.tables.append(tbl)

    if hasattr(self, 'converged') and not self.converged:
        smry.tables.append(SimpleTable([["* convergence not achieved"]]))

    # xtreg
    if hasattr(self, 'command') and self.command=='xtreg':
        rsq = self.xt_rsq
        fmt = '%.4f'
        tbl = SimpleTable(
            [[''],
             [fmt % rsq['r2_w']],
             [fmt % rsq['r2_b']],
             [fmt % rsq['r2_o']]],
            stubs = ['R-squared:',
                     ' Within  =',
                     ' Between =',
                     ' Overall ='],
        )
        if hasattr(self, 'sigma_e'):
            fmt = '%.5f'
            tbl2 = SimpleTable(
                [[fmt % self.sigma],
                 [fmt % self.sigma_u],
                 [fmt % self.sigma_e],
                 [fmt % self.lamb]],
                stubs = [' sigma   =',
                         ' sigma_u =',
                         ' sigma_e =',
                         ' lambda  =']
            )
            tbl.extend_right(tbl2)

        tbl2 = SimpleTable(
            [[''],
             ['{:>10,}'.format(int(self.ntotobs))],
             ['{:>10,}'.format(int(self.ngroups))],
             ['']],
            stubs = [' Number of',
                     '  obs    =',
                     '  groups =', '']
        )
        tbl.extend_right(tbl2)
        tbl2 = SimpleTable(
            [[''],
             ['{:>6,}'.format(int(self.tmin))],
             ['{:,.1f}'.format(self.tavg).rjust(6)],
             ['{:>6,}'.format(int(self.tmax))]
            ],
            stubs = [' Obs per group:',
                     '          min =',
                     '          avg =',
                     '          max =']
        )
        tbl.extend_right(tbl2)
        smry.tables.append(tbl)

    # add warnings/notes, added to text format only
    etext = []
    if not self.k_constant:
        etext.append(
            "R² is computed without centering (uncentered) since the "
            "model does not contain a constant."
        )
    if hasattr(self, 'cov_type'):
        etext.append(self.cov_kwds['description'])
    if self.model.exog.shape[0] < self.model.exog.shape[1]:
        wstr = "The input rank is higher than the number of observations."
        etext.append(wstr)
    if eigvals[-1] < 1e-10:
        wstr = "The smallest eigenvalue is %6.3g. This might indicate "
        wstr += "that there are\n"
        wstr += "strong multicollinearity problems or that the design "
        wstr += "matrix is singular."
        wstr = wstr % eigvals[-1]
        etext.append(wstr)
    elif condno > 100000:  # TODO: what is recommended? - annoying
        wstr = "The condition number is large, %6.3g. This might "
        wstr += "indicate that there are\n"
        wstr += "strong multicollinearity or other numerical "
        wstr += "problems."
        wstr = wstr % condno
        etext.append(wstr)

    mod = self.model
    if hasattr(mod, 'dropped_names') and len(mod.dropped_names) > 0:
        etext.append(
            f'{", ".join(self.model.dropped_names)} '
            'omitted because of collinearity.'
        )

    if etext:
        etext = [f"{i + 1}. {text}"
                 for i, text in enumerate(etext)]
        etext.insert(0, "Notes:")
        smry.add_extra_txt(etext)

    return smry
