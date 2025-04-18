import numpy as np
from statsmodels.regression.linear_model import GLS
from .sm_fast_lm import _fast_scalar_reg, _fast_ols

class Prais(GLS):
    __doc__ = """
    Prais-Winsten AR(1) FGLS and Cochrane-Orcutt Regression
    """

    def __init__(self, endog, exog=None, missing='none', hasconst=None,
                 rho=None, twostep=False, corc=False,
                 rhotype = 'regress', maxiter=100, **kwargs):
        self.corc = corc
        self.twostep = twostep
        self.maxiter = 1 if twostep else maxiter
        self.rho = rho
        self.rhotype = rhotype
        super(Prais, self).__init__(
            endog, exog, missing=missing, hasconst=hasconst, **kwargs
        )
        self.title = 'Prais-Winsten AR(1) regression'

    def _fast_ols(self,corc):
        X = self.wexog
        y = self.wendog
        if self.rho is not None and corc:
            X = X[1:]
            y = y[1:]
        return np.linalg.solve(
            np.dot(X.T, X),
            np.dot(X.T, y)
        )

    def _estimate_rho(self,rhotype,params,n=None,k=None):
        """
        Estimate rho
        """
        if n is None or k is None: n,k = self.exog.shape
        u = self.endog - np.dot(self.exog, params)
        if rhotype=='regress':
            rho = _fast_scalar_reg(u[:-1], u[1:])
        elif rhotype=='freg':
            rho = _fast_scalar_reg(u[1:], u[:-1])
        elif rhotype=='tscorr' or rhotype=='theil':
            rho = np.dot(u[:-1], u[1:])/np.sum(np.square(u))
            if rhotype=='theil': rho = rho*(n-k)/n
        elif rhotype=='dw' or rhotype=='nagar':
            dw = np.sum(np.square(u[1:]-u[:-1]))/np.sum(np.square(u))
            rho = 1.0-dw/2.0
            if rhotype=='nagar': rho = (rho*n**2 + k**2)/(n**2-k**2)
        self.rho = rho

    def iterative_fit(self, corc=None, twostep=None,
                      maxiter=None, rhotype=None, tol=1e-6,
                      quiet=False, **kwargs):
        """
        Iterative fit
        """
        from statsmodels.stats.stattools import durbin_watson
        self.check_collinearity()
        self.quiet = quiet
        if corc is None: corc = self.corc
        if twostep is None: twostep = self.twostep
        if maxiter is None: maxiter = self.maxiter
        if rhotype is None: rhotype = self.rhotype
        if twostep: maxiter = 1

        n,k = self.exog.shape
        if not quiet:
            print(
                'Iteration %d:  rho = %.4f' %
                (0, 0.0 if self.rho is None else self.rho)
            )
        converged = False

        # If self.rho is None, do OLS, maxiter--.
        dw0 = None
        if self.rho is None:
            i = 1
            params = self._fast_ols(corc)
            self._estimate_rho(rhotype, params, n, k)
            if not quiet: print('Iteration %d:  rho = %.4f' % (i, self.rho))
            maxiter -= 1
            u = self.endog - np.dot(self.exog, params)
            dw0 = durbin_watson(u)

        # iteratively estimate
        for i in range(maxiter):
            last_rho = self.rho
            self.initialize()
            params = self._fast_ols(corc)
            self._estimate_rho(rhotype, params, n, k)
            if not quiet: print('Iteration %d:  rho = %.4f' % (i+2, self.rho))
            if dw0 is None:
                u = self.endog - np.dot(self.exog, params)
                dw0 = durbin_watson(u)
            if abs(self.rho-last_rho) < tol:
                converged = True
                break

        self.initialize()
        kwargs['iter'] = i+1
        kwargs['rho'] = self.rho
        kwargs['converged'] = converged
        kwargs['dw_orig'] = dw0
        results = self.fit(**kwargs)
        if corc:
            self.title = 'Cochrane-Orcutt AR(1) regression'
        if twostep:
            self.title += ' with twostep estimates'
        else:
            self.title += ' with iterative estimates'
        return results

    def whiten(self,x):
        x = np.asarray(x, np.float64)
        _x = x.copy()

        if self.rho is not None:
            _x[1:] = x[1:] - self.rho*x[:-1]
            if not self.corc:
                _x[0] = np.sqrt(1-self.rho**2)*x[0]
        if self.rho is not None and self.corc:
            return _x[1:]
        else:
            return _x

