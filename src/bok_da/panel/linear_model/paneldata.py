import statsmodels.api as sm
import pandas as pd
import numpy as np
#from ...more import collinearity
from bok_da.patch import collinearity

# def _cor(x,y):
#     """Pearson correlation coefficient"""
#     import numpy
#     return numpy.corrcoef(x.squeeze(), y.squeeze())[0,1]

def _cor_sq(x,y):
    """Pearson correlation coefficient squared"""
    import numpy as np
    import pandas as pd
    if isinstance(x, pd.DataFrame): x = x.to_numpy()
    if isinstance(y, pd.DataFrame): y = y.to_numpy()
    x = x.flatten()
    y = y.flatten()
    x = x - np.mean(x)
    y = y - np.mean(y)
    n = len(y)
    return np.dot(x,y)**2/np.sum(np.square(x))/np.sum(np.square(y))

def _count_t(x):
    from collections import Counter
    a = Counter(x).values()
    return min(a), sum(a), max(a), len(a)

def _set_extra_info(obj,id_vec, **kwargs):
    tmin,tsum,tmax,ngrps = _count_t(id_vec)
    obj.set_extra_attributes(
        ntotobs = tsum,
        ngroups = ngrps,
        tmin = tmin,
        tavg = tsum/ngrps,
        tmax = tmax,
        **kwargs
    )

def _get_xt_rsq(obj,y,X,id_vec):
    if hasattr(obj.model, 'dropped_index') and obj.model.dropped_index:
        X = X.iloc[:,obj.model.kept_index]
    yhat = pd.DataFrame({'yhat': np.matmul(X,obj.params)})
    ybar,yhatbar = xtmean(y,yhat,id_vec)
    yd,yhatd = xtdev(y,yhat,id_vec)
    rsq = {}
    rsq['r2_w'] = _cor_sq(yd,yhatd)
    rsq['r2_b'] = _cor_sq(ybar, yhatbar)
    rsq['r2_o'] = _cor_sq(y, yhat)
    return rsq

def _to_dataframe(x):
    if isinstance(x, pd.Series):
        x = id.to_frame()
    elif isinstance(x, list):
        x = pd.DataFrame(x)
    elif isinstance(x, np.ndarray):
        if x.ndim==1:
            x = pd.DataFrame(id)
        elif x.ndim==2 and x.shape[1]==1:
            x = pd.DataFrame(x[:,0])
        else:
            raise ValueError('Invalid id argument dimension')
    elif not isinstance(id, pd.DataFrame):
        raise ValueError('Unsupported id argument type')
    return x

def _concat(*args):
    dfs = []
    for arg in args:
        if isinstance(arg, pd.Series):
            arg = arg.to_frame()
        elif isinstance(arg, np.ndarray):
            arg = pd.DataFrame(arg)
        dfs.append(arg)
    return pd.concat(dfs, axis=1)

def xtdev(y, X, id_vec):
    """Return within-group deviations"""
    if y is None or not len(y):
        w = pd.concat([id_vec, X], axis=1)
    else:
        w = pd.concat([id_vec, y, X], axis=1)
    grouped = w.groupby(id_vec.name)
    #wd = grouped.transform(lambda x: x-x.mean())
    wd = w-grouped.transform('mean')
    if y is None or not len(y):
        return wd
    else:
        yvar = y.columns.tolist() if isinstance(y, pd.DataFrame) else y.name
        return (wd[yvar], wd[X.columns.tolist()])

def xtmean(y, X, id_vec, expand = False):
    """Return group means"""
    w = pd.concat([id_vec, y, X], axis=1)
    grouped = w.groupby(id_vec.name)
    wm = grouped.transform('mean') if expand else grouped.mean()
    if y is None or not len(y):
        return wm
    else:
        yvar = y.columns if isinstance(y, pd.DataFrame) else y.name
        return (wm[yvar], wm[X.columns.tolist()])

class PanelData:
    """Panel data analysis"""

    def __init__(self, data=None, panelvar=None, timevar=None, delta=1):
        self.data = None
        self.panelvar = None
        self.timevar = None
        self.delta = 0
        self.oordvar = '--orig-order--'
        if not data is None: self.use(data)
        if panelvar is not None:
            self.xtset(panelvar, timevar, delta)

    def __repr__(self):
        if self.data is None:
            s = 'Empty PanelData class. Use use.'
        else:
            s = (
                'PanelData class with '
                f'{len(self.data)}x{self.data.shape[1]-1} '
                'data set'
            )
            if self.is_xtset():
                s = s + f'. panelvar = "{self.panelvar}", '
                s = s + f'timevar = "{self.timevar}" (delta = {self.delta})'
            else:
                s = s + '. Still need xtset.'
        return s

    def __str__(self):
        return self.__repr__()

    def use(self, df):
        import os
        if isinstance(df, str):
            if os.path.isfile(df):
                if df.endswith('.dta'):
                    df = pd.read_stata(df)
                elif df.endswith('.csv'):
                    df = pd.read_csv(df)
                elif df.endswith('.xlsx'):
                    df = pd.read_excel(df)
                elif df.endswith('.xls'):
                    df = pd.read_excel(df)
                else:
                    print(f'*** Cannot read {df}. Stop.')
                    return

        self.data = df

    def xtset(self, panelvar, timevar=None, delta = 1):
        if self.data is None:
            print('*** No data set loaded. Stop. Use -use-')
            return

        varlist = self.data.columns.tolist()
        if panelvar in varlist:
            self.panelvar = panelvar
        else:
            print(f'*** {panelvar} is not in data set. Stop.')
            return

        if timevar is not None:
            if timevar in varlist:
                self.timevar = timevar
                self.delta = delta
            else:
                print(f'*** {timevar} is not in data set. Stop.')
                return

        self.data[self.oordvar] = range(len(self.data))
        if timevar is None:
            self.data.sort_values(by=[self.panelvar])
        else:
            self.data.sort_values(by=[self.panelvar, self.timevar])

    def xtunset(self):
        self.panelvar = None
        self.timevar = None
        self.delta = 0
        self.data.sort_values(by=[self.oordvar], ascending = True)
        del self.data[self.oordvar]

    def is_xtset(self):
        return self.panelvar is not None and self.data is not None

    def _xtsum(self, variable):
        df = self.data[~np.isnan(self.data[variable].to_numpy())]
        df = df[[self.panelvar, self.timevar, variable]]
        ans = {}

        ans['N'] = len(df)
        ans['n'] = len(set(df[self.panelvar]))
        a = df.groupby(self.panelvar).count()
        ans['Tbar'] = np.mean(a[self.timevar])

        ## Overall
        x = df[variable].to_numpy()
        ans['mean'] = np.mean(x)
        ans['sd'] = np.std(x)
        ans['min'] = np.min(x)
        ans['max'] = np.max(x)

        ## Between
        xb = df.groupby(self.panelvar).mean()[variable]
        ans['sd_b'] = np.std(xb)
        ans['min_b'] = np.min(xb)
        ans['max_b'] = np.max(xb)

        ## Within
        m = df.groupby(self.panelvar)[variable].transform('mean')
        xw = x-m+ans['mean']
        ans['sd_w'] = np.std(xw)
        ans['min_w'] = np.min(xw)
        ans['max_w'] = np.max(xw)

        return ans

    def xtsum(self, varlist = None):
        if not self.is_xtset():
            print('*** xtset first.')
            return None
        if varlist is None:
            return None

        if isinstance(varlist, str): varlist = [varlist]

        ans = {}
        for v in varlist: ans[v] = self._xtsum(v)

        return ans

    @classmethod
    def hausman(self, fe, re, sigma='less'):
        '''
        Hausman test for FE vs RE
        '''
        import scipy.stats

        if re.cov_type != 'nonrobust' or fe.cov_type != 'nonrobust':
            raise ValueError(
                'Hausman test cannot be combined with robust se.'
            )

        vcov1 = fe.cov_params()
        vcov0 = re.cov_params()
        vlist1 = vcov1.columns.tolist()
        vlist0 = vcov0.columns.tolist()
        vlist1.remove('Intercept')
        vv = [elem for elem in vlist1 if elem in vlist0]

        b1 = fe.params[vv]
        b0 = re.params[vv]
        bdiff = b1-b0

        scale1 = 1
        scale0 = 1

        if sigma=='more': scale1 = re.scale / fe.scale
        if sigma=='less': scale0 = fe.scale / re.scale

        vdiff = scale1*vcov1.loc[vv,vv] - scale0*vcov0.loc[vv,vv]
        _df = np.linalg.matrix_rank(vdiff.values, tol=1e-9)

        _bd = bdiff.values
        _Vd = vdiff.values
        _se = np.sqrt(np.diag(_Vd))

        chi2 = np.dot(np.dot(_bd.T, np.linalg.pinv(_Vd, rcond=1e-9)), _bd)

        return {
            'rank': _df,
            'chi2': chi2,
            'df': _df,
            'p': scipy.stats.chi2.sf(chi2,_df)
        }

def _xtreg_pols(
        y, X, id_vec, vce='o', return_rsq=True, **kwargs
):
    """Pooled OLS"""
    model = sm.OLS(y,X)
    model.set_extra_attributes(title='Pooled OLS Regression')
    if vce=='o':
        ans = model.fit(**kwargs)
    elif vce=='r':
        ans = model.fit(
            cov_type='cluster',
            cov_kwds={'groups': id_vec},
            **kwargs
        )
    else:
        ans = model.fit(cov_type=vce, **kwargs)

    # R-squareds
    rsq = _get_xt_rsq(ans, y, X, id_vec) if return_rsq else {}
    _set_extra_info(ans, id_vec, method = 'pols', xt_rsq = rsq)
    return ans

def _xtreg_be(
        y, X, id_vec, vce='o', return_rsq=True, **kwargs
):
    """Between-group regression"""
    (ybar, Xbar) = xtmean(y, X, id_vec)

    model = sm.OLS(ybar, Xbar)
    model.set_extra_attributes(title = 'Between-effects Regression')
    if vce=='o':
        ans = model.fit(method='qr', **kwargs)
    elif vce=='r':
        ans = model.fit(method='qr', cov_type='HC1', **kwargs)
    else:
        ans = model.fit(method='qr', cov_type=vce, **kwargs)

    # R-squareds
    rsq = _get_xt_rsq(ans, y, X, id_vec) if return_rsq else {}
    _set_extra_info(ans, id_vec, method = 'be', xt_rsq = rsq)
    return ans

def _xtreg_fe(
        y, X, id_vec, vce='o', return_rsq=True, **kwargs
):
    """Within-group (FE) regression"""
    # Within-group deviations
    yd,Xd = xtdev(y,X,id_vec)
    # Intercept
    yd = yd + y.mean()
    for v in Xd.columns: Xd[v] = Xd[v] + X[v].mean()

    model = sm.OLS(yd,Xd)
    model.set_extra_attributes(title = 'Fixed-effects Regression')

    # Adjust degrees of freedom
    n1 = len(set(id_vec))-1
    model.df_model = model.df_model+n1
    model.df_resid = model.df_resid-n1

    if vce=='o':
        ans = model.fit(method='qr', **kwargs)
    elif vce=='r':
        ans = model.fit(
            method='qr',
            cov_type='cluster',
            cov_kwds={'groups': id_vec},
            **kwargs
        )
    else:
        ans = model.fit(method='qr', cov_type=vce, **kwargs)

    s2e = ans.scale
    X1 = X
    if hasattr(model, 'dropped_index') and model.dropped_index:
        X1 = X.iloc[:,model.kept_index]
    resid = y.squeeze() - X1.dot(ans.params.squeeze())
    u_i = resid.groupby(id_vec).mean()
    u_i = u_i - np.mean(u_i)
    s2u = np.dot(u_i,u_i) / (len(u_i)-1)
    s2 = s2u + s2e

    # R-squareds
    rsq = _get_xt_rsq(ans, y, X, id_vec) if return_rsq else {}
    _set_extra_info(
        ans, id_vec, method='fe', xt_rsq=rsq,
        sigma_u=np.sqrt(s2u),
        sigma_e=np.sqrt(s2e),
        sigma=np.sqrt(s2),
        lamb=s2u/(s2u+s2e),
    )
    return ans

def _xtreg_re(
        y, X, id_vec, vce='o', return_rsq=True, verbose=True,
        **kwargs
):
    """Random-effects FGLS"""
    fe = _xtreg_fe(y,X,id_vec, return_rsq=False, quiet=True)
    s2e = fe.scale
    be = _xtreg_be(y,X,id_vec, return_rsq=False, quiet=True)
    s2b = be.scale
    n = len(set(id_vec))
    Ti = id_vec.groupby(id_vec).count()
    T_hbar = n/np.sum(1/Ti)
    Ti_exp = id_vec.groupby(id_vec).transform('count')
    s2u = np.max([0,s2b-s2e/T_hbar])
    theta = 1-1/np.sqrt(1+Ti_exp.to_numpy()*(s2u/s2e))

    ym,Xm = xtmean(y,X,id_vec, expand=True)
    Ti_exp = pd.DataFrame(Ti_exp)
    yre = y.sub(ym.mul(theta, axis=0), axis=0)
    Xre = X.sub(Xm.mul(theta, axis=0), axis=0)

    model = sm.OLS(yre,Xre)
    model.set_extra_attributes(title = 'Random-effects Regression')

    if vce=='o':
        ans = model.fit(method='qr', **kwargs)
    elif vce=='r':
        ans = model.fit(
            method='qr',
            cov_type='cluster',
            cov_kwds={'groups': id_vec},
            **kwargs,
        )
    else:
        ans = model.fit(method='qr', cov_type=vce, **kwargs)

    # R-squareds
    rsq = _get_xt_rsq(ans, y, X, id_vec) if return_rsq else {}
    _set_extra_info(
        ans, id_vec, method = 're', xt_rsq = rsq,
        theta = theta,
        sigma_e = np.sqrt(s2e), sigma_u = np.sqrt(s2u),
        sigma = np.sqrt(s2u+s2e),
        lamb = s2u/(s2u+s2e)
    )
    return ans

def xtreg(formula, xtdata, method = 'fe', subset=None, vce='o', **kwargs):
    '''POLS, be, fe, re'''
    import patsy
    df = xtdata.data if subset is None else xtdata.data[subset]
    y,X = patsy.dmatrices(formula, df, return_type = 'dataframe')
    id_vec = xtdata.data.loc[X.index, xtdata.panelvar]
    if method=='pols' or method=='POLS':
        ans = _xtreg_pols(y, X, id_vec, vce=vce, **kwargs)
    elif method=='be' or method=='BE':
        ans = _xtreg_be(y, X, id_vec, vce=vce, **kwargs)
    elif method=='fe' or method=='FE':
        ans = _xtreg_fe(y, X, id_vec, vce=vce, **kwargs)
    elif method=='re' or method=='RE':
        ans = _xtreg_re(y, X, id_vec, vce=vce, **kwargs)
    else:
        raise ValueError(f'method={method} is not implemented.')
    ans.set_extra_attributes(
        xt = xtdata,
        command = 'xtreg'
    )
    return ans
