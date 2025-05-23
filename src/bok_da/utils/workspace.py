import bok_da

# COLOR_RED = '\033[91m'
# COLOR_GREEN = '\033[92m'
# COLOR_YELLOW = '\033[93m'
# COLOR_BLUE = '\033[94m'
# COLOR_MAGENTA = '\033[95m'
# COLOR_CYAN = '\033[96m'
# COLOR_RESET = '\033[0m'  # Reset text formatting
# BOLD_BEGIN = '\033[1m'
# BOLD_END = '\033[0;0m'

# _b_beg_str = '**'
# _b_end_str = '**'
#_b_beg_str = '\033[94m'
_b_beg_str = '\033[1m'
_b_end_str = '\033[0;0m'

def function_exists(function_name):
    return function_name in globals() and callable(globals()[function_name])

def beautifier_strings(beautiful):
    if beautiful:
        return _b_beg_str, _b_end_str
    else:
        return '', ''

def beautifier_function(beautiful):
    if beautiful:
        return lambda x: _b_beg_str + x + _b_end_str
    else:
        return lambda x: x

def is_statsmodels(obj):
    name = obj.__class__.__name__
    if name=='IVResults':
        return False
    elif name=='OLSResults' or name=='RegressionResults':
        return True
    else:
        raise ValueError('Cannot determine the package')

def pretty_number(num, total_length):
    import numpy as np
    import re
    absval = np.abs(num)
    integer_length = int(np.log10(max(1,absval))) + 2
    decimal_length = total_length-1-integer_length
    if absval < 1: decimal_length += 1
    fmt = '%%.%df' % (decimal_length)
    ans = fmt % num
    ans = re.sub('0+$', '', ans)
    if absval < 1 and num > 0:
        ans = ans.replace(' 0.', ' .')
        ans = re.sub(r'^(\s*)0\.', r'\1.', ans)
    elif absval < 1 and num < 0:
        ans = ans.replace('-0.', '-.')
    fmt = '%%%ds' % total_length
    return fmt % ans

def flex_len(x,beautiful=True):
    if beautiful:
        beg_str,end_str = beautifier_strings(beautiful)
        return len(x.replace(beg_str,'').replace(end_str,''))
    else:
        return len(x)

def join_str_lists(*args, beautiful=True):
    # args contain lists or strings
    lens = []
    for arg in args:
        if isinstance(arg,str): lens.append(1)
        else: lens.append(len(arg))
    maxcount = max(lens) # max number of elements in args
    if not len(args): return None

    # begin with list of empty strings
    ans = ['']*maxcount
    for arg in args:
        if isinstance(arg,str): arg = [arg]*maxcount
        n = len(arg) # number of elements in arg
        if n < maxcount:
            arg += ['']*(maxcount-len(arg)) # pad '' if list is shorter
        mlen = max([flex_len(v) for v in arg]) # max strlen
        for j in range(len(arg)):
            n1 = flex_len(arg[j])
            if n1 < mlen:
                arg[j] += ' '*(mlen-n1)
        ans = [x+y for x,y in zip(ans,arg)]
    return ans

def conf_int(level,b,se,dof): # alpha is like 0.95
    from scipy import stats
    import pandas as pd
    if dof is None:
        # Standard normal distribution
        q = stats.norm.ppf(1-level/200.)
    else:
        # t distribution
        q = stats.t.ppf(1-level/200., dof)
    lower = b - q * se
    upper = b + q * se
    return pd.concat([lower,upper], axis=1)

def format_simple_anova(obj,left_len=12, beautiful=True):
    '''Make ANOVA table vector'''
    df_model = obj.df_model
    df_resid = obj.df_resid
    df_total = df_model + df_resid
    ss_model = obj.ess
    ss_resid = obj.ssr
    ss_total = obj.centered_tss
    ms_model = obj.mse_model
    ms_resid = obj.mse_resid
    ms_total = obj.mse_total

    fmt = '%%%ds' % left_len

    _b = beautifier_function(beautiful)

    def fmt_line(title, ss, df, ms):
        s = fmt % title
        s += ' |  ' + _b(pretty_number(ss, 10))
        s += _b('  %8d' % df)
        s += '  ' + _b(pretty_number(ms, 10))
        return s

    sep = ('-'*left_len) + '-+----------------------------------'

    s = []
    s.append((fmt % 'Source') + " |       SS           df       MS   ")
    s.append(sep)
    s.append(fmt_line('Model', ss_model, df_model, ms_model))
    s.append(fmt_line('Residual', ss_resid, df_resid, ms_resid))
    s.append(sep)
    s.append(fmt_line('Total', ss_total, df_total, ms_total))
    return s

def format_reg_sum(obj, items=['nobs'], sep=' = ', pkg='sm', beautiful=True):
    '''Make summary stats string vector'''
    import numpy as np
    robust = obj.cov_type != 'nonrobust'
    s1 = []
    s3 = []
    for item in items:
        if item=='nobs':
            s1.append('Number of obs ')
            s3.append('%8s' % format(obj.nobs, ","))
        elif item=='f':
            s1.append('F(%d, %d)' % (obj.df_model, obj.df_resid))
            s3.append('%8.2f' % obj.fvalue)
        elif item=='pf':
            s1.append('Prob > F')
            s3.append('%8.4f' % obj.f_pvalue)
        elif item=='r2':
            s1.append('R-squared')
            s3.append('%8.4f' % obj.rsquared)
        elif item=='r2a':
            s1.append('Adj R-squared')
            s3.append('%8.4f' % obj.rsquared_adj)
        elif item=='wald_chi2':
            s1.append('Wald ' + obj.f_statistic.dist_name)
            s3.append('%8.2f' % obj.f_statistic.stat)
        elif item=='pchi2':
            s1.append('Prob > chi2')
            s3.append('%8.4f' % obj.f_statistic.pval)
        elif item=='rmse':
            s1.append('Root MSE')
            rmse = np.nan
            if pkg=='sm':
                rmse = np.sqrt(obj.mse_resid)
            elif pkg=='lm':
                rmse = np.sqrt(obj.resid_ss/obj.nobs)
            s3.append(' ' + pretty_number(rmse,7))

    # print("\n".join(join_str_lists(s1, ' = ', s3)))

    b_beg_str, b_end_str = beautifier_strings(beautiful)
    ans = join_str_lists(s1,sep, b_beg_str, s3, b_end_str)
    return ans

def format_coef_tbl(obj, left_len=12, level=95, beautiful=True):
    '''Make coefficient table'''
    import pandas as pd
    from scipy import stats
    import numpy as np
    import re
    line_length = left_len+66
    top_bot = '-'*line_length
    midrule = '-'*left_len + '-+-' + '-'*63
    lhs_fmt = '%%%ds' % left_len
    mod = obj.model
    ans = []

    _b = beautifier_function(beautiful)
    b_beg_str, b_end_str = beautifier_strings(beautiful)
    nb = len(b_beg_str)
    ne = len(b_end_str)

    alpha = level/100
    if obj.cmd=='regress':
        alpha = 1.-alpha
    if obj.cov_type=='cluster':
        s = '(Std. err. adjusted for %s%d%s clusters' % (
            b_beg_str, obj.ngroups, b_end_str
        )
        m = 1
        if obj.grpvar is None:
            s += ' in the loaded data'
        else:
            s += ' in ' + _b(obj.grpvar)
            m += 1
        s += ')'
        ans.append(' '*(line_length-len(s)+m*nb+m*ne)+s)
        # Must calculate conf int again
        if obj.use_t:
            dof = getattr(obj, 'df_resid_inference', obj.ngroups-1)
        else:
            dof = None

        ci = conf_int(level/100., obj.params, obj.bse, dof)
    else:
        ci = obj.conf_int(alpha)

    ans.append(top_bot)
    if obj.cov_type != 'nonrobust':
        ans.append(' '*left_len + ' |               Robust')
    yname = ''
    if hasattr(mod, 'endog_names'): yname = mod.endog_names
    if hasattr(mod, 'dependent'): yname = mod.dependent.labels[1][0]
    dstr = 'z'
    if getattr(obj,'use_t',False):
        dstr = 't'
    ans.append(
        lhs_fmt % yname + 
        f' | Coefficient  std. err.      {dstr}    P>|{dstr}|     ' +
        f'[%g%% conf. interval]' % level
    )
    ans.append(midrule)

    def fmt_line(vname, *args):
        s = lhs_fmt % vname
        s += ' | '
        if len(args):
            s += ' ' + _b(pretty_number(args[0], 9))
            s += '  ' + _b(pretty_number(args[1], 9))
            s += _b('%9.2f' % args[2])
            s += _b('%8.3f' % args[3])
            s += '    ' + _b(pretty_number(args[4], 9))
            s += '   ' + _b(pretty_number(args[5], 9))
        else:
            s += ' (omitted)'
        return s

    cmd = getattr(obj, 'cmd', '')
    if cmd=='regress':
        vlist = mod.full_names
    elif cmd=='ivregress':
        vlist = mod.endog.labels[1] + mod.exog.labels[1]
    else:
        vlist = ['Intercept']
    if vlist[0]=='Intercept':
        vlist = vlist[1:] + vlist[:1]
    icept = 'Intercept'
    if icept in vlist:
        vlist.remove(icept)
        vlist.append(icept)

    dropped_names = getattr(mod, 'dropped_names', [])
    if hasattr(obj, 'bse'):
        stderr = obj.bse
        tstats = obj.tvalues
    elif hasattr(obj, 'std_errors'):
        stderr = obj.std_errors
        tstats = obj.tstats
    for v in vlist:
        if v in dropped_names:
            s = fmt_line(v)
        else:
            s = fmt_line(
                v,
                obj.params[v],
                stderr[v],
                tstats[v],
                obj.pvalues[v],
                ci.loc[v,ci.columns[0]],
                ci.loc[v,ci.columns[1]],
            )
        ans.append(s)

    ans.append(top_bot)
    return ans

def _varname_len(*args, minimum=12):
    xx = []
    for arg in args:
        if isinstance(arg,str):
            xx.append(arg)
        elif isinstance(arg,int):
            xx.append(' '*arg)
        else:
            xx += arg
    lens = [len(v) for v in xx] + [minimum]
    return max(lens)

def disp_regress(obj, **kwargs):
    '''Display RegressionResults'''
    level = kwargs.pop('level', 95)
    beautiful = kwargs.pop('beautiful', True)
    vname_len = _varname_len(obj.model.exog_names, obj.model.endog_names)
    su_items = ['nobs', 'f', 'pf', 'r2', 'r2a', 'rmse']
    if obj.cov_type=='nonrobust':
        s = format_simple_anova(obj, vname_len, beautiful=beautiful)
        sep = '  =  '
    else:
        fmt = '%%-%ds' % (vname_len+33)
        s = [fmt % 'Linear regression']
        su_items.remove('r2a')
        sep = '    =   '

    ans = join_str_lists(
        s,
        '   ',
        format_reg_sum(obj, su_items, sep=sep, beautiful = beautiful)
    )
    ans.append("")
    ans += format_coef_tbl(obj, vname_len, level=level, beautiful = beautiful)
    return ans

def disp_ivregress(obj, **kwargs):
    '''Display RegressionResults'''
    level = kwargs.pop('level', 95)
    beautiful = kwargs.pop('beautiful', True)
    Xnames = obj.model.exog.labels[1]
    Ynames = obj.model.endog.labels[1]
    Znames = obj.model.instruments.labels[1]
    ynames = obj.model.dependent.labels[1]
    vname_len = _varname_len(Xnames, Ynames, ynames)
    fmt = '%%-%ds' % (vname_len+33)
    s = [fmt % 'Instrumental variable 2SLS regression']

    su_items = ['nobs', 'wald_chi2', 'pchi2', 'r2', 'rmse']
    ans = join_str_lists(
        s, '     ',
        format_reg_sum(
            obj, su_items, sep='  =   ', pkg='lm', beautiful=beautiful
        )
    )
    ans.append("")
    ans += format_coef_tbl(obj, vname_len, level=level, beautiful=beautiful)
    exog = Xnames+Znames
    exog.remove('Intercept')
    _b = beautifier_function(beautiful)
    ans.append('Endogenous: ' + _b(" ".join(Ynames)))
    ans.append('Exogenous:  ' + _b(" ".join(exog)))
    return ans

def insert_extra_bold(text):
    import re
    text = re.sub(
        r' F\((.*?),( ?)(.*?)\)',
        f' F({_b_beg_str}\\1{_b_end_str},\\2{_b_beg_str}\\3{_b_end_str})',
        text
    )
    text = re.sub(
        r' chi2\((.*?)\)',
        f' chi2({_b_beg_str}\\1{_b_end_str})',
        text
    )
    return text

def disp_eval_forecasts(obj, **kwargs):
    '''Display forecast performance'''
    import re
    beautiful = kwargs.pop('beautiful', True)
    beg_str,end_str = beautifier_strings(beautiful)
    my_keys = list(obj.keys())
    m = len(obj[my_keys[0]])
    hdr = ['%9s' % ' Predictor', ' |'] + ['%10s' % s for s in my_keys]
    hdr[2] = hdr[2][1:]
    hdr = "".join(hdr) + " "
    hdr = re.sub(' (MA?PE) ', ' \\1%', hdr)
    sep = '-'*10 + '-+-' + '-'*60
    lft = [' %5d    ' % k for k in range(m)]
    lft = join_str_lists(lft, [' |']*m)
    tbl = ''
    for key in my_keys:
        val = obj[key]
        if val is None:
            tmp = ['     .  ']*m
        else:
            tmp = [pretty_number(x,8) for x in val]
        tbl = join_str_lists(tbl, '  ', tmp)

    ans = [hdr, sep] + join_str_lists(lft, beg_str, tbl, end_str)
    return ans

def disp_dm_test(obj, **kwargs):
    '''Display Diebold-Mariano test results'''
    beautiful = kwargs.pop('beautiful', True)
    beg_str, end_str = beautifier_strings(beautiful)
    stats = obj['stats']
    pvals = obj['pvals']
    vlist = stats.columns.tolist()
    vname_len = max(max([len(s) for s in vlist]), 8)
    lhs_fmt = '%%%ds' % vname_len
    top_bot = '-'*(vname_len + 3 + 12*len(vlist))

    def get_value(key, default='default', func = lambda x: x):
        if obj[key] is None:
            return default
        else:
            return func(obj[key])

    alternative = get_value('alternative')
    h = get_value('h', func = lambda x: "%d" % x)
    pow = get_value('power', func = lambda x: "%d" % x)

    s1 = ['Diebold-Mariano tests with ']
    s2 = ["alternative", "    horizon", "      power"]
    fmt_tmp = "%s"
    s3 = [
        fmt_tmp % alternative,
        fmt_tmp % h,
        fmt_tmp % pow
    ]
    ans = s1
    lines = join_str_lists(s2, ' = ', beg_str, s3, end_str)
    ans.append('  ' + "  ".join([s.strip() for s in lines]))
    #ans.append('')
    ans.append(top_bot)
    ans.append(
        (lhs_fmt % 'DM test') + ' | ' + ''.join([' %8s   ' % s for s in vlist])
    )
    ans.append('-'*vname_len + '-+-' + '-'*(12*len(vlist)))
    for v1 in vlist:
        line = (lhs_fmt % v1) + ' | ' + beg_str
        for v2 in vlist:
            if v1==v2:
                line += ' '*12
            else:
                s = stats.loc[v1,v2]
                p = pvals.loc[v1,v2]
                star = ''
                star += ('*' if p < 0.10 else ' ')
                star += ('*' if p < 0.05 else ' ')
                star += ('*' if p < 0.01 else ' ')
                line += ' ' + pretty_number(s,8) + star
        line += end_str
        ans.append(line)
    ans.append(top_bot)
    ans.append('note. ***p<0.01, **p<0.05, *p<0.10')
    return ans

def disp_cw_test(obj, **kwargs):
    '''Display Clark-West test results'''
    beautiful = kwargs.pop('beautiful', True)
    beg_str, end_str = beautifier_strings(beautiful)
    stats = obj['stats']
    pvals = obj['pvals']
    vlist = stats.columns.tolist()
    vname_len = max(max([len(s) for s in vlist]), 8)
    lhs_fmt = '%%%ds' % vname_len
    top_bot = '-'*(vname_len + 3 + 12*len(vlist))

    def get_value(key, default='default', func = lambda x: x):
        if obj[key] is None:
            return default
        else:
            return func(obj[key])

    s1 = ['Clark-West tests']
    ans = s1
    ans.append(top_bot)
    ans.append(
        (lhs_fmt % 'CW test') + ' | ' + ''.join([' %8s   ' % s for s in vlist])
    )
    ans.append('-'*vname_len + '-+-' + '-'*(12*len(vlist)))
    for v1 in vlist:
        line = (lhs_fmt % v1) + ' | ' + beg_str
        for v2 in vlist:
            if v1==v2:
                line += ' '*12
            else:
                s = stats.loc[v1,v2]
                p = pvals.loc[v1,v2]
                star = ''
                star += ('*' if p < 0.10 else ' ')
                star += ('*' if p < 0.05 else ' ')
                star += ('*' if p < 0.01 else ' ')
                line += ' ' + pretty_number(s,8) + star
        line += end_str
        ans.append(line)
    ans.append(top_bot)
    ans.append('note. ***p<0.01, **p<0.05, *p<0.10')
    return ans

class Workspace:
    '''BOK workplace'''

    def __init__(self, verbose=True, beautiful=True, **kwargs):
        self._verbose = kwargs.pop('verbose', verbose)

        # data
        data = kwargs.pop('using', None)
        if data is None:
            data = kwargs.pop('data', None)
        else:
            if not isinstance(data, str):
                import inspect
                frame = inspect.currentframe().f_back
                argvals = inspect.getargvalues(frame).locals
                data_arg_name = [k for k, v in argvals.items() if v is data]
                if len(data_arg_name):
                    kwargs['data_name'] = data_arg_name[0]

        self.use(data, **kwargs)

        # others
        self.beautiful = beautiful
        self.results = None

    def __repr__(self):
        if self.data is None:
            s = 'Empty workspace'
        else:
            s = (
                'Workspace with data with '
                f'{self.data.shape[0]} rows of {self.data.shape[1]} variables'
            )
            if self.data_name is not None:
                s += f' (file: {self.data_name})'

        s += f'; {"verbose" if self.verbose else "quiet"}'
        return s

    def set_attr(self, **kwargs):
        for key,val in kwargs.items():
            setattr(self, key, val)

    def push_results(self, obj):
        self.results = obj

    def clear_results(self):
        self.results = None

    @property
    def last_results(self):
        return self.results

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        self._verbose = value

    def use(self, data, **kwargs):
        import pandas as pd
        if data is None:
            self.clear_results()
            self.data = data
            return

        if isinstance(data, str):
            self.data_name = data
            if data.endswith('.dta'):
                self.data = pd.read_stata(data, **kwargs)
            elif data.endswith('.xlsx'):
                self.data = pd.read_excel(data, **kwargs)
            elif data.endswith('.csv'):
                self.data = pd.read_csv(data, **kwargs)
            else:
                raise ValueError(f'Cannot read {data}')
        else:
            self.data_name = kwargs.pop('data_name', None)
            self.data = data

        self.clear_results()

    def set_stata(self, path, edition, splash=None, **kwargs):
        import warnings
        from .cstata import Stata
        if splash is None: splash = self.verbose
        self.stata = Stata(path, edition, splash)
        self.stata.get_ready()
        if not kwargs.get('verbose', self.verbose):
            self.stata.be_quiet()

        clear = kwargs.get('clear', False)
        if clear:
            self.stata.run('clear all')

        if kwargs.get('load', False):
            if self.data is None:
                warnings.warn('Workspace has no data to load.')
            else:
                self.stata.use(self.data, force = True)
                if self.verbose:
                    print('Data in workspace has been loaded into Stata.')

    def regress(self, *args, **kwargs):
        '''OLS and WLS'''
        kwargs.setdefault('data', self.data)
        level = kwargs.pop('level',95)
        verbose = kwargs.pop('verbose', self.verbose)
        beautiful = kwargs.pop('beautiful', self.beautiful)
        ans = bok_da.linear.lm.regress(*args, **kwargs)
        if verbose:
            print("")
            text = "\n".join(
                disp_regress(ans, level=level, beautiful=beautiful)
            )
            if beautiful:
                text = insert_extra_bold(text)
                #text = bok.md2term(text)
            print(text)
        self.push_results(ans)
        return ans

    def ivregress(self, *args, **kwargs):
        '''2SLS and LIML'''
        kwargs.setdefault('data', self.data)
        level = kwargs.pop('level', 95)
        verbose = kwargs.pop('verbose', self.verbose)
        beautiful = kwargs.pop('beautiful', self.beautiful)
        ans = bok_da.linear.lm.ivregress(*args, **kwargs)
        if verbose:
            print("")
            text = "\n".join(
                disp_ivregress(ans, level=level, beautiful=beautiful)
            )
            if beautiful:
                text = insert_extra_bold(text)
                #text = bok.md2term(text)
            print(text)
        self.push_results(ans)
        return ans

    def col_evaluation_metrics(self, target, *args, **kwargs):
        '''Evaluate forecasts'''
        #import bok.forecast as fcast
        import bok_da
        from bok_da.valid.pred_perf import col_evaluation_metrics
        verbose = kwargs.pop('verbose', self.verbose)
        beautiful = kwargs.pop('beautiful', self.beautiful)
        #ans = fcast.eval_forecasts(target, *args, **kwargs)
        ans = col_evaluation_metrics(target, *args, **kwargs)
        if verbose and function_exists('disp_eval_forecasts'):
            print('')
            print("\n".join(disp_eval_forecasts(ans, beautiful=beautiful)))
        self.push_results(ans)
        return ans

    def diebold_mariano(self, target, *args, **kwargs):
        '''Diebold-Mariano test'''
        import bok_da
        from bok_da.valid.pred_perf import diebold_mariano
        verbose = kwargs.pop('verbose', self.verbose)
        beautiful = kwargs.pop('beautiful', self.beautiful)
        ans = diebold_mariano(target, *args, **kwargs)
        if verbose:
            if function_exists('disp_dm_test'):
                print('')
                print("\n".join(disp_dm_test(ans, beautiful=beautiful)))
            else:
                print('note. Display function for dm_test not implemented.')
        self.push_results(ans)
        return ans

    def clark_west(self, target, *args, **kwargs):
        '''Clark-West test'''
        import bok_da
        from bok_da.valid.pred_perf import clark_west
        verbose = kwargs.pop('verbose', self.verbose)
        beautiful = kwargs.pop('beautiful', self.beautiful)
        ans = clark_west(target, *args, **kwargs)
        if verbose:
            if function_exists('disp_cw_test'):
                print('')
                print("\n".join(disp_cw_test(ans, beautiful=beautiful)))
            else:
                print('note. Display function for cw_test not implemented.')
        self.push_results(ans)
        return ans
