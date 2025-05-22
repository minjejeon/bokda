import pandas as pd
import numpy as np
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

_b_beg_str = '\033[1m'
_b_end_str = '\033[0;0m'

def beautifier_strings(beautiful):
    if beautiful:
        return _b_beg_str, _b_end_str
    else:
        return '', ''
    
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

def flex_len(x,beautiful=True):
    if beautiful:
        beg_str,end_str = beautifier_strings(beautiful)
        return len(x.replace(beg_str,'').replace(end_str,''))
    else:
        return len(x)
    
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

def _args_to_dataframe(args):
    '''
    Function to use to convert *args to a pandas dataframe
    '''
    import numpy as np
    import pandas as pd

    def local_list_to_df(x):
        data = {f'y{i+1}': arg for i, arg in enumerate(x)}
        df = pd.DataFrame(data)
        df.columns = range(len(df.columns))
        return df

    if len(args)==0:
        return None
    elif isinstance(args[0], list):
        #print('args[0] is a list')
        if isinstance(args[0][0], (np.ndarray, list)):
            return local_list_to_df(args[0])
        elif isinstance(args[0][0], pd.Series):
            return pd.concat(args[0], axis=1)
    elif isinstance(args[0], pd.DataFrame):
        #print('1st arg is a pandas dataframe')
        return args[0]
    elif isinstance(args[0], pd.Series):
        #print('1st arg is a pandas series')
        return pd.concat(args, axis=1)
    elif isinstance(args[0], np.ndarray):
        if len(args[0].shape)==1:
            #print('1st arg is a one-dimensional numpy ndarray')
            return local_list_to_df(args)
        else:
            #print('1st arg is a multi-dimensional numpy ndarray')
            return pd.DataFrame(args[0])
    warnings.warning('Something wrong')
    return None

def _acf(x, lag_max=None):
    n = len(x)
    if lag_max is None:
        lag_max = n-1
    xbar = np.mean(x)
    if isinstance(x, (np.ndarray, pd.Series)):
        xd = x-xbar
    else:
        xd = [x1-xbar for x1 in x]
    lag_max = min(lag_max, n-1)
    lag_max += 1
    ans = pd.Series(index=range(lag_max))
    for j in range(lag_max):
        xd1 = xd[j:]
        xd2 = xd[:(n-j)]
        ans[j] = np.sum([x*y for x,y in zip(xd1,xd2)])
    return ans/float(n)

def _dm(e1, e2, alternative='two', h=1, power=2, varestimator='acf'):
    
    from scipy.stats import t
    h = int(h)
    if h < 1:
        raise ValueError('h must be at least 1')
    if h > len(e1):
        raise ValueError(
            'h cannot be longer than the number of forecast errors'
        )
    d = np.abs(e1)**power - np.abs(e2)**power
    n = len(d)
    d_cov = _acf(d, lag_max=h-1)
    if varestimator=='acf' or h==1:
        d_var = (d_cov[0] + 2*np.sum(d_cov[1:]))/n
    else:
        d_var = (d_cov[0] + 2*np.sum(pd.Series(1-range(h)/n)*d_cov[1:]))/n

    if d_var > 0:
        stat = np.mean(d)/np.sqrt(d_var)
    elif h==1:
        raise ValueError('Variance of DM statistic is zero')
    else:
        warnings.warning(
            'Variance is negative. Try varestimator==bartlett. '
            'Proceeding with horizon h=1.'
        )
        return _dm(e1, e2, alternative, 1, power, varestimator)

    k = np.sqrt(((n+1-2*h+h/n*(h-1))/n))
    stat = stat*k
    if alternative.startswith('two'):
        pval = 2*t.cdf(-np.abs(stat), n-1)
    elif alternative=='less':
        pval = t.cdf(stat, n-1)
    elif alternative=='greater':
        pval = 1-t.cdf(stat, n-1)
    return stat,pval

def _cw(y, pred0, pred1, vcov_func = np.var):
    from scipy.stats import norm
    if y is not None and isinstance(y,list): y = np.array(y)
    n = len(pred0)
    if isinstance(pred0,list): pred0 = np.array(pred0)
    if isinstance(pred1,list): pred1 = np.array(pred1)
    if y is not None:
        pred0 = pred0 - y
        pred1 = pred1 - y
    x = pred0**2 - pred1**2 + (pred0-pred1)**2
    mu = x.mean()
    se = np.sqrt(vcov_func(x)/n)
    z_stat = mu/se
    pval = 1-norm.cdf(z_stat)
    return mu, se, z_stat, pval

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
    #lft = [' %5d    ' % k for k in range(m)]
    lft = [f'  Model {k+1}' for k in range(m)]
    lft = join_str_lists(lft, ['  |']*m)
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

def col_evaluation_metrics(y, *args, **kwargs):
    '''
    Evaluate forecasts.
    '''
    print_res = kwargs.pop('print_res', True)
    
    if not len(args):
        return None
    perr = _args_to_dataframe(args)
    # forecast errors
    if y is not None:
        perr = perr.sub(y, axis=0)

    mse = perr.apply(lambda x: (x**2).mean())
    mae = perr.apply(lambda x: np.abs(x).mean())
    rmse = mse.apply(np.sqrt)
    mape = None
    mpe = None
    rsq = None
    if y is not None:
        if isinstance(y, pd.Series):
            y = y.to_numpy()
        elif isinstance(y, list):
            y = np.array(y)
        ppe = perr.div(y, axis=0)
        mape = ppe.apply(lambda x: np.abs(x).mean())
        mpe = ppe.apply(lambda x: x.mean())
        mst = ((y-y.mean())**2).mean()
        rsq = 1-mse/mst

    ans = {
        'MSE': mse.to_numpy(),
        'RMSE': rmse.to_numpy(),
        'MAE': mae.to_numpy(),
        'MAPE': None if mape is None else mape.to_numpy()*100,
        'MPE': None if mpe is None else mpe.to_numpy()*100,
        'Rsq': None if rsq is None else rsq.to_numpy(),
    }
    
    if print_res:
        print("\n".join(disp_eval_forecasts(ans, beautiful=True)))

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
    s3 = [fmt_tmp % alternative, fmt_tmp % h, fmt_tmp % pow]
    
    ans = s1
    lines = join_str_lists(s2, ' = ', beg_str, s3, end_str)
    ans.append('  ' + ", ".join([s.strip() for s in lines]))
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

def diebold_mariano(target, *args, **kwargs):
    
    if len(args)==0:
        warnings.warn('Nothing to do.')
        return None

    print_res = kwargs.pop('print_res', True)
    beautiful = kwargs.pop('beautiful', True)
    
    h = kwargs.get('h', 1)
    power = kwargs.get('power', 2)
    drop = kwargs.pop('drop', False)
    perr = _args_to_dataframe(args)
    
    if target is not None:
        perr = perr.sub(target, axis=0)

    cols = perr.columns
    n = len(cols)
    if n < 2:
        warnings.warn('DM test requres at least two forecasts.')
        return None
    stats = pd.DataFrame(index = cols, columns = cols)
    pvals = stats.copy()
    direc = kwargs.get('alternative', 'two_sided')
    twosided = direc.startswith('two')

    for i in range(1,n):
        e1 = perr.iloc[:,i]
        for j in range(i):
            e2 = perr.iloc[:,j]
            stat,pval = _dm(e1,e2,**kwargs)
            stats.iloc[i,j] = stat
            pvals.iloc[i,j] = pval
            stats.iloc[j,i] = -stat
            pvals.iloc[j,i] = pval if twosided else 1-pval
    
    ans = {'stats': stats, 'pvals': pvals, 'alternative': direc, 'h': h, 'power': power}
    
    if print_res:
        print("\n".join(disp_dm_test(ans, beautiful=beautiful)))
    
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


def clark_west(target, *args, **kwargs):
    """
    Clark-West tests
    """
    print_res = kwargs.pop('print_res', True)
    beautiful = kwargs.pop('beautiful', True)
    if len(args)==0:
        warnings.warn('Nothing to do.')
        return None

    perr = _args_to_dataframe(args)
    if target is not None:
        perr = perr.sub(target, axis=0)

    cols = perr.columns
    n = len(cols)
    if n < 2:
        warnings.warn('CW test requres at least two forecasts.')
        return None
    means = pd.DataFrame(index = cols, columns = cols)
    stderrs = means.copy()
    stats = stderrs.copy()
    pvals = stats.copy()

    for i in range(n):
        e1 = perr.iloc[:,i]
        for j in range(n):
            if i==j: continue
            e2 = perr.iloc[:,j]
            mu,se,stat,pval = _cw(None,e1,e2,**kwargs)
            means.iloc[i,j] = mu
            stderrs.iloc[i,j] = se
            stats.iloc[i,j] = stat
            pvals.iloc[i,j] = pval
    
    ans = {'means': means,
           'stderrs': stderrs,
           'stats': stats,
           'pvals': pvals,}

    if print_res:
        print("\n".join(disp_cw_test(ans, beautiful=beautiful)))

    return ans