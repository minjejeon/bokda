def _check(X, vlist, eps=1e-12, pivoting=False, ret_names=True):
    """Check collinearity"""
    import numpy.linalg
    import scipy.linalg
    if X is None:
        return {'keep': None, 'drop': None}

    if pivoting:
        R,P = scipy.linalg.qr(X, mode='r', pivoting=True)
        keep = abs(R.diagonal()) > eps
        to_keep = list(P[keep])
        to_drop = list(P[keep==False])
        to_keep.sort()
        to_drop.sort()
    else:
        _,R = numpy.linalg.qr(X)
        keep = abs(R.diagonal()) > eps
        keep = list(keep)
        to_keep = [i for i,val in enumerate(keep) if val]
        to_drop = [i for i,val in enumerate(keep) if not val]

    # idx = [(i in to_keep) for i,val in enumerate(vlist)]

    if ret_names:
        to_keep = [vlist[i] for i in to_keep]
        to_drop = [vlist[i] for i in to_drop]

    return to_keep,to_drop,vlist
    #return {'keep': to_keep, 'drop': to_drop, 'full': vlist}

def check(obj, data=None, eps=1e-12, pivoting=False, ret_names=True):
    """Check collinearity"""
    import pandas
    import patsy
    import numpy
    what = type(obj).__name__
    if isinstance(obj, str):
        _,X = patsy.dmatrices(obj, data, return_type="dataframe")
        vlist = X.columns.tolist()
    elif isinstance(obj, numpy.matrix) or isinstance(obj, numpy.ndarray):
        X = obj
        vlist = list(range(obj.shape[1]))
    elif isinstance(obj, pandas.DataFrame):
        X = obj
        vlist = obj.columns.tolist()
    elif what=='OLS' or what=='WLS' or what=='GLS':
        X = obj.exog
        vlist = obj.exog_names
    elif what=='DesignMatrix':
        X = obj
        vlist = obj.design_info.column_names
    else:
        raise Exception("Cannot handle this type.")

    return _check(X,vlist,eps=eps,pivoting=pivoting,ret_names=ret_names)

