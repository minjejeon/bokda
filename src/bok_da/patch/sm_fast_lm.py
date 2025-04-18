import numpy as np

def _fast_scalar_reg(x,y):
    """(x'y)/(x'x)"""
    return np.sum(x*y)/np.sum(x**2)


def _fast_ols(X,y):
    """inv(X'X)*X'y"""
    Q,R = np.linalg.qr(X)
    return np.linalg.solve(R, np.dot(Q.T, y))
