import numpy as np
from scipy.stats import norm

from .operator import rows, sumc, det, is_invertible, chol, lndet1, cols


def lnpdfn(x: np.ndarray, 
           mu: np.ndarray, 
           sig2vec: np.ndarray) -> np.ndarray:
    """
    Log PDF of normal distribution.

    Args:
        x (np.ndarray): Normal variates.
        mu (np.ndarray): Vector of means.
        sig2vec (np.ndarray): Vector of variances.

    Returns:
        y: Computed log probability density function values.
    """
    # if not isinstance(x, np.ndarray) or not isinstance(mu, np.ndarray) or not isinstance(sig2vec, np.ndarray):
    #     raise ValueError("Inputs must be numpy arrays.")
    if not isinstance(x, np.ndarray):
        x = np.array(x).reshape(-1, 1)
    
    if len(x) > 1 and len(mu) == 1:
        mu = np.ones((rows(x), 1)) * mu

    if len(x) > 1 and len(sig2vec) == 1:
        sig2vec = np.ones((rows(x), 1)) * sig2vec

    c = -0.5 * np.log(2 * sig2vec * np.pi)
    e = x - mu
    e2 = np.multiply(e, e)

    y = c - np.divide(0.5 * e2, sig2vec)
    return y


def lnpdfmvn(x: np.ndarray, 
             m: np.ndarray, 
             C: np.ndarray) -> float:
    """    
    GAUSSIAN_PROB Evaluate a multivariate Gaussian density.
    
    Args:
        x (np.ndarray): d by 1 vector to evaluate the logPDF
        m (np.ndarray): mean vector
        C (np.ndarray): covariance matrix
    
    Returns:
        logp (float) or None: log probability density value
    """
    x = x.reshape(-1, 1)
    d = rows(x)
    denom0 = np.power(2*np.pi, d/2)
    denom = denom0 * np.sqrt(np.abs(det(C)))
    diff = x - m

    if C.shape == ():
        # handle scalar
        invC = 1 / C
    else:
        if is_invertible(C):
            invC = np.linalg.inv(C)
        else:
            raise ValueError("the Covariance matrix is singular")
    
    S1 = diff.T
    S0 = S1 @ invC
    S = np.multiply(S0, S1)
    mahal = sumc(S.T) # Chris Bergler's trick
    logp = -0.5 * mahal - np.log(denom)

    return logp


def lnpdfn1(e):
   '''
   log pdf of standard normal 
   '''
   c = -0.5*np.log(2*np.pi);  	
   y = c - 0.5*np.multiply(e,e)   
   return y


def lnpdfmvn1(y,mu,P):	
    """
    로그 정규밀도
    uses precision instead of var
    P = precision matrix
    """
    C = chol(P) # the matrix that makes the y uncorrelated 
    e = C*(y-mu) # standard normals: k times m matrix 
    y = 0.5*lndet1(C) + sumc(lnpdfn1(e));  # the log of the density 

    return y


def cdfn(x):
    '''
    cdf of standard normal   
    x = matrix
    y = matrix
    ''' 
    x = np.asmatrix(x)
    rs = rows(x)
    cs = cols(x)
    y = x.copy()
    for c in np.arange(cs):
        for r in np.arange(rs):
            y[r, c] = norm.cdf(x[r, c], loc=0, scale=1)

    return y