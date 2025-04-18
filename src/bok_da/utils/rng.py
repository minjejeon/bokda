import numpy as np

from .operator import sumc, reshape, rows, ones, chol, inv, zeros


def rand(r: int, 
          c: int) -> np.ndarray:  
    """
    uniform distribution
    """
    rc = r * c
    x = np.random.uniform(0, 1, rc)
    y = reshape(x.T, r, c)
    return y


def randn(r: int, 
          c: int) -> np.ndarray:
    """
    Sample from a standard normal distribution.

    Args:
        r (int): The number of rows in the output matrix.
        c (int): The number of columns in the output matrix.

    Returns:
        np.ndarray: An r by c array of samples from a standard normal distribution.
    """
    y = np.random.normal(0, 1, size=(r, c))
    return y


def randitg(m: int, 
            n: int) -> np.ndarray:
    """
    Generate an n by 1 vector of random integers from a discrete uniform distribution.

    Args:
        m (int): The upper bound of the distribution range [1, m].
        n (int): The number of random integers to generate.

    Returns:
        np.ndarray: An n by 1 array of random integers from 1 to m.
    """
    y = np.random.randint(1, m + 1, size=(n, 1))
    return y


def randbeta(a: float, 
             b: float, 
             m: int=1, 
             n: int=1) -> np.ndarray:
    """
    RNG: beta
    a, b = 베타분포 파라메터
    y = m by n 행렬
    """
    mn = m * n

    if mn == 1:
        A = rand(a,1)
        B = rand(b,1)
        A = sumc(np.log(A))
        B = sumc(np.log(B))
        y = A / (A+B)
        
    else:
        A = rand(a,mn)
        B = rand(b,mn)
        A = sumc(np.log(A)) # mn by 1
        B = sumc(np.log(B)) # mn by 1
        y = np.divide(A, A + B)  # mn by 1
        
    return y


def randig(shape,scale,r=1,c=1):
    ''' 
    RNG: inverse gamma
    shape = shape parameter
    scale = scale parameter
    Generate random numbers from the Inverse Gamma distribution
    '''
    
    rc = r*c
 
    gam = np.random.gamma(shape, 1/scale, rc)
    gam = np.asmatrix(gam)
   
    gam = reshape(gam.T, r, c)
    y = np.divide(ones(r,c), gam)
    return y


def randwishart(Omega, nu):
    '''
    sampling Wishart dist
    nu = 자유도(df)
    평균, E(V) = Omega*nu
    '''

    k = rows(Omega)
    Chol_omg = chol(Omega)
    V = Chol_omg.T*randn(k,nu)  # % k by nu
    V = V*V.T  # k by k

    return V


def randiw(Omega0_df, df):
    '''
    RNG: Inverse Wishart
    Omega0_df = Omega(inverse wishart 분포)의 기대값 X 자유도(df)
    inverse wishart 분포를 따르는 Omega의 기대값 = Omega0_df/df;
    df = 자유도, 자유도가 클수록 강한 prior
    '''
    Omega0_df_inv = inv(Omega0_df)
    Omega0_df_inv = 0.5*(Omega0_df_inv + Omega0_df_inv.T)
    Omega_inv = randwishart(Omega0_df_inv, df)
    Omega = inv(Omega_inv)

    return Omega


def randb(p,n):
    '''
    RNG: 베르누이분포 샘플링
    p = 확률
    y = n by 1 벡터
    '''
    tmp = rand(n,1)  # nu by n
    y = zeros(n, 1)
    for j in range(0, n):
        y[j, 0] = int(tmp[j,0] < p) # n by 1
    
    y = np.asarray(y)
    return y