import numpy as np

# revised
def rows(x: np.ndarray) -> int:
    """
    array의 행(row)의 갯수 계산하기.
    """
    if np.isscalar(x):
        rs = 1
        return rs
    elif is_array(x):
        if x.size == 0:
            rs = 0
        else:
            rs = x.shape[0]
        return rs
    elif type(x) == np.matrix:
        rs, _ = x.shape
        return rs
    else:
        print("type of x is ", type(x))
        print(x)
        return None


# revised
def cols(x: np.ndarray) -> int:
    """
    array의 열(cols)의 갯수 계산하기.

    Args:
        x (np.ndarray): Input array whose columns are to be counted.

    Returns:
        cs: The number of columns in the array.
    """
    if np.isscalar(x):
        cs = 1
        return cs
    elif is_array(x): 
        if x.ndim == 1:
            x = np.asmatrix(x)
            _, cs = x.shape
        else:
            size = x.shape
            cs = size[-1]
        return cs
    elif type(x) == np.matrix:
        _, cs = x.shape
        return cs
    else:
        print("type of x is ", type(x))
        print(x)
        return None


def meanc(x: np.ndarray) -> np.ndarray:
    """
    열별로 평균계산하기.

    Args:
        x (np.ndarray): T by K matrix.

    Returns:
        y: Column vector, K by 1, representing the mean of each column.
    """
    if not isinstance(x, np.ndarray):
        raise ValueError("Input must be a numpy array.")
    y = np.mean(x, axis=0)
    return y


def demeanc(X: np.ndarray) -> np.ndarray:
    """
    열별로 평균을 제거하기.

    Args:
        X (np.ndarray): Input array from which the mean of each column is to be removed.

    Returns:
        np.ndarray: The array after removing the mean from each column.
    """
    if not isinstance(X, np.ndarray):
        raise ValueError("Input must be a numpy array.")

    # Handle 1D array case
    if X.ndim == 1:
        return X - np.mean(X)

    cs = cols(X)
    Y = X.copy()
    for j in np.arange(cs):
        Y[:, j] = X[:, j] - meanc(X[:, [j]]).flatten()
    return Y


def stdc(X):
    """
    열별로 표준편차 계산하기.

    Args:
        X (np.ndarray): Input array.

    Returns:
        np.ndarray: The standard deviation of each column in the array.
    """
    if not isinstance(X, np.ndarray):
        raise ValueError("Input must be a numpy array.")
    
    # Handle 1D array case
    if X.ndim == 1:
        return np.array([[np.std(X, ddof=1)]])

    y = np.zeros((cols(X), 1))
    N = rows(X)
    for j in np.arange(cols(X)):
        Xj = X[:, [j]]
        Xj = demeanc(Xj)
        y[j, 0] = np.sqrt(np.sum(np.square(Xj)) / (N - 1))

    return y


def diag(X: np.ndarray) -> np.ndarray:
    """
    Convert a square matrix to a diagonal column vector or a column vector to a diagonal matrix.

    Args:
        X (np.ndarray): A square matrix or a column vector.

    Returns:
        y: If X is a square matrix, returns a diagonal column vector.
                  If X is a column vector, returns a diagonal matrix with X on the diagonal.
    """
    if not isinstance(X, np.ndarray):
        raise ValueError("Input must be a numpy array.")

    # Check if X is a square matrix
    if X.ndim == 2 and X.shape[0] == X.shape[1]:
        y = np.diag(X)
        y = y.T
        return y
    # Check if X is a column vector
    elif X.ndim == 2 and X.shape[1] == 1:
        y = np.diagflat(X)
        return y
    else:
        if cols(X) > 1: # 행렬이라면 대각값을 열벡터로 추출
            if cols(X) == rows(X):
                y = np.asmatrix(np.diag(X))
                y = y.T
                return y
            else:
                print('정방행렬이 아닙니다')
        elif cols(X)==1: # 벡터라면 X가 대각인 행렬. 비대각은 0 
            y = np.asmatrix(np.diagflat(X))
            return y

        raise ValueError("Input must be either a square matrix or a column vector.")


def sumc(x: np.ndarray) -> np.ndarray:
    """
    열별로 합하기
    
    Args: 
        x (np.ndarray): T by K
    
    Returns:
        y (np.ndarray): K by 1 벡터
    """
    y = x.sum(0)
    y = y.T.copy()
    return y
    

def zeros(r: int, 
          c: int, 
          h: int=None) -> np.ndarray: 
    """
    영(0) 행렬 만들기

    Args: 
        r: row of the array
        c: column of the array
        h: hight(3-dim) of the array, default None

    Returns:
        y: r by c (by h) zero matrix
    """
    if h is None: # 2차원이면
       y = np.zeros((r, c))
    else: # 3차원이면
       y = np.zeros((h, r, c))
    return y


def ones(r: int, 
         c: int, 
         h: int=None) -> np.ndarray:
    """
    일(1) 행렬 만들기
    """
    if h is None: # 2차원이면
        y = np.ones((r, c))
    else: # 3차원이면
        y = np.ones((h, r, c))
    return y


def eye(x:int) -> np.ndarray:
    """
    항등행렬 만들기
    """
    y = np.eye(x)
    return y


def det(X): 
    """
    determinant 계산하기
    """
    y = np.linalg.det(X)
    return y


def inv(X: np.ndarray) -> np.ndarray:
    """
    Calculate the inverse of a square numpy array if it is invertible.

    Args:
        X (np.ndarray): A square numpy array.

    Returns:
        np.ndarray or None: The inverse of X if it is square and invertible, None otherwise.
    """
    if X.ndim == 1:
        return 1/X
    if cols(X) == rows(X):  # Check if the matrix is square
        if is_invertible(X):
            return np.linalg.inv(X)
        else:
            print('The matrix is singular and cannot be inverted.')
            return None
    else:
        print('The input is not a square matrix.')
        return None


def vec(x: np.ndarray) -> np.ndarray:
    """
    Vectorize a matrix
    
    ex)
    input:  x = [1, 2, 3,
                 4, 5, 6]: 2 by 3 행렬
    output: y = [1, 4, 2, 5, 3, 6]': 6 by 1 행렬
    """
    y1 = x.T.copy()
    y = y1.copy()

    if x.ndim > 1:
        rs, cs = x.shape
        y2 = np.reshape(x.T.copy(), (cs*rs, 1))
        y = y2.copy()
        
    return y


def reshape(x: np.ndarray, 
            rs: int, 
            cs: int) -> np.ndarray:
    """
    열벡터를 행렬로 변환하기
    
    Args:
        x (np.ndarray): The array to be reshaped. Can be a 1D column vector or a 2D array.
        rs (int): The number of rows in the resulting matrix.
        cs (int): The number of columns in the resulting matrix.

    Returns:
        y (np.ndarray): The reshaped matrix.
    """
    if x.size != rs*cs:
        raise ValueError("인수의 크기(length)가 (행크기*열크기)와 다릅니다.")
    
    y = np.zeros((rs, cs))

    if cols(x) > 1:
        x_vec = vec(x)
        for j in range(cs):  
                y[:,j] = x_vec[j*rs:(j+1)*rs].flatten()
    else:
        for j in range(cs):  
                y[:,j] = x[j*rs:(j+1)*rs].flatten()
    return y


def maxc(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    열별로 최댓값 계산
    
    Args: 
        x: 행렬

    Returns: 
        y: 열별 최댓값, 열벡터
        maxidx: 열별 최댓값이 있는 행, 열벡터
    """
    y = x.max(0)
    maxidx = x.argmax(axis=0)
    maxidx = maxidx.T
    return y.T, maxidx


def minc(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    열별로 최솟값 계산
    
    Args: 
        x: 행렬

    Returns: 
        y: 열별 최솟값, 열벡터
        minidx: 열별 최솟값이 있는 행, 열벡터
    """
    y = x.min(0)
    minidx = x.argmin(axis=0)
    minidx = minidx.T
    return y.T, minidx


def length(x: np.ndarray) -> int:
    """
    행과 열 중 큰 수를 반환
    """
    r = rows(x)
    c = cols(x)
    return max(r, c)


def chol(x: np.ndarray) -> np.ndarray:
    """
    Cholesky decomposition
    Y(return) is upper triangular matrix.
    """
    if cols(x) != rows(x):
        raise ValueError("정방행렬이 아닙니다.")
    if not is_symmetric:
        raise ValueError("대칭행렬이 아닙니다.")
    
    try:
        L = np.linalg.cholesky(x)
        Y = L.T
        return Y
    except np.linalg.LinAlgError:
        raise ValueError("positive definite이 아닙니다.")


def rev(x: np.ndarray) -> np.ndarray:
    """열별로 순서 뒤집기
    """
    if rows(x) > 1:
        y = np.flipud(x)
    else:
        y = x
    return y


def eig(X): 
    '''
    eigenvalue decomposition
    V = 특성벡터
    D = 특성근이 대각인 정방행렬
    '''
    eigenvalues, eigenvectors = np.linalg.eig(X)

    V = np.asarray(eigenvectors)
    eigenvalues = np.asarray(eigenvalues)
    D = diag(eigenvalues.T)
    return V, D


def recserar(x: np.ndarray, 
             y0: np.ndarray, 
             a: np.ndarray) -> np.ndarray:
    """
    Constructs a recursive time series.

    Args:
        x (np.ndarray): An N*K array for the time series.
        y0 (np.ndarray): A P*K array for initial values.
        a (np.ndarray): A P*K array for coefficients.

    Returns:
        y (np.ndarray): The resulting N*K recursive time series.
    """
    p = rows(y0)
    n = rows(x)
    k = cols(x)

    y = zeros(n, k)
    y[:p, :] = y0
    aReverse = rev(a)

    for i in range(p+1, n+1):
        y_prod = np.multiply(y[i-p-1:i-1, :], aReverse)
        y_sum = sumc(y_prod)
        y[i-1, :] = y_sum.T + x[i-1, :]
        
    return y


def MA(Data, order):
    Data = Data.reshape(-1, 1)
    Data_MA = Data.copy()
    T = rows(Data)

    if order > 1:
        for t in range(order):
            Data_MA[t, :] = meanc(Data[0:t, :]).T
    
        for t in range(order, T):
            Data_MA[t, :] = meanc(Data[t-order+1:t, :]).T
    
    return Data_MA


def is_array(X: any) -> bool:
    """check whether the type of X is array or not
    """
    if type(X) == np.ndarray:
        return True
    else:
        return False
    

def is_invertible(a: np.ndarray) -> bool:
    """
    Check if a numpy array is invertible.

    Args:
        a (np.ndarray): A numpy array.

    Returns:
        bool: True if a is square and invertible, False otherwise.
    """
    if a.ndim == 1:
        raise ValueError("1D arrays are not applicable for matrix inversion.")
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]


def is_nan(X: np.ndarray) -> np.ndarray:
    """
    행렬의 각 argument가 number type인지의 여부를 반환
    nan일 경우 1, 아닐 경우 0의 인자를 갖는 동일 사이즈의 행렬
    """
    X = np.isnan(X)
    Y = X*1
    
    return Y


def is_None(x: np.ndarray) -> bool:
    """
    check if any element of x is None
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    y = np.any(x == None)
    return y


def is_symmetric(x: np.ndarray) -> bool:
    """
    check if a square matrix is symmetric
    """
    # if x is not square
    if cols(x) != rows(x):
        raise ValueError("정방행렬이 아닙니다.")
    # check symmetry
    return np.allclose(x, x.T)


def is_pos_def(x: np.ndarray) -> bool:
    """
    check if a square matrix is positive definite.
    """
    # if x is not square
    if cols(x) != rows(x):
        return ValueError("정방행렬이 아닙니다.")
    
    if np.all(np.linalg.eigvals(x) > 0):
        return True
    else:
        return False
    

def array(X):
    """요소(entry)가 실수인 array 만들기
    """
    Y = np.array(X, dtype=float)
    return Y


def matrix(X):
    """요소(entry)가 실수인 행렬만들기
    """
    Y = np.matrix(X, dtype=float)
    return Y


def standdc(x):
    """ 
    (정규화)표준화 하기. 평균=0, 표준편차=1
    """
    if rows(x) > 1:
        x = demeanc(x) # T by K
        sd = stdc(x);  # K by 1
        sd = np.kron(sd.T, ones(rows(x), 1)) # T by K
        y = np.divide(x, sd)
        return y
    else:
        print('행의 크기가 1보다 커야 합니다(standdc)')


def lndet1(C):
    y = 2*sumc(np.log(diag(C)))
    return y