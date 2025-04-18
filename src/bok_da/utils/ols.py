import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from typing import Union

from .operator import rows, cols, ones, inv, array, meanc, demeanc, zeros, matrix

def plot_HD(Y: np.matrix, 
            X: np.matrix, 
            bhat: np.matrix, 
            ehat: np.matrix):
    """ plot historical decomposition

    Args:
        Y: dependent variables
        X: independent variables
        bhat: estimated coefficients
        ehat: estimated residuals
    """
    Xbhat = np.multiply(X, bhat.T)
    combined = np.hstack((Xbhat, ehat))
    
    T = rows(Y)
    K = cols(X)

    df = pd.DataFrame(Y)
    df_decomp = pd.DataFrame(combined)

    variindex = list(range(1, T+1, 1))
    palette = sns.color_palette("muted", df_decomp.shape[1])

    df.index = df.index + 1
    df_decomp.index = df_decomp.index + 1

    plt.figure(figsize=(20, 5))
    sns.lineplot(df, legend=False, palette=['black'], linewidth=1, label='total', marker="o")

    # bottom = np.zeros(len(df_decomp)) # use if it only contains positive values
    bottom_pos = np.zeros(T)
    bottom_neg = np.zeros(T)

    for i, color in zip(range(df_decomp.shape[1]), palette):
        if i < K: label_i = str(f"X_{i+1}")
        else: label_i = str("residual")

        values = df_decomp.iloc[:, i]

        # bottom based on whether values are positive or negative
        bottom = np.where(values >= 0, bottom_pos, bottom_neg)
        plt.bar(variindex, df_decomp.iloc[:, i], bottom=bottom, color=color, label=f'{label_i}')
        
        # bottom update depending on the values are positive or negative
        bottom_pos = np.where(values >= 0, bottom_pos + values, bottom_pos)
        bottom_neg = np.where(values < 0, bottom_neg + values, bottom_neg)

    plt.legend()
    plt.grid()
    plt.show()


def logdiff(data:np.array, 
            freq:str, 
            diff:str, 
            percentize:bool=True, 
            annualize:bool=True) -> np.array:
    """ log-differencing the data
    
    Args:
        data: data to difference, 
            if the data contains under 0, 
            the function automatically uses normal differencing
        freq: m(monthly), q(quarterly), y(yearly)
        diff: mom, qoq, yoy
        
        percentize: True then percentize
        annualize: True then annualize

    Returns:
        Y: log-differenced data
    """
    if percentize: perc = 100 
    else: perc = 1
    
    if freq == 'm':
        if diff == 'mom':
            a = 1
            b = 12
        elif diff == 'qoq':
            a = 4
            b = 4
        elif diff == 'yoy':
            a = 12
            b = 1
        else:
            raise Exception("잘못된 차분값입니다. diff에 'mom', 'qoq', 'yoy' 중 하나를 입력하세요.")
    elif freq == 'q':
        if diff == 'qoq':
            a = 1
            b = 4
        elif diff == 'yoy':
            a = 4
            b = 1
        else:
            raise Exception("잘못된 차분값입니다. diff에 'qoq', 'yoy' 중 하나를 입력하세요.")
    elif freq == 'y':
        if diff == 'yoy':
            a = 1
            b = 1
        else: 
            raise Exception("잘못된 차분값입니다. diff에 'yoy' 중 하나를 입력하세요.")
    else:
        raise Exception("잘못된 데이터 frequancy입니다. freq에 'm', 'q', 'y' 중 하나를 입력하세요.")

    if ~annualize: b = 1

    X = np.array(data)
    T, K = X.shape

    # check whether data containing under 0
    if np.any(X <= 0):
        Y = np.zeros((T-a))
        for iter in range(T-a):
            Xta = float(X[iter+a])
            Xt = float(X[iter])
            Y[iter] = perc * b * ((Xta - Xt) / Xt)
    
    else:
        Y = np.zeros((T-a))
        for iter in range(T-a):
            Xta = float(X[iter+a])
            Xt = float(X[iter])
            Y[iter] = perc * b * (np.log(Xta) - np.log(Xt))

    return Y


def detrend(Y, N=1):
    '''  
    # N차 시간 다항식으로 추세 추정과 추세 제거하기
    # N = order
    # Yhat = 순환치
    # Trend = 시간추세
    ''' 
    T = rows(Y)
    M = cols(Y)
    xt = np.asmatrix(np.arange(T))
    xt = xt.T 
    X = np.hstack((ones(T, 1), xt))
    if N > 1:
        for i in range(2, N+1):
            xt = np.multiply(xt, xt)
            X = np.hstack((X, xt))

    XX = X.T*X
    invXX = inv(XX)
    if M == 1:
        xy = X.T*Y
        b = invXX*xy
        Yhat = Y - X*b
    else:
        Yhat = Y.copy()
        for i in np.arange(M):
            xy = X.T*Y[:, i]
            b = invXX*xy
            Yhat[:, i] = Y[:, i] - X*b
    
    Trend = Y - Yhat
    return Yhat, Trend