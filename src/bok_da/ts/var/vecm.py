import numpy as np
import scipy.linalg as splin
import matplotlib.pyplot as plt
import warnings

from ...utils.operator import ones, array, inv, zeros, rows, eye, chol
from .var import irf_estimate
from .var import order_var

def VECM_MLE(y,r,option,p=None):
    '''Johansen의 MLE 기법으로 추정한 VECM 모형, Johansen (1992)
    
    Args:
        y = 사용하는 VECM 모형의 반응변수
        r = 추정, 또는 가정된 VECM 모형의 rank
        option = 사용하고자하는 모형의 종류
            - 1: 상수항이 존재하는 모형 (상수항이 공적분 관계에 대한 제약이 없는 VECM 모형)
            - 2: 상수항이 존재하지 않는 모형 (상수항이 공적분 관계에 대한 제약이 있는 VECM 모형)
        p = VECM 모형의 최적 시차 (default = None)
                        
    Returns:
        alpha = 오차 수정항 (Error correction term)
        beta = 공적분 벡터
            alpha @ beta.T = Pi
        Gamma = 차분항들의 계수 추정치
        Lam = 내림차순으로 정렬된 고유값, 공적분 검정에 사용
        Sigmau = 백색잡음과정 오차항에 대한 분산-공분산 추정치
        u_hat = 잔차항
    '''
    y = array(y)
    if y.shape[1] < 2 :
        y = y.reshape(-1,1)
        print("y 변수가 단변수 (univariate) 입니다.")
    
    T,K = y.shape
    
    if p == None:
        #p_seq, p_aic, p_bic, p_hq = order_VAR(y)
        p_res = order_var(y, verbose=True)
        p = p_res.p_bic
    else:
        p = p
    
    # 1. Specify the Model
    Y0 = y[1:T,:].copy()
    Y_lag = y[0:T-1,:].copy()
    ydif = Y0 - Y_lag

    y = y.T.copy()
    ydif = ydif.T.copy()

    Dy = ydif[:,p-1:T-1].copy()
    DX = ones(1,T-p)
    for i in range(p-1):
        DX = np.vstack((DX, ydif[:,p-i-2:T-i-2]))

    if option == 1:
        Y_1 = y[:,p-1:T-1].copy()
    elif option == 2:
        Y_1 = y[:,p-1:T-1].copy()
        Y_1 = np.vstack((Y_1,ones(1,T-p))).copy()
        DX = DX[1:T,:].copy()
    else:
        print("경고: 올바르지 않은 option값이 사용되었습니다.")
        print("option은 상수항이 존재하지 않는 모형의 경우 1, 상수항이 존재하지 않는 모형의 경우 2여야합니다.")

    # 2. Specify Eigenvalue
    R0_temp1 = Dy @ DX.T
    R0_temp2 = DX @ DX.T
    R0 = Dy - np.linalg.lstsq(R0_temp2.T,R0_temp1.T, rcond=None)[0].T @ DX
    R1temp1 = Y_1@DX.T
    R1temp2 = DX@DX.T
    R1 = Y_1 - np.linalg.lstsq(R1temp2.T,R1temp1.T,rcond=None)[0].T @ DX

    S00 = R0 @ R0.T / (T-p)
    S11 = R1 @ R1.T / (T-p)
    S01 = R0 @ R1.T / (T-p)

    S11_invsq = inv(splin.sqrtm(S11))
    temp = S11_invsq @ S01.T @ inv(S00) @ S01 @ S11_invsq
    Lam, B = np.linalg.eig(temp) # eigenvector and eigenvalues

    lam = np.diag(Lam)
    ind_lam = np.argsort(-Lam,kind='quicksort').T
    lamsort = np.sort(-Lam).T
    Lam = np.diag(lamsort)
    Lam = -Lam # Return initial signs of Lambda

    # 3. Get the maximizers of the log-likelihood function
    B = B[:,ind_lam]
    beta = S11_invsq @ B[:,0:r]
    alpha = S01 @ beta @ inv(beta.T @ S11 @ beta)
    Gamma_temp1 = (Dy - alpha@beta.T@Y_1) @ DX.T
    Gamma_temp2 = DX @ DX.T
    Gamma = np.linalg.lstsq(Gamma_temp2.T,Gamma_temp1.T, rcond=None)[0].T
    U = Dy - alpha@beta.T@Y_1 - Gamma@DX
    Sigmau = (U@U.T) / (T-p)
    
    u_hat = U.T

    return alpha, beta, Gamma, Lam, Sigmau, u_hat, p


def recover_VAR(y,p,alpha,beta,Gamma,Sigmau,option):
    '''추정된 VECM 모형으로부터 축약형 VAR 모형 복원
    
    Args:
        y = 사용하는 VECM 모형의 반응변수
        p = VECM 모형의 시차
        alpha = 오차 수정항 추정치 (Error correction term)
        beta = 공적분 벡터 추정치
            Pi = alpha @ beta.T
        Gamma = 차분항들의 계수 추정치
        Sigmau = 백색잡음과정 오차항에 대한 분산-공분산 추정치
        option = 사용하고자하는 모형의 종류
            - 1: 상수항이 존재하는 모형 (상수항이 공적분 관계에 대한 제약이 없는 VECM 모형)
            - 2: 상수항이 존재하지 않는 모형 (상수항이 공적분 관계에 대한 제약이 있는 VECM 모형)
    
    Returns:
        Ahat = 축약형 VAR 모형의 계수 추정치
        Sigma_hat = 축약형 VAR 모형의 분산-공분산 행렬 추정치
    '''
    y = array(y)
    if y.shape[1] < 2 :
        y = y.reshape(-1,1)
        print("y 변수가 단변수 (univariate) 입니다.")
        
    T,K = y.shape
    
    Pi = alpha @ beta.T
    
    if option == 1:
        mu = Gamma[:,0].reshape(-1,1)
        Gammam = Gamma[:,1:] # K*(p-1)
    elif option == 2:
        Gammam = Gamma # K*(p-1)
    else:
        print("경고: 올바르지 않은 option값이 사용되었습니다.")
        print("option값은 상수항이 존재하는 모형의 경우 1, 상수항이 존재하지 않는 모형의 경우 2여야합니다.")

    # Linear Transformation
    J = zeros(K,K*p)
    J[0:K,0:K] = eye(K)

    temp = ones(p-1,1)
    temp = np.diagflat(temp,-1)
    K1 = rows(temp)

    for i in range(K1):
        if i == 0:
            temp[i,i] = 1
        else:
            temp[i,i] = -1

    W = np.kron(temp,eye(K))

    coef = np.hstack((Pi,Gammam))
    Am = coef @ W + J
    if option == 1:
        Am = np.hstack((mu,Am))

    Ahat = Am.T # reduced-form coefficient matrix
    Sigma_hat = Sigmau
    
    return Ahat, Sigma_hat


def VECM_IRF(y,H,rank,option,p=None):
    '''VECM 모형의 충격반응함수 추정 (신뢰구간은 추정하지 않음)
    cf) 단기 제약만 고려하였음.
    
    Args:
        y = 사용하는 VECM 모형의 반응변수
        H = 충격반응함수 도출 시 고려하는 최대 예측시차
        rank = 공적분 rank 추정치
        option = 사용하고자하는 모형의 종류
            - 1: 상수항이 존재하는 모형 (상수항이 공적분 관계에 대한 제약이 없는 VECM 모형)
            - 2: 상수항이 존재하지 않는 모형 (상수항이 공적분 관계에 대한 제약이 있는 VECM 모형)
        p = VECM 모형의 최적 시차 (default = None)
    
    Returns:
        Theta = 충격반응함수, H+1 by K^2
    '''
    y = array(y)
    if y.shape[1] < 2 :
        y = y.reshape(-1,1)
        print("y 변수가 단변수 (univariate) 입니다.")
        
    T,K = y.shape

    # <Step 1>: Estimate the VECM
    alpha,beta,Gamma,Lambda,Sigma,u_hat,phat = VECM_MLE(y,rank,option,p)
    psave = p
    p = phat

    # <Step 2>: Recover the Reduced-form VAR
    Ahat, Sigma_hat = recover_VAR(y,p,alpha,beta,Gamma,Sigma,option)

    # <Step 3>: Obtain the B0inv by short-run restrictions
    B0inv = chol(Sigma_hat).T

    # <Step 4>: Obtain the impulse response function
    if option == 1: # with intercept
        F1 = Ahat[1:,:].T 
        F2 = np.hstack((eye((p-1)*K), zeros(K*(p-1),K)))
        F = np.vstack((F1, F2))  # p*k by p*k
    elif option == 2: # without intercept
        F1 = Ahat.T
        F2 = np.hstack((eye(p-1)*K,zeros(K*(p-1),K)))
        F = np.vstack((F1,F2))
    else:
        print("경고: 올바르지 않은 option값이 사용되었습니다.")
        print("option값은 상수항이 존재하는 모형의 경우 1, 상수항이 존재하지 않는 모형의 경우 2여야합니다.")

    # Initial IRF
    Theta = irf_estimate(F,p,H,B0inv)
    
    if psave == None:
        print(" ")
        print("BIC로 추정한 VAR 최적 시차 = ", phat)
    else:
        print(" ")
        print("사용자가 입력한 VAR 최적 시차 = ", phat)

    '''
    Plotting Results
    '''
    name = []
    for idx in range(1,K+1):
        for idx2 in range(1,K+1):
            warnings.filterwarnings("ignore")
            name1 = r"$e_{idx2} \Rightarrow y_{idx}$"
            name.append(name1)
    name = np.array(name)
    time = np.matrix(np.arange(0,H+1))
    plt.figure(figsize=[3.3*K,3.3*K])  
    for i in range(1,K**2+1):
        plt.subplot(K,K,i)
        plt.plot(time.T, Theta[i-1,:].T, '-k', time.T, zeros(H+1,1), '-r')
        plt.xlim([0,H+1])
        plt.title(name[i-1])

    plt.legend(['IRF'])
    plt.show()
    
    return Theta