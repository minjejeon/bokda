import bok_da as bd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from typing import Union

from ...utils.operator import zeros, ones, eye, diag, inv, sumc, meanc, stdc, rows, cols, length, MA
from ...utils.pdf import lnpdfn
from .optimizer import trans_SSM, paramconst_SSM, SA_Newton, Gradient
#import cython_SSM as cbp
from . import cython_SSM as cbp

warnings.filterwarnings("ignore", category=RuntimeWarning)
    
class UnobservedComponentModel:
    def __init__(self, drift: str='time-varying', lag=1, lamb=3):
        self.drift = drift
        self.lag = lag
        self.lamb = lamb
        
    def fit(self, data, verbose: bool=False):
        self.results = bd.ts.ssm.uc(data, self.lag, self.lamb, self.drift, verbose)
        return self.results

class UCResult:

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        
    def __repr__(self):
        return (f"UCResult(drift={self.drift} - Drift 시계열 추정치, "
                f"drift_se={self.drift_se} - Drift의 표준오차 시계열 추정치, "
                f"trend={self.trend} - 추세 시계열 추정치, "
                f"trend_se={self.trend_se} - 추세 표준오차 시계열 추정치, "
                f"cycle={self.cycle} - 순환 시계열 추정치, "
                f"cycle_se={self.cycle_se} - 순환 표준오차 시계열 추정치, "
                f"table={self.table} - 모형 파라미터 추정결과, "
                f"lnL={self.lnL} - 로그우도(log likelihood), "
                f"bic={self.bic} - 베이지안 정보기준(Bayesian Information Criterion))")
    
    def get_description(self):
        desc = {
            "결과": ['drift', 'drift_se', 'trend', 'trend_se', 'cycle', 'cycle_se', 'table', 'lnL', 'bic'],
            "설명": ['Drift 시계열 추정치', 'Drift의 표준오차 시계열 추정치', '추세 시계열 추정치', 
                     '추세 표준오차 시계열 추정치', '순환 시계열 추정치', '순환 표준오차 시계열 추정치', 
                     '모형 파라미터 추정결과', '로그우도(log likelihood)', '베이지안 정보기준(Bayesian Information Criterion)']
        }
        df = pd.DataFrame(desc)
        df.set_index("결과", inplace=True)
        df.index.name = None
        #df = df.style.set_properties(**{'text-align': 'left'})
        #df = df.set_table_styles([{'selector': 'th', 'props': [('text-align', 'left')]}])
        
        return df
    
    def get_coef_bound(self):
        
        param = pd.concat([self.drift, self.trend, self.cycle], axis=1)
        param_se = pd.concat([self.drift_se, self.trend_se, self.cycle_se], axis=1)
        param_se.columns = param.columns
        lowb = param - param_se*2
        uppb = param + param_se*2
        return lowb, uppb
            
    def plot_uc_components(self, figsize: tuple=(8,5), title: bool=True,
                        title_fontsize: int=14, ncol: int=1, **kwargs):
        
        if 'drift' in kwargs['comp'] and hasattr(self, 'drift'):
            self._plot_uc_components(self.drift, figsize, title, title_fontsize, ncol)
        if 'trend' in kwargs['comp'] and hasattr(self, 'trend'):
            self._plot_uc_components(self.trend, figsize, title, title_fontsize, ncol)
        if 'cycle' in kwargs['comp'] and hasattr(self, 'cycle'):
            self._plot_uc_components(self.cycle, figsize, title, title_fontsize, ncol)

    def _plot_uc_components(self, data, figsize, title, title_fontsize, ncol):
        
        lower_bound, upper_bound = self.get_coef_bound()
        
        pp = bd.viz.Plotter(xmargin=0, figsize=figsize)
        pp.line(data.index, data, color='b', label=data.columns)
        pp.line(data.index, [lower_bound[data.columns[0]], upper_bound[data.columns[0]]], linestyle='--', color='k', label=['lower','upper'])
        pp.set_xaxis('year')
        if title:
            pp.set_title(title=f'{data.columns[0]}', fontsize=title_fontsize)
        pp.legend(ncol=ncol)
        
    
def lnlik_UC(para, Spec):
    # drift = "constant" - w_drift, "zero" - wo_drift, "time-varying" - w_TV_drift
    
    Y = Spec['Y']
    para1 = trans_SSM(para, Spec)
    sig2index = Spec['sig2index']
    sig2 = para1[sig2index]
    lamb = Spec['lamb']
    p = Spec['Lag']
    drift = Spec['drift']
    
    K = p + 2 if drift == "time-varying" else p + 1 # w_drift, wo_drift
    
    Omega = zeros(K, K)
    Omega[0, 0] = sig2
    Omega[1, 1] = lamb*sig2

    if drift == "time-varying":
        w2index = Spec['w2index'] # w_TV_drift
        w2 = para1[w2index] # w_TV_drift
        Omega[K-1, K-1] = w2 # w_TV_drift
    
    mu = para1[0] if drift == "constant" else 0
    Mu = zeros(K, 1)
    Mu[0, 0] = mu
    
    if p == 0:
        phi = 0
    else:
        phi = para1[1:] if drift == "zero" else para1[2:]

    if p > 1:
        Phi1 = phi.T
        Phi2 = np.hstack((eye(p-1), zeros(p-1, 1)))
        Phi = np.vstack((Phi1, Phi2))
    else:
        Phi = phi
    
    F = eye(K)
    if drift == "time-varying":
        F[1:K-1, 1:K-1] = Phi # w_TV_drift
        F[0, K-1] = 1 # w_TV_drift
    else:
        F[1:, 1:] = Phi # w_drift, wo_drift

    Sigma = zeros(1, 1).reshape(-1, 1)

    H = zeros(1, K)
    H[0, 0] = 1
    H[0, 1] = 1

    U_LL = zeros(K, 1)
    U_LL[0, 0] = Y[0]
    P_LL = diag(ones(K, 1)*sig2)

    lnLm, U_ttm, P_ttm, U_tLm, P_tLm = cbp.ckalman_filter_UC(Y, K, F, Mu, Omega, Sigma, U_LL, P_LL, H)
    lnL = sumc(lnLm[5:])  # 번인 제외
    return lnL, U_ttm, P_ttm, U_tLm, P_tLm


def lnlik_UC_w_drift(para, Spec):
    Y = Spec['Y']
    sig2index = Spec['sig2index']
    lamb = Spec['lamb']
    p = Spec['Lag']
    para1 = trans_SSM(para, Spec)
    mu = para1[0]
    sig2 = para1[sig2index]
    if p == 0:
        phi = 0
    else:
        phi = para1[2:]
    K = p + 1
    F = eye(K)

    if p > 1:
        Phi1 = phi.T
        Phi2 = np.hstack((eye(p-1), zeros(p-1, 1)))
        Phi = np.vstack((Phi1, Phi2))
    else:
        Phi = phi

    F[1:, 1:] = Phi

    Mu = zeros(K, 1)
    Mu[0,0] = mu
    Omega = zeros(K, K)
    Omega[0,0] = sig2
    Omega[1,1] = lamb*sig2

    Sigma = zeros(1,1)

    H = zeros(1, K)
    H[0,0] = 1
    H[0,1] = 1

    U_LL = zeros(K, 1)
    U_LL[0,0] = Y[0]
    P_LL = diag(ones(K, 1)*sig2)

    T = rows(Y)
    lnLm = zeros(T, 1)
    U_ttm = zeros(T, K)
    P_ttm = zeros(K, K, T)

    U_tLm = zeros(T, K)
    P_tLm = zeros(K, K, T)

    for t in range(T):
        
        U_tL = Mu + F @ U_LL
        P_tL = F @ P_LL @ F.T + Omega

        y_tL = H @ U_tL
        f_tL = H @ P_tL @ H.T + Sigma

        y_t = Y[t, 0]
        lnp = lnpdfn(y_t, y_tL, f_tL)
        lnLm[t] = lnp

        invf_tL = inv(f_tL)
        U_tt = U_tL + P_tL @ H.T @ invf_tL @ (y_t - y_tL)
        P_tt = P_tL - P_tL @ H.T @ invf_tL @ H @ P_tL
        P_tt = (P_tt + P_tt.T)/2

        U_ttm[t, :] = U_tt.T
        P_ttm[t, :, :] = P_tt

        U_tLm[t, :] = U_tL.T
        P_tLm[t, :, :] = P_tL

        U_LL = U_tt
        P_LL = P_tt

    lnL = sumc(lnLm[5:])  # 번인 제외
    return lnL, U_ttm, P_ttm, U_tLm, P_tLm


def lnlik_UC_wo_drift(para, Spec):
    Y = Spec['Y']
    sig2index = Spec['sig2index']
    lamb = Spec['lamb']
    p = Spec['Lag']
    para1 = trans_SSM(para, Spec)
    mu = 0
    sig2 = para1[sig2index]
    if p == 0:
        phi = 0
    else:
        phi = para1[1:]
    K = p + 1
    F = eye(K)

    if p>1:
        Phi1 = phi.T
        Phi2 = np.hstack((eye(p-1), zeros(p-1, 1)))
        Phi = np.vstack((Phi1, Phi2))
    else:
        Phi = phi

    F[1:, 1:] = Phi

    Mu = zeros(K, 1)
    Mu[0,0] = mu
    Omega = zeros(K, K)
    Omega[0,0] = sig2
    Omega[1,1] = lamb*sig2

    Sigma = zeros(1,1)

    H = zeros(1, K)
    H[0,0] = 1
    H[0,1] = 1

    U_LL = zeros(K, 1)
    U_LL[0,0] = Y[0]
    P_LL = diag(ones(K, 1)*sig2)

    T = rows(Y)
    lnLm = zeros(T, 1)
    U_ttm = zeros(T, K)
    P_ttm = zeros(K, K, T)

    U_tLm = zeros(T, K)
    P_tLm = zeros(K, K, T)

    for t in range(T):
        
        U_tL = Mu + F @ U_LL
        P_tL = F @ P_LL @ F.T + Omega

        y_tL = H @ U_tL
        f_tL = H @ P_tL @ H.T + Sigma

        y_t = Y[t, 0]
        lnp = lnpdfn(y_t, y_tL, f_tL)
        lnLm[t] = lnp

        invf_tL = inv(f_tL)
        U_tt = U_tL + P_tL @ H.T @ invf_tL @ (y_t - y_tL)
        P_tt = P_tL - P_tL @ H.T @ invf_tL @ H @ P_tL
        P_tt = (P_tt + P_tt.T)/2

        U_ttm[t, :] = U_tt.T
        P_ttm[t, :, :] = P_tt

        U_tLm[t, :] = U_tL.T
        P_tLm[t, :, :] = P_tL

        U_LL = U_tt
        P_LL = P_tt

    lnL = sumc(lnLm[5:])  # 번인 제외
    return lnL, U_ttm, P_ttm, U_tLm, P_tLm


def lnlik_UC_w_TV_drift(para, Spec):
    Y = Spec['Y']
    w2index = Spec['w2index']
    sig2index = Spec['sig2index']
    lamb = Spec['lamb']
    p = Spec['Lag']
    para1 = trans_SSM(para, Spec)
    mu = 0
    w2 = para1[w2index]
    sig2 = para1[sig2index]
    if p == 0:
        phi = 0
    else:
        phi = para1[2:]
    K = p + 2
    F = eye(K)

    if p>1:
        Phi1 = phi.T
        Phi2 = np.hstack((eye(p-1), zeros(p-1, 1)))
        Phi = np.vstack((Phi1, Phi2))
    else:
        Phi = phi

    F[1:K-1, 1:K-1] = Phi
    F[0, K-1] = 1

    Mu = zeros(K, 1)
    Mu[0, 0] = mu
    Omega = zeros(K, K)
    Omega[0, 0] = sig2
    Omega[1, 1] = lamb*sig2
    Omega[K-1, K-1] = w2

    Sigma = zeros(1, 1)

    H = zeros(1, K)
    H[0, 0] = 1
    H[0, 1] = 1

    U_LL = zeros(K, 1)
    U_LL[0, 0] = Y[0]
    P_LL = diag(ones(K, 1)*sig2)

    T = rows(Y)
    lnLm = zeros(T, 1)
    U_ttm = zeros(T, K)
    P_ttm = zeros(K, K, T)

    U_tLm = zeros(T, K)
    P_tLm = zeros(K, K, T)

    for t in range(T):
        
        U_tL = Mu + F @ U_LL # k by 1
        P_tL = F @ P_LL @ F.T + Omega # k by k

        y_tL = H @ U_tL # 1 by 1
        f_tL = H @ P_tL @ H.T + Sigma # 1 by 1

        y_t = np.array([Y[t, 0]])
        lnp = lnpdfn(y_t, y_tL, f_tL)
        lnLm[t] = lnp

        invf_tL = inv(f_tL)
        U_tt = U_tL + P_tL @ H.T * invf_tL * (y_t - y_tL) # k by 1
        P_tt = P_tL - P_tL @ H.T @ invf_tL @ H @ P_tL # k by k
        P_tt = (P_tt + P_tt.T)/2

        U_ttm[t, :] = U_tt.T
        P_ttm[t, :, :] = P_tt

        U_tLm[t, :] = U_tL.T
        P_tLm[t, :, :] = P_tL

        U_LL = U_tt
        P_LL = P_tt

    lnL = sumc(lnLm[5:])  # 번인 제외
    return lnL, U_ttm, P_ttm, U_tLm, P_tLm


def Kalman_Smoother_UC(F, Beta_ttm, P_ttm, Beta_tLm, P_tLm):
    T = rows(Beta_ttm) 
    K = cols(Beta_ttm)
    Beta_tTm = Beta_ttm.copy()
    P_tTm = zeros(T, K)
    P_tTm[T-1, :] = diag(P_ttm[T-1, :, :]).T # % 대각만 저장하기
    P_t1T = P_ttm[T-1, :, :]; # 1기 이후 분산-공분산
    t = T-2

    while t >= 0:
        P_tt = P_ttm[t, :, :] # k by k
        P_t1t = P_tLm[t+1, :,:] # k by k
        invPt1t = inv(P_t1t)

        weight = P_tt @ F.T @ invPt1t # k by k

        beta_tT = Beta_ttm[t, :].T.reshape(-1, 1) + weight @ (Beta_tTm[t+1, :].T - Beta_tLm[t+1, :].T).reshape(-1, 1)
        Beta_tTm[t, :] = beta_tT.T

        P_tT = P_tt + P_tt @ F.T @ invPt1t @ (P_t1T - P_t1t) @ invPt1t @ F @ P_tt
        P_tTm[t, :] = diag(P_tT).T  # % 대각행렬(분산)만 저장하기

        P_t1T = P_tT  # % 다음기에 사용될 것, K by K
        t = t - 1

    return Beta_tTm, P_tTm

def uc(Y: Union[pd.Series, pd.DataFrame], 
       lag: int = 1, 
       lamb: int = 1, 
       drift: str = 'time-varying',
       verbose: bool = False
      ):
    
    """
    상태공간모형 중 하나인 은닉인자(Unobserved Component) 모형 추정함수.
    시계열 데이터 Y의 수준, 추세, 순환 요인을 추정.
    
    하이퍼 파라미터
    ---------------
    Y: pd.Series 또는 pd.DataFrame
        시계열 데이터가 판다스 시리즈 또는 DataFrame.
    lag: int, 기본값=1
        순환의 AR 과정 시차를 결정하는 파라미터, 1이상 4이하 자연수.
    lamb: int, 기본값=1
        모형의 평활화(smoothing)와 관련된 파라미터, 값이 클수록 추세의 변동성에 비해 순환부분의 변동성이 커짐.
    drift: str, 기본값='time-varying'
        드리프트 항의 유형을 결정, 'zero', 'constant', 'time-varying' 중 하나를 선택.
    verbose: bool, 기본값='False'
        모형 추정결과 그림 표시 옵션.
        
    반환값
    ------
    UCResult
        은닉인자모형의 추정결과를 포함하는 객체.
        UCResult 객체는 다음의 속성과 메서드를 포함:
        
        - Drift: Drift 시계열 추정치
        - Drift_SE: Drift의 표준오차 시계열 추정치
        - Trend: 추세 시계열 추정치
        - Trend_SE: 추세 표준오차 시계열 추정치
        - Cycle: 순환 시계열 추정치
        - Cycle_SE: 순환 표준오차 시계열 추정치
        - Table_frame: 모형 파라미터 추정결과
        - lnL: 로그우도(log likelihood)
        - BIC: 베이지안 정보기준(Bayesian Information Criterion)
        - get_description(): uc 함수에서 생성된 반환값(속성 및 메서드)을 설명하는 메서드
        - get_coef_bound(): 추정된 요인의 lower and upper bound를 출력하는 메서드
        - plot_uc_components(): 추정된 요인의 그림을 그리는 메서드
        
    예제:
    ---------------------
    >>> import bok_da as bd
    >>> uc_results = bd.ssm.uc(Y, lag=1, lamb=1, drift='time-varying')
    >>> uc_results.get_description()
    >>> uc_results.Table_frame
    """
    Lag = lag
    index = Y.index
    columns = Y.columns
    
    Y = np.array(Y, dtype=float)

    if cols(Y) > 1:
        print('입력된 자료는 ', cols(Y), '(=열의 수)개 입니다')
        print(Y)
        raise ValueError('자료는 일변수 시계열이어야 합니다')

    if drift == 'constant':
        para0 = zeros(2+Lag, 1)
        growth = Y[1:, 0] - Y[0:-1, 0]
        sig2index = 1
        w2index = sig2index

        para0[0] = meanc(growth)  # 추세 초기값
        para0[sig2index] = np.log(np.square(stdc(growth))/2)  # 변동성
        if Lag > 0:
            para0[sig2index+1] = 0.7   # AR 계수
        Spec = {'Y': Y,
                'Lag': Lag,
                'sig2index': sig2index,
                'w2index': w2index,
                'lamb': lamb}
        
        # 추정
        para0_hat, fmax, G, V, Vinv = SA_Newton(lnlik_UC_w_drift, paramconst_SSM, para0, Spec)

        para1_hat = trans_SSM(para0_hat, Spec)

        lnL, U_ttm, P_ttm, U_tLm, P_tLm = lnlik_UC_w_drift(para0_hat, Spec)

        narg = length(para0_hat)
        T = rows(Y)
        BIC = narg*np.log(T) - 2*lnL

        Grd = Gradient(trans_SSM, para0_hat, Spec)

        Var = Grd @ V @ Grd.T
        SE = np.round(np.sqrt(np.abs(diag(Var))), 3)

        ## F 만들기
        if Lag > 0:
            phi = para1_hat[sig2index+1:]
        else:
            phi = 0

        K = Lag + 1
        F = eye(K)

        if Lag > 1:
            Phi1 = phi.T
            Phi2 = np.hstack((eye(Lag-1), zeros(Lag-1, 1)))
            Phi = np.vstack((Phi1, Phi2))
        else:
            Phi = phi

        F[1:, 1:] = Phi

        U_tTm, P_tTm = Kalman_Smoother_UC(F, U_ttm, P_ttm, U_tLm, P_tLm)

        Drift = ones(T, 1)*para1_hat[0] 
        Drift_SE = ones(T, 1)*SE[0] 
        Trend = U_tTm[:, 0]
        Trend_SE = np.sqrt(P_tTm[:, 0]) 
        Cycle = U_tTm[:, 1]
        Cycle_SE = np.sqrt(P_tTm[:, 1])

    if drift == 'zero':
        para0 = zeros(1+Lag, 1)
        growth = Y[1:, 0] - Y[0:-1, 0]
        sig2index = 0
        para0[sig2index] = np.log(np.square(stdc(growth))/2)  # 변동성
        if Lag > 0:
            para0[sig2index+1] = 0.7   # AR 계수
        w2index = sig2index
        Spec = {'Y': Y,
                'Lag': Lag,
                'sig2index': sig2index,
                'w2index': w2index,
                'lamb': lamb}
        # 추정
        para0_hat,fmax,G,V,Vinv = SA_Newton(lnlik_UC_wo_drift,paramconst_SSM, para0, Spec)
        para1_hat = trans_SSM(para0_hat, Spec)
        lnL, U_ttm, P_ttm, U_tLm, P_tLm = lnlik_UC_wo_drift(para0_hat, Spec)

        narg = length(para0_hat)
        T = rows(Y)
        BIC = narg*np.log(T) - 2*lnL

        Grd = Gradient(trans_SSM, para0_hat, Spec)
        Var = Grd @ V @ Grd.T
        SE = np.round(np.sqrt(np.abs(diag(Var))), 3)

        ## F 만들기
        if Lag > 0:
            phi = para1_hat[sig2index+1:]
        else:
            phi = 0
                
        K = Lag + 1
        F = eye(K)

        if Lag>1:
            Phi1 = phi.T
            Phi2 = np.hstack((eye(Lag-1), zeros(Lag-1, 1)))
            Phi = np.vstack((Phi1, Phi2))
        else:
            Phi = phi

        F[1:, 1:] = Phi

        U_tTm, P_tTm = Kalman_Smoother_UC(F, U_ttm, P_ttm, U_tLm, P_tLm)

        Drift = zeros(T, 1) 
        Drift_SE = zeros(T, 1) 
        Trend = U_tTm[:, 0]
        Trend_SE = np.sqrt(P_tTm[:, 0]) 
        Cycle = U_tTm[:, 1]
        Cycle_SE = np.sqrt(P_tTm[:, 1]) 

    if drift == 'time-varying':
        para0 = zeros(2+Lag, 1)
        growth = Y[1:, 0] - Y[0:-1, 0]

        drift0 = MA(growth, 16)
        drift0 = drift0[17:]
        w2index = 0
        sig2index = 1

        para0[w2index] = np.log(np.square(stdc(drift0)))  # 추세 초기값
        para0[sig2index] = np.log(np.square(stdc(growth))/2)  # 변동성
        if Lag > 0:
            para0[sig2index+1] = 0.7   # AR 계수

        Spec = {'Y': Y,
                'Lag': Lag,
                'sig2index': sig2index,
                'w2index': w2index,
                'lamb': lamb, 
                'drift': drift}
        # 추정
        para0_hat,fmax,G,V,Vinv = SA_Newton(lnlik_UC_w_TV_drift,paramconst_SSM, para0, Spec)

        para1_hat = trans_SSM(para0_hat, Spec).reshape(-1, 1)

        lnL, U_ttm, P_ttm, U_tLm, P_tLm = lnlik_UC_w_TV_drift(para0_hat, Spec)

        narg = length(para0_hat)
        T = rows(Y)
        BIC = narg*np.log(T) - 2*lnL

        Grd = Gradient(trans_SSM, para0_hat, Spec)

        Var = Grd @ V @ Grd.T
        SE = np.round(np.sqrt(np.abs(diag(Var))), 3).reshape(-1, 1)

        ## F 만들기
        if Lag > 0:
            phi = para1_hat[sig2index+1:]
        else:
            phi = 0

        K = Lag + 2
        F = eye(K)

        if Lag>1:
            Phi1 = phi.T
            Phi2 = np.hstack((eye(Lag-1), zeros(Lag-1, 1)))
            Phi = np.vstack((Phi1, Phi2))
        else:
            Phi = phi

        F[1:K-1, 1:K-1] = Phi
        F[0, K-1] = 1

        U_tTm, P_tTm = Kalman_Smoother_UC(F, U_ttm, P_ttm, U_tLm, P_tLm)

        Drift = U_tTm[:, K-1] 
        Drift_SE = np.sqrt(P_tTm[:, K-1]) 
        Trend = U_tTm[:, 0]
        Trend_SE = np.sqrt(P_tTm[:, 0]) 
        Cycle = U_tTm[:, 1]
        Cycle_SE = np.sqrt(P_tTm[:, 1]) 

    if drift == 'zero':
        TMP = zeros(1, 2)
        TMP[0,0] = lamb*para1_hat[sig2index]
        TMP[0,1] = lamb*SE[sig2index]
    
        Table = np.hstack((para1_hat, SE.reshape(-1, 1)))
        Table = np.vstack((TMP, Table))
        Table = np.round(Table, 3)
        paraName = ['순환 충격 분산','추세 충격 분산', 'AR(1)']
        if Lag > 1:
            paraName += ['AR(2)']
        if Lag > 2:
            paraName += ['AR(3)']
        if Lag > 3:
            paraName += ['AR(4)']

    if drift == 'constant':
        TMP = zeros(1, 2)
        TMP[0,0] = lamb*para1_hat[sig2index]
        TMP[0,1] = lamb*SE[sig2index]
    
        Table = np.hstack((para1_hat, SE.reshape(-1, 1)))
        Table = np.vstack((TMP, Table))
        Table = np.round(Table, 3)
        paraName = ['순환 충격 분산','Drift 항', '추세 충격 분산', 'AR(1)']
        if Lag > 1:
            paraName += ['AR(2)']
        if Lag > 2:
            paraName += ['AR(3)']
        if Lag > 3:
            paraName += ['AR(4)']

    if drift == 'time-varying':
        TMP = zeros(1, 2)
        TMP[0,0] = lamb*para1_hat[sig2index]
        TMP[0,1] = lamb*SE[sig2index]
    
        Table = np.hstack((para1_hat, SE))
        Table = np.vstack((TMP, Table))
        Table = np.round(Table, 3)
        paraName = ['순환 충격 분산','Drift 충격 분산', '추세 충격 분산', 'AR(1)']
        if Lag > 1:
            paraName += ['AR(2)']
        if Lag > 2:
            paraName += ['AR(3)']
        if Lag > 3:
            paraName += ['AR(4)']

    Table_frame = pd.DataFrame(Table, index = paraName,columns=['추정치', '표준오차'])
    
    if verbose:
        print('-----------------------------------------------')
        print('로그 우도 = ', lnL)
        print('BIC = ', BIC)
        print('-----------------------------------------------')


        ## 추정결과 그림 그리기
        T = rows(Trend)
        x = range(T)

        plt.subplot(2, 2, 1)
        Drift_low = Drift - Drift_SE*2
        Drift_upp = Drift + Drift_SE*2
        plt.plot(x, Drift, linestyle='--', color = 'blue')
        plt.plot(x, Drift_low, linestyle=':', color = 'black')
        plt.plot(x, Drift_upp, linestyle=':', color = 'black')
        plt.title('Drift')

        plt.subplot(2, 2, 2)
        Trend_low = Trend - Trend_SE*2
        Trend_upp = Trend + Trend_SE*2
        plt.plot(x, Y, linestyle='-', color = 'black')
        plt.plot(x, Trend, linestyle='--', color = 'blue')
        plt.plot(x, Trend_low, linestyle=':', color = 'black')
        plt.plot(x, Trend_upp, linestyle=':', color = 'black')
        plt.title('Trend')

        plt.subplot(2, 2, 3)
        Cycle_low = Cycle - Cycle_SE*2
        Cycle_upp = Cycle + Cycle_SE*2
        plt.plot(x, zeros(T, 1), linestyle='-', color = 'black')
        plt.plot(x, Cycle, linestyle='-', color = 'blue')
        plt.plot(x, Cycle_low, linestyle=':', color = 'black')
        plt.plot(x, Cycle_upp, linestyle=':', color = 'black')
        plt.title('Cycle')
        plt.show()
    
    Drift = pd.DataFrame(Drift, index=index, columns=['drift'])
    Drift_SE = pd.DataFrame(Drift_SE, index=index, columns=['drift_se'])
    Trend = pd.DataFrame(Trend, index=index, columns=['trend'])
    Trend_SE = pd.DataFrame(Trend_SE, index=index, columns=['trend_se'])
    Cycle = pd.DataFrame(Cycle, index=index, columns=['cycle'])
    Cycle_SE = pd.DataFrame(Cycle_SE, index=index, columns=['cycle_se'])
    lnL = pd.DataFrame(lnL, index=['value'], columns=['log likelihood'])
    BIC = pd.DataFrame(BIC, index=['value'], columns=['BIC'])
    
    return UCResult(drift=Drift, drift_se=Drift_SE, trend=Trend, trend_se=Trend_SE, cycle=Cycle, cycle_se=Cycle_SE, 
                    table=Table_frame, lnL=lnL, bic=BIC)


def uc2(Y: np.ndarray, 
        Lag: int = 1, 
        lamb: int = 1, 
        drift: str = 'constant'):

    if cols(Y) > 1:
        print('입력된 자료는 ', cols(Y), '(=열의 수)개 입니다')
        print(Y)
        raise ValueError('자료는 일변수 시계열이어야 합니다')

    if drift == "constant":
        lagindex = 2
        sig2index = 1
        w2index = sig2index
        kindex = 1
        paraName = ['순환 충격 분산', 'Drift 항', '추세 충격 분산', 'AR(1)']
    elif drift == "zero":
        lagindex = 1
        sig2index = 0
        w2index = sig2index
        kindex = 1
        paraName = ['순환 충격 분산', '추세 충격 분산', 'AR(1)']
    elif drift == "time-varying":
        lagindex = 2
        sig2index = 1
        w2index = 0
        kindex = 2
        paraName = ['순환 충격 분산', 'Drift 충격 분산', '추세 충격 분산', 'AR(1)']
    else:
        raise ValueError("drift must be either `constant`, `zero`, or `time-varying`.")
    
    paraName += [f'AR({i})' for i in range(2, Lag + 1)]

    para0 = zeros(lagindex + Lag, 1)
    growth = Y[1:, 0] - Y[0:-1, 0]

    if drift == "constant": para0[0] = meanc(growth)  # 추세 초기값
    if drift == "time-varying": 
        drift0 = MA(growth, 16)
        drift0 = drift0[17:]
        para0[w2index] = np.log(np.square(stdc(drift0))) # 추세 초기값
    
    para0[sig2index] = np.log(np.square(stdc(growth))/2)  # 변동성

    if Lag > 0:
        para0[sig2index+1] = 0.7   # AR 계수

    Spec = {'Y': Y,
            'Lag': Lag,
            'sig2index': sig2index,
            'w2index': w2index,
            'lamb': lamb, 
            'drift': drift}
    
    # 추정
    para0_hat, fmax, G, V, Vinv = SA_Newton(lnlik_UC, paramconst_SSM, para0, Spec)
    para1_hat = trans_SSM(para0_hat, Spec).reshape(-1, 1)
    lnL, U_ttm, P_ttm, U_tLm, P_tLm = lnlik_UC(para0_hat, Spec)

    narg = length(para0_hat)
    T = rows(Y)
    BIC = narg*np.log(T) - 2*lnL

    Grd = Gradient(trans_SSM, para0_hat, Spec)
    Var = Grd @ V @ Grd.T
    SE = np.round(np.sqrt(np.abs(diag(Var))), 3).reshape(-1, 1)
    phi = para1_hat[sig2index+1:] if Lag > 0 else 0 # F 만들기

    K = Lag + kindex
    F = eye(K)

    if Lag > 1:
        Phi1 = phi.T
        Phi2 = np.hstack((eye(Lag - 1), zeros(Lag - 1, 1)))
        Phi = np.vstack((Phi1, Phi2))
    else:
        Phi = phi

    if drift == "time-varying":
        F[1:K-1, 1:K-1] = Phi
        F[0, K-1] = 1
    else:
        F[1:, 1:] = Phi

    U_tTm, P_tTm = cbp.cKalman_Smoother_TVP(F, U_ttm, P_ttm, U_tLm, P_tLm)


    if drift == "constant":
        Drift = ones(T, 1)*para1_hat[0] 
        Drift_SE = ones(T, 1)*SE[0] 
    elif drift == "zero":
        Drift = zeros(T, 1) 
        Drift_SE = zeros(T, 1) 
    elif drift == "time-varying":
        Drift = U_tTm[:, K-1] 
        Drift_SE = np.sqrt(P_tTm[:, K-1]) 
    else:
        raise ValueError("drift must be either `constant`, `zero`, or `time-varying`.")
    
    Trend = U_tTm[:, 0]
    Trend_SE = np.sqrt(P_tTm[:, 0]) 
    Cycle = U_tTm[:, 1]
    Cycle_SE = np.sqrt(P_tTm[:, 1])


    TMP = zeros(1, 2)
    TMP[0,0] = lamb*para1_hat[sig2index]
    TMP[0,1] = lamb*SE[sig2index]

    Table = np.hstack((para1_hat, SE))
    Table = np.vstack((TMP, Table))
    Table = np.round(Table, 3)

    Table_frame = pd.DataFrame(Table, index = paraName,columns=['추정치', '표준오차'])

    print('-----------------------------------------------')
    print('로그 우도 = ', lnL)
    print('BIC = ', BIC)
    print('-----------------------------------------------')

    ## 추정결과 그림 그리기
    T = rows(Trend)
    x = range(T)

    components = [("Drift", Drift, Drift_SE),
                  ("Trend", Trend, Trend_SE),
                  ("Cycle", Cycle, Cycle_SE)]

    for i, (title, component, stder) in enumerate(components, start=1):
        plt.subplot(2, 2, i)
        lower_bound = component - stder * 2
        upper_bound = component + stder * 2

        if title == "Cycle":
            plt.plot(x, np.zeros(len(x)), linestyle='-', color='black')
        elif title == "Trend":
            plt.plot(x, Y, linestyle='-', color='black')  # Assuming Y is defined

        plt.plot(x, component, linestyle='--', color='blue')
        plt.plot(x, lower_bound, linestyle=':', color='black')
        plt.plot(x, upper_bound, linestyle=':', color='black')
        plt.title(title)

    plt.tight_layout()
    plt.show()

    return Drift, Drift_SE, Trend, Trend_SE, Cycle, Cycle_SE, Table_frame, lnL, BIC