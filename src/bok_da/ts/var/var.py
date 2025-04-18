import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import seaborn as sns
import scipy.stats as sp
import warnings

from typing import Union

from ...utils.operator import array, inv, ones, rows, meanc, demeanc, zeros, diag, eye, rev, vec, chol, reshape, matrix, det, sumc, minc
from ...utils.pdf import cdfn
from ...utils.rng import rand
from dataclasses import dataclass, field

@dataclass
class OrderVarResult:
    seq: int = field(default=0)
    aic: int = field(default=0)
    bic: int = field(default=0)
    hq: int = field(default=0)
    aic_value: float = field(default=0.0)
    bic_value: float = field(default=0.0)
    hq_value: float = field(default=0.0)
    description: str = field(default='')

def order_var(y, lag_max: int=None, verbose: bool=False):
    '''VAR(p) 모형의 최적 시차 p 추정
    
    Args:
        y = 사용하는 VAR(p) 모형의 반응변수 벡터
        lag_max = 예상하는 VAR(p)모형의 최대 시차
        
    Returns:
        seq = 순차적 검정으로 추정된 VAR(p) 모형의 최적 시차
        aic = Akaike 정보기준 (AIC)로 추정된 VAR(p) 모형의 최적 시차
        bic = Bayesian 정보기준 (BIC)로 추정된 VAR(p) 모형의 최적 시차
        hq = Hannan-Quinn 정보기준 HQIC)로 추정된 VAR(p) 모형의 최적 시차
    '''
    y = np.array(y)
    if y.shape[1] < 2 :
        y = y.reshape(-1,1)
        print("y 변수가 단변수 (univariate) 입니다.")
    
    T, K = y.shape
    
    pmax = lag_max
    
    if pmax == None:
        pmax = round(12*((T/100)**(1/4))) # Rule of thumb, Ng and Perron (2001)

    # For Saving
    lnSig = zeros(pmax+1,1)

    # Preparing
    Y = y[pmax:T,:] # Initial LHS
    Z = ones(T-pmax,1) # Initial Constant Term
    ZZ = Z.T @ Z
    B = inv(ZZ) @ (Z.T @ Y) # OLS Estimator
    U = Y - Z @ B # Residual
    Sig = (U.T @ U) / (T-pmax) # Initial Sigma_hat
    lnSig[0,0] = np.log(det(Sig))

    for p in np.arange(pmax):
        Z = np.hstack((Z,y[pmax-p-1:T-p-1,:]))
        ZZ = Z.T @ Z
        B = inv(ZZ) @ (Z.T @ Y)
        U = Y - Z @ B
        Sig = (U.T @ U) / (T-pmax)
        lnSig[p+1,0] = np.log(det(Sig))
    
    # Sequential Testing
    LRstat = (T-pmax)*(lnSig[0:pmax,0]-lnSig[1:pmax+1,0]) # Sequence of LR test statistics
    cv = sp.chi2.ppf(0.95,K**2)
    reject = (LRstat>cv)
    if sumc(reject) == 0: # fail to reject in all cases
        p_seq = 0
    else:
        p_seq = np.max(np.nonzero(reject))
    p_seq = p_seq+1
    
    # Information Criteria
    pones = matrix(np.arange(pmax+1))
    AIC = lnSig + (2*(K**2*pones.T + K*ones(pmax+1,1)))/(T-pmax)
    BIC = lnSig + (np.log(T-pmax)*(K**2*pones.T + K*ones(pmax+1,1)))/(T-pmax)
    HQ = lnSig + (2*np.log(np.log(T-pmax))*(K**2*pones.T + K*ones(pmax+1,1)))/(T-pmax)
    aic_val, ind_aic = minc(AIC)
    bic_val, ind_bic = minc(BIC)
    hq_val, ind_hq = minc(HQ)
    
    p_aic = ind_aic[0,0]
    p_bic = ind_bic[0,0]
    p_hq = ind_hq[0,0]
    
    if verbose:
        print('   ')
        print(' VAR Order Selected by Sequential testing = ', p_seq)
        print(' VAR Order Selected by AIC =                ', p_aic)
        print(' VAR Order Selected by BIC =                ', p_bic)
        print(' VAR Order Selected by HQ =                 ', p_hq)
    
    description = (
        'seq = 순차적 검정으로 추정된 VAR(p) 모형의 최적 시차\n'
        'aic = Akaike 정보기준 (AIC)로 추정된 VAR(p) 모형의 최적 시차\n'
        'bic = Bayesian 정보기준 (BIC)로 추정된 VAR(p) 모형의 최적 시차\n'
        'hq = Hannan-Quinn 정보기준 HQIC)로 추정된 VAR(p) 모형의 최적 시차\n'
        'aic_value = AIC 값\n'
        'bic_value = AIC 값\n'
        'hq_value = AIC 값\n'
    )
    
    seq = p_seq
    aic = p_aic
    bic = p_bic
    hq = p_hq
    aic_value = aic_val
    bic_value = bic_val
    hq_value = hq_val
    
    return OrderVarResult(seq, aic, bic, hq, aic_value, bic_value, hq_value, description)


class VarResult:
    def __init__(self, phi_hat, lag, **kwargs):
        self.phi_hat = phi_hat
        self.lag = lag
        self.k = phi_hat.shape[1]
        
        #self.phi_hat = [phi_hat[i*self.k:(i+1)*self.k, :] for i in range(self.lag)]
        
        for key, value in kwargs.items():
            setattr(self, key, value)
            
    def get_description(self):
        desc = {
            "결과": ['phi_hat', 'omega_hat', 'f_mat', 'u_hat', 'y0', 'y_lag', 'y_pred', 'lag'],
            "설명": ['축약형 VAR(p) 모형의 계수 행렬 추정량', '축약형 VAR(p) 모형의 분산-공분산 행렬 추정량', 
                     '축약형 VAR(p) 모형의 동반행렬 (Companion Matrix) 형태의 계수 행렬 추정량',
                     '축약형 VAR(p) 모형의 잔차', '추정에 사용된 반응변수 행렬', '추정에 사용된 설명변수 행렬',
                     '예측치', '시차(None이면 BIC를 기준으로 선택된 값, None이 아니면 입력된 값)']
        }

        df = pd.DataFrame(desc)
        df.set_index("결과", inplace=True)
        df.index.name = None
        #df = df.style.set_properties(**{'text-align': 'left'})
        #df = df.set_table_styles([{'selector': 'th', 'props': [('text-align', 'left')]}])

        return df

def var(y: pd.DataFrame, lag: int=None, h: int=None):
    '''축약형 VAR(p) 모형 추정
    
    Args:
        y = 사용하는 VAR(p) 모형의 반응변수 행렬
        P = VAR(p) 모형의 최적 시차 (default = None)
        H = 최대 예측 시차 (default = None)
        
    Returns:
        phi_hat = 축약형 VAR(p) 모형의 계수 행렬 추정량
        omega_hat = 축약형 VAR(p) 모형의 분산-공분산 행렬 추정량
        f_mat = 축약형 VAR(p) 모형의 동반행렬 (Companion Matrix) 형태의 계수 행렬 추정량
        u_hat = 축약형 VAR(p) 모형의 잔차
        y0 = 추정에 사용된 반응변수 행렬
        y_lag = 추정에 사용된 설명변수 행렬
        y_pred = 예측된 값
        lag = 사용된 시차 (None이면 추정된 BIC값, None이 아니면 입력된 값)
    '''
    P=lag
    H=h
    y = np.array(y)
    if y.shape[1] < 2 :
        y = y.reshape(-1,1)
        print("y 변수가 단변수 (univariate) 입니다.")
        
    if P == None:
        p_res = order_var(y)
        P = p_res.bic
    else:
        P = P
    
    Y_ = meanc(y) # 변수별 평균
    Y = demeanc(y) # 평균제거하기
    T,K = Y.shape # 표본크기, 변수의 수

    Y0 = Y[P:,:] # 종속변수 T-P by K
    
    Y_lag = zeros(T-P, P*K) # 설명변수를 저장할 방
    for j in np.arange(P):
        Y_lag[:, j*K:(j+1)*K] = Y[P-1-j:T-1-j,:]

    Y_lag2 = Y_lag.T @ Y_lag
    phi_hat = inv(Y_lag2) @ (Y_lag.T @ Y0) # P*K by K
 
    if P > 1:
        F1 = phi_hat.T 
        F2 = np.hstack((eye((P-1)*K), zeros(K*(P-1),K)))
        F = np.vstack((F1, F2))  # p*k by p*k
    elif P==1:
        F = phi_hat.T

    T0 = rows(Y0)

    # Omega_hat 
    U_hat = Y0 - Y_lag @ phi_hat # T-p by k 
    Omega_hat = U_hat.T @ U_hat/(T0-P*K)  # k by k

    # 예측
    if H is None:
        Y_predm = None
    else:
        Y_predm = zeros(H, K)
        Y_Lag = rev(Y[-P:,:]) # 예측에 사용될 설명변수, 최근값이 위로 오도록 역순하기
        FF = F
        for h in range(0, H):
            Vec_Y_Lag = vec(Y_Lag.T)
            Y_h = FF*Vec_Y_Lag
            y_h = Y_h[0:K, 0] 
            Y_predm[h, :] = y_h.T
            Y_Lag = np.vstack((y_h.T, Y_Lag[0:P-1, :])) 
                 # Y(t-1)을 1기-얘측치로 대체
                 # 대신 Y(t-P)는 제외하기 
            FF = FF*F

        Y_predm = Y_predm + np.kron(ones(H, 1), Y_.T)  # 표본평균 보정(+)해주기

    return VarResult(phi_hat=phi_hat, omega_hat=Omega_hat, f_mat=F, u_hat=U_hat, y0=Y0, y_lag=Y_lag, y_pred=Y_predm, lag=P)


def B0invSolve(phi_hat, Omega_hat, restrict):
    '''단기 또는 장기 제약을 통한 구조 VAR(p) 모형 식별
    
    Args:
        phi_hat : 추정된 축약형 VAR(p) 모형의 계수 행렬 (장기제약 시 사용됨)
        Omega_hat : 추정된 축약형 VAR(p) 모형의 분산-공분산 행렬 (장,단기 제약 모두에 사용됨)
        restrict : 장,단기 제약에 대한 옵션
                    'short' 또는 'long'으로 입력할 것, string
    
    Returns:
        B0inv : 구조형 VAR(p) 모형의 식별에 사용되는 B0의 역행렬 (B0inv)
    '''
    # 1. Short-run Restriction
    if restrict == 'short':
        B0inv = chol(Omega_hat).T
    # 2. Long-run Restriction
    elif restrict == 'long':
        A = phi_hat.T
        K,n = A.shape
        p = n/K
        p = int(p)
        A1 = eye(K)
        
        for j in range(p):
            A1 = A1 - A[:,j*K:(j+1)*K]
            
        Phi = inv(A1) # Phi(1)
        temp = Phi @ Omega_hat @ Phi.T
        temp = 0.5*(temp+temp.T)
        Lbar = chol(temp).T
        # Lbar[1,1] = -Lbar[1,1].copy(); for Blanchard and Quah replication
        B0inv = A1 @ Lbar
    else:
        print("경고: 올비르지 못한 졔약조건(restriction)입니다.")
        print("restrict 변수는 단기제약의 경우 'short', 장기제약의 경우 'long', 일반화된 충격반응함수의 경우 'general' 이어야 합니다.")
        
    return B0inv


def irf_estimate(F,p,H,B0inv):
    '''충격반응함수 추정
    
    Args:
        F : VAR(p) 모형의 동반행렬 (Companion Form) 형태의 계수 행렬 추정치
        p : VAR(p) 모형의 최적 시차
        H : 충격반응함수 도출 시 사용되는 예측 시차
        B0inv : 식별제약에 따른 B0 역행렬
        
    Returns:
        Theta : 추정된 충격반응함수, H+1 by K^2
    '''
    K = rows(B0inv)
    FF = eye(p*K)
    vecB0inv = vec(B0inv.T)
    Theta = vecB0inv.copy() # at h=0, (K^2 * 1) is the response
    for h in np.arange(H):
        FF = F @ FF
        theta = FF[0:K,0:K] @ B0inv
        theta = vec(theta.T)
        Theta = np.hstack((Theta, theta))
        
    return Theta


def randper(y):
    ''' 행렬 원소의 위치 랜덤하게 바꾸기
    
    Args:
        y : 원소의 위치를 바꾸고자하는 변수 (행렬 또는 벡터 모두 가능)
        
    Returns:
        z : 원소의 위치가 바뀐 변수
    '''
    n = rows(y)
    z = y.copy()

    k = n-1
    while k > 1:
        i = np.ceil(rand(1,1)*k) # matrix
        i = i[0,0]
        i = int(i)
        zi = z[i,:].copy() # interchange values
        zk = z[k,:].copy() # change zi to zk
        z[i,:] = zk
        z[k,:] = zi
        k = k-1

    return z


class VectorAutoRegression:
    def __init__(self, lag: int=1):
        self.lag = lag
        self.res = None
        
    def fit(self, data, irf: str=None, h: int=16, q: float=90, n: int=2000, verbose: bool=False):
        if irf is None:
            self.res = var(data, lag=self.lag, h=h)
        elif irf in ['short', 'long']:
            self.res = var_irf_bootstrap(data, lag=self.lag, h=h, irf=irf, q=q, n=n, verbose=verbose)
        elif irf == 'generalized':
            self.res = var_girf_bootstrap(data, lag=self.lag, h=h, q=q, n=n, verbose=verbose)
        return self.res
    
    def variance_decomposition(self, data, irf: str='short', h: int=16, verbose: bool=False, figsize=(20,5)):
        self.res = var(data, lag=self.lag, h=h)
        self.result = var_decomp(self.res.phi_hat, self.res.omega_hat, self.res.f_mat, irf, self.lag, h, verbose, figsize, col_names=data.columns)
        return self.result
            
    def historical_decomposition(self, data, irf: str='short', verbose: bool=False, figsize=(20,5)):
        self.res = var(data, lag=self.lag)
        self.result = hist_decomp(data, self.res.phi_hat, self.res.omega_hat, self.res.f_mat, self.res.u_hat, irf, self.lag, verbose, figsize)
        return self.result
            

#class VectorAutoRegression:
#    def __init__(self, lag: int=1, h: int=24, irf_type: str='short', qt=None):
#        self.lag = lag
#        self.h = h
#        self.restrict = irf_type
#        self.qt = qt
#        self.res = None
#        
#    def fit(self, data, h: int=None):
#        self.res = var(data, lag=self.lag, h=h)
#        return self.res
#    
#    def irf(self, data, n_iter: int=2000, verbose: bool=False):
#        if self.restrict in ['short', 'long']:
#            self.results = var_irf_bootstrap(data, lag=self.lag, h=self.h, restrict=self.restrict, qt=self.qt, n_iter=n_iter, verbose=verbose)
#        elif self.restrict == 'general':
#            self.results = var_girf_bootstrap(data, lag=self.lag, h=self.h, qt=self.qt, n_iter=n_iter, verbose=verbose)
#        return self.results
#    
#    def variance_decomposition(self, data, verbose: bool=False, figsize=(20,5)):
#        if self.res is not None:
#            self.result = var_decomp(self.res.phi_hat, self.res.omega_hat, self.res.f_mat, self.restrict, self.lag, self.h, verbose, figsize, col_names=data.columns)
#            return self.result
#        else:
#            raise ValueError("The model has not been fitted yet.")
#            
#    def historical_decomposition(self, data, verbose: bool=False, figsize=(20,5)):
#        if self.res is not None:
#            self.result = hist_decomp(data, self.res.phi_hat, self.res.omega_hat, self.res.f_mat, self.res.u_hat, self.restrict, self.lag, verbose, figsize)
#            return self.result
#        else:
#            raise ValueError("The model has not been fitted yet.")

class VarIrfResult:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
            
    def get_description(self):
        desc = {
            "결과": ['theta', 'cilv', 'cihv', 'cum_theta', 'cum_cilv', 'cum_cihv',
                     'phi_hat', 'omega_hat', 'f_mat', 'u_hat', 'y0', 'y_lag', 'y_pred', 'lag'],
            "설명": ['추정된 VAR(p) 모형의 충격반응함수', '충격반응함수 신뢰구간의 하한', 
                     '충격반응함수 신뢰구간의 상한', '추정된 VAR(p) 모형의 누적된 충격반응함수', '누적된 충격반응함수 신뢰구간의 하한',
                     '누적된 충격반응함수 신뢰구간의 상한',
                     '축약형 VAR(p) 모형의 계수 행렬 추정량', '축약형 VAR(p) 모형의 분산-공분산 행렬 추정량', 
                     '축약형 VAR(p) 모형의 동반행렬 (Companion Matrix) 형태의 계수 행렬 추정량',
                     '축약형 VAR(p) 모형의 잔차', '추정에 사용된 반응변수 행렬', '추정에 사용된 설명변수 행렬',
                     '예측치', '시차(None이면 BIC를 기준으로 선택된 값, None이 아니면 입력된 값)']
        }

        df = pd.DataFrame(desc)
        df.set_index("결과", inplace=True)
        df.index.name = None
        #df = df.style.set_properties(**{'text-align': 'left'})
        #df = df.set_table_styles([{'selector': 'th', 'props': [('text-align', 'left')]}])

        return df
    
    #def plot_irf(self, restrict: str='short', cum: bool=False, h: int=16, title: bool=True, title_fontsize: int=14):
    def plot_irf(self, cum: bool=False, title: bool=True, title_fontsize: int=14):
        if cum:
            self._plot_irf(self.cum_theta, self.cum_cihv, self.cum_cilv, self.__restrict, cum, self.__h, title, title_fontsize)
        else:
            self._plot_irf(self.theta, self.cihv, self.cilv, self.__restrict, cum, self.__h, title, title_fontsize)
        
    def _plot_irf(self, theta, cihv, cilv, restrict, cum, h, title, title_fontsize):
        
        col_names = self.__col_names
        K = len(col_names)
        H = h
        
        Theta = theta
        CIHv = cihv
        CILv = cilv
        
        name = []
        for idx in range(K):
            for idx2 in range(K):
                name1 = f'{col_names[idx2]} to {col_names[idx]}'
                name.append(name1)

        name = np.array(name)
        time = np.matrix(np.arange(0,H+1))

        plt.figure(figsize=(K*3.3, K*3.3))  
        for i in range(1,K**2+1):
            plt.subplot(K,K,i)
            plt.plot(time.T, Theta[i-1,:].T, '-k', time.T, CIHv[i-1,:].T, '-.b', time.T, CILv[i-1,:].T, '-.b', time.T, zeros(H+1,1), '-r')
            plt.xlim([0,H+1])
            plt.title(name[i-1])
            
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        
        if title and cum:
            plt.suptitle('Cumulative IRF Estimation Results', fontsize=title_fontsize)
        elif title and not cum:
            plt.suptitle('IRF Estimation Results', fontsize=title_fontsize)

        if restrict == 'general':
            plt.legend(['GenIRF', 'Bootstrap'])
        else:
            plt.legend(['IRF', 'Bootstrap'])
            
    #def variance_decomposition(self, verbose: bool=False, figsize=(20,5)):
    #    self.result = var_decomp(self.__phi_hat, self.__omega_hat, self.__f_mat, self.__restrict, self.__lag, self.__h, verbose, figsize, self.__col_names)
    #    return self.result
        
    #def historical_decomposition(self, verbose: bool=False, figsize=(20,5)):
    #    self.result = hist_decomp(self.__df, self.__phi_hat, self.__omega_hat, self.__f_mat, self.__u_hat, self.__restrict, self.__lag, verbose, figsize)
    #    return self.result
            

def var_irf_bootstrap(y, lag: int=1, h: int=16, irf: str='short', q: float=90, n: int=2000, verbose: bool=False):
    '''VAR 충격반응함수 및 부트스래핑 (Bootstrapping)을 통한 충격반응함수의 신뢰구간 도출
    
    Args:
        y : VAR(p) 모형에 사용되는 반응변수 행렬
        lag : VAR(p) 모형의 최적 시차 (default = None)
        h : 충격반응함수 도출 시 고려하는 최대 예측시차
        irf: 사용하고자하는 식별 방법 ('short' = 단기제약, 'long' = 장기제약)
        q : 충격반응함수의 신뢰구간 (confidence interval)
            입력하지 않을 시 default = 90 (90% 신뢰구간)
        n : 부트스트래핑 횟수
        
    Returns:
        theta : 추정된 VAR(p) 모형의 충격반응함수
        cilv : 충격반응함수 신뢰구간의 하한
        cihv : 충격반응함수 신뢰구간의 상한
        cum_theta : 추정된 VAR(p) 모형의 누적된 충격반응함수
        cum_cilv : 누적된 충격반응함수 신뢰구간의 하한
        cum_cihv : 누적된 충격반응함수 신뢰구간의 상한
    
    Steps:
        <1 단계>: 축약형 VAR(p) 모형 추정 (demean 사용)
        <2 단계>: 단기, 또는 장기 제약으로 B0inv 도출
        <3 단계>: 잔차를 섞어 새로운 잔차를 랜덤 샘플링
        <4 단계>: 부트스트랩 샘플을 얻은 후 축약형 VAR(p) 모형을 다시 추정
        <5 단계>: 충격반응함수를 추정
        <6 단계>: <3 단계>부터 <5단계>까지 반복 한 이후 신뢰구간을 도출
    '''
    df = y.copy()
    p = lag
    H = h
    restrict = irf
    qt = q
    col_names = y.columns
    
    y = np.array(y)
    if y.shape[1] < 2 :
        y = y.reshape(-1,1)
        print("y 변수가 단변수(univariate) 입니다.")
    
    T,K = y.shape
    
    # <Step 1>: Estimate Reduced-form VAR (with demeaning)
    res = var(y,p)
    phi_hat = res.phi_hat
    Omega_hat = res.omega_hat
    F = res.f_mat
    U_hat = res.u_hat
    Y0 = res.y0
    Y_lag = res.y_lag
    Y_predm = res.y_pred
    phat = res.lag

    psave = p
    p = phat

    # <Step 2>: Obtain B0inv by short-run restriction
    B0inv = B0invSolve(phi_hat, Omega_hat, restrict)

    # Save the initial IRF
    Theta = irf_estimate(F,p,H,B0inv)
    cumTheta = zeros(K**2,H+1)
    for c in range(K**2):
        cumtemp = np.cumsum(Theta[c,:])
        cumTheta[c,:] = cumtemp

    # CI (Bootstrapping)
    n_bootstrap = n

    IRFm = zeros(n,K**2*(H+1))
    cumIRFm = zeros(n,K**2*(H+1))

    for iter in range(n_bootstrap):
        # <Step 3>: Obtain a new set of errors
        indu = matrix(range(T-p)) # 0 from 93
        indu = indu.T
        indu = randper(indu)
        indu = indu.astype(int) # make matrix as an integer

        ustar = U_hat[indu,:]

        ystar = zeros(T,K)
        ind_ystar = np.fix(rand(1,1)*(T-p+1))+1
        ind_ystar = ind_ystar[0,0]
        ind_ystar = ind_ystar.astype(int)
        ystar[0:p-1,:] = y[ind_ystar:ind_ystar+p-1,:]

        # <Step 4>: Obtain the Bootstrap Sample and estimate again
        for it in range(p,T):
            ystar[it,:] = ustar[it-p,:]
            for jt in range(p):
                ystar[it,:] = ystar[it,:] + ystar[it-jt-1,:] @ phi_hat[jt*K:(jt+1)*K,:]

        res_star = var(ystar,p)
        phi_star = res_star.phi_hat
        Omega_star = res_star.omega_hat
        F_star = res_star.f_mat
        U_star = res_star.u_hat
        Y0_star = res_star.y0
        Y_lag_star = res_star.y_lag
        Y_predm = res_star.y_pred
        phat_star = res_star.lag

        B0inv_star = B0invSolve(phi_star, Omega_star, restrict)

        # <Step 5>: Compute the Impulse Response Function
        IRFboot = irf_estimate(F_star, p, H, B0inv_star)
        cumIRFboot = zeros(K**2,H+1)
        for c in range(K**2):
            cumIRFboot[c,:] = np.cumsum(IRFboot[c,:])

        IRFm[iter,:] = vec(IRFboot.T).T
        cumIRFm[iter,:] = vec(cumIRFboot.T).T

    # <Step 6>: Obtain Quantiles
    if qt is None:
        CIlow = np.quantile(np.array(IRFm), 0.05, axis=0) # must be an option (default = 5%)
        CIhigh = np.quantile(np.array(IRFm), 0.95, axis=0) # must be an option (default = 95%)
        cumCIlow = np.quantile(np.array(cumIRFm), 0.05, axis=0) # must be an option (default = 5%)
        cumCIhigh = np.quantile(np.array(cumIRFm), 0.95, axis=0) # must be an option (default = 95%)
    else:
        qtlow = ((100-qt)/2)*0.01
        qthigh = (100 - qtlow)*0.01
        CIlow = np.quantile(np.array(IRFm), qtlow, axis=0)
        CIhigh = np.quantile(np.array(IRFm), qthigh, axis=0)
        cumCIlow = np.quantile(np.array(cumIRFm), qtlow, axis=0)
        cumCIhigh = np.quantile(np.array(cumIRFm), qthigh, axis=0)

    CIlow_mat = matrix(CIlow)
    CIhigh_mat = matrix(CIhigh)
    cumCIlow_mat = matrix(cumCIlow)
    cumCIhigh_mat = matrix(cumCIhigh)

    CILv = reshape(CIlow_mat.T, H+1, K**2).T
    CIHv = reshape(CIhigh_mat.T, H+1, K**2).T
    cumCILv = reshape(cumCIlow_mat.T,H+1,K**2).T
    cumCIHv = reshape(cumCIhigh_mat.T,H+1,K**2).T

    if verbose:
        if psave == None:
            print(" ")
            print("BIC로 추정한 VAR 최적 시차 = ", phat)
        else:
            print(" ")
            print("사용자가 입력한 VAR 최적 시차 = ", phat)
    
    if verbose:
        '''
        Plotting Results
        '''
        warnings.filterwarnings('ignore')
        
        name = []
        for idx in range(K):
            for idx2 in range(K):
                name1 = f'{col_names[idx2]} to {col_names[idx]}'
                name.append(name1)

        name = np.array(name)
        time = np.matrix(np.arange(0,H+1))

        plt.figure(figsize=(K*3.3, K*3.3))  
        for i in range(1,K**2+1):
            plt.subplot(K,K,i)
            plt.plot(time.T, Theta[i-1,:].T, '-k', time.T, CIHv[i-1,:].T, '-.b', time.T, CILv[i-1,:].T, '-.b', time.T, zeros(H+1,1), '-r')
            plt.xlim([0,H+1])
            plt.title(name[i-1])
            
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        plt.suptitle('IRF Estimation Results', fontsize=14)

        if restrict == 'general':
            plt.legend(['GenIRF', 'Bootstrap'])
        else:
            plt.legend(['IRF', 'Bootstrap'])

        plt.show()

        plt.figure(figsize=(K*3.3, K*3.3))  
        for i in range(1,K**2+1):
            plt.subplot(K,K,i)
            plt.plot(time.T, cumTheta[i-1,:].T, '-k', time.T, cumCIHv[i-1,:].T, '-.b', time.T, cumCILv[i-1,:].T, '-.b', time.T, zeros(H+1,1), '-r')
            plt.xlim([0,H+1])
            plt.title(name[i-1])
            
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        plt.suptitle('Cumulative IRF Estimation Results', fontsize=14)

        plt.legend(['cumIRF', 'Bootstrap'])
        plt.show()
    
    return VarIrfResult(theta=Theta,  cilv=CILv, cihv=CIHv, cum_theta=cumTheta, cum_cilv=cumCILv, cum_cihv=cumCIHv,
                        phi_hat = phi_hat, omega_hat=Omega_hat, f_mat=F, u_hat=U_hat, y0=Y0, y_lag=Y_lag, y_pred=Y_predm, lag=phat,
                        _VarIrfResult__restrict=restrict, _VarIrfResult__lag=lag, _VarIrfResult__h=h, _VarIrfResult__df=df,
                        _VarIrfResult__col_names=col_names, _VarIrfResult__phi_hat=phi_hat, _VarIrfResult__omega_hat=Omega_hat,
                        _VarIrfResult__f_mat=F, _VarIrfResult__u_hat=U_hat)


class VarGeneralizedIrfResult:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
            
    def get_description(self):
        desc = {
            "결과": ['theta', 'cilv', 'cihv', 'cum_theta', 'cum_cilv', 'cum_cihv', 'for_var', 'hist_decomp'],
            "설명": ['추정된 VAR(p) 모형의 충격반응함수', '충격반응함수 신뢰구간의 하한', 
                     '충격반응함수 신뢰구간의 상한', '추정된 VAR(p) 모형의 누적된 충격반응함수', '누적된 충격반응함수 신뢰구간의 하한',
                     '누적된 충격반응함수 신뢰구간의 상한', '예측 오차 분산 분해 결과 (K by (H+1))',
                     '각 반응변수에 대한 역사적 분해 결과']
        }

        df = pd.DataFrame(desc)
        df.set_index("결과", inplace=True)
        df.index.name = None
        df = df.style.set_properties(**{'text-align': 'left'})
        df = df.set_table_styles([{'selector': 'th', 'props': [('text-align', 'left')]}])

        return df
    
    def plot_irf(self, cum: bool=False, title: bool=True, title_fontsize: int=14):
        if cum:
            self._plot_girf(self.cum_theta, self.cum_cihv, self.cum_cilv, cum, self.__h, title, title_fontsize)
        else:
            self._plot_girf(self.theta, self.cihv, self.cilv, cum, self.__h, title, title_fontsize)
        
    def _plot_girf(self, theta, cihv, cilv, cum, h, title, title_fontsize):
        
        col_names = self.__col_names
        K = len(col_names)
        H = h
        
        Theta = theta
        CIHv = cihv
        CILv = cilv
        
        name = []
        for idx in range(K):
            for idx2 in range(K):
                name1 = f'{col_names[idx2]} to {col_names[idx]}'
                name.append(name1)

        name = np.array(name)
        time = np.matrix(np.arange(0,H+1))

        plt.figure(figsize=(K*3.3, K*3.3))  
        for i in range(1,K**2+1):
            plt.subplot(K,K,i)
            plt.plot(time.T, Theta[i-1,:].T, '-k', time.T, CIHv[i-1,:].T, '-.b', time.T, CILv[i-1,:].T, '-.b', time.T, zeros(H+1,1), '-r')
            plt.xlim([0,H+1])
            plt.title(name[i-1])
            
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        
        if title and cum:
            plt.suptitle('Cumulative IRF Estimation Results', fontsize=title_fontsize)
        elif title and not cum:
            plt.suptitle('IRF Estimation Results', fontsize=title_fontsize)

        plt.legend(['GenIRF', 'Bootstrap'])

def var_girf_bootstrap(y, lag: int=1, h: int=16, q: float=90, n: int=2000, verbose: bool=False):
    '''VAR 일반화된 충격반응함수 및 부트스트래핑을 통한 신뢰구간 도출
    
    Args:
        y : VAR(p) 모형에 사용되는 반응변수
        H : 충격반응함수 도출 시 고려하는 최대 예측시차
        qt : 충격반응함수의 신뢰구간 (confidence interval)
            입력하지 않을 시 default = 90 (90% 신뢰구간)
        p : VAR(p) 모형의 최적 시차
            입력하지 않을 시 default = None (자동 추정)
    
    Returns:
        Theta : VAR 모형의 일반화된 충격반응함수
        CILv : 충격반응함수 신뢰구간의 하한
        CIHv : 충격반응함수 신뢰구간의 상한
        cumTheta : 추정된 VAR(p) 모형의 누적된 충격반응함수
        cumCILv : 누적된 충격반응함수 신뢰구간의 하한
        cumCIHv : 누적된 충격반응함수 신뢰구간의 상한
    
    Steps:
        <1 단계>: 축약형 VAR(p) 모형 추정 (demean 사용)
        <2 단계>: Omega_hat./sqrt(sigma_{ii})을 통한 C 도출
        <3 단계>: 잔차를 섞어 새로운 잔차를 랜덤 샘플링
        <4 단계>: 부트스트랩 샘플을 얻은 후 축약형 VAR(p) 모형을 다시 추정
        <5 단계>: 충격반응함수를 추정
        <6 단계>: <3 단계>부터 <5단계>까지 반복 한 이후 신뢰구간을 도출
    '''
    p = lag
    H = h
    qt = q
    col_names = y.columns
    y = np.array(y)
    if y.shape[1] < 2 :
        y = y.reshape(-1,1)
        print("y 변수가 단변수 (univariate) 입니다.")
    
    T,K = y.shape

    # <Step 1>: Estimate Reduced-form VAR (with demeaning)
    res = var(y, p)
    phi_hat = res.phi_hat
    Omega_hat = res.omega_hat
    F = res.f_mat
    U_hat = res.u_hat
    Y0 = res.y0
    Y_lag = res.y_lag
    Y_predm = res.y_pred
    phat = res.lag
    
    psave = p
    p = phat

    # <Step 2>: Obtain C
    C = Omega_hat/np.kron(ones(rows(Omega_hat),1),np.sqrt(np.diag(Omega_hat)).T) # 1 standard dev shock

    # Save the initial IRF
    Theta = irf_estimate(F,p,H,C)
    cumTheta = zeros(K**2,H+1)
    for c in range(K**2):
        cumtemp = np.cumsum(Theta[c,:])
        cumTheta[c,:] = cumtemp

    # CI (Bootstrapping)
    n_bootstrap = n

    IRFm = zeros(n,K**2*(H+1))
    cumIRFm = zeros(n,K**2*(H+1))

    for iter in range(n_bootstrap):
        # <Step 3>: Obtain a new set of errors
        indu = matrix(range(T-p)) # 0 from 93
        indu = indu.T
        indu = randper(indu)
        indu = indu.astype(int) # make matrix as an integer
    
        ustar = U_hat[indu,:]

        ystar = zeros(T,K)
        ind_ystar = np.fix(rand(1,1)*(T-p+1))+1
        ind_ystar = ind_ystar[0,0]
        ind_ystar = ind_ystar.astype(int)
        ystar[0:p-1,:] = y[ind_ystar:ind_ystar+p-1,:]

        # <Step 4>: Obtain the Bootstrap Sample and estimate again
        for it in range(p,T):
            ystar[it,:] = ustar[it-p,:]
            for jt in range(p):
                ystar[it,:] = ystar[it,:] + ystar[it-jt-1,:] @ phi_hat[jt*K:(jt+1)*K,:]
        
        res_star = var(ystar,p)
        phi_star = res_star.phi_hat
        Omega_star = res_star.omega_hat
        F_star = res_star.f_mat
        U_star = res_star.u_hat
        Y0_star = res_star.y0
        Y_lag_star = res_star.y_lag
        Y_predm = res_star.y_pred
        phat_star = res_star.lag

        C_star = Omega_star/np.kron(ones(rows(Omega_star),1),np.sqrt(np.diag(Omega_star)).T)

        # <Step 5>: Compute the Impulse Response Function
        IRFboot = irf_estimate(F_star, p, H, C_star)
        cumIRFboot = zeros(K**2,H+1)
        for c in range(K**2):
            cumIRFboot[c,:] = np.cumsum(IRFboot[c,:])
        
        IRFm[iter,:] = vec(IRFboot.T).T
        cumIRFm[iter,:] = vec(cumIRFboot.T).T

    # <Step 6>: Obtain Quantiles
    if qt is None:
        CIlow = np.quantile(np.array(IRFm), 0.05, axis=0) # must be an option (default = 5%)
        CIhigh = np.quantile(np.array(IRFm), 0.95, axis=0) # must be an option (default = 95%)
        cumCIlow = np.quantile(np.array(cumIRFm), 0.05, axis=0) # must be an option (default = 5%)
        cumCIhigh = np.quantile(np.array(cumIRFm), 0.95, axis=0) # must be an option (default = 95%)
    else:
        qtlow = ((100-qt)/2)*0.01
        qthigh = (100 - qtlow)*0.01
        CIlow = np.quantile(np.array(IRFm), qtlow, axis=0)
        CIhigh = np.quantile(np.array(IRFm), qthigh, axis=0)
        cumCIlow = np.quantile(np.array(cumIRFm), qtlow, axis=0)
        cumCIhigh = np.quantile(np.array(cumIRFm), qthigh, axis=0)
    
    CIlow_mat = matrix(CIlow)
    CIhigh_mat = matrix(CIhigh)
    cumCIlow_mat = matrix(cumCIlow)
    cumCIhigh_mat = matrix(cumCIhigh)

    CILv = reshape(CIlow_mat.T, H+1, K**2).T
    CIHv = reshape(CIhigh_mat.T, H+1, K**2).T
    cumCILv = reshape(cumCIlow_mat.T,H+1,K**2).T
    cumCIHv = reshape(cumCIhigh_mat.T,H+1,K**2).T
    
    if verbose:
        if psave == None:
            print(" ")
            print("BIC로 추정한 VAR 최적 시차 = ", phat)
        else:
            print(" ")
            print("사용자가 입력한 VAR 최적 시차 = ", phat)
            
    if verbose:
        '''
        Plotting Results
        '''
        warnings.filterwarnings('ignore')
        
        name = []
        for idx in range(K):
            for idx2 in range(K):
                name1 = f'{col_names[idx2]} to {col_names[idx]}'
                name.append(name1)

        name = np.array(name)
        time = np.matrix(np.arange(0,H+1))

        plt.figure(figsize=(K*3.3, K*3.3))  
        for i in range(1,K**2+1):
            plt.subplot(K,K,i)
            plt.plot(time.T, Theta[i-1,:].T, '-k', time.T, CIHv[i-1,:].T, '-.b', time.T, CILv[i-1,:].T, '-.b', time.T, zeros(H+1,1), '-r')
            plt.xlim([0,H+1])
            plt.title(name[i-1])
            
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        plt.suptitle('IRF Estimation Results', fontsize=14)


        plt.legend(['IRF', 'Bootstrap'])

        plt.figure(figsize=(K*3.3, K*3.3))  
        for i in range(1,K**2+1):
            plt.subplot(K,K,i)
            plt.plot(time.T, cumTheta[i-1,:].T, '-k', time.T, cumCIHv[i-1,:].T, '-.b', time.T, cumCILv[i-1,:].T, '-.b', time.T, zeros(H+1,1), '-r')
            plt.xlim([0,H+1])
            plt.title(name[i-1])
            
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        plt.suptitle('Cumulative IRF Estimation Results', fontsize=14)

        plt.legend(['cumIRF', 'Bootstrap'])
        plt.show()
    
    return VarGeneralizedIrfResult(theta=Theta, cilv=CILv, cihv=CIHv, cum_theta=cumTheta, cum_cilv=cumCILv, cum_cihv=cumCIHv,
                                   phi_hat = phi_hat, omega_hat=Omega_hat, f_mat=F, u_hat=U_hat, y0=Y0, y_lag=Y_lag, y_pred=Y_predm, lag=phat,
                                   _VarGeneralizedIrfResult__lag=lag, _VarGeneralizedIrfResult__h=h,
                                   _VarGeneralizedIrfResult__col_names=col_names, _VarGeneralizedIrfResult__phi_hat=phi_hat, _VarGeneralizedIrfResult__omega_hat=Omega_hat,
                                   _VarGeneralizedIrfResult__f_mat=F, _VarGeneralizedIrfResult__u_hat=U_hat)


def var_decomp(phi_hat, Omega_hat, F, irf, lag, h, verbose, figsize=(20, 5), col_names=None):
    '''VAR(p) 모형의 예측오차 분산분해 결과 도출
    
    Args:
        phi_hat = 축약형 VAR(p) 모형의 계수 행렬 추정량
        Omega_hat = 축약형 VAR(p) 모형의 분산-공분산 행렬 추정량
        F : VAR(p) 모형의 동반행렬 (companion matrix) 형태의 계수 행렬 추정치
        restrict: 사용하고자하는 식별 방법 (short=단기제약, long=장기제약)
        p : VAR(p) 모형의 최적 시차
        H : 예측 오차 도출 시 사용되는 예측 시차
        
    Returns:
        ForVar : 예측 오차 분산 분해 결과 (K by (H+1))
            - K개의 행은 각 충격이 k번째 변수에 기여하는 정도
            - (H+1)개의 열은 각 예측 시차
    '''
    p = lag
    H = h
    restrict = irf
    B0inv = B0invSolve(phi_hat, Omega_hat, restrict)

    K = rows(B0inv)
    FF = eye(p*K)

    ForVar = zeros(K**2,H+1)

    Theta0 = B0inv
    varmat = np.power(Theta0,2)
    Rvarmat = varmat / np.kron(ones(K,1), varmat.sum(axis=1)).T
    ForVar[:,0] = vec(Rvarmat.T).reshape(1,-1)

    for h in range(H):
        FF = F @ FF
        Theta = FF[0:K,0:K] @ B0inv
        varmat = varmat + np.power(Theta,2)
        Rvarmat = varmat / np.kron(ones(K,1), varmat.sum(axis=1)).T
        ForVar[:,h+1] = vec(Rvarmat.T).reshape(1,-1)
    
    if verbose:
        name = []
        for k in range(K):
            #name1 = f"e_{k}"
            name1 = f'{col_names[k]}'
            name.append(name1)

        for k in range(1,K+1):
            combined = pd.DataFrame(ForVar[(k-1)*K:k*K,:].T)
            df_decomp = pd.DataFrame(combined)

            variindex = list(range(0, H+1, 1))
            palette = sns.color_palette("muted", df_decomp.shape[1])

            df_decomp.index = df_decomp.index + 1

            plt.figure(figsize=figsize)

            bottom = np.zeros(len(df_decomp))
            for i, color in zip(range(df_decomp.shape[1]), palette):
                plt.bar(variindex, df_decomp.iloc[:, i], bottom=bottom, color=color)
                plt.xlim([-1,H+1])
                plt.ylim([0,1])
                plt.title(f'Forecast Error Variance for {col_names[k-1]}')
                bottom += df_decomp.iloc[:, i]
            plt.grid()
            plt.legend(name)

        plt.show()
    
    return ForVar


def hist_decomp(y, phi_hat, Omega_hat, F, U_hat, irf, lag, verbose, figsize=(20,5)):
    '''역사적 분해 (Historical Decomposition)
    
    Args:
        y : VAR(p) 모형을 구성하는 반응변수
        p : VAR(p) 모형의 최적 시차
        phi_hat : 축약형 VAR(p) 모형의 계수 행렬 추정량
        Omega_hat : 축약형 VAR(p) 모형의 분산-공분산 행렬 추정량
        F : VAR(p) 모형의 동반행렬 (companion matrix) 형태의 계수 행렬 추정치
        restrict: 사용하고자하는 식별 방법 (short=단기제약, long=장기제약)
        
    Returns:
        HDm : 각 반응변수에 대한 역사적 분해 결과
    '''
    p = lag
    restrict = irf
    col_names = y.columns
    index = y.index
    y = np.array(y)
    if y.shape[1] is None:
        y = y.reshape(-1,1)
        print("y 변수가 단변수 (univariate) 입니다.")
    T,K = y.shape
    
    # <Step 1>: Estimate the reduced-form VAR
    # 함수 밖에서 이루어짐.
    
    # <Step 2>: Identify the structural-form VAR
    B0inv = B0invSolve(phi_hat,Omega_hat,restrict)

    # <Step 3>: Obtain the structural form residual
    e_hat = U_hat @ inv(B0inv)

    # <Step 4>: Obtain the Impulse response functions
    Theta = irf_estimate(F,p,T-p-1,B0inv)

    # <Step 5>: Obtain the weights for the effect of a given shock
    yhatm = zeros(K,T-p-1,K) # shock, T, variable

    for ind_var in range(K):
        yhats = zeros(T-p-1,K)
        for t in range(T-p-1):
            for ind_shock in range(K):
                dot_temp = Theta[ind_var+ind_shock*K,:t+1] @ e_hat[t::-1,ind_shock]
                yhats[t,ind_shock] = dot_temp

        temp = zeros(1,T-p-1)
        for ind_shock in range(K):
            temp = np.vstack((temp,yhats[:,ind_shock].T))

        temp = temp[1:,:]
        yhatm[:,ind_var,:] = temp

    T1,K = U_hat.shape
    
    if verbose:
        # Plotting
        name = []
        for k in range(K):
            #name1 = f"e_{k}"
            name1 = f'{col_names[k]}'
            name.append(name1)

        for ind_var in range(0,K):
            HD = yhatm[:,ind_var,:]
            df_decomp = pd.DataFrame(HD.T)
            variindex = list(range(1,T1,1))
            palette = sns.color_palette("muted", df_decomp.shape[1])
            df_decomp.index = df_decomp.index + 1
            #df_decomp.index = index[p+1:]
            plt.figure(figsize=figsize)

            bottom_pos = np.zeros(T1-1)
            bottom_neg = np.zeros(T1-1)

            for i, color in zip(range(df_decomp.shape[1]), palette):

                values = df_decomp.iloc[:,i]
                bottom = np.where(values >= 0, bottom_pos, bottom_neg)

                plt.bar(variindex, values, bottom=bottom, color=color)
                #plt.xlim([-1,T1+1])
                plt.title(f'Historical Decomposition for {col_names[ind_var]}')

                bottom_pos = np.where(values >= 0, bottom_pos + values, bottom_pos)
                bottom_neg = np.where(values < 0, bottom_neg + values, bottom_neg)

            plt.legend(name)
            plt.grid()
            ax = plt.gca()
            ax.set_xticks(variindex)
            ax.set_xticklabels(index[p+1:].strftime('%Y-%m'))
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, prune='both', nbins=10))

        plt.show()
    
    HDm = yhatm
    
    return HDm