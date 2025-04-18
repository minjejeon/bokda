import numpy as np
import pandas as pd
import scipy.stats as sp
import matplotlib.pyplot as plt

from typing import Union

from ...utils.operator import rows, ones, matrix, array, zeros, minc, sumc, inv, det, cols, vec, reshape, eye, meanc, demeanc

#from ...utils.ols import autocov

from ..var.var import var
from ..var.vecm import VECM_MLE

from dataclasses import dataclass, field


class AutoCorResult:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
            
    def get_description(self):
        desc = {
            "결과": ['corr', 'bound'],
            "설명": ['추정된 자기상관함수 행렬(tau_max by 1)', '2 표준오차 bound']
        }

        df = pd.DataFrame(desc)
        df.set_index("결과", inplace=True)
        df.index.name = None
        #df = df.style.set_properties(**{'text-align': 'left'})
        #df = df.set_table_styles([{'selector': 'th', 'props': [('text-align', 'left')]}])

        return df
    
    def plot_acf(self, title: bool=True, title_fontsize: int=14, fontsize: int=12, colors: str='b', linestyles: str='-'):
        
        self._plot_acf(self.corr, self.bound, title, title_fontsize, fontsize, colors, linestyles)
        
    def _plot_acf(self, corr, bound, title, title_fontsize, fontsize, colors, linestyles):
        
        var_name = self.var_name
        plt.figure(3)
        plt.stem(np.arange(len(corr)), corr, markerfmt='o', linefmt='-')
        if title:
            plt.title(fr'Sample Autocorrelation Function for {var_name[0]}', fontsize=title_fontsize)
        plt.xlabel(r'Lag', fontsize=fontsize)
        plt.xlim(-0.15,rows(self.corr)-0.9)
        plt.hlines(bound[:, 0], 0, len(corr)-1, colors=colors, linestyles=linestyles)
        plt.hlines(bound[:, 1], 0, len(corr)-1, colors=colors, linestyles=linestyles)
        plt.grid(True)
        plt.show()


def autocor(X: Union[pd.Series, pd.DataFrame], tau_max=None, verbose=False):
    '''단변수 확률과정의 자기상관함수 추정 및 백색잡음과정 단순 가설 검정
    
    Args:
        X : 추정하고자하는 변수(단변수)
        tau_max : 자기상관함수 추정에 사용되는 최대 시차 (default = 20)
    
    Returns:
        corr : 추정된 자기상관함수 행렬, tau_max by 1
        bound : 2 표준 오차 한도, Box et al. (2015), pp. 33
    '''
    if isinstance(X, pd.Series):
        X = X.to_frame()
        
    var_name = X.columns
    X = np.array(X)
    
    #if X.shape[1] < 2:
    #if X.ndim < 2:
    #    X = X.reshape(-1,1)
    #else:
    #    print("경고: 입력되는 X는 단변수여야합니다.")
    
    if X.shape[1] >= 2:
        print("경고: 입력되는 X는 단변수여야 합니다.")
        
    T,K = X.shape
    
    if tau_max is None:
        tau_max = 20
    else:
        tau_max = tau_max
    
    if K > 1:
        print("경고: X는 단변수 데이터프레임 또는 시리즈이어야 합니다.")
    else:
        cov = autocov(X, tau_max)
        corr = cov/cov[0,0]

    bound = matrix((2 * np.sqrt(1/T), -2 * np.sqrt(1/T))) # Box et al. (2015), pp. 33

    if verbose:
        plt.figure(3)
        plt.stem(np.arange(len(corr)), corr, markerfmt='o', linefmt='-')
        plt.title(r'Sample Autocorrelation Function', fontsize=14)
        plt.xlabel(r'Lag', fontsize=12)
        plt.ylabel(r'Sample Autocorrelation', fontsize=12)
        plt.xlim(-0.15,rows(corr)-0.9)
        plt.hlines(bound[:, 0], 0, len(corr)-1, colors='b', linestyles='-')
        plt.hlines(bound[:, 1], 0, len(corr)-1, colors='b', linestyles='-')
        plt.grid(True)
        plt.show()

    return AutoCorResult(corr=corr, bound=bound, var_name=var_name)


def autocov(X,tau_max=None):
    '''단변수 확률과정의 자기공분산함수 추정
    
    Args:
        X : 추정하고자하는 변수 (단변수여야 함)
        tau_max : 최대 시차 (default = 20)
    
    Returns:
        cov : 추정된 자기공분산 함수 행렬, tau_max by 1
    '''
    X = array(X)
    if X.shape[1] < 2:
        X = X.reshape(-1,1)
    else:
        print("경고: 입력되는 X는 단변수여야합니다.")
    
    T,K = X.shape
    
    if tau_max is None:
        tau_max = 20
    else:
        tau_max = tau_max
    
    if K > 1:
        print("경고: X는 열벡터여야 합니다.")
    else:
        X_ = meanc(X)
        Xd = demeanc(X)
        cov = zeros(tau_max+1,1)
        for i in range(tau_max+1):
            cov[i,:] = (1/T) * Xd[i:T,:].T @ Xd[0:T-i,:]
    
    return cov


class WNResult:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_description(self):
        desc = {
            "결과": ['box_pierce', 'pval_bp', 'ljung_box', 'pval_lb'],
            "설명": ['Box-Pierce 통계량 값', 'Box-Pierce 통계량의 p-값', 'Ljung-Box 통계량 값', 'Ljung-Box 통계량의 p-값']
        }

        df = pd.DataFrame(desc)
        df.set_index("결과", inplace=True)
        df.index.name = None
        #df = df.style.set_properties(**{'text-align': 'left'})
        #df = df.set_table_styles([{'selector': 'th', 'props': [('text-align', 'left')]}])

        return df

    def get_table(self):
        index = ['stat', 'p-val']
        cols = ['Box-Pierce', 'Ljung-Box']
    
        table = pd.DataFrame(np.around(self.stat_p,4), index=index, columns=cols)
        print(table)
        print(" ")
        print("H0:r(1)=r(2)=...r(h)=0; 백색잡음과정이다.")
        print("유의수준 < p-값 = 유의수준 하에서 귀무가설 (H0)을 기각, i.e. 백색잡음과정이 아니다.")


def white_noise(X: Union[pd.Series, pd.DataFrame], tau_max=None, verbose=False):
    '''백색잡음과정 결합 가설 검정 (Box-Pierce와 Ljung-Box 검정)
    
    Args:
        X : 추정하고자하는 변수 (단변수여야 함)
        tau_max : 추정에 사용하는 최대 시차 (default = 20)
    
    Returns:
        box_pierce : Box-Pierce 통계량 값
        pval_bp : Box-Pierce 통계량의 p-값
        ljung_box : Ljung-Box 통계량 값
        pval_lb : Ljung-Box 통계량의 p-값
        
    H0: r(1)=r(2)=...=r(h)=0 (백색잡음과정이다)
    H1: 어떠한 1<=s<=h에 대해 등호가 성립하지 않는다 (백색잡음과정이 아니다)
    '''
    if isinstance(X, pd.Series):
        X = X.to_frame()
        
    var_name = X.columns
    X = np.array(X)
    
    if X.shape[1] >= 2:
        print("경고: 입력되는 X는 단변수여야합니다.")
        
    T,K = X.shape
    
    if tau_max is None:
        tau_max = 20
    else:
        tau_max = tau_max

    if K > 1:
        print("경고: X는 열벡터여야 합니다.")
    else:
        cov = autocov(X,tau_max)
        corr = cov[1:rows(cov)]/cov[0,0]

    BoxPierce = T * corr.T @ corr
    pval1 = 1 - sp.chi2.cdf(BoxPierce, tau_max)
    test1 = np.vstack((BoxPierce, pval1))

    s = ones(tau_max,1) * T - matrix(np.arange(1,tau_max+1)).T
    LjungBox = T*(T+2) * (corr/s).T @ corr
    pval2 = 1 - sp.chi2.cdf(LjungBox, tau_max)
    test2 = np.vstack((LjungBox, pval2))
    
    #index = range(0,2)
    index = ['stat', 'p-val']
    cols = ['Box-Pierce', 'Ljung-Box']
    
    data = np.hstack((test1, test2))
    
    if verbose:
        result = pd.DataFrame(np.around(data,4), index=index, columns=cols)
        print(result)
        print(" ")
        print("H0:r(1)=r(2)=...r(h)=0; 백색잡음과정이다.")
        print("유의수준 < p-값 = 유의수준 하에서 귀무가설 (H0)을 기각, i.e. 백색잡음과정이 아니다.")
    
    return WNResult(box_pierce=BoxPierce, pval_bp=pval1, ljung_box=LjungBox, pval_lb=pval2, stat_p=data)


def granger_causality_test(y1, y2, lag=None):
    '''그레인저 인과관계 검정 (Granger Causality Test)
    
    Args:
        y1 : 그레인저 인과되는 (Granger caused) 변수
        y2 : 그레인저 인과하는 (Granger causes) 변수
        p : VAR(p) 모형의 최적 시차 (default = None)
        
    Returns:
        Wald : Wald 검정 통계량
        pval : p-값 (틀린 기각의 확률)

    H0: y2가 y1을 그레인저 인과하지 않는다, i.e. VAR(p) 모형의 계수행렬의 모든 우측 상방 블록이 모두 영행렬이다.
    귀무가설 하에서 왈드 검정 통계량은 카이제곱 분포를 따른다.
    '''
    col1_names = y1.columns
    col2_names = y2.columns
    p = lag
    
    y1 = np.array(y1)
    if y1.shape[1] < 2 :
        y1 = y1.reshape(-1,1)
        
    y2 = np.array(y2)
    #if y2.shape[1] < 2 :
    if y2.ndim < 2:
        y2 = y2.reshape(-1,1)
    
    K1 = cols(y1)
    K2 = cols(y2)
    K = K1 + K2

    y = np.hstack((y1,y2))
    res = var(y,p)
    phi_hat = res.phi_hat
    Omega_hat = res.omega_hat
    F = res.f_mat
    U_hat = res.u_hat
    Y0 = res.y0
    Y_lag = res.y_lag
    Y_predm = res.y_pred
    phat = res.lag
    
    p = phat
    Y_lag2 = Y_lag.T @ Y_lag

    alpha = vec(phi_hat) # vec(B)
    OMEGA = np.kron(Omega_hat, inv(Y_lag2))

    C = zeros(1,K*p*K)
    Cmat = zeros(K,K*p)

    for ik in range(K1):
        for ip in range(p):
            for jk in range(K2):
                Ctemp = zeros(1,K*p*K)
                Ctemp[0,(ik*K*p)+(ip*K)+K1+(jk)] = 1
                Cmat = Cmat + reshape(Ctemp.T, K*p, K).T
                C = np.vstack((C,Ctemp))
            
    C = C[1:rows(C),:]
    Wald = (C@alpha).T @ inv(C@OMEGA@C.T) @ (C@alpha)
    pval = 1 - sp.chi2.cdf(Wald,rows(C))
    print(f'H0: {col2_names.values}(y2)가 {col1_names.values}(y1)을 그레인져 인과 (Granger Cause)하지 않는다.')
    print(' ')
    print(f'The p-value is {np.around(pval[0,0],4)}')
    
    return Wald, pval


def mic(y: Union[pd.Series, pd.DataFrame], 
        m: str='trend',
        ic: str='mbic',
        print_res: bool=False):
    '''수정된 정보기준 (Modified Information Criteria, MIC)를 통한 최적 시차 추정
    
    Args:
        y : 추정하고자하는 변수, 단변수만 가능
        m : 사용하고자하는 모형의 종류
            m = 'constant': 상수항만 존재하는 모형
            m = 'trend' : 상수항과 선형 추세항이 존재하는 모형
        ic : MIC의 종류
            ic = 'maic' : MAIC, penalty = 2
            ic = 'mbic' : MBIC, penalty = log(T-pmax-1)
            
    Returns:
        phat : 추정된 최적 시차
    '''
    y = np.array(y)
    #if y.shape[1] < 2 :
    if y.ndim < 2:
        y = y.reshape(-1,1)
    
    T,K = y.shape
    pmax = round(12*((T/100)**(1/4))) # Rule of thumb - Ng and Perron (2001), pp. 1536
    
    # <Step 0>: Specifying the method and model
    if ic == 'maic': # MAIC
        CT = 2
    elif ic == 'mbic': # MBIC (default)
        CT = np.log(T-pmax-1)
    
    if m == 'constant':
        z = ones(T,1)
    elif m == 'trend':
        z = np.hstack((ones(T,1),matrix(range(1,T+1)).T))
    else:
        print("경고: 올바르지 않은 m값이 사용되었습니다.")
        print("m은 상수항만 존재하는 모형의 경우 'constant'이거나 상수항과 선형 추세항이 모두 존재하는 모형의 경우 'trend'여야 합니다.")
    
    # <Step 1>: OLS Detrending, Qu and Perron (2007)
    gam_OLS = inv(z.T@z) @ (z.T@y) # OLS estimate
    vhat = y - z @ gam_OLS # OLS residual
    v_lag = vhat[pmax:T-1]

    # <Step 2>: Obtain the differenced terms
    Dv = zeros(T-pmax-1,pmax)
    for i in range(pmax):
        temp = vhat[pmax-i:T-i-1,0] - vhat[pmax-i-1:T-i-2,0]
        Dv[:,i] = temp.reshape(1,-1)

    # <Step 3>: Obtain MIC for each p = 0,...,pmax
    MICm = zeros(pmax+1,1)
    for t in range(pmax+1):
        if t == 0:
            dv = 0
        else:
            dv = Dv[:,0:t]
        
        if t == 0:
            v0 = vhat[pmax+1:T,0]
            v_1 = vhat[pmax:T-1,0]
        
            aux_y = v0 - v_1
            aux_x = v_1 - 0
        else:
            v0 = vhat[pmax+1:T,0]
            v_1 = vhat[pmax:T-1,0]
        
            aux_y = v0 - v_1
            coef1 = inv(dv.T @ dv) @ (dv.T @ aux_y)
            aux_y = aux_y - dv @ coef1
        
            coef2 = inv(dv.T @ dv) @ (dv.T @ v_1)
            aux_x = v_1 - dv @ coef2
        
        temp1 = aux_x.T @ aux_x
        temp2 = temp1.reshape(1,-1)
        temp3 = aux_x.T @ aux_y
        temp4 = temp3.reshape(1,-1)
        b0 = inv(temp2) @ (temp4)
        resid = aux_y - aux_x * b0
        resid = resid.reshape(-1,1)
        sig2 = (resid.T @ resid) / (T-pmax-1)
    
        tau = (b0**2/sig2) @ (v_lag.T @ v_lag)
    
        mic = np.log(sig2) + CT *(tau+t)/(T-pmax-1)
        MICm[t,0] = mic[0,0]

    min_val, phat = minc(MICm)
    phat = phat[0]
    if ic == 'maic':
        name = 'Modified AIC'
    elif ic == 'mbic':
        name = 'Modified BIC'
        
    if print_res:
        print(f'{name}를 사용하여 추정된 ADF 모형의 AR 최적 시차는 {phat}')
    
    return phat


def adf_test_gls_detrending(y: Union[pd.Series, pd.DataFrame],
                            m: str='trend',
                            ic: str='mbic',
                            print_res: bool=False):
    '''GLS 추세 제거를 한 ADF 단위근 검정
    
    Args:
        y : 추정하고자하는 변수, 단변수만 가능
        m : 사용하고자하는 모형의 종류
            m = 'constant' : 상수항만 존재하는 모형
            m = 'trend' : 상수항과 선형 추세항이 존재하는 모형
        ic : ADF 모형 최적시차 추정을 위한 정보기준 인자
        print_res : 결과 출력 인자
    
    Returns:
        res : ADF GLS 단위근 검정의 검정 통계량
        
    모형에 따른 임계값표 (ref: Ng and Perron 2001, pp. 1524)
    1% CV = -2.58 for m=1 and -3.42 for m=2
    5% CV = -1.98 for m=1 and -2.91 for m=2
    10% CV = -1.62 for m=1 and -2.62 for m=2
    
    H0: AR 단위근이 존재한다.
    |검정 통계량| > |임계값|의 경우 AR 단위근이 존재한다는 귀무가설을 기각
    '''
    y = np.array(y)
    #if y.shape[1] < 2 :
    if y.ndim < 2:
        y = y.reshape(-1,1)

    T,K = y.shape
    
    p = mic(y, m, ic, print_res) # Estimate p by MIC

    # <Step 0>: Specifying the model
    if m == 'constant':
        z = ones(T,1)
        a = 1 - (7/T)
        term = 1
    elif m == 'trend':
        z = np.hstack((ones(T,1),matrix(range(1,T+1)).T))
        a = 1 - (13.5/T)
        term = 2
    else:
        print("경고: 올바르지 못한 m 값이 입력되었습니다.")
        print("m은 상수항만 존재하는 모형의 경우 'constant', 상수항과 선형 추세항이 모두 존재하는 모형의 경우 'trend'이어야 합니다.")
    
    # <Step 1>: Apply GLS to ystar and zstar
    ystar = y - a * np.vstack((0,y[0:T-1,:]))
    zstar = z - a * np.vstack((zeros(1,term),z[0:T-1,:]))

    gam_GLS = inv(zstar.T @ zstar) @ (zstar.T @ ystar) # GLS estimator
    vtilde = y - z * gam_GLS # GLS detrended residual

    # <Step 2>: Regress the DF residual and obtain alpha (Frisch-Waugh Theorem is usded)
    #   model: v0 = v_1*alpha + X*beta + e
    #   (1) aux_y = v0 - X * coef1
    #   (2) aux_x = v_1 - X * coef2
    #   (3) alpha = inv(aux_x' @ aux_x) @ (aux_x' @ aux_y)

    Dv = zeros(T-p-1,p)
    for i in range(p):
        temp = vtilde[p-i:T-i-1,0] - vtilde[p-i-1:T-i-2,0]
        Dv[:,i] = temp.reshape(1,-1)
    
    if p == 0:
        aux_y = vtilde[p+1:T,0]
        aux_x = vtilde[p:T-1,0]
    else:
        v0 = vtilde[p+1:T,0]
        coef1 = inv(Dv.T @ Dv) @ (Dv.T @ v0)
        aux_y = v0 - Dv @ coef1
    
        v_1 = vtilde[p:T-1,0]
        coef2 = inv(Dv.T @ Dv) @ (Dv.T @ v_1)
        aux_x = v_1 - Dv @ coef2
    
    temp1 = aux_x.T @ aux_x
    temp2 = temp1.reshape(1,-1)
    temp3 = aux_x.T @ aux_y
    temp4 = temp3.reshape(1,-1)
    alpha = inv(temp2) @ (temp4)

    # <Step 3>: Derive the test statistic
    resid = aux_y - aux_x * alpha
    resid = resid.reshape(-1,1)
    s2 = (resid.T @ resid) / (T-p-1)

    test = (alpha-1)/np.sqrt(s2/(aux_x.T @ aux_x))
    
    if m == 'constant':
        CV = matrix([-2.58, -1.98, -1.62])
    elif m == 'trend':
        CV = matrix([-3.42, -2.91, -2.62])
    else:
        print("경고: 올바르지 못한 m값이 입력되었습니다.")
        print("m은 상수항만 존재하는 모형의 경우 'constant', 상수항과 선형 추세항이 모두 존재하는 모형의 경우 'trend'이어야 합니다.")
    
    if m == 'constant':
        text = '상수항만 있는'
    elif m == 'trend':
        text = '상수항과 선형추세항이 함께 있는'
    if print_res:
        print(" ")
        print(f'{text} 모형을 사용한 ADF-GLS 검정통계량은 {round(test[0,0],4)}')
        print("CV 1% = ",CV[0,0]," 5% = ",CV[0,1]," 10% = ",CV[0,2])
        print(" ")
        print("귀무가설 (H0): AR 단위근이 존재한다.")
        print("|test statistic| > |CV| : 귀무가설을 유의수준 x % 하에서 기각한다.")
        print("|test statistic| < |CV| : 귀무가설을 유의수준 x % 하에서 기각하지 못한다.")
        print(" ")
        print(" ref: Ng and Perron (2001), <Table I>, pp. 1524")

    return test


def read_table_VECM(y,option,c,sig_level,LR_test):
    '''Johansen System Method로 귀무가설 검정 시 Osterwald-Lenum (1992)의 임계값 표에 
    기반한 절편항 제외 및 절편항 포함 VECM 모형의 rank 추정
    
    Args:
        y = 사용하는 VECM 모형의 반응변수
        option = 사용하고자하는 모형의 종류
            - 1: 상수항이 존재하는 모형 (상수항이 공적분 관계에 대한 제약이 없는 VECM 모형)
            - 2: 상수항이 존재하지 않는 모형 (상수항이 공적분 관계에 대한 제약이 있는 VECM 모형)
        c = 사용하고자하는 검정 통계량의 종류
            - 1 : LR 대각합 (trace) 통계량
            - 2 : LR 최대 고유값 (maximum eigenvalue) 통계량
        sig_level = 검정에 사용할 유의수준 (잘못된 기각의 확률)
            - 10 : 10% 유의수준
            - 5 : 5% 유의수준
            - 1 : 1% 유의수준
        LR_test : LR 검정 통계량 (대각합 통계량 또는 최대 고유값 통계량)
        
    Returns:
        rank = 공적분 rank 추정치
            - value of r when fail to reject the null hypothesis such that rank(Pi) = r
        
    refer to Table 1.1* in Osterwald-Lenum (1992) for option = 1 case
    refer to Table 1* in Osterwald-Lenum (1992) for option = 2 case
    '''
    # (1) Unrestricted, max_eig
    table_unrestrict_maxeig = matrix([[2.43, 4.82, 6.50, 8.18, 9.72, 11.65],
                                        [7.53, 10.77, 12.91, 14.90, 17.07, 19.19],
                                        [12.65, 16.51, 18.90, 21.07, 22.89, 25.75],
                                        [17.83, 22.16, 24.78, 27.14, 29.16, 32.14],
                                        [23.20, 28.09, 30.84, 33.32, 35.80, 38.78],
                                        [28.26, 33.35, 36.35, 39.43, 41.86, 44.59],
                                        [33.63, 39.00, 42.06, 44.91, 47.59, 51.30],
                                        [39.11, 45.03, 48.43, 51.07, 53.85, 57.07],
                                        [44.42, 50.43, 54.01, 57.00, 59.80, 63.37],
                                        [49.57, 55.75, 59.19, 62.42, 64.98, 68.61],
                                        [54.70, 61.25, 65.07, 68.27, 70.69, 74.36]])

    # (2) Unrestricted, trace
    table_unrestrict_trace = matrix([[2.43, 4.82, 6.50, 8.18, 9.72, 11.65],
                                       [9.39, 13.21, 15.66, 17.95, 20.08, 23.52],
                                       [20.11, 25.39, 28.71, 31.52, 34.48, 37.22],
                                       [34.91, 41.65, 45.23, 48.28, 51.54, 55.43],
                                       [53.81, 61.75, 66.49, 70.60, 74.04, 78.87],
                                       [75.93, 85.38, 90.39, 85.18, 99.32, 104.20],
                                       [102.16, 113.05, 118.99, 124.25, 129.75, 136.06],
                                       [132.63, 144.70, 151.38, 157.11, 162.75, 168.92],
                                       [166.07, 179.54, 186.54, 192.84, 198.06, 204.79],
                                       [202.77, 217.86, 226.34, 232.49, 238.26, 246.27],
                                       [244.21, 260.85, 269.53, 277.39, 283.84, 292.65]])

    # (3) Restricted, max_eig
    table_restrict_maxeig = matrix([[3.40, 5.91, 7.52, 9.24, 10.80, 12.97],
                                        [8.27, 11.54, 13.75, 15.67, 17.63, 20.20],
                                        [13.47, 17.40, 19.77, 22.00, 24.07, 26.81],
                                        [18.70, 22.95, 25.56, 28.14, 30.32, 33.24],
                                        [23.78, 28.76, 31.66, 34.40, 36.90, 39.79],
                                        [29.08, 34.25, 37.45, 40.30, 43.22, 46.82],
                                        [34.73, 40.13, 43.25, 46.45, 48.99, 51.91],
                                        [39.70, 45.53, 48.91, 52.00, 54.71, 57.95],
                                        [44.97, 50.73, 54.35, 57.42, 60.50, 63.71],
                                        [50.21, 56.52, 60.25, 63.57, 66.24, 69.94],
                                        [55.70, 62.38, 66.02, 69.74, 72.64, 76.63]])

    # (4) Restricted, trace
    table_restrict_trace = matrix([[3.40, 5.91, 7.52, 9.24, 10.80, 12.97],
                                       [11.25, 15.25, 17.85, 19.96, 22.05, 24.60],
                                       [23.28, 28.75, 32.00, 34.91, 37.61, 41.07],
                                       [38.84, 45.65, 49.65, 53.12, 56.06, 60.16],
                                       [58.46, 66.91, 71.86, 76.07, 80.06, 84.45],
                                       [81.90, 91.57, 97.18, 102.14, 106.74, 111.01],
                                       [109.17, 120.35, 126.58, 131.70, 136.49, 143.09],
                                       [139.83, 152.56, 159.48, 165.58, 171.28, 177.20],
                                       [174.88, 198.08, 196.37, 202.92, 208.81, 215.74],
                                       [212.93, 228.08, 236.54, 244.15, 251.30, 257.68],
                                       [254.84, 272.82, 282.45, 291.40, 298.31, 307.64]])

    if option == 1 and c == 1:
        table = table_unrestrict_trace
    elif option == 1 and c == 2:
        table = table_unrestrict_maxeig
    elif option == 2 and c == 1:
        table = table_restrict_trace
    elif option == 2 and c == 2:
        table = table_restrict_maxeig
    else:
        print("경고: 올바르지 않은 option 및 c값이 사용되었습니다.")
        print("option값은 상수항이 존재하는 모형의 경우 1, 상수항이 존재하지 않는 모형의 경우 2여야합니다.")
        print("c 값은 LR 대각합 통계량의 경우 1, LR 최대 고유값 통계량의 경우 2여야합니다.")

    table = np.hstack((table[:,2],table[:,3],table[:,5])) # 10% 5% 1% CV

    if sig_level == 10:
        table = table[:,0]
    elif sig_level == 5:
        table = table[:,1]
    elif sig_level == 1:
        table = table[:,2]
    else:
        print("경고: 올바르지 않은 sig_level 값이 사용되었습니다.")
        print("sig_level은 1% 유의수준의 경우 1, 5% 유의수준의 경우 5, 10% 유의수준의 경우 10이어야합니다.")

    if c == 1:
        test_stat = LR_test[:,0]
    elif c == 2:
        test_stat = LR_test[:,1]
    else:
        print("경고: 올바르지 않은 c값이 사용되었습니다.")
        print("c 값은 LR 대각합 통계량의 경우 1, LR 최대 고유값 통계량의 경우 2여야합니다.")

    test_stat = test_stat.reshape(-1,1)

    y = array(y)
    if y.shape[1] < 2 :
        y = y.reshape(-1,1)
        print("y 변수가 단변수 (univariate) 입니다.")
        
    T,K = y.shape

    for r in range(K-1):
        row = K-r
        CV = table[row,:]
        if abs(test_stat[r,:]) < abs(CV): # fail to reject the null
            break

    if r == K:
        print("사용하는 반응변수가 I(0)입니다. 수준 변수들로 구성된 VAR 모형을 사용하세요.")

    rank = r+1
    
    return rank


def COINT_test(y,option,c,sig_level,p=None):
    '''VAR 모형의 공적분 검정, Johansen system method
    
    Args:
        y = 사용하는 VAR 모형의 반응변수
        option = 사용하고자하는 모형의 종류
            - 1: 상수항이 존재하는 모형 (상수항이 공적분 관계에 대한 제약이 없는 VECM 모형)
            - 2: 상수항이 존재하지 않는 모형 (상수항이 공적분 관계에 대한 제약이 있는 VECM 모형)
        c = 사용하고자하는 검정 통계량의 종류
            - 1 : LR 대각합 (trace) 통계량
            - 2 : LR 최대 고유값 (maximum eigenvalue) 통계량
        sig_level = 검정에 사용할 유의수준 (잘못된 기각의 확률)
            - 10 : 10% 유의수준
            - 5 : 5% 유의수준
            - 1 : 1% 유의수준
        p = VAR 모형의 최적 시차 (default = None)
            
    Returns:
        rank : 공적분 rank 추정치
        LR_test : LR 검정 통계량 (대각합 통계량 또는 최대 고유값 통계량)
    '''
    y = array(y)
    if y.shape[1] < 2 :
        y = y.reshape(-1,1)
        print("y 변수가 단변수 (univariate) 입니다.")
    
    T,K = y.shape
    r = K
    
    alpha, beta, Gamma, Lam, Sigmau, u_hat, phat = VECM_MLE(y,r,option,p)
    
    K1,K2 = Lam.shape
    Lam1 = eye(K1) - Lam
    lamb_temp = zeros(K,1)
    for i in range(K):
        lamb_temp[i,:] = Lam1[i,i]
    
    lamb = -(T-phat) * np.log(lamb_temp)

    LR = zeros(K,1)
    for i in range(K):
        LR[i,:] = sumc(lamb[i:,:])
    
    LR_trace = LR
    LR_max = lamb

    LR_test = np.hstack((LR_trace,LR_max))
    
    rank = read_table_VECM(y,option,c,sig_level,LR_test)
    
    return rank, LR_test

def longrun_var(y: Union[pd.Series, pd.DataFrame],
             window: str='quadratic'):
    '''장기 분산 추정
    
    Args:
        y : 추정하고자하는 변수(들), 단변수 또는 다변수 모두 가능
        window : 사용하고자하는 윈도우 종류
            - 'bartlett' = Bartlett Window를 사용 (추정된 장기 분산이 Newey-West 추정량이라고도 부름)
            - 'quadratic' = AR(1) 과정을 통해 근사한 밴드 넓이 파라미터 (bandwidth parameter)를 사용한 Quadratic Window (default)
              reference: Andrews (1991)
    
    Returns:
        jhat : 장기 분산 추정치
    '''
    
    y = np.array(y)
    #if y.shape[1] < 2 :
    if y.ndim < 2:
        y = y.reshape(-1,1)
    
    vhat_ = meanc(y)
    vhat = demeanc(y)

    T, K = vhat.shape

    # 1. Get estimator of autocovariance function
    Gamma = zeros(T*K,K)

    for j in range(T):
        gamma = (1/T) * vhat[j:T,:].T @ vhat[0:T-j,:]
        Gamma[K*j:K*(j+1),0:K] = gamma
    
    # 2. Select the bandwidth parameter (by AR(1) approximation)
    rho = zeros(K,1)
    sigma = zeros(K,1)

    # 2-(1). AR(1) regression
    for i in range(K):
        v = vhat[:,i].reshape(-1,1)
        v0 = v[1:T,:]
        v_lag = v[0:T-1,:]
        v_lag2 = v_lag.T @ v_lag
    
        r = inv(v_lag2) @ (v_lag.T @ v0)
        rho[i,:] = r
    
        e_hat = v0 - v_lag @ r
        sigma[i,:] = (e_hat.T @ e_hat) / T
    
    # 2-(2) AR(1) approximation
    denom1 = 0
    numer1 = 0
    for ind in range(K):
        numer1 = numer1 + 4*(rho[ind,:]**2)*(sigma[ind,:]**2) / ((1-rho[ind,:])**6 * (1+rho[ind,:])**2)
        denom1 = denom1 + (sigma[ind,:]**2) / ((1-rho[ind,:])**4)
    
    alpha1 = numer1 / denom1
    
    denom2 = 0
    numer2 = 0

    for ind in range(K):
        numer2 = numer2 + 4*(rho[ind,:]**2)*(sigma[ind,:]**2) / ((1-rho[ind,:])**8)
        denom2 = denom2 + (sigma[ind,:]**2) / ((1-rho[ind,:])**4)

    alpha2 = numer2 / denom2

    # 2-(3) Bandwidth parameter
    if window == 'bartlett': # Bartlett Kernel (Newey-West)
        temp = alpha1*T
        s_T1 = 1.1447 * temp**(1/3)
        s_T = np.floor(s_T1)
        s_T = int(s_T)
    elif window == 'quadratic': # Quadratic Kernel (Andrews 1991)
        temp = alpha2*T
        s_T = 1.3221 * temp**(1/5)
    
    # 3. Applying Window
    J = Gamma[0:K,:] # Gamma(0)

    if window == 'bartlett':
        for j in np.arange(1,s_T+1):
            weight = 1 - np.abs(j/s_T1)
            J = J + weight*Gamma[j*K:j*K+K,:]
            
        for j in np.arange(1,s_T+1):
            weight = 1 - np.abs(j/s_T1)
            J = J + weight*Gamma[j*K:j*K+K,:].T
        
    elif window == 'quadratic':
        for j in np.arange(1,T-1):
            d = (6*np.pi*(j/s_T)) / 5
            weight = (25/(12*(np.pi**2)*(j/s_T)**2)) * ((np.sin(d)/d) - np.cos(d))
            J = J + weight*Gamma[j*K:j*K+K,:]
        
        for j in np.arange(1,T-1):
            d = (6*np.pi*(-j/s_T)) / 5
            weight = (25/(12*np.pi**2*(j/s_T)**2)) * ((np.sin(d)/d) - np.cos(d))
            J = J + weight*Gamma[j*K:j*K+K,:].T
        
    Jhat = J * (T/(T-K))
    
    return Jhat

def dfuller(*args, **kwargs):
    '''Augmented Dicker Fuller test'''
    from statsmodels.tsa.stattools import adfuller
    verbose = kwargs.pop('verbose', False)
    ans = adfuller(*args, **kwargs)
    if verbose:
        regresults = kwargs.get('regresults', False)
        if regresults:
            stat,pval,cv,regout = ans
            nobs = regout.nobs
            lags = regout.usedlag
        else:
            stat,pval,lags,nobs,cv,_ = ans
        print('Augmented Dickery-Fuller test for unit root')
        print('')
        print('Number of obs  = %5d' % nobs)
        print('Number of lags = %5d' % lags)
        print('')
        style = kwargs.get('regression', 'c')
        if style=='n':
            desc = 'no constant, no trend'
        elif style=='c':
            desc = 'constant only'
        elif style=='ct':
            desc = 'constant and trend'
        elif style=='ctt':
            desc = 'constant, and linear and quadratic trend'
        print(f'H0: Random walk ({desc})')
        print('')
        print('     Test      -------- critical value ---------')
        print('statistic          10%           5%           1%')
        print('------------------------------------------------')
        print('%9.3f    %9.3f    %9.3f    %9.3f' % \
              (stat, cv['10%'], cv['5%'], cv['1%']))
        print('------------------------------------------------')
        print('p-value = %.4f' % pval)
    return ans

def kpss(*args, **kwargs):
    '''KPSS test (with statsmodels branched)'''
    import warnings
    from statsmodels.tsa.stattools import kpss
    verbose = kwargs.pop('verbose', False)
    if verbose:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ans = kpss(*args, **kwargs)
    else:
        ans = kpss(*args, **kwargs)
    if verbose:
        regresults = kwargs.get('store', False)
        if regresults:
            stat,pval,lags,cv,regout = ans
            nobs = regout.nobs
        else:
            stat,pval,lags,nobs,cv = ans
        print('KPSS test for stationarity')
        print('')
        print('Number of obs  = %5d' % nobs)
        print('Number of lags = %5d' % lags)
        print('')
        style = kwargs.get('regression', 'c')
        if style=='c':
            desc = 'constant'
        elif style=='ct':
            desc = 'trend'
        print(f'H0: Stationary around a {desc}')
        print('')
        print('     Test      --------------- critical value ---------------')
        print('statistic          10%           5%         2.5%           1%')
        print('-------------------------------------------------------------')
        print('%9.3f    %9.3f    %9.3f    %9.3f    %9.3f' % \
              (stat, cv['10%'], cv['5%'], cv['2.5%'], cv['1%']))
        print('-------------------------------------------------------------')
        if stat < cv['10%']:
            print('p-value > 0.1')
        elif stat > cv['1%']:
            print('p-value < 0.01')
        else:
            print('p-value = %.4f' % pval)
    return ans

#def var(*args, **kwargs):
#    '''Vector autoregression'''
#    from statsmodels.tsa.api import VAR
#    maxlags = kwargs.pop('maxlags',1)
#    model = VAR(*args, **kwargs)
#    return model.fit(maxlags=maxlags)

def vecrank(*args, **kwargs):
    '''Johansen cointegration test'''
    from statsmodels.tsa.vector_ar.vecm import (
        coint_johansen,
        select_coint_rank
    )
    verbose = kwargs.pop('verbose', False)
    #ans = coint_johansen(*args, **kwargs)
    ans = select_coint_rank(*args, **kwargs)
    if verbose: print(ans)
    return ans

def vecm(*args, **kwargs):
    '''Vector Error Correction'''
    from statsmodels.tsa.vector_ar.vecm import VECM
    model = VECM(*args, **kwargs)
    ans = model.fit()
