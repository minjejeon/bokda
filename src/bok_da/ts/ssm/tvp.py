import bok_da as bd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union

from ...utils.operator import inv, eye, cols, diag, rows, zeros, ones, sumc, minc, demeanc, stdc, length
from .optimizer import SA_Newton, Gradient
from . import cython_SSM as cbp
    
class TimeVaryingParameterModel:
    def __init__(self, tvp_index: list, vsf: int=5, bvsf: int=2, vtf: int=100, iwf: int=3):
        self.tvp_index = tvp_index
        self.vsf = vsf
        self.bvsf = bvsf
        self.vtf = vtf
        self.iwf = iwf
        
    def fit(self, dep, indep, verbose: bool=False):
        self.results = bd.ts.ssm.tvp(dep, indep, self.tvp_index, self.vsf, self.bvsf, self.vtf, self.iwf, verbose)
        return self.results
    
class TVPResult:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_description(self):
        desc = {
            "결과": ['params', 'stderr', 'table', 'fitted', 'resid', 'lnL', 'bic '],
            "설명": ['TVP 추정치(칼만 스무더), T by len(TVP_index)', 'TVP 추정치의 표준오차, T by len(TVP_index)', '모형 파라미터 추정결과',
                     'Y 중 설명변수에 의해서 설명되는 부분', '잔차항', '로그우도(log likelihood)', '베이지안 정보기준(Bayesian Information Criterion)']
        }

        df = pd.DataFrame(desc)
        df.set_index("결과", inplace=True)
        df.index.name = None
        #df = df.style.set_properties(**{'text-align': 'left'})
        #df = df.set_table_styles([{'selector': 'th', 'props': [('text-align', 'left')]}])

        return df

    def get_coef_bound(self):
        lower_bound = self.params - self.stderr*2
        upper_bound = self.params + self.stderr*2
        return lower_bound, upper_bound

    def plot_tvp_estimates(self,
                           figsize: tuple=(10,6),
                           title: bool=True,
                           title_fontsize: int=14,
                           ncol: int=1
                          ):

        lowb, uppb = self.get_coef_bound()

        for i, label in enumerate(self.params.columns):
            pp = bd.viz.Plotter(xmargin=0, figsize=figsize)
            pp.line(self.params.index, self.params.iloc[:,i], color='r', label=label)
            pp.line(self.params.index, [lowb.iloc[:,i], uppb.iloc[:,i]], linestyle='--', color='k', label=['lower','upper'])
            pp.set_xaxis('year')
            if title:
                pp.set_title(title=f'TVP Estimates for {label}', fontsize=title_fontsize)
            pp.legend(ncol=ncol)

    def plot_fitted_resid(self,
                          figsize: tuple=(10,6),
                          title: bool=True,
                          title_fontsize: int=14,
                          ncol: int=1
                         ):

        fitted_resid = pd.concat([self.fitted, self.resid], axis=1)

        for i, label in enumerate(fitted_resid.columns):
            pp = bd.viz.Plotter(xmargin=0, figsize=figsize)
            pp.line(fitted_resid.index, fitted_resid.iloc[:, i], color='b', label=label)
            #pp.legend(ncol=ncol)
            if title:
                pp.set_title(title=f'{label}', fontsize=title_fontsize)
            else:
                pp.legend(ncol=ncol)

def OLS_TVP(Y, X):

    X2 = X.T @ X
    X2 = 0.5*(X2 + X2.T)
    XY = X.T @ Y

    bhat = inv(X2) @ XY 
    return bhat


def trans_TVP(para0, Spec):

    para1 = para0.copy()

    Var_index = Spec['Var_index']
    para1[Var_index] = np.exp(para0[Var_index])
    
    return para1


def lnlik_TVP(para, Spec):
    Y = Spec['Y']
    X = Spec['X']
    Z = Spec['Z']
    K = cols(X)
    Ng = cols(Z)
    para1 = trans_TVP(para, Spec)
    Sigma = para1[0].reshape(-1, 1)  # 오차항의 분산
    sig2beta = para1[1:K+1] # 시변계수의 분산  
    gam = para1[K+1:] # constant beta
     
    F = eye(K)

    Omega = diag(sig2beta)
    Mu = zeros(K, 1)
  
    U_LL = zeros(K, 1)
    P_LL = diag(ones(K, 1)*10000)
    
    if K == 1:
        Mu = Mu.reshape(-1, 1)
        Omega = Omega.reshape(-1, 1)
        U_LL = U_LL.reshape(-1, 1)
        P_LL = P_LL.reshape(-1, 1)

    lnLm, U_ttm, P_ttm, U_tLm, P_tLm = cbp.ckalman_filter_TVP(Y, X, Z, K, F, Mu, gam, Omega, Sigma, U_LL, P_LL)
    
    # T = rows(Y)
    # lnLm = zeros(T, 1)
    # U_ttm = zeros(T, K)
    # P_ttm = zeros(K, K, T)

    # U_tLm = zeros(T, K)
    # P_tLm = zeros(K, K, T)

    # for t in range(T):
        
    #     U_tL = Mu + F @ U_LL # k by 1
    #     P_tL = F @ P_LL @ F.T + Omega # k by k

    #     H = X[t, :].reshape(1, -1) # 1 by k
        
    #     y_tL = H @ U_tL + Z[t, :].reshape(1, -1) @ gam # 1 by 1
    #     f_tL = H @ P_tL @ H.T + Sigma # 1 by 1

    #     y_t = Y[t, :].T
    #     # y_t, y_tL, f_tL:  (1,) (1, 1) (1, 1)
    #     lnp = lnpdfn(y_t, y_tL, f_tL)
    #     lnLm[t] = lnp

    #     invf_tL = np.asarray(inv(f_tL))
    #     U_tt = U_tL + P_tL @ H.T * invf_tL * (y_t - y_tL) # k by 1
    #     P_tt = P_tL - P_tL @ H.T @ invf_tL @ H @ P_tL # k by k
    #     P_tt = (P_tt + P_tt.T)/2
        
    #     U_ttm[t, :] = U_tt.T
    #     P_ttm[t, :, :] = P_tt

    #     U_tLm[t, :] = U_tL.T
    #     P_tLm[t, :, :] = P_tL

    #     U_LL = U_tt
    #     P_LL = P_tt

    lnL = sumc(lnLm[5:])  # 번인 제외

    return lnL, U_ttm, P_ttm, U_tLm, P_tLm


def paramconst_TVP(para, Spec):
    
    para1 = trans_TVP(para, Spec)
    validm = ones(10, 1)
    
    Var_index = Spec['Var_index']
    sig2_Res = Spec['sig2_Res']
    sig2m = para1[Var_index]
    msig2, tmp = minc(sig2m)
    validm[1] = msig2 > 0.0000001
    validm[2] = sig2m[0] > sig2_Res

    valid, tmp = minc(validm)
    
    return valid


def Kalman_Smoother_TVP(F, Beta_ttm, P_ttm, Beta_tLm, P_tLm):
# Beta_tTm, P_tTm 둘다 T by K
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

        weight = P_tt @ F.T @ inv(P_t1t) # k by k

        beta_tT = Beta_ttm[t, :].T.reshape(-1, 1) + weight @ (Beta_tTm[t+1, :].T - Beta_tLm[t+1, :].T).reshape(-1, 1)
        Beta_tTm[t, :] = beta_tT.T

        P_tT = P_tt + P_tt @ F.T @ inv(P_t1t) @ (P_t1T - P_t1t) @ inv(P_t1t) @ F @ P_tt
        P_tTm[t, :] = diag(P_tT).T  # % 대각행렬(분산)만 저장하기

        P_t1T = P_tT  # % 다음기에 사용될 것, K by K
        t = t - 1

    return Beta_tTm, P_tTm
            
    
def tvp(Dependent: Union[pd.Series, pd.DataFrame],
        Independent: Union[pd.Series, pd.DataFrame],
        tvp_index: list, vsf: int=5, bvsf: int=2, vtf: int=100, iwf: int=3,
        verbose: bool=False):
    
    """
    상태공간모형 중 하나인 시변계수(Time-Varying Parameter) 추정 함수.
    
    주어진 종속변수 Y와 설명변수 X를 사용하여 시변계수 모형 추정.
    TVP_index는 주어진 설명변수에 대해 시변계수를 지정하는 파라미터.
    
    하이퍼 파라미터
    ---------------
    Y: pd.Series 또는 pd.DataFrame
        종속변수 데이터, pandas 시리즈 또는 데이터프레임.
    X: pd.Series 또는 pd.DataFrame
        설명변수 데이터, pandas 시리즈 또는 데이터프레임.
    TVP_index: list
        설명변수에 대한 시변계수 인덱스 지정.
    vsf: int
        variance scaling factor, 분산 초기값 스케일을 조정하는 파라미터
    bvsf: int
        beta variance scaling factor, 시변계수 베타의 분산 초기값 스케일을 조정하는 파라미터
    vtf: int
        varinace threshold factor, 분산의 하한선을 조정하는 파라미터
    iwf: int
        initial window factor, 전체 기간 중 초기 구간의 비율을 설정하는 파라미터
    verbose: bool
        모형 추정결과 그림 표시 옵션.
        
    반환값
    ------
    TVPResult
        시변계수 모형의 추정결과를 포함하는 객체.
        TVPResult 객체는 다음의 속성과 메서드를 포함:
        
        - TV_param = TVP 추정치(칼만 스무더), T by len(TVP_index)
        - TV_param_SE = TVP 추정치의 표준오차, T by len(TVP_index)
        - Table_frame: 모형 파라미터 추정결과
        - Fitted = Y 중 설명변수에 의해서 설명되는 부분
        - Residuals = 잔차항
        - lnL: 로그우도(log likelihood)
        - BIC: 베이지안 정보기준(Bayesian Information Criterion)
        - get_description(): tvp 함수에서 생성된 반환값(속성 및 메서드)을 설명하는 메서드
        - get_coef_bound(): 추정된 계수의 lower and upper bound 출력하는 메서드
        - plot_tvp_estimates(): 추정된 계수의 그림을 그리는 메서드
        - plot_fitted_resid(): fitted 값과 잔차항 그림을 그리는 메서드
        
    예제:
    ---------------------
    >>> import bok as bd
    >>> uc_results = bd.ssm.tvp(df[y], df[x], TVP_index = ['tr_3y', 'esi'], verbose=False)
    >>> uc_results.get_description()
    >>> uc_results.Table_frame
    """

    #if type(Dependent) != np.ndarray: Dependent = np.array(Dependent)
    #if type(Independent) != np.ndarray: Independent = np.array(Independent)
    TVP_index = tvp_index
    index = Dependent.index
    columns = Independent.columns
    Dependent = np.array(Dependent, dtype=float)
    Independent = np.array(Independent, dtype=float)

    if rows(Dependent) != rows(Independent):
        print('종속변수와 설명변수의 표본크기가 다릅니다')
        print(f"종속변수의 표본크기는 각각 {rows(Dependent)}와 {rows(Independent)}입니다")
        return
    
    tvp_name = TVP_index.copy()
    const_name = [i for i in columns if i not in tvp_name]
    TVP_index = [columns.get_loc(i) for i in columns if i in TVP_index]

    Y = Dependent.reshape(-1, 1)
    X = Independent[:, TVP_index]
    N = cols(Independent)
    Indp_Index = range(N)
    
    Const_index = [x for x in Indp_Index if x not in TVP_index]
    Z = Independent[:, Const_index]
    T = rows(Y)
    Z = np.hstack((Z, ones(T, 1)))

    if len(TVP_index) == 0:
        print('시변계수은 최소 1개 이상이어야 합니다')
        return
    else:
        K = len(TVP_index)
    
    para0 = zeros(cols(Independent)+2, 1)

    ## 분산의 초기값
    Var_index = range(K+1)
    Y_ = demeanc(Y)
    X_ = demeanc(Independent)
    bhat = OLS_TVP(Y_, X_)
    ehat = Y_ - X_ @ bhat
    sig2_hat = (ehat.T @ ehat)/T
    para0[0] = np.log(sig2_hat/vsf)  # 변동성, vsf = variance scaling factor
    sig2_Res = sig2_hat/vtf # vtf = variance threshold factor

    T0 = int(T/iwf) # iwf = initial window factor
    betam = zeros(T-T0, K)
    for t in range(T0, T):
        Yt = demeanc(Y[t-T0:t])
        Xt = demeanc(X[t-T0:t, :])
        bhat = OLS_TVP(Yt, Xt)
        betam[t-T0, :] = bhat.T

    d_betam = betam[1:, :] - betam[0:-1, :]
    para0[1:K+1] = np.log(np.square(stdc(d_betam)*bvsf))  # beta 시변계수 변동성

    Spec =  {'Y': Y,
             'X': X,
             'Z': Z,
             'N': N,
             'sig2_Res': sig2_Res,
             'Var_index': Var_index}

    para0_hat,fmax,G,V,Vinv = SA_Newton(lnlik_TVP, paramconst_TVP, para0, Spec, verbose)
    para1_hat = trans_TVP(para0_hat, Spec).reshape(-1, 1)
    lnL, U_ttm, P_ttm, U_tLm, P_tLm = lnlik_TVP(para0_hat, Spec)

    narg = length(para0_hat)
    T = rows(Y)
    BIC = narg*np.log(T) - 2*lnL

    Grd = Gradient(trans_TVP, para0_hat, Spec)
    if Grd.ndim == 1:
        Grd = Grd.reshape(-1, 1)

    Var = Grd @ V @ Grd.T
    SE = np.round(np.sqrt(np.abs(diag(Var))), 3).reshape(-1, 1)

    ## F 만들기
    F = eye(K)

    TV_param, P_tTm = cbp.cKalman_Smoother_TVP(F, U_ttm, P_ttm, U_tLm, P_tLm)
    TV_param_SE = np.sqrt(P_tTm)
    
    if verbose:
        print('=================================================')
    paraName = [f'{tvp_name[0]} 시변계수 분산']
    for j in range(1,K): # range(2,K+1)
        paraName += [f"{tvp_name[j]} 시변계수 분산"]
        
    NZ = cols(Z)
    if NZ > 1:
        for j in range(K+1,K+NZ):
            paraName += [f"{const_name[j-K-1]} 계수"]

    paraName += ['절편항']
    paraName += ['오차항 분산']
    
    Table_para = np.hstack((para1_hat, SE))
    Table_para = np.vstack((Table_para[1:, :], Table_para[0, :]))
    Table_para = np.round(Table_para, 3)

    Table_frame = pd.DataFrame(Table_para, index = paraName,columns=['추정치', '표준오차'])
         
    if verbose:
        
        print('-----------------------------------------------')
        print('로그 우도 = ', lnL)
        print('BIC = ', BIC)
        print('-----------------------------------------------')

        ## 추정결과 그림 그리기
        T = rows(TV_param)
        x = np.arange(T)

        for j in range(K):
            plt.subplot(int(np.ceil(K/2)), 2, int(j+1))
            plt.plot(x, TV_param[:, j], linestyle='-', color = 'blue')

            Lowb = TV_param[:, j] - TV_param_SE[:, j]*2
            Uppb = TV_param[:, j] + TV_param_SE[:, j]*2
            plt.plot(x, Lowb, linestyle='--', color = 'black')
            plt.plot(x, Uppb, linestyle='--', color = 'black')
            plt.title(f"time-varying parameter {j}")
        plt.show()


    TV_part = np.multiply(X, TV_param) # T by 2
    TV_part = sumc(TV_part.T).reshape(-1, 1) # T by 1
    
    Fitted = TV_part + Z @ para1_hat[K+1:K+NZ+1]
    Residuals = Y - Fitted

    if verbose:
        plt.subplot(2, 1, 1)
        plt.plot(x, Y, linestyle='-', color = 'blue')
        plt.plot(x, Fitted, linestyle='--', color = 'black')
        plt.title("Dependent (solid) and Fitted values (dashed)") 

        plt.subplot(2, 1, 2)
        plt.plot(x, Residuals, linestyle='-', color = 'blue')
        plt.plot(x, zeros(T, 1), linestyle='--', color = 'black')   
        plt.title("Residuals")     
        plt.show()
    
    TV_param = pd.DataFrame(TV_param, index=index, columns=columns[TVP_index])
    TV_param_SE = pd.DataFrame(TV_param_SE, index=index, columns=columns[TVP_index])
    Fitted = pd.DataFrame(Fitted, index=index, columns=['Fitted'])
    Residuals = pd.DataFrame(Residuals, index=index, columns=['Residuals'])
    return TVPResult(params = TV_param, stderr = TV_param_SE, table = Table_frame, fitted = Fitted, resid = Residuals, lnL = lnL, bic = BIC)

