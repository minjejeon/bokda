import bok_da as bd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union

from ...utils.operator import ones, maxc, minc, rows, cols, zeros, diag, eye, sumc, demeanc, meanc, stdc, length
from ...utils.pdf import lnpdfn
from .optimizer import SA_Newton, Gradient
from . import cython_SSM as cbp


class DynamicFactorModel:
    def __init__(self, regional_countries: list=[], local_countries: list=[], lag: int=1):
        self.regional_countries = regional_countries
        self.local_countries = local_countries
        self.lag = lag
        
    def fit(self, data, verbose: bool=False):
        self.results = bd.ts.ssm.dfm(data, self.lag, self.regional_countries, self.local_countries, verbose)
        return self.results

class DFMResult:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def get_description(self):
        desc = {
            "결과": ['global_factor', 'regional_factor', 'local_factor', 'individual_factors', 'table_var_decomp',
                     'table_common_ar', 'table_common_var', 'table_idio_ar', 'table_idio_var', 'table_global_factorloading',
                     'table_regional_factorloading', 'table_local_factorloading', 'lnL', 'bic'],
            "설명": ['공통요인(global factor) 추정치, T by 1', '지역요인(regional factor) 추정치, T by 1', '로컬요인(local factor) 추정치, T by 1',
                     '변수고유요인, T by N', '분산분해 결과', '공통요인의 AR 계수 추정치와 표준오차. 만약 P = 0이면 Table_Common_AR은 empty',
                     '공통요인 충격의 분산 추정치와 표준오차', '변수고유요인의 AR 계수 추정치와 표준오차, N by 2. 만약 P = 0이면 Table_Idio_AR은 empty',
                     '변수고유요인 충격의 분산 추정치와 표준오차, N by 2', '공통요인 계수, N by 2', '지역요인 계수, N by 2', '로컬요인 계수, N by 2', 
                     '로그우도(log likelihood)', '베이지안 정보기준(Bayesian Information Criterion)']
        }

        df = pd.DataFrame(desc)
        df.set_index("결과", inplace=True)
        df.index.name = None
        #df = df.style.set_properties(**{'text-align': 'left'})
        #df = df.set_table_styles([{'selector': 'th', 'props': [('text-align', 'left')]}])
        
        return df
            
    def plot_dfm_factors(self, figsize: tuple=(8,5), title: bool=True,
                         title_fontsize: int=14, legend: bool=True, ncol: int=1, **kwargs):
        
        if 'global' in kwargs['factor'] and hasattr(self, 'global_factor'):
            self._plot_dfm_factors(self.global_factor, figsize, title, title_fontsize, legend, ncol)
        if 'regional' in kwargs['factor'] and hasattr(self, 'regional_factor'):
            self._plot_dfm_factors(self.regional_factor, figsize, title, title_fontsize, legend, ncol)
        if 'local' in kwargs['factor'] and hasattr(self, 'local_factor'):
            self._plot_dfm_factors(self.local_factor, figsize, title, title_fontsize, legend, ncol)
        if 'individual' in kwargs['factor'] and hasattr(self, 'individual_factors'):
            self._plot_dfm_factors(self.individual_factors, figsize, title, title_fontsize, legend, ncol)
            
    def _plot_dfm_factors(self, data, figsize, title, title_fontsize, legend, ncol):
        pp = bd.viz.Plotter(xmargin=0, figsize=figsize)
        y = data.copy()
        
        if y.shape[1] == 1:
            pp.line(y.index, y, color='b', label=y.columns)
            if title:
                y.columns = y.columns.str.replace('_', ' ', regex=False)
                pp.set_title(title=f'{y.columns[0]}', fontsize=title_fontsize)
        else:
            pp.line(y.index, [y[col] for col in y.columns], label=y.columns)
            if title:
                pp.set_title(title='Individual factors', fontsize=title_fontsize)
                    
        pp.set_xaxis('year')
        
        if legend:
            pp.legend(ncol=ncol)


def trans_DFM(para0: np.ndarray, 
              Spec: dict) -> np.ndarray:
    """
    Transform parameters for DFM

    Args:
        para0 (np.ndarray): initial parameters
        Spec (dict): dictionary of indices and parameters
    
    Returns:
        para1 (np.ndarray)
    """
    para1 = para0.copy()

    Var_index = Spec['Var_index']
    Phi_index = Spec['Phi_index']
    N = Spec['N']
    NC = Spec['NC']
    para1[Var_index] = np.exp(para0[Var_index])
    
    p = Spec['Lag']
    if p > 0:
        ExpPhi = np.exp(para0[Phi_index])
        para1[Phi_index] = np.divide(ExpPhi, (ones(N+NC, 1) + ExpPhi))

    return para1


def paramconst_DFM(para: np.ndarray, 
                   Spec: dict) -> np.ndarray:
    """Check parameter constraints for a Dynamic Factor Model.
    """
    para1 = trans_DFM(para, Spec)
    validm = ones(10, 1)
    
    p = Spec['Lag']
    
    if p > 0:
        Phi_index = Spec['Phi_index']
        phi = para1[Phi_index]
        maxabsphi, _ = maxc(phi)
        validm[0] = maxabsphi < 0.95
    
    Var_index = Spec['Var_index']
    sig2m = para1[Var_index]
    msig2, _ = minc(sig2m)
    validm[1] = msig2 > 0

    valid, tmp = minc(validm)
    
    return valid


def kalman_filter_DFM(Y, K, H, F, Mu, Omega, Sigma, U_LL, P_LL):
    T = rows(Y)
    lnLm = np.zeros(T)
    U_ttm = np.zeros((T, K))
    P_ttm = np.zeros((T, K, K))
    U_tLm = np.zeros((T, K))
    P_tLm = np.zeros((T, K, K))

    for t in range(T):
        U_tL = Mu + F @ U_LL
        P_tL = F @ P_LL @ F.T + Omega

        y_tL = H @ U_tL
        f_tL = H @ P_tL @ H.T + Sigma

        y_t = Y[t, :].T.reshape(-1, 1)
        lnp = cbp.clnpdfmvn(y_t, y_tL, f_tL)
        lnLm[t] = lnp

        y_tL = y_tL.reshape(-1, 1)
        invf_tL = np.linalg.inv(f_tL)

        U_tt = U_tL + P_tL @ H.T @ invf_tL @ (y_t - y_tL)
        P_tt = P_tL - P_tL @ H.T @ invf_tL @ H @ P_tL
        P_tt = (P_tt + P_tt.T) / 2

        U_ttm[t, :] = U_tt.T
        P_ttm[t, :, :] = P_tt
        U_tLm[t, :] = U_tL.T
        P_tLm[t, :, :] = P_tL

        U_LL, P_LL = U_tt, P_tt

    return lnLm, U_ttm, P_ttm, U_tLm, P_tLm


def lnlik_DFM(para, Spec):
    # print(Spec['Regional_countries'])
    
    Y = Spec['Data']
    N = cols(Y)
    NC, NR, NL, p = Spec['NC'], Spec['NR'], Spec['NL'], Spec['Lag']
    # Regional_countries = np.array(Spec['Regional_countries']).T
    Regional_countries = Spec['Regional_countries']
    # np.array(Regional_countries)[:, np.newaxis]
    # Local_countries = np.array(Spec['Local_countries']).T
    Local_countries = Spec['Local_countries']
    Phi_index, G_index = Spec['Phi_index'], Spec['G_index']
    R_index, L_index = Spec['R_index'], Spec['L_index']
    para1 = trans_DFM(para, Spec)
    sig2C = para1[0: NC] # 공통인자의 분산
    sig2E = para1[NC: NC + N] # 고유요인의 분산
    
    K = NC + N

    if p > 0:
        F = diag(para1[Phi_index])
    else:
        F = zeros(K, K)

    Mu = zeros(K, 1)
    factor_sig2 = np.vstack((sig2C, sig2E))
    Omega = diag(factor_sig2)

    Sigma = zeros(N,N)
  
    H = zeros(N, K)
    H[0, 0] = 1
    H[1:, 0] = para1[G_index].flatten()

    if NC > 1:
        # print(Regional_countries)
        H[Regional_countries[0], 1] = 1
        H[Regional_countries[1:], 1] = para1[R_index].flatten()

        if NC > 2:
            H[Local_countries[0], 2] = 1
            H[Local_countries[1:], 2] = para1[L_index].flatten()
    
    H[:, NC:] = eye(N)
    U_LL, P_LL = cbp.cmake_R0(Mu, F, Omega)

    lnLm, U_ttm, P_ttm, U_tLm, P_tLm = cbp.ckalman_filter_DFM(Y, K, H, F, Mu, Omega, Sigma, U_LL, P_LL)

    lnL = sumc(lnLm)  # 번인 제외
    prior_var = 1
    lnprior = sumc(lnpdfn(para1[G_index], zeros(N-1, 1), ones(N-1, 1)*prior_var))
    if NR > 0:
        lnprior = lnprior + sumc(lnpdfn(para1[R_index], zeros(NR-1, 1), ones(NR-1, 1)*prior_var)) 
        if NL > 0:
            lnprior = lnprior + sumc(lnpdfn(para1[L_index], zeros(NL-1, 1), ones(NL-1, 1)*prior_var))

    lnL = lnL + lnprior
    return lnL, U_ttm, P_ttm, U_tLm, P_tLm



def dfm(Data: Union[pd.Series, pd.DataFrame],
        lag: int=1,
        regional_countries: list=[], # 기본값은 empty 리스트
        local_countries: list=[],
        verbose: bool=False
        ):

    """
    상태공간모형 중 하나인 동태적요인모형(Dynamic Factor Model, DFM) 추정 함수.
    시계열 변수로 구성된 Data로부터 공통요인(global factor), 지역요인(regional factor), 로컬요인(local factor)을 추출.
    
    하이퍼 파라미터
    ---------------
    Data: pd.DataFrame
        모형 추정에 사용할 시계열 변수로 구성된 DataFrame. 각 열은 서로 다른 변수를 나타내며, 각 행은 시점을 나타냄.
    lag: int, 기본값=1
        요인과정(factor process)의 AR 시차를 결정하는 파라미터. 0 또는 1로 설정.
    regional_countries: list, 기본값=[]
        지역요인에 영향을 받는 시계열 변수 이름(str)들로 구성된 list. 이 list를 통해 어떤 변수들이 지역요인에 영향 받는지 설정.
        만약 지역요인이 없다고 가정한다면, list=[]로 설정.
    local_countries: list, 기본값=[]
        로컬요인에 영향을 받는 시계열 변수 이름(str)들로 구성된 list. 이 list를 통해 어떤 변수들이 로컬요인에 영향 받는지 설정.
        만약 로컬요인이 없다고 가정한다면, list=[]로 설정.
    verbose: bool, 기본값='False'
        모형 추정결과 그림 표시 옵션.
        
    반환값
    ------
    DFMResult
        동태적요인모형의 추정결과를 포함하는 객체.
        UCResult 객체는 다음의 속성과 메서드를 포함:
        
        - global_factor: 공통요인(global factor) 추정치, T by 1, T = 표본크기
        - regional_factor: 지역요인(regional factor) 추정치, T by 1, T = 표본크기
        - local_factor: 로컬요인(local factor) 추정치, T by 1, T = 표본크기
        - individual_factors: 변수고유요인(individual facotrs) 추정치, T by N, T = 표본크기, N = 변수의 수
        - table_var_decomp: 분산분해 결과
        - table_common_ar: 공통요인의 AR 계수 추정치와 표준오차, 만약 Lag=0이면, empty
        - table_common_var: 공통요인 충격의 분산 추정치와 표준오차
        - table_idio_ar: 변수고유요인의 AR 계수 추정치와 표준오차, N by 2, 만약 Lag=0이면, empty
        - table_idio_var: 변수고유요인 충격의 분산 추정치와 표준오차, N by 2
        - table_global_factorloading: 측정식에서 공통요인 계수 추정치
        - table_regional_factorloading: 측정식에서 지역요인 계수 추정치
        - table_local_factorloading: 측정식에서 로컬요인 계수 추정치
        - lnL: 로그우도(log likelihood)
        - bic: 베이지안 정보기준(Bayesian Information Criterion)
        - get_description(): dfm 함수에서 생성된 반환값(속성 및 메서드)을 설명하는 메서드
        - plot_dfm_factors(): 추정된 요인들의 그림을 그리는 함수
        
    예제: Data = 지역별 전년동월대비 주택가격지수
    ---------------------
    >>> import bok as bd
    >>> res = bd.ts.ssm.dfm(Data, Lag=1, Regional_countries=['서울','경기','인천'], Local_countries=['경기','인천'], verbose=False)
    >>> res.get_description()
    >>> res.Global_factor
    """    
    
    index = Data.index
    columns = Data.columns
    Lag = lag
    Regional_countries = regional_countries
    Local_countries = local_countries
    
    # ensure that the Data is np.ndarray type
    Data = np.array(Data, dtype=float) # np.asarray(Data)
    
    Regional_countries = [columns.get_loc(i) for i in columns if i in Regional_countries]
    Local_countries = [columns.get_loc(i) for i in columns if i in Local_countries]

    Y = demeanc(Data)
    N = cols(Data)
    if len(Regional_countries) == 0:
        NR = 0
        Local_countries = [ ]
    else:
        NR =len(Regional_countries)

    if len(Local_countries) == 0:
        NL = 0
    else:
        NL = len(Local_countries)


    if NR >= N:
        print('regional_countries의 수가 전체 국가 수보다 작아야 합니다')
        print('전체 국가 수 = ', N)
        print('regional country의 수 = ', NR)

    if NL >= NR and NL>0:
        print('local_country의 수가 Regional_country 수보다 커야 합니다')
        print('local country의 수 = ', NL)
        print('regional country의 수 = ', NR)

    if NL >0:
        for i in Local_countries:
            if i not in Regional_countries:
                print('regional country가 local_countries을 포함하지 않습니다')

    if NL==1:
        print('local_country 수는 1보다 커야 합니다')

    if NR==1:
        print('regional_country 수는 1보다 커야 합니다')

    NC = 1  # 공통요인의 수
    N_gam = N-1 # 팩터로딩 파라메터의 수
    if NR > 0:
        NC = 2
        N_gam = N_gam + (NR-1) # 팩터로딩 파라메터의 수
        if NL > 0:
            NC = 3
            N_gam = N_gam + (NL-1) # 팩터로딩 파라메터의 수

    # 공통요인의 분산 = NC
    # 고유요인의 분산 = N
    # 고유요인과 공통요인의 AR = NC + N
    
    if Lag == 1:
        para0 = zeros(NC + N + NC + N + N_gam, 1)
    else:
        para0 = zeros(NC + N + N_gam, 1)


    Global_proxy = meanc(Y.T)

    ## 분산의 초기값
    Var_index = range(NC+N)
    para0[0] = np.log(np.square(stdc(Global_proxy)))  # 변동성
    
    if NR > 0:
        if Lag == 0:
            para0[1] = np.log(np.square(stdc(Global_proxy)))  # 변동성
        else:
            para0[1] = np.log(np.square(stdc(Global_proxy))/10)  # 변동성
            
            if NL > 0:
                if Lag == 0:
                    para0[2] = np.log(np.square(stdc(Global_proxy)))  # 변동성
                else:
                    para0[2] = np.log(np.square(stdc(Global_proxy))/10)  # 변동성

    para0[NC:NC+N] = np.log(np.square(stdc(Y))/4)  # 변동성
    
    ## Phi의 초기값
    if Lag == 1:
        N_var_phi = 2*(NC+N)
        Phi_index = range(NC+N, N_var_phi) 
        para0[Phi_index] = 0.5 # Phi 초기값
    else:
        N_var_phi = NC + N
        Phi_index = [ ]
        

    ## 팩터로딩의 초기값
    G_index = range(N_var_phi, N_var_phi+N-1)
    para0[G_index] = 1

    if NR > 0:
        R_index = range(N_var_phi+N-1, N_var_phi+N-1+NR-1)
        para0[R_index] = 0.5
        if NL > 0:
            L_index = range(N_var_phi+N-1+NR-1, N_var_phi+N-1+NR-1+NL-1)
            para0[L_index] = 0.5
        else:
            L_index = [ ]
    else:
        R_index = [ ]
        L_index = [ ]

    Spec =  {
            'Data': Y,
            'Lag': Lag,
            'NC': NC,
            'N': N,
            'NR': NR,
            'NL': NL,
            'Regional_countries': Regional_countries,
            'Local_countries': Local_countries,
            'Var_index': Var_index,
            'Phi_index': Phi_index,
            'G_index': G_index,
            'R_index': R_index,
            'L_index': L_index
        }
    
    # 추정
    para0_hat,fmax,G,V,Vinv = SA_Newton(lnlik_DFM, paramconst_DFM, para0, Spec)

    para1_hat = trans_DFM(para0_hat, Spec)

    lnL, U_ttm, P_ttm, U_tLm, P_tLm = lnlik_DFM(para0_hat, Spec)

    narg = length(para0_hat)
    T = rows(Y)
    BIC = narg*np.log(T) - 2*lnL

    Grd = Gradient(trans_DFM, para0_hat, Spec)
    Var = Grd @ V @ Grd.T
    SE = np.round(np.sqrt(np.abs(diag(Var))), 3)

    ## F 만들기
    K = NC + N
    
    if Lag>0:
        Phi = para1_hat[Phi_index]
        F = diag(Phi)
    else:
        F = zeros(K, K)
        
    U_tTm, P_tTm = cbp.cKalman_Smoother_DFM(F, U_ttm, P_ttm, U_tLm, P_tLm)

    ## 파라메터 추정결과
    SE = SE.reshape(-1, 1)
    Table_para = np.hstack((para1_hat, SE))
    Table_para = np.round(Table_para, 3)

    paraCommonName = ['공통요인(global factor)']
    if NC > 1:
        paraCommonName += ['지역요인(regional factor)']
    
    if NC > 2:
        paraCommonName += ['로컬요인(local factor)']
        
    Table_Common_Var = Table_para[0:NC, :]
    Table_Common_Var = pd.DataFrame(Table_Common_Var, index = paraCommonName, columns=['추정치', '표준편차'])

    paraVariName = ['변수 1']
    for j in range(1,N):
        paraVariName += [f"변수 {j+1}"]

    Table_Idio_Var = Table_para[NC:K, :]
    Table_Idio_Var = pd.DataFrame(Table_Idio_Var, index = columns, columns=['추정치', '표준편차'])

    if Lag > 0:
        Phi_SE = SE[Phi_index]
        Table_Phi = np.hstack((Phi, Phi_SE))
        Table_Common_AR = Table_Phi[0:NC, :]
        Table_Common_AR = pd.DataFrame(Table_Common_AR, index = paraCommonName, columns=['추정치', '표준편차'])
        Table_Idio_AR = Table_Phi[NC:K, :]
        Table_Idio_AR = pd.DataFrame(Table_Idio_AR, index = columns, columns=['추정치', '표준편차'])
    else:
        Table_Idio_AR = [ ]
        Table_Common_AR = [ ]

    Gam1_hat = para1_hat[G_index]
    Gam1_SE = SE[G_index]

    GAM1_hat = ones(N, 1)
    GAM1_hat[1:] = Gam1_hat

    GAM1_SE = zeros(N, 1)
    GAM1_SE[1:] = Gam1_SE

    Table_Global_factorloading = np.hstack((GAM1_hat, GAM1_SE))
    Table_Global_factorloading = np.round(Table_Global_factorloading, 3)
    Table_Global_factorloading = pd.DataFrame(Table_Global_factorloading, index = columns, columns=['추정치', '표준편차'])

    GAM2_hat = zeros(N, 1)

    if NR > 0:
        Gam2_hat = para1_hat[R_index]
        
        GAM2_hat[Regional_countries[0]] = 1
        GAM2_hat[Regional_countries[1:]] = Gam2_hat
        Gam2_SE = SE[R_index]

        GAM2_SE = zeros(N, 1)
        GAM2_SE[Regional_countries[1:]] = Gam2_SE

        Table_Regional_factorloading = np.hstack((GAM2_hat, GAM2_SE))
        Table_Regional_factorloading = np.round(Table_Regional_factorloading, 3)
        Table_Regional_factorloading = pd.DataFrame(Table_Regional_factorloading, index = columns, columns=['추정치', '표준편차'])
    
    else:
        Table_Regional_factorloading = [ ]

    GAM3_hat = zeros(N, 1)
    if NL > 0:
        Gam3_hat = para1_hat[L_index]
        
        GAM3_hat[Local_countries[0]] = 1
        GAM3_hat[Local_countries[1:]] = Gam3_hat
        Gam3_SE = SE[L_index]
        
        GAM3_SE = zeros(N, 1)
        GAM3_SE[Local_countries[1:]] = Gam3_SE
        
        Table_Local_factorloading = np.hstack((GAM3_hat, GAM3_SE))
        Table_Local_factorloading = np.round(Table_Local_factorloading, 3)
        Table_Local_factorloading = pd.DataFrame(Table_Local_factorloading, index = columns, columns=['추정치', '표준편차'])
    
    else:
        Table_Local_factorloading = [ ]

    sig2C_hat = para1_hat[0:NC] # 공통인자의 분산
    sig2E_hat = para1_hat[NC:NC+N] # 고유요인의 분산
    

    factor_sig2_hat = np.vstack((sig2C_hat, sig2E_hat))
    Omega_hat = diag(factor_sig2_hat)

    tmp, P_LL_hat = cbp.cmake_R0(zeros(K, 1), F, Omega_hat)

    diag_P_LL_hat = diag(P_LL_hat)
    Var_G = diag_P_LL_hat[0]
    Var_R = 0
    Var_L = 0
    if NC > 1:
        Var_R = diag_P_LL_hat[1]
        if NC > 2:
            Var_L = diag_P_LL_hat[2]
    
    Var_E = diag_P_LL_hat[NC:] 
    
    Total_Var_G = np.multiply(GAM1_hat, GAM1_hat)*Var_G # N by 1
    Total_Var_R = np.multiply(GAM2_hat, GAM2_hat)*Var_R # N by 1
    Total_Var_L = np.multiply(GAM3_hat, GAM3_hat)*Var_L # N by 1
    Total_Var_E = Var_E.reshape(-1, 1)                  # N by 1

    Total_Var_vec = Total_Var_G + Total_Var_R + Total_Var_L + Total_Var_E # N by 1
    Total_Var = np.kron(ones(1, 4), Total_Var_vec)
    Table_Var_Decomp = np.hstack((Total_Var_G*100, Total_Var_R*100, Total_Var_L*100, Total_Var_E*100))

    Table_Var_Decomp = np.divide(Table_Var_Decomp, Total_Var)

    Table_Var_Decomp = np.round(Table_Var_Decomp, 1)

    Table_Var_Decomp = pd.DataFrame(Table_Var_Decomp, index = columns, 
                                    columns=['공통요인(global factor)', '지역요인(regional factor)', '로컬요인(local factor)', '변수고유요인(individual factors)'])
    
    if verbose:
        print('=================================================')
        print('로그 우도 = ', lnL)
        print('BIC = ', BIC)

    ## 추정결과 그림 그리기
    T = rows(U_tTm)
    x = np.arange(T)
    
    if verbose:
        ## 공통인자 그림
        plt.figure(figsize=(9, NC*6)) 
        if NC == 1:
            plt.plot(x, U_tTm[:, 0], linestyle='--', color = 'blue')  
            plt.title('Global common factor')
            plt.show()
        elif NC > 1:
            for j in range(NC):
                plt.subplot(int(NC), 1, int(j+1))
                plt.plot(x, U_tTm[:, j], linestyle='--', color = 'blue')  
                if j == 0:
                    plt.title('Global common factor')
                elif j == 1:
                    plt.title('Regional common factor')
                elif j == 2:
                    plt.title('Local common factor')
            plt.show()

        ## 개별요인 그림
        plt.figure(figsize=(9, N*6)) 
        for j in range(N):
            plt.subplot(int(N), 1, int(j+1))
            plt.plot(x, U_tTm[:, NC+j], linestyle='--', color = 'blue')  
            plt.title(f"Variable {j}'s individual factor")
        plt.show()

    Global_factor = U_tTm[:, 0]
    Regional_factor = zeros(T, 1)
    Local_factor = zeros(T, 1)
    Individual_factors = U_tTm[:, NC:]

    if NC > 1:
        Regional_factor = U_tTm[:, 1]
        if NC > 2:
            Local_factor = U_tTm[:, 2]
            
    Global_factor = pd.DataFrame(Global_factor, index=index, columns=['Global_factor'])
    Regional_factor = pd.DataFrame(Regional_factor, index=index, columns=['Regional_factor'])
    Local_factor = pd.DataFrame(Local_factor, index=index, columns=['Local_factor'])
    Individual_factors = pd.DataFrame(Individual_factors, index=index, columns=columns)
    lnL = pd.DataFrame(lnL, index=['value'], columns=['log likelihood'])
    BIC = pd.DataFrame(BIC, index=['value'], columns=['BIC'])

    return DFMResult(global_factor=Global_factor, regional_factor=Regional_factor, local_factor=Local_factor, 
                     individual_factors=Individual_factors, table_var_decomp=Table_Var_Decomp, table_common_ar=Table_Common_AR,
                     table_common_var=Table_Common_Var, table_idio_ar=Table_Idio_AR, table_idio_var=Table_Idio_Var, 
                     table_global_factorloading=Table_Global_factorloading,
                     table_regional_factorloading=Table_Regional_factorloading,
                     table_local_factorloading=Table_Local_factorloading, lnL=lnL, bic=BIC)
