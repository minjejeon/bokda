import numpy as np
from tqdm import tqdm
import scipy as sc
from numpy.random import default_rng
import os
from statsmodels.tools.numdiff import approx_hess
### 기초 변수 생성 작업 ###


from . import functions as func
from .. import container_class


def EXOv_total_maker(Parameters, Raw_Data):
    """
    주어진 Parameters와 Raw_Data를 사용하여 EXOv_total, EXOv, EXOv_AR를 생성하여 Raw_Data에 추가한다.

    Parameters
    ----------
    Parameters : dict
        - 'Trend' : int
            1이면 상수항(constant term)만 포함, 2이면 선형 추세(linear trend) 포함, 3이면 이차 추세(quadratic trend) 포함
        - 'n' : int
            전체 관측치의 수
        - 'T' : int
            분석에 사용되는 관측치의 수

    Raw_Data : dict
        - 함수 내에서 'EXOv_total', 'EXOv', 'EXOv_AR'를 생성하여 추가한다.

    Returns
    -------
    Raw_Data : dict
        - 'EXOv_total' : np.matrix
            전체 기간에 대한 외생변수 행렬
        - 'EXOv' : np.matrix
            분석 기간(T)에 대한 외생변수 행렬
        - 'EXOv_AR' : np.matrix
            AR(4) 모형 추정을 위한 외생변수 행렬

    함수 동작
    --------
    - Parameters.Trend 값에 따라 외생변수를 생성한다.
        - Trend == 1 : 상수항만 포함
        - Trend == 2 : 상수항과 시간 추세 포함
        - Trend == 3 : 상수항, 시간 추세, 시간의 제곱항 포함
    - Raw_Data 딕셔너리에 EXOv_total, EXOv, EXOv_AR 키를 생성하여 외생변수를 저장한다.
    """
    if Parameters.Trend == 1:
        Raw_Data.EXOv_total = np.matrix(np.ones(Parameters.n)).T
    elif Parameters.Trend == 2:
        Raw_Data.EXOv_total = np.matrix(np.ones(Parameters.n)).T
        Raw_Data.EXOv_total = np.column_stack(
            (Raw_Data.EXOv_total, np.matrix(np.arange(1, Parameters.n + 1)).T))
    elif Parameters.Trend == 3:
        Raw_Data.EXOv_total = np.matrix(np.ones(Parameters.n)).T
        Raw_Data.EXOv_total = np.column_stack(
            (Raw_Data.EXOv_total, np.matrix(np.arange(1, Parameters.n + 1)).T))
        Raw_Data.EXOv_total = np.column_stack(
            (Raw_Data.EXOv_total, np.matrix(np.arange(1, Parameters.n + 1) ** 2).T))

    if Parameters.Trend == 1:
        Raw_Data.EXOv = np.matrix(np.ones(Parameters.T)).T
    elif Parameters.Trend == 2:
        Raw_Data.EXOv = np.matrix(np.ones(Parameters.T)).T
        Raw_Data.EXOv = np.column_stack(
            (Raw_Data.EXOv, np.matrix(np.arange(1, Parameters.T + 1)).T))
    elif Parameters.Trend == 3:
        Raw_Data.EXOv = np.matrix(np.ones(Parameters.T)).T
        Raw_Data.EXOv = np.column_stack(
            (Raw_Data.EXOv, np.matrix(np.arange(1, Parameters.T + 1)).T))
        Raw_Data.EXOv = np.column_stack(
            (Raw_Data.EXOv, np.matrix(np.arange(1, Parameters.T + 1) ** 2).T))

    TT = Parameters.n - 4
    if Parameters.Trend == 1:
        Raw_Data.EXOv_AR = np.matrix(np.ones(TT)).T
    elif Parameters.Trend == 2:
        Raw_Data.EXOv_AR = np.matrix(np.ones(TT)).T
        Raw_Data.EXOv_AR = np.column_stack(
            (Raw_Data.EXOv_AR, np.matrix(np.arange(1, TT + 1)).T))
    elif Parameters.Trend == 3:
        Raw_Data.EXOv_AR = np.matrix(np.ones(TT)).T
        Raw_Data.EXOv_AR = np.column_stack(
            (Raw_Data.EXOv_AR, np.matrix(np.arange(1, TT + 1)).T))
        Raw_Data.EXOv_AR = np.column_stack(
            (Raw_Data.EXOv_AR, np.matrix(np.arange(1, TT + 1) ** 2).T))

    return Raw_Data


def LBVAR_variable_maker(Raw_Data, Parameters):
    """
    LBVAR 모델을 위한 변수 Z와 Y를 생성하여 Raw_Data에 추가한다.

    Parameters
    ----------
    Raw_Data : dict
        - 'Set' : np.matrix
            시계열 데이터 매트릭스 (n x nvar)
        - 'EXOv' : np.matrix
            외생변수 행렬 (T x c), EXOv_total_maker 함수에서 생성됨

    Parameters : dict
        - 'T' : int
            분석에 사용되는 관측치의 수
        - 'k' : int
            VAR 모델의 총 변수 개수 (nvar x p)
        - 'p' : int
            VAR 모델의 시차 수
        - 'nvar' : int
            변수의 수

    Returns
    -------
    Raw_Data : dict
        - 'Z' : np.matrix
            회귀분석의 독립변수 행렬 (T x (k + c))
        - 'Y' : np.matrix
            종속변수 행렬 (T x nvar)

    함수 동작
    --------
    - VAR 모델의 독립변수 행렬 Z를 생성한다.
        - Z는 시차가 있는 변수들과 외생변수를 포함한다.
    - 종속변수 행렬 Y를 생성한다.
    - Raw_Data 딕셔너리에 Z와 Y를 추가한다.
    """
    Z = np.matrix(np.empty((Parameters.T, Parameters.k)))
    for i in range(0, Parameters.p):
        Z[:, i * Parameters.nvar:(i + 1) * Parameters.nvar] = Raw_Data.Set[Parameters.p - (i + 1):(
            Raw_Data.Set.shape[0]) - (i + 1), :]

    Z = np.column_stack((Raw_Data.EXOv, Z))

    Y = Raw_Data.Set[Parameters.p:, :]

    Raw_Data.Z = Z
    Raw_Data.Y = Y



    return Raw_Data


def As_LBVAR_AR_sigma_maker(Raw_Data, Parameters, Prior):
    """
    각 변수별로 AR(4) 모형을 추정하여 추정된 오차분산을 Prior.Sigma_hat에 저장한다.

    Parameters
    ----------
    Raw_Data : dict
        - 'Set' : np.matrix
            시계열 데이터 매트릭스 (n x nvar)
        - 'EXOv_AR' : np.matrix
            AR(4) 모형 추정을 위한 외생변수 행렬

    Parameters : dict
        - 'nvar' : int
            변수의 수
        - 'n' : int
            전체 관측치의 수

    Prior : dict
        - 함수 내에서 'Sigma_hat' 키를 생성하여 오차분산 추정치를 저장한다.

    Returns
    -------
    Prior : dict
        - 'Sigma_hat' : np.ndarray
            각 변수의 오차분산 추정치를 담은 대각행렬 (nvar x nvar)

    함수 동작
    --------
    - 각 변수에 대해 AR(4) 모형을 추정한다.
        - 각 변수별로 OLS를 통해 계수를 추정하고 오차분산을 계산한다.
    - 추정된 오차분산을 Prior.Sigma_hat에 저장한다.
    """
    Sigma_hat_dum = np.zeros((Parameters.nvar, Parameters.nvar))

    for i in range(0, Parameters.nvar):
        Y_AR = Raw_Data.Set
        Y_AR = Y_AR[:, i]
        X_AR = np.empty((Parameters.n - 4, 4))
        for j in range(0, 4):
            X_AR[:, j] = Y_AR[4 - (j + 1):Y_AR.shape[0] - (j + 1), :].reshape(-1)
        X_AR = np.column_stack((Raw_Data.EXOv_AR, X_AR))
        Y_AR = Y_AR[4:, :]
        Beta_AR = np.matmul(np.linalg.inv(np.matmul(X_AR.T, X_AR)), np.matmul(X_AR.T, Y_AR))
        Tem_sq_sum = np.power((Y_AR - np.matmul(X_AR, Beta_AR)), 2)
        Sigma_hat_dum[i, i] = np.mean(Tem_sq_sum)

    Prior.Sigma_hat = Sigma_hat_dum

    return Prior


def Prior_Maker(Raw_Data, Parameters, Prior, hyperparameters):
    """
    주어진 데이터와 하이퍼파라미터를 사용하여 Prior 분포에 필요한 변수들을 생성한다.

    Parameters
    ----------
    Raw_Data : dict
        - 'Set' : np.matrix
            시계열 데이터 매트릭스
        - 'EXOv_AR' : np.matrix
            AR 모형 추정을 위한 외생변수 행렬

    Parameters : dict
        - 'nvar' : int
            변수의 수
        - 'num_of_parameter' : int
            VAR 모델의 계수 수
        - 기타 필요한 파라미터들

    Prior : dict
        - 함수 내에서 'nu', 'S', 'mbeta', 'Vbeta', 'ma', 'maV'를 생성한다.

    hyperparameters : np.ndarray
        - 하이퍼파라미터 값들의 배열

    Returns
    -------
    Prior : dict
        - 'Sigma_hat' : np.ndarray
            각 변수의 오차분산 추정치 (As_LBVAR_AR_sigma_maker 함수에서 생성)
        - 'nu' : np.ndarray
            자유도 파라미터
        - 'S' : np.ndarray
            스케일 파라미터
        - 'mbeta' : np.ndarray
            베타의 사전 평균
        - 'Vbeta' : np.ndarray
            베타의 사전 분산
        - 'ma' : np.ndarray
            계수 a의 사전 평균
        - 'maV' : np.ndarray
            계수 a의 사전 분산
        - 'Vbeta_reduced' : np.ndarray
            베타의 Reduced Form 하이퍼파라미터를 통해 구한 사전 분산

    함수 동작
    --------
    - As_LBVAR_AR_sigma_maker 함수를 호출하여 Prior.Sigma_hat를 생성한다.
    - 각 변수에 대해 nu_i, S_i, mbeta_i, Vbeta_reduced를 생성한다.
        - nu_i_maker, S_i_maker, LBVAR_minnesota_beta, Vbeta_i_maker 함수를 사용
    - 마지막 변수에 대해 ma_i와 maV_i를 생성한다.
        - ma_i_maker, maV_i_maker 함수를 사용
    - 구축된 값들을 이용하여 Structural Form Vbeta를 생성한다.
        - Struct_Vbeta_i_maker 함수를 사용
    """
    Prior = As_LBVAR_AR_sigma_maker(Raw_Data, Parameters, Prior)

    Prior.nu = np.empty((Parameters.nvar, 1))
    Prior.S = np.empty((Parameters.nvar, 1))
    Prior.mbeta = np.empty((Parameters.num_of_parameter, Parameters.nvar))
    Prior.Vbeta_reduced = np.empty((Parameters.num_of_parameter, Parameters.nvar))
    Prior.Vbeta = np.empty((Parameters.num_of_parameter, Parameters.nvar))

    for i in range(0, Parameters.nvar):
        Prior.nu[i] = nu_i_maker(i + 1)
        Prior.S[i] = S_i_maker(i, Prior.Sigma_hat)
        Prior.mbeta[:, i] = LBVAR_minnesota_beta(i, Parameters)
        Prior.Vbeta_reduced[:,i] = Vbeta_i_maker(i, Parameters, Prior, hyperparameters)

        if i == (Parameters.nvar - 1):
            Prior.ma = ma_i_maker(Parameters)
            Prior.maV = maV_i_maker(Prior, Parameters)

    for i in range(0, Parameters.nvar) :
        Prior.Vbeta[:,i] = Struct_Vbeta_i_maker(i, Parameters, Prior)

    return Prior



def nu_i_maker(i):
    """
    각 변수에 대한 자유도 파라미터 nu_i를 생성한다.

    Parameters
    ----------
    i : int
        변수의 인덱스 (0부터 시작)

    Returns
    -------
    nu_i : float
        변수 i에 대한 자유도 파라미터

    함수 동작
    --------
    - nu_i = 1 + (i / 2)
    - 변수의 인덱스를 사용하여 nu_i를 계산한다.
    """
    nu_i = 1 + (i / 2)
    return nu_i


def S_i_maker(i, Sigma_hat):
    """
    각 변수에 대한 스케일 파라미터 S_i를 생성한다.

    Parameters
    ----------
    i : int
        변수의 인덱스 (0부터 시작)
    Sigma_hat : np.ndarray
        오차분산 추정치 행렬 (nvar x nvar)

    Returns
    -------
    S_i : float
        변수 i에 대한 스케일 파라미터

    함수 동작
    --------
    - S_i = Sigma_hat[i, i] / 2
    - 변수 i의 오차분산 추정치를 사용하여 S_i를 계산한다.
    """
    S_i = (Sigma_hat[i, i]) / 2
    return S_i


def LBVAR_minnesota_beta(i, Parameters):
    """
    변수 i에 대한 Minnesota 사전 평균 벡터를 생성한다.

    Parameters
    ----------
    i : int
        변수의 인덱스 (0부터 시작)
    Parameters : dict
        - 'num_of_parameter' : int
            전체 파라미터의 수 (nvar * p + c)
        - 'RV_list' : list
            레벨 변수의 인덱스 리스트
        - 'c' : int
            외생변수의 수
        - 'beta' : float
            레벨 변수에 대한 사전 평균 값

    Returns
    -------
    Minnesota_beta : np.ndarray
        변수 i에 대한 사전 평균 벡터 (길이: num_of_parameter)

    함수 동작
    --------
    - vec_minnesota_beta_prior_dum을 0으로 초기화한다.
    - 변수 i가 RV_list에 포함되어 있으면 해당 위치에 beta 값을 할당한다.
    """
    vec_minnesota_beta_prior_dum = np.zeros(Parameters.num_of_parameter)
    if i in Parameters.RV_list:
        vec_minnesota_beta_prior_dum[Parameters.c + i] = Parameters.beta
    Minnesota_beta = vec_minnesota_beta_prior_dum
    return Minnesota_beta


def Vbeta_i_maker(i, Parameters, Prior, hyperparameters):
    """
    변수 i에 대한 베타의 사전 분산 Vbeta_i를 생성한다.

    Parameters
    ----------
    i : int
        변수의 인덱스 (0부터 시작)
    Parameters : dict
        - 'c' : int
            외생변수의 수
        - 'p' : int
            VAR 모델의 시차 수
        - 'nvar' : int
            변수의 수
    Prior : dict
        - 'Sigma_hat' : np.ndarray
            오차분산 추정치 행렬 (nvar x nvar)
    hyperparameters : np.ndarray
        - hyperparameters[0]: own-lag 분산 조정 계수 (lambda_0)
        - hyperparameters[1]: cross-lag 분산 조정 계수 (lambda_1)
        - hyperparameters[2]: 상수항 분산 조정 계수 (lambda_c)

    Returns
    -------
    Vbeta_i : np.ndarray
        변수 i에 대한 베타의 사전 분산 벡터

    함수 동작
    --------
    - 외생변수 부분은 hyperparameters[2]로 초기화한다.
    - 각 시차와 변수에 대해 분산을 계산하여 Vbeta_i에 추가한다.
    """
    Vbeta_i = np.zeros((Parameters.c, 1))
    Vbeta_i[0:Parameters.c] = hyperparameters[2]
    for o in range(0, Parameters.p):
        for j in range(0, Parameters.nvar):
            if j == i:
                Vbeta_i_k = hyperparameters[0] / (Prior.Sigma_hat[j, j] * ((o + 1) ** 2))
            else:
                Vbeta_i_k = hyperparameters[1] / (Prior.Sigma_hat[j, j] * ((o + 1) ** 2))
            Vbeta_i = np.row_stack((Vbeta_i, Vbeta_i_k))

    Vbeta_i = Vbeta_i.reshape(-1)
    return Vbeta_i


def ma_i_maker(Parameters):
    """
    변수 i에 대한 a의 사전 평균 벡터를 생성한다.

    Parameters
    ----------
    Parameters : dict
        - 'nvar' : int
            변수의 수

    Returns
    -------
    ma_i : np.ndarray
        변수 i에 대한 a의 사전 평균 벡터 (길이: nvar - 1)

    함수 동작
    --------
    - ma_i를 0으로 초기화한다.
    """
    ma_i = np.zeros((Parameters.nvar - 1))
    return ma_i


def maV_i_maker(Prior, Parameters):
    """
    변수 i에 대한 a의 사전 분산 행렬을 생성한다.

    Parameters
    ----------
    Prior : dict
        - 'Sigma_hat' : np.ndarray
            오차분산 추정치 행렬
    Parameters : dict
        - 'nvar' : int
            변수의 수

    Returns
    -------
    maV_i : np.ndarray
        변수 i에 대한 a의 사전 분산 행렬 (대각 행렬)

    함수 동작
    --------
    - Sigma_hat의 대각 원소의 역수를 취하여 대각 행렬을 생성한다.
    """
    maV_i = Prior.Sigma_hat[:Parameters.nvar, :Parameters.nvar]
    maV_i = np.where(maV_i == 0, np.inf, maV_i)  # NOTE: 0인 경우 np.inf로 대체
    maV_i = 1 / maV_i
    maV_i = np.diag(maV_i)
    return maV_i

def Struct_Vbeta_i_maker(i, Parameters, Prior):
    """
    Reduced Form의 Hyperparameter들을 Structural Hyperparameter로 역산한 후 Variance를 구함

    Parameters
    --------
    i : int
        변수의 인덱스 (0부터 시작)
    Parameters : dict
        - 'num_of_parameter' : int
            전체 파라미터의 수 (nvar * p + c)
    Prior : dict
        - 'Sigma_hat' : np.ndarray
            각 변수의 오차분산 추정치 (As_LBVAR_AR_sigma_maker 함수에서 생성)
        - 'mbeta' : np.ndarray
            베타의 사전 평균
        - 'Vbeta_reduced' : np.ndarray
            베타의 Reduced Form 하이퍼파라미터 하에서의 사전 분산

    Returns
    --------
    Vbeta : np.ndarray
        변수 i에 대한 베타의 사전 분산 벡터 (Structural)
    
    함수 동작
    --------
    - 첫 번째 변수의 경우 기존 분산을 그대로 사용
    - 두 번째 변수의 경우 누적합 및 제곱을 진행 (Chan 2022; Asymmetric conjugate priors for large Bayesian VARs Appendix 6p 참고)
    """

    diag_sigma = np.diag(Prior.Sigma_hat)
    diag_sigma_inv = 1 / diag_sigma 

    Vbeta = np.zeros(Parameters.num_of_parameter)

    for j in range(Parameters.num_of_parameter):
        if i == 0:
            Vbeta[j] = Prior.Vbeta_reduced[j, i] 
        else:
            Vbeta[j] = Prior.Vbeta_reduced[j, i] + np.sum(
                Prior.Vbeta_reduced[j, :i] + (Prior.mbeta[j, :i] ** 2) * diag_sigma_inv[:i]
            )
    
    return Vbeta

def Posterior_Draw(Parameters, Raw_Data, Prior):
    """
    후방분포로부터 파라미터의 샘플을 추출한다.

    Parameters
    ----------
    Parameters : dict
        - 'nvar' : int
            변수의 수
        - 'ndraws' : int
            샘플링 횟수
        - 'num_of_parameter' : int
            총 파라미터 수
        - 'k' : int
            VAR 모델의 총 변수 개수 (nvar x p)
        - 'c' : int
            외생변수의 수
        - 'T' : int
            관측치의 수
    Raw_Data : dict
        - 'Z' : np.ndarray
            독립변수 행렬
        - 'Y' : np.ndarray
            종속변수 행렬
    Prior : dict
        - 'Vbeta' : np.ndarray
            베타의 사전 분산
        - 'S' : np.ndarray
            스케일 파라미터
        - 'mbeta' : np.ndarray
            베타의 사전 평균
        - 'maV' : np.ndarray
            a의 사전 분산
        - 'ma' : np.ndarray
            a의 사전 평균
        - 'nu' : np.ndarray
            자유도 파라미터

    Returns
    -------
    Draw : dict
        - 'theta_is' : list
            각 변수별 파라미터 샘플
        - 'Sigma_i_sq' : np.ndarray
            오차 분산의 샘플
        - 'A_matrix' : np.ndarray
            A 행렬의 샘플
        - 'Bet' : np.ndarray
            베타의 샘플
        - 'Sigma_struct' : np.ndarray
            구조적 오차 분산 행렬의 샘플
        - 'Sigma' : np.ndarray
            오차 분산 행렬의 샘플

    함수 동작
    --------
    - 각 변수에 대해 후방분포에서 파라미터를 샘플링한다.
    - 샘플링된 값을 사용하여 A_matrix, Bet 등을 계산한다.
    """
    Draw = container_class.Container()
    Draw.theta_is = np.empty(Parameters.nvar, dtype=object)
    Draw.Sigma_i_sq = np.empty((Parameters.ndraws, Parameters.nvar))
    for i in range(0, Parameters.nvar):
        Draw.theta_is[i] = np.empty((Parameters.num_of_parameter + i, Parameters.ndraws))

    for i in tqdm(range(0, Parameters.nvar)):
        if i == 0:
            V = Prior.Vbeta[:, i]
            X_i = Raw_Data.Z
            Y_i = Raw_Data.Y[:, i]
            S_i = Prior.S[i, 0]
            M_i = Prior.mbeta[:, i]
        else:
            V = np.hstack((Prior.Vbeta[:, i], Prior.maV[0:i]))
            X_i = np.column_stack((Raw_Data.Z, -Raw_Data.Y[:, 0:i]))
            Y_i = Raw_Data.Y[:, i]
            S_i = Prior.S[i, 0]
            M_i = np.hstack((Prior.mbeta[:, i], Prior.ma[0:i]))

        M_i = M_i.reshape(M_i.shape[0], 1)
        V_i = np.diag(V)
        inv_V_i = np.linalg.solve(V_i, np.identity(V_i.shape[0]))
        K_theta_i = inv_V_i + X_i.T @ X_i
        Chol_K_theta_i = np.linalg.cholesky(K_theta_i)
        theta_hat_i = sc.linalg.solve_triangular(Chol_K_theta_i, (inv_V_i @ M_i + X_i.T @ Y_i), lower=True)
        theta_hat_i = sc.linalg.solve_triangular(Chol_K_theta_i.T, theta_hat_i)
        S_hat_i = S_i + (Y_i.T @ Y_i + (M_i.T @ inv_V_i @ M_i) - theta_hat_i.T @ K_theta_i @ theta_hat_i) / 2

        for d in range(0, Parameters.ndraws):
            Draw.Sigma_i_sq[d,i] = 1/(np.random.gamma(Prior.nu[i] + (Parameters.T/2), 1/S_hat_i))
            Draw.theta_is[i][:,d] = np.array(theta_hat_i).reshape(-1) + sc.linalg.solve_triangular(Chol_K_theta_i.T, np.random.normal(0, np.sqrt(Draw.Sigma_i_sq[d,i]), theta_hat_i.shape[0])) 


    I = np.eye(Parameters.nvar)
    Draw.A_matrix = np.tile(I[:,:,np.newaxis], (1,1,Parameters.ndraws))
    Draw.Bet = np.empty((Parameters.nvar, Parameters.k + Parameters.c, Parameters.ndraws))
    Draw.Sigma_struct = np.zeros((Parameters.nvar, Parameters.nvar, Parameters.ndraws))
    Draw.Sigma = np.zeros((Parameters.nvar, Parameters.nvar, Parameters.ndraws))
    for d in range(0, Parameters.ndraws) :
        I = np.eye(Parameters.nvar)
        Bet = np.empty((Parameters.nvar, Parameters.k + Parameters.c))
        
        for i in range(0, Parameters.nvar) :
            I[i,0:i] = Draw.theta_is[i][Parameters.k + Parameters.c:,d]
            Bet[i,:] = Draw.theta_is[i][0:Parameters.k + Parameters.c,d]
        
        Draw.A_matrix[:,:,d] = I
        Draw.Bet[:,:,d] = np.linalg.inv(I) @ Bet
        Draw.Sigma_struct[:,:,d] = np.diag(Draw.Sigma_i_sq[d,:])
        Draw.Sigma[:,:,d] = np.linalg.inv(Draw.A_matrix[:,:,d]) @ Draw.Sigma_struct[:,:,d] @ np.linalg.inv(Draw.A_matrix[:,:,d]).T

    return Draw


def log_pdf_inv_gamma(a, b, x):
    """
    Inverse Gamma 분포의 확률 밀도 함수의 로그 값을 계산하는 함수입니다.

    Parameters
    ----------
    a : float
        Inverse Gamma 분포의 shape 파라미터 (a > 0).
    b : float
        Inverse Gamma 분포의 scale 파라미터 (b > 0).
    x : float 또는 np.ndarray
        확률 밀도를 계산할 위치 (x > 0).

    Returns
    -------
    log_pdf : float 또는 np.ndarray
        주어진 x에서의 Inverse Gamma 분포의 로그 확률 밀도 값.

    함수 동작
    --------
    - Inverse Gamma 분포의 확률 밀도 함수는 다음과 같습니다:
        f(x; a, b) = [b^{a} / Γ(a)] * x^{-a-1} * e^{-b/x}
    - 이 함수는 위 식의 로그 값을 계산합니다.
    - scipy.special.gamma 함수를 사용하여 감마 함수 Γ(a)를 계산합니다.
    - 로그 계산을 통해 매우 작은 값에 대한 언더플로우 문제를 방지합니다.
    """
    log_pdf = a * np.log(b) - np.log(sc.special.gamma(a)) - (a + 1) * np.log(x) - b / x
    return log_pdf


def Log_function(a):
    """
    주어진 값의 자연로그를 계산하는 함수입니다.

    Parameters
    ----------
    a : float 또는 np.ndarray
        로그를 계산할 값.

    Returns
    -------
    result : float 또는 np.ndarray
        입력값의 자연로그 값.

    함수 동작
    --------
    - np.log 함수를 사용하여 자연로그를 계산합니다.
    - 계산 결과가 NaN인 경우 0으로 처리합니다.
    """
    result = np.log(a)
    # NaN 값을 0으로 대체하려면 아래 주석을 해제하세요.
    # result = np.where(np.isnan(result), 0, result)
    return result


def Marginal_Likelihood(Raw_Data, Parameters, Prior, candi_hyperparameters):
    """
    주어진 하이퍼파라미터에 대한 주변우도를 계산하는 함수입니다.

    Parameters
    ----------
    Raw_Data : dict
        - 'Z' : np.ndarray
            독립변수 행렬 (T x k)
        - 'Y' : np.ndarray
            종속변수 행렬 (T x nvar)
    Parameters : dict
        - 'nvar' : int
            변수의 수
        - 'T' : int
            관측치의 수
    Prior : dict
        - 함수 내에서 업데이트되는 Prior 정보를 포함합니다.
    candi_hyperparameters : np.ndarray
        - hyperparameters[0]: own-lag 분산 조정 계수 (lambda_0)
        - hyperparameters[1]: cross-lag 분산 조정 계수 (lambda_1)
        - hyperparameters[2]: 상수항 분산 조정 계수 (lambda_c)

    Returns
    -------
    Kernel : float
        주변우도의 음수 값 (최소화 문제를 위해 음수로 반환).

    함수 동작
    --------
    - 주어진 하이퍼파라미터를 사용하여 Prior를 업데이트합니다.
    - 각 변수에 대한 부분 로그 우도를 계산하고 합산합니다.
    - Gamma 분포의 로그 확률 밀도를 사용하여 하이퍼파라미터의 사전 분포 로그 값을 계산합니다.
    - 전체 로그 주변우도를 계산하고 음수로 반환합니다 (최적화를 위한 최소화 문제로 변환).
    """
    Prior = Prior_Maker(Raw_Data, Parameters, Prior, candi_hyperparameters)

    Partials = np.empty(Parameters.nvar)
    for i in range(0, Parameters.nvar):
        if i == 0:
            V = Prior.Vbeta[:, i]
            X_i = Raw_Data.Z
            Y_i = Raw_Data.Y[:, i]
            S_i = Prior.S[i, 0]
            M_i = Prior.mbeta[:, i]
        else:
            V = np.hstack((Prior.Vbeta[:, i], Prior.maV[0:i]))
            X_i = np.column_stack((Raw_Data.Z, -Raw_Data.Y[:, 0:i]))
            Y_i = Raw_Data.Y[:, i]
            S_i = Prior.S[i, 0]
            M_i = np.hstack((Prior.mbeta[:, i], Prior.ma[0:i]))

        M_i = M_i.reshape(M_i.shape[0], 1)
        V_i = np.diag(V)
        inv_V_i = np.linalg.solve(V_i, np.identity(V_i.shape[0]))
        K_theta_i = inv_V_i + X_i.T @ X_i
        Chol_K_theta_i = np.linalg.cholesky(K_theta_i)
        theta_hat_i = np.linalg.solve(Chol_K_theta_i, (inv_V_i @ M_i + X_i.T @ Y_i))
        theta_hat_i = np.linalg.solve(Chol_K_theta_i.T, theta_hat_i)
        S_hat_i = S_i + (Y_i.T @ Y_i + (M_i.T @ inv_V_i @ M_i) - theta_hat_i.T @ K_theta_i @ theta_hat_i) / 2

        Partials[i] = -(1 / 2) * (Log_function(np.linalg.det(V_i)) + Log_function(np.linalg.det(K_theta_i))) \
                      + Log_function(sc.special.gamma(Prior.nu[i] + (Parameters.T / 2))) + Prior.nu[i] * Log_function(S_i) \
                      - Log_function(sc.special.gamma(Prior.nu[i])) - (Prior.nu[i] + Parameters.T / 2) * Log_function(S_hat_i)

    log_p = (-(Parameters.T * Parameters.nvar / 2) * np.log(2 * np.pi)) + np.sum(Partials)
    pi_p = func.log_pdf_gamma_1(1.0733, 0.6825, candi_hyperparameters[0]) \
           + func.log_pdf_gamma_1(1.0159, 0.3137, candi_hyperparameters[1]) \
           + func.log_pdf_gamma_1(101, 1, candi_hyperparameters[2])

    Kernel = -(log_p + pi_p)
    return Kernel


def Optimization(Raw_Data, Parameters, Prior, candi_hyperparameters):
    """
    하이퍼파라미터 최적화를 위한 목적함수를 계산하는 함수입니다.

    Parameters
    ----------
    Raw_Data : dict
        - 'Z' : np.ndarray
            독립변수 행렬 (T x k)
        - 'Y' : np.ndarray
            종속변수 행렬 (T x nvar)
    Parameters : dict
        - 'nvar' : int
            변수의 수
        - 'T' : int
            관측치의 수
    Prior : dict
        - 함수 내에서 업데이트되는 Prior 정보를 포함합니다.
    candi_hyperparameters : np.ndarray
        - hyperparameters[0]: own-lag 분산 조정 계수 (lambda_0)
        - hyperparameters[1]: cross-lag 분산 조정 계수 (lambda_1)
        - hyperparameters[2]: 상수항 분산 조정 계수 (lambda_c)

    Returns
    -------
    Kernel : float
        목적함수 값 (최소화 문제를 위해 음수 로그 주변우도와 사전 분포 로그 값을 반환).

    함수 동작
    --------
    - Marginal_Likelihood 함수와 동일하게 주변우도를 계산합니다.
    - 하이퍼파라미터의 사전 분포 로그 값을 더하여 전체 목적함수를 계산합니다.
    - 계산된 값을 반환합니다.
    """
    Prior = Prior_Maker(Raw_Data, Parameters, Prior, candi_hyperparameters)

    Partials = np.empty(Parameters.nvar)
    for i in range(0, Parameters.nvar):
        if i == 0:
            V = Prior.Vbeta[:, i]
            X_i = Raw_Data.Z
            Y_i = Raw_Data.Y[:, i]
            S_i = Prior.S[i, 0]
            M_i = Prior.mbeta[:, i]
        else:
            V = np.hstack((Prior.Vbeta[:, i], Prior.maV[0:i]))
            X_i = np.column_stack((Raw_Data.Z, -Raw_Data.Y[:, 0:i]))
            Y_i = Raw_Data.Y[:, i]
            S_i = Prior.S[i, 0]
            M_i = np.hstack((Prior.mbeta[:, i], Prior.ma[0:i]))

        M_i = M_i.reshape(M_i.shape[0], 1)
        V_i = np.diag(V)
        inv_V_i = np.linalg.solve(V_i, np.identity(V_i.shape[0]))
        K_theta_i = inv_V_i + X_i.T @ X_i
        Chol_K_theta_i = np.linalg.cholesky(K_theta_i)
        theta_hat_i = np.linalg.solve(Chol_K_theta_i, (inv_V_i @ M_i + X_i.T @ Y_i))
        theta_hat_i = np.linalg.solve(Chol_K_theta_i.T, theta_hat_i)
        S_hat_i = S_i + (Y_i.T @ Y_i + (M_i.T @ inv_V_i @ M_i) - theta_hat_i.T @ K_theta_i @ theta_hat_i) / 2

        Partials[i] = -(1 / 2) * (Log_function(np.linalg.det(V_i)) + Log_function(np.linalg.det(K_theta_i))) \
                      + Log_function(sc.special.gamma(Prior.nu[i] + (Parameters.T / 2))) + Prior.nu[i] * Log_function(S_i) \
                      - Log_function(sc.special.gamma(Prior.nu[i])) - (Prior.nu[i] + Parameters.T / 2) * Log_function(S_hat_i)

    log_p = (-(Parameters.T * Parameters.nvar / 2) * np.log(2 * np.pi)) + np.sum(Partials)
    pi_p = func.log_pdf_gamma_1(1.0733, 0.6825, candi_hyperparameters[0]) \
           + func.log_pdf_gamma_1(1.0159, 0.3137, candi_hyperparameters[1]) \
           + func.log_pdf_gamma_1(101, 1, candi_hyperparameters[2])

    Kernel = -(log_p + pi_p)
    return Kernel, Prior


def Log_Kernel(Raw_Data, Parameters, Prior, candi_hyperparameters):
    """
    MCMC를 위한 로그 커널 값을 계산하는 함수입니다.

    Parameters
    ----------
    Raw_Data : dict
        - 'Z' : np.ndarray
            독립변수 행렬 (T x k)
        - 'Y' : np.ndarray
            종속변수 행렬 (T x nvar)
    Parameters : dict
        - 'nvar' : int
            변수의 수
        - 'T' : int
            관측치의 수
    Prior : dict
        - 함수 내에서 업데이트되는 Prior 정보를 포함합니다.
    candi_hyperparameters : np.ndarray
        - hyperparameters[0]: own-lag 분산 조정 계수 (lambda_0)
        - hyperparameters[1]: cross-lag 분산 조정 계수 (lambda_1)
        - hyperparameters[2]: 상수항 분산 조정 계수 (lambda_c)

    Returns
    -------
    Kernel : float
        로그 커널 값 (로그 주변우도와 하이퍼파라미터의 로그 사전 분포 합).

    함수 동작
    --------
    - Marginal_Likelihood 함수와 동일하게 로그 주변우도를 계산합니다.
    - 하이퍼파라미터의 로그 사전 분포 값을 더하여 로그 커널 값을 계산합니다.
    - 계산된 값을 반환합니다.
    """
    Prior = Prior_Maker(Raw_Data, Parameters, Prior, candi_hyperparameters)

    Partials = np.empty(Parameters.nvar)
    for i in range(0, Parameters.nvar):
        if i == 0:
            V = Prior.Vbeta[:, i]
            X_i = Raw_Data.Z
            Y_i = Raw_Data.Y[:, i]
            S_i = Prior.S[i, 0]
            M_i = Prior.mbeta[:, i]
        else:
            V = np.hstack((Prior.Vbeta[:, i], Prior.maV[0:i]))
            X_i = np.column_stack((Raw_Data.Z, -Raw_Data.Y[:, 0:i]))
            Y_i = Raw_Data.Y[:, i]
            S_i = Prior.S[i, 0]
            M_i = np.hstack((Prior.mbeta[:, i], Prior.ma[0:i]))

        M_i = M_i.reshape(M_i.shape[0], 1)
        V_i = np.diag(V)
        inv_V_i = np.linalg.solve(V_i, np.identity(V_i.shape[0]))
        K_theta_i = inv_V_i + X_i.T @ X_i
        Chol_K_theta_i = np.linalg.cholesky(K_theta_i)
        theta_hat_i = np.linalg.solve(Chol_K_theta_i, (inv_V_i @ M_i + X_i.T @ Y_i))
        theta_hat_i = np.linalg.solve(Chol_K_theta_i.T, theta_hat_i)
        S_hat_i = S_i + (Y_i.T @ Y_i + (M_i.T @ inv_V_i @ M_i) - theta_hat_i.T @ K_theta_i @ theta_hat_i) / 2

        Partials[i] = -(1 / 2) * (Log_function(np.linalg.det(V_i)) + Log_function(np.linalg.det(K_theta_i))) \
                      + Log_function(sc.special.gamma(Prior.nu[i] + (Parameters.T / 2))) + Prior.nu[i] * Log_function(S_i) \
                      - Log_function(sc.special.gamma(Prior.nu[i])) - (Prior.nu[i] + Parameters.T / 2) * Log_function(S_hat_i)

    log_p = (-(Parameters.T * Parameters.nvar / 2) * np.log(2 * np.pi)) + np.sum(Partials)
    pi_p = func.log_pdf_gamma_1(1.0733, 0.6825, candi_hyperparameters[0]) \
           + func.log_pdf_gamma_1(1.0159, 0.3137, candi_hyperparameters[1]) \
           + func.log_pdf_gamma_1(101, 1, candi_hyperparameters[2])

    Kernel = log_p + pi_p
    return Kernel


def Hyperparameter_MCMC(Raw_Data, Parameters, Prior, hyperparameters, hessian, verbose=True, n_draws=10000, n_burnin=1000):
    """
    MCMC를 사용하여 하이퍼파라미터를 추정하는 함수입니다.

    Parameters
    ----------
    Raw_Data : dict
        - 'Z' : np.ndarray
            독립변수 행렬 (T x k)
        - 'Y' : np.ndarray
            종속변수 행렬 (T x nvar)
    Parameters : dict
        - 'nvar' : int
            변수의 수
        - 'T' : int
            관측치의 수
    Prior : dict
        - 함수 내에서 업데이트되는 Prior 정보를 포함합니다.
    hyperparameters : np.ndarray
        초기 하이퍼파라미터 값
    hessian : np.ndarray
        초기 하이퍼파라미터에 대한 Hessian 행렬

    Returns
    -------
    Medians : np.ndarray
        MCMC 결과에서 추정된 하이퍼파라미터의 중간값

    함수 동작
    --------
    - 주어진 Hessian 행렬을 사용하여 하이퍼파라미터의 제안 분포의 공분산을 설정합니다.
    - MCMC를 수행하여 새로운 하이퍼파라미터 샘플을 생성합니다.
    - 사후 분포에서 중간값을 계산하여 최종 하이퍼파라미터로 반환합니다.
    """
    if np.isnan(hessian).any():
        print("> NaN detected in Hessian matrix. Exiting function.")
        print(f"> Hyperparameter is selected as {np.round(hyperparameters,3)}")
        return hyperparameters

    try:
        VARIANCE = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        print("> Hessian matrix isn't Positive Definite. Exiting function.")
        print(f"> Hyperparameter is selected as {np.round(hyperparameters,3)}")
        return hyperparameters

    MCMC_hyperparameters = np.empty((3, 11000))
    MCMC_accept = np.zeros(11000)
    candi_hyperparameters_ = hyperparameters.copy()

    VARIANCE = (VARIANCE.T + VARIANCE) / 2
    if verbose:
        print("> Hyperparameter Optimization Start")
    try:
        # print("============Hyperparameter Optimization Start============")
        if verbose:
            iters = tqdm(range(n_draws), desc="Hyperparameter MCMC")
        else:
            iters = range(n_draws)
        for i in iters:
            Draw = False

            while not Draw:
                candi_hyperparameters = np.random.multivariate_normal(candi_hyperparameters_, VARIANCE)
                if all(candi_hyperparameters > 0):
                    Draw = True

            alpha = min(Log_Kernel(Raw_Data, Parameters, Prior, candi_hyperparameters) - Log_Kernel(Raw_Data, Parameters, Prior, candi_hyperparameters_), 0)
            U = np.log(np.random.rand())
            if U < alpha:
                MCMC_hyperparameters[:, i] = candi_hyperparameters
                candi_hyperparameters_ = candi_hyperparameters.copy()
                MCMC_accept[i] = 1
            else:
                MCMC_hyperparameters[:, i] = candi_hyperparameters_.copy()


        MCMC_results = MCMC_hyperparameters[:, 1001:]
        Medians = np.median(MCMC_results, axis=1)
        if verbose:
            print("> Hyperparameter Optimization Clear")
            print(f"> Opt Hyperparameters are {np.round(Medians,3)}")
        return Medians
    except Exception as e:
        print("MCMC isn't clearly working")
        print(f"Hyperparameter is selected as {np.round(hyperparameters,3)}")
        return hyperparameters


def Random_Search(object_function, init, K, P, alpha, verbose=True):
    """
    Random Search 알고리즘을 사용하여 최적화를 수행하는 함수입니다.

    Parameters
    ----------
    object_function : callable
        최적화할 목적함수
    init : np.ndarray
        초기 파라미터 값
    K : int
        전체 반복 횟수
    P : int
        각 반복에서의 시도 횟수
    alpha : float 또는 str
        스텝 사이즈 (float) 또는 'diminishing' (str)
    verbose: bool

    Returns
    -------
    Optimizers : np.ndarray
        최적화된 파라미터 값
    w_box : np.ndarray
        각 반복에서의 파라미터 값 기록
    min_box : np.ndarray
        각 반복에서의 목적함수 값 기록

    함수 동작
    --------
    - 주어진 초기값에서 시작하여 Random Search를 수행합니다.
    - 각 반복에서 여러 방향으로 파라미터를 변경하여 목적함수 값을 계산합니다.
    - 목적함수 값을 최소화하는 파라미터를 선택합니다.
    - 최적화된 파라미터와 각 반복의 기록을 반환합니다.
    """
    rng = default_rng()
    w = init.copy()
    init_size = init.shape[0]
    min_val = object_function(w)
    w_box = np.zeros((init_size, K))
    min_box = np.zeros(K)

    if verbose:
        iters = tqdm(range(K), desc="")
    else:
        iters = range(K)

    for k in iters:
        if k == 0 or k % 10 == 0:
            print_str = f"Current Optimization {k}/{K}: {np.round(min_val, 3)}, {np.round(w, 3)}"
            if verbose:
                iters.set_description(print_str)

        if alpha == "diminishing":
            a = 1 / (k + 1)
        else:
            a = alpha

        w_box[:, k] = w
        min_box[k] = min_val

        if k % 100 == 0 and k > 199:
            if np.abs(min_val - min_box[k - 100]) < 0.01:
                if verbose:
                    print(f"> Early return: Optimization converged at iteration {k}")
                w_box = w_box[:, :k]
                min_box = min_box[:k]
                return w, w_box, min_box

        min_temp_box = []
        w_temp_box = []

        for _ in range(P):
            directions = rng.normal(0, 1, init_size)
            norms = np.linalg.norm(directions)
            directions /= norms
            ww = w + a * directions

            if any(ww < 0):
                continue

            try:
                values = object_function(ww)
                if np.isnan(values) or np.isinf(values):
                    continue
            except Exception:
                continue

            w_temp_box.append(ww)
            min_temp_box.append(values)

        if min_temp_box:
            idx = np.argmin(min_temp_box)
            if min_temp_box[idx] < min_val:
                w = w_temp_box[idx]
                min_val = min_temp_box[idx]
        else:
            pass  # 새로운 후보가 없을 경우 현재 w와 min_val 유지

    Optimizers = w_box[:, np.argmin(min_box)]
    if verbose:
        print(f"> Calculated Optimization is : {np.round(np.min(min_box),3)}, {np.round(Optimizers, 3)}")
    return Optimizers, w_box, min_box