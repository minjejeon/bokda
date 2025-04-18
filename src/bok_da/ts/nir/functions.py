import numpy as np
import math as math
import matplotlib.pyplot as plt
import pandas as pd
from scipy.linalg import lu 


#============================================================
# Utils
#============================================================

 
def mergeOptions(defaultOptions, inputOptions):
    """
    기본 옵션과 입력 옵션을 병합해 새로운 옵션 딕셔더리 생성

    Parameters
    ----------
    defaultOptions : dict
                     기본 옵션 딕셔너리
    inputOptions : dict
                   사용자 지정 옵션 딕셔너리

    Returns
    -------
    mergedOptions : dict
                    병합된 옵션 딕셔너리
    """
    mergedOptions = defaultOptions.copy()

    for key, value in inputOptions.items():
        if key in defaultOptions:
            mergedOptions[key] = value
            
    return mergedOptions
 

def sample_statistics(sample, lower_p, upper_p):
    """
    주어진 sample 데이터에 대한 통계량 계산
    # 지금은 sample의 행별 계산 -> 사용하는거 상황봐서 변경필요

    Parameters
    ----------
    sample : ndarray 
             샘플 데이터.
    lower_p : float
              하위 백분위수
    upper_p : float
              상위 백분위수

    Returns
    -------
    avg : float
          샘플의 평균
    med : float
          샘플 중앙값
    perc_lower : float
                 하위 백분위수
    perc_upper : float
                 상한 백분위수
    """
    avg = np.mean(sample, axis=0)
    med = np.median(sample, axis=0)
    perc_lower = np.percentile(sample, lower_p, axis=0)
    perc_upper = np.percentile(sample, upper_p, axis=0)
    
    return avg, med, perc_lower, perc_upper


#============================================================
# Functions
#============================================================


def GenerateSlopeCoefficients(Y_data, X_data, sigma2, Inv_V_prior, Inv_V_mean_prior):
    """
    기울기 계수의 posterior 샘플 추출

    Parameters
    ----------
    Y_data : ndarray of shape (n,)
             종속 변수 데이터
    X_data : ndarray of shape (n,p)
             설명 변수 데이터
    sigma2 : float
             분산
    Inv_V_prior : ndarray of shape (p,p)
                  Prior 분포 분산의 역행렬
    Inv_V_mean_prior : ndarray of shape (p,)
                       Prior 분포 평균의 역행렬

    Returns
    -------
    Coefficients_draw : ndarray of shape (p,)
                        추출된 기울기 계수의 posterior 샘플
    Mean_posterior : ndarray of shape (p,)
                     Posterior 분포의 평균
    Variance_posterior : ndarray of shape (p,p)
                         Posterior 분포의 분산
    """
    Variance_posterior = np.linalg.inv(Inv_V_prior + (X_data.T @ X_data) / sigma2)
    Mean_posterior = (Variance_posterior @ (Inv_V_mean_prior + (X_data.T @ Y_data) / sigma2)).reshape(-1,1)
    Coefficients_draw = Mean_posterior + np.linalg.cholesky(Variance_posterior).T @ np.random.randn(Mean_posterior.shape[0],1)

    return Coefficients_draw, Mean_posterior, Variance_posterior
 
def hpfilter(x, lamb=1600):
    """
    Hodrick-Prescott filter.

    Parameters
    ----------
    x : ndarray(d,1)
        The time series to filter, 1-d.
    lamb : float
        The Hodrick-Prescott smoothing parameter.

    Returns
    -------
    trend : ndarray(d,1)
        The estimated trend in the data given lamb.
    """
    nobs = len(x)
    x = np.asarray(x)

    # Identity matrix
    I = np.eye(nobs)

    # Creating K matrix
    K = np.zeros((nobs - 2, nobs))
    for i in range(nobs - 2):
        K[i, i] = 1
        K[i, i + 1] = -2
        K[i, i + 2] = 1

    # Solve for trend using the normal equation
    KtK = K.T @ K
    trend = np.linalg.solve(I + lamb * KtK, x)

    # Calculate cycle
    cycle = x - trend
    # pd.Series(cycle, name='cycle') 는 제외. only obtain trend component
    return pd.Series(trend, name='trend')
 
def HP_filter(y, lamb=1600):
    """
    분기별 출력 데이터에 대한 Hodrick-Prescott 필터를 적용합니다.

    Parameters
    ----------
    y : ndarray
        시계열 데이터.
    lamb : float, optional
        스무딩 파라미터. 분기별 데이터의 경우 기본값은 1600입니다.

    Returns
    -------
    trend : ndarray
        시계열의 추세 성분.
    cyclical : ndarray
        시계열의 순환 성분.
    """
    T = len(y)
    I = np.eye(T)
    data = np.array([1, -2, 1])
    D = np.zeros((T - 2, T))
    for i in range(T - 2):
        D[i, i:i + 3] = data
    trend = np.linalg.inv(I + lamb * D.T @ D) @ y
    cyclical = y - trend
    return trend, cyclical

def logdet(A, op=None):
    """
    행렬의 로그 행렬식을 계산

    Parameters
    ----------
    A : ndarray
        입력 행렬
    op : str, optional
         'chol'을 사용하여 양의 정부호 행렬에 대해 Cholesky 분해를 수행

    Returns
    -------
    v : float
        로그 행렬식 값
    """
    assert isinstance(A, np.ndarray) and A.ndim == 2 and A.shape[0] == A.shape[1], \
        "A는 정방행렬이어야 합니다."
    
    if op is None:
        use_chol = False
    else:
        if op.lower() == 'chol':
            use_chol = True
        else:
            raise ValueError("두 번째 인자는 'chol' 문자열이어야 합니다.")
    
    # 계산 수행
    if use_chol:
        v = 2 * np.sum(np.log(np.diag(np.linalg.cholesky(A))))
    else:
        P, L, U = lu(A) 
        du = np.diag(U)
        c = np.linalg.det(P) * np.prod(np.sign(du))
        v = np.log(c) + np.sum(np.log(np.abs(du)))
    
    return v
 
def Prior_LatentVariables(Data_full, SSRatio, Ytivs, Ygivs, Izivs):
    """
    잠재 변수의 초기값에 대한 사전 분포를 계산합니다.

    Parameters
    ----------
    Data_full : ndarray
        'output'이 첫 번째 열에 포함된 데이터 행렬.
    SSRatio : float
        사전 계산에 사용할 샘플 비율.
    Ytivs : float
        출력 추세의 초기값에 대한 prior 분산의 스케일 계수.
    Ygivs : float
        출력 장기 성장률의 초기값에 대한 prior 분산의 스케일 계수.
    Izivs : float
        이자율 구성요소의 초기값에 대한 prior 분산 스케일 계수.

    Returns
    -------
    PriorLatent : dict
        잠재 변수의 prior 정보 (평균과 분산).
    """
    # 샘플 크기 측정
    T = Data_full.shape[0]
    Tpercent = int(round(SSRatio * T))

    # Hodrick-Prescott 필터 적용
    output = Data_full[:, 0]
    trend, cycle = HP_filter(output, lamb=1600)
    HP_DTrend = trend[1:] - trend[:-1]

    # 사이클의 분산 계산
    cycle_var = np.var(cycle[:Tpercent])

    # 잠재 변수에 대한 사전 평균 계산 (t=5부터 추정 시작)
    PriorLatent = {}
    PriorLatent['yt_mean'] = trend[1:4]  # 인덱스 1부터 3까지 (Python은 0부터 시작)
    PriorLatent['g_mean'] = HP_DTrend[0:3]
    PriorLatent['z_mean'] = np.zeros(3)

    # 잠재 변수에 대한 사전 분산 계산
    PriorLatent['yt_var'] = cycle_var * Ytivs
    PriorLatent['g_var'] = (1 ** 2) * Ygivs  # 장기 성장률에 대한 1%의 불확실성
    PriorLatent['z_var'] = (1 ** 2) * Izivs  # z 구성요소에 대한 1%의 불확실성

    # 전체 사전 분포 구성
    PriorLatent['mean'] = np.concatenate((PriorLatent['yt_mean'],
                                          PriorLatent['g_mean'],
                                          PriorLatent['z_mean']))
    PriorLatent['var'] = np.zeros((9, 9))

    # 마지막 기간 (MATLAB 인덱스 3,6,9는 Python 인덱스 2,5,8)
    PriorLatent['var'][2, 2] = PriorLatent['yt_var']
    PriorLatent['var'][5, 5] = PriorLatent['g_var']
    PriorLatent['var'][8, 8] = PriorLatent['z_var']

    # 마지막 기간 + 1
    PriorLatent['var'][1, 1] = PriorLatent['yt_var'] + PriorLatent['g_var'] + (1 ** 2)
    PriorLatent['var'][1, 2] = PriorLatent['yt_var']
    PriorLatent['var'][2, 1] = PriorLatent['yt_var']

    PriorLatent['var'][4, 4] = PriorLatent['g_var'] + (0.1 ** 2)
    PriorLatent['var'][4, 5] = PriorLatent['g_var']
    PriorLatent['var'][5, 4] = PriorLatent['g_var']

    PriorLatent['var'][7, 7] = PriorLatent['z_var'] + (0.1 ** 2)
    PriorLatent['var'][7, 8] = PriorLatent['z_var']
    PriorLatent['var'][8, 7] = PriorLatent['z_var']

    # 마지막 기간 + 2
    PriorLatent['var'][0, 0] = PriorLatent['var'][1, 1] + PriorLatent['var'][4, 4] + (1 ** 2)
    PriorLatent['var'][0, 1] = PriorLatent['var'][1, 1]
    PriorLatent['var'][1, 0] = PriorLatent['var'][1, 1]
    PriorLatent['var'][0, 2] = PriorLatent['var'][2, 2]
    PriorLatent['var'][2, 0] = PriorLatent['var'][2, 2]

    PriorLatent['var'][3, 3] = PriorLatent['var'][4, 4] + (0.1 ** 2)
    PriorLatent['var'][3, 4] = PriorLatent['var'][4, 4]
    PriorLatent['var'][3, 5] = PriorLatent['var'][5, 5]
    PriorLatent['var'][4, 3] = PriorLatent['var'][4, 4]
    PriorLatent['var'][5, 3] = PriorLatent['var'][5, 5]

    PriorLatent['var'][6, 6] = PriorLatent['var'][7, 7] + (0.1 ** 2)
    PriorLatent['var'][6, 7] = PriorLatent['var'][7, 7]
    PriorLatent['var'][6, 8] = PriorLatent['var'][8, 8]
    PriorLatent['var'][7, 6] = PriorLatent['var'][7, 7]
    PriorLatent['var'][8, 6] = PriorLatent['var'][8, 8]

    return PriorLatent
 

def DrawVarianceIG(ErrorData, shape_prior, scale_prior):
    """
    Inverse Gamma 분포로부터 분산 파라미터의 posterior 샘플을 추출합니다.

    Parameters
    ----------
    ErrorData : ndarray
        오차 데이터.
    shape_prior : float
        사전 분포의 shape 파라미터.
    scale_prior : float
        사전 분포의 scale 파라미터.

    Returns
    -------
    variance_sample : float
        추출된 분산 샘플.
    """
    T = len(ErrorData)
    shape_posterior = (shape_prior + T) / 2
    scaling_posterior = (scale_prior + np.dot(ErrorData, ErrorData)) / 2
    
    variance_sample = 1 / np.random.gamma(shape_posterior, 1 / scaling_posterior)
    return variance_sample

def AdaptiveMH_IG(Y, X, fieldName, modelName, ProposalType_idx, MHSample, ParaSample,
                  PriorLatent, pr_shape, pr_scale, ProposalVar_0, SizeInformation, last_loglike=None, s_data=None):
    """
    Inverse Gamma 분포에 대한 Adaptive Metropolis-Hastings 알고리즘을 구현합니다.

    Parameters
    ----------
    Y : ndarray
        종속 변수 데이터.
    X : ndarray
        독립 변수 데이터.
    fieldName : str
        샘플링할 파라미터의 이름.
    modelName : str
        사용할 칼만 필터 모델 이름 ('HLW', 'FC' 또는 'Covid').
    ProposalType_idx : int
        제안 분포의 타입 인덱스 (0 또는 1).
    MHSample : ndarray
        이전 MH 샘플들.
    ParaSample : dict
        현재 파라미터 샘플.
    PriorLatent : dict
        잠재 변수의 prior 정보.
    pr_shape : float
        Prior 분포의 shape 파라미터.
    pr_scale : float
        Prior 분포의 scale 파라미터.
    ProposalVar_0 : float
        제안 분포의 초기 분산.
    SizeInformation : dict
        데이터의 크기 정보.

    s_data : ndarray | Default = None

    Returns
    -------
    para_out : float
        새로운 파라미터 샘플.
    MHSample : ndarray
        업데이트된 MH 샘플들.
    accept_idx : int
        MH 알고리즘에서의 수락 여부 (1이면 수락, 0이면 거부).
    """
    
    # 마지막 파라미터 값
    para_val = MHSample[-1]

    # 제안 분포의 공분산 계산
    dimension_proposal = 1
    if ProposalType_idx == 0:
        cand_var = ProposalVar_0
    else:
        delta_proposal = 2.38 / np.sqrt(dimension_proposal)
        cand_var = (delta_proposal ** 2) * (np.var(MHSample, ddof=1) + 1e-5)

    # 제안 분포에서 후보 샘플 생성
    pr_cand = para_val + np.sqrt(cand_var) * np.random.randn()

    current_loglike = last_loglike

    # Prior 평가
    if pr_cand < 0:
        para_out = para_val
        MHSample = np.append(MHSample[1:], para_val)
        accept_idx = 0
    else:
        # 우도 함수 평가
        ParaSample_cand = ParaSample.copy()
        ParaSample_cand[fieldName] = pr_cand

        if modelName == 'HLW':
            LogLike_cand, _, _, _, _ = Kalman_filter_HLW(Y, X, ParaSample_cand, PriorLatent, SizeInformation)

            if last_loglike is None:
                LogLike_old, _, _, _, _ = Kalman_filter_HLW(Y, X, ParaSample, PriorLatent, SizeInformation)
                last_loglike = LogLike_old

        elif modelName == 'FC':
            LogLike_cand, _, _, _, _ = Kalman_filter_FC(Y, X, ParaSample_cand, PriorLatent, SizeInformation)

            if last_loglike is None:
                LogLike_old, _, _, _, _ = Kalman_filter_FC(Y, X, ParaSample, PriorLatent, SizeInformation)
                last_loglike = LogLike_old

        elif modelName == 'Covid':
            LogLike_cand, _, _, _, _ = Kalman_filter_Covid(Y, X, s_data, ParaSample_cand, PriorLatent, SizeInformation)

            if last_loglike is None:
                LogLike_old, _, _, _, _ = Kalman_filter_Covid(Y, X, s_data, ParaSample, PriorLatent, SizeInformation)
                last_loglike = LogLike_old

        else:
            raise ValueError("Invalid modelName. Use 'HLW', 'FC', or 'Covid'.")

        lnden_pr_new = pr_shape * np.log(pr_scale) - math.lgamma(pr_shape) - (pr_shape + 1) * np.log(pr_cand) - (pr_scale / pr_cand)
        lnden_pr_old = pr_shape * np.log(pr_scale) - math.lgamma(pr_shape) - (pr_shape + 1) * np.log(para_val) - (pr_scale / para_val)

        # 수락 확률 계산
        ln_accept = (LogLike_cand + lnden_pr_new) - (last_loglike + lnden_pr_old)
        if np.log(np.random.rand()) < ln_accept:
            para_out = pr_cand
            MHSample = np.append(MHSample[1:], para_out)
            accept_idx = 1
            current_loglike = LogLike_cand
        else:
            para_out = para_val
            MHSample = np.append(MHSample[1:], para_val)
            accept_idx = 0

    return para_out, MHSample, accept_idx, current_loglike

def AdaptiveMH_Normal(Y, X, fieldName, modelName, ProposalType_idx, MHSample, ParaSample,
                      PriorLatent, pr_mean, pr_var, pr_lndetvar, ProposalVar_0, SizeInformation, last_loglike=None, s_data=None):
    """
    정규 분포에 대한 Adaptive Metropolis-Hastings 알고리즘을 구현합니다.

    Parameters
    ----------
    Y : ndarray
        종속 변수 데이터.
    X : ndarray
        독립 변수 데이터.
    fieldName : str
        샘플링할 파라미터의 이름.
    modelName : str
        칼만 필터 모델 이름 ('HLW' 또는 'FC' 또는 'Covid').
    ProposalType_idx : int
        제안 분포의 타입 인덱스 (0 또는 1).
    MHSample : ndarray
        이전 MH 샘플들.
    ParaSample : dict
        현재 파라미터 샘플.
    PriorLatent : dict
        잠재 변수의 prior 정보.
    pr_mean : ndarray
        Prior 분포의 평균.
    pr_var : ndarray
        Prior 분포의 분산.
    pr_lndetvar : float
        Prior 분포 분산의 로그 행렬식.
    ProposalVar_0 : float
        제안 분포의 초기 분산.
    SizeInformation : dict
        데이터의 크기 정보.

    s_data : ndarray
        시간 가변적 변동성을 나타내는 데이터 | Default: None
        - Covid에서만 사용됨

    Returns
    -------
    para_out : ndarray
        새로운 파라미터 샘플.
    MHSample : ndarray
        업데이트된 MH 샘플들.
    accept_idx : int
        MH 알고리즘에서의 수락 여부 (1이면 수락, 0이면 거부).
    """
    # 마지막 파라미터 값
    para_val = MHSample[-1, :]

    dimension_proposal = para_val.shape[0]
    if ProposalType_idx == 0:
        cand_var = ProposalVar_0 * np.eye(dimension_proposal)
    else:
        delta_proposal = 2.38 / np.sqrt(dimension_proposal)
        cand_var = (delta_proposal ** 2) * (np.cov(MHSample, rowvar=False) + 0.00001 * np.eye(dimension_proposal))

    pr_cand = para_val + np.random.multivariate_normal(np.zeros(dimension_proposal), cand_var)

    if fieldName.lower() == 'yc':
        phi1_cand = pr_cand[0] - pr_cand[1]
        phi2_cand = pr_cand[1]
        beta_cand = pr_cand[2]
        ParaSample_cand = ParaSample.copy()
        ParaSample_cand['phi1_yc'] = phi1_cand
        ParaSample_cand['phi2_yc'] = phi2_cand
        ParaSample_cand['beta_yc'] = beta_cand
        persistency_cand = pr_cand[0]
    elif fieldName.lower() == 'zc':
        ParaSample_cand = ParaSample.copy()
        ParaSample_cand['delta_z'] = pr_cand[0]
        ParaSample_cand['phi_z'] = pr_cand[1]
        persistency_cand = pr_cand[1]
    elif fieldName.lower() == 'z':
        ParaSample_cand = ParaSample.copy()
        ParaSample_cand['delta_z'] = 0
        ParaSample_cand['phi_z'] = pr_cand[0]
        persistency_cand = pr_cand[0]
    elif fieldName.lower() == 'p':
        ParaSample_cand = ParaSample.copy()
        ParaSample_cand['phi_p'] = pr_cand[0]
        ParaSample_cand['beta_p'] = pr_cand[1]
        persistency_cand = pr_cand[0]
    elif fieldName.lower() == 'f':
        delta_cand = pr_cand[0]
        phi1_cand = pr_cand[1] - pr_cand[2]
        phi2_cand = pr_cand[2]
        beta_cand = pr_cand[3]
        ParaSample_cand = ParaSample.copy()
        ParaSample_cand['delta_f'] = delta_cand
        ParaSample_cand['phi1_f'] = phi1_cand
        ParaSample_cand['phi2_f'] = phi2_cand
        ParaSample_cand['beta_f'] = beta_cand
        persistency_cand = pr_cand[1]
    elif fieldName.lower() == 'yt':
        ParaSample_cand = ParaSample.copy()
        ParaSample_cand['gamma_yt'] = pr_cand
        persistency_cand = 0
    else:
        raise ValueError(f"Unknown fieldName: {fieldName}")

    current_loglike = last_loglike

    if np.abs(persistency_cand) > 0.98:
        para_out = para_val
        MHSample = np.vstack([MHSample[1:], para_val])
        accept_idx = 0
    else:
        if modelName.lower() == 'hlw':
            LogLike_cand, _, _, _, _ = Kalman_filter_HLW(Y, X, ParaSample_cand, PriorLatent, SizeInformation)

            if last_loglike is None:
                LogLike_old, _, _, _, _ = Kalman_filter_HLW(Y, X, ParaSample, PriorLatent, SizeInformation)
                last_loglike = LogLike_old

        elif modelName.lower() == 'fc':
            LogLike_cand, _, _, _, _ = Kalman_filter_FC(Y, X, ParaSample_cand, PriorLatent, SizeInformation)

            if last_loglike is None:
                LogLike_old, _, _, _, _ = Kalman_filter_FC(Y, X, ParaSample, PriorLatent, SizeInformation)
                last_loglike = LogLike_old

        elif modelName.lower() == 'covid':
            LogLike_cand, _, _, _, _ = Kalman_filter_Covid(Y, X, s_data, ParaSample_cand, PriorLatent, SizeInformation)

            if last_loglike is None:
                LogLike_old, _, _, _, _ = Kalman_filter_Covid(Y, X, s_data, ParaSample, PriorLatent, SizeInformation)
                last_loglike = LogLike_old
        else:
            raise ValueError(f"Invalid modelName: {modelName}")

        diff_cand = pr_cand - pr_mean
        diff_old = para_val - pr_mean

        # NOTE: yt 계산 과정에서 type error 가 나서 (matrix 가 아닌 float 로 input 이 들어오는 문제)
        # 아래의 if 문으로 분기를 만들어 따로 scalar 연산을 실행해 줌.
        if fieldName.lower() == 'yt':
            lnden_pr_new = -0.5 * dimension_proposal * np.log(2 * np.pi) - 0.5 * pr_lndetvar - 0.5 * diff_cand.T / (pr_var) * diff_cand
            lnden_pr_old = -0.5 * dimension_proposal * np.log(2 * np.pi) - 0.5 * pr_lndetvar - 0.5 * diff_old.T / (pr_var) * diff_old
        else:
            lnden_pr_new = -0.5 * dimension_proposal * np.log(2 * np.pi) - 0.5 * pr_lndetvar - 0.5 * diff_cand.T @ np.linalg.inv(pr_var) @ diff_cand
            lnden_pr_old = -0.5 * dimension_proposal * np.log(2 * np.pi) - 0.5 * pr_lndetvar - 0.5 * diff_old.T @ np.linalg.inv(pr_var) @ diff_old

        ln_accept = (LogLike_cand + lnden_pr_new) - (last_loglike + lnden_pr_old)
        if np.log(np.random.rand()) < ln_accept:
            para_out = pr_cand
            MHSample = np.vstack([MHSample[1:], para_out])
            accept_idx = 1
            current_loglike = LogLike_cand
        else:
            para_out = para_val
            MHSample = np.vstack([MHSample[1:], para_val])
            accept_idx = 0

    return para_out, MHSample, accept_idx, current_loglike


#============================================================
# HLW 
#============================================================

def GenerateData_HLW(X, ParaSample, PriorLatent, SizeInformation):
    """
    주어진 매개변수와 사전 정보를 기반으로 데이터를 생성

    Parameters
    ----------
    X : array-like
            설명 변수 데이터
    ParaSample : dict
                 매개변수 샘플을 포함하는 딕셔너리
    PriorLatent : dict
                  잠재 변수의 사전 정보를 포함하는 딕셔너리
    SizeInformation : dict
                      사이즈 정보를 포함하는 딕셔너리


    Returns
    -------
    Yg : ndarray
         생성된 데이터
    LVg : ndarray
          생성된 잠재 변수
    Eg : ndarray
         생성된 오류
    """
    
    # Define intercept matrix (M) for the transition equation
    M_yt = np.array([[0], [0], [0]])
    M_g = np.array([[0], [0], [0]])
    M_z = np.array([[ParaSample['delta_z']], [0], [0]])

    M = np.vstack([M_yt, M_g, M_z])  # Shape: 9x1

    # Define transition matrix (F)
    F_yt_yt = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0]])
    F_g_yt = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    F_z_yt = np.zeros((3, 3))

    F_g_g = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0]])
    F_yt_g = np.zeros((3, 3))
    F_z_g = np.zeros((3, 3))

    F_z_z = np.array([[ParaSample['phi_z'], 0, 0], [1, 0, 0], [0, 1, 0]])
    F_yt_z = np.zeros((3, 3))
    F_g_z = np.zeros((3, 3))

    F = np.block([
        [F_yt_yt, F_g_yt, F_z_yt],
        [F_yt_g, F_g_g, F_z_g],
        [F_yt_z, F_g_z, F_z_z]
    ])  # Shape: 9x9

    # Define selection matrix (G)
    G_yt = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    G_g = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
    G_z = np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]])

    G = np.vstack([G_yt, G_g, G_z])  # Shape: 9x3

    # Define covariance matrix (Q)
    Sigma_transition = np.diag([ParaSample['sig2_yt'], ParaSample['sig2_g'], ParaSample['sig2_z']])
    Q = G @ Sigma_transition @ G.T  # Shape: 9x9

    # Define relation matrix (H)
    # float() 자료형 명시
    H_yt_y = np.array([1, -1 * float(ParaSample['phi1_yc']), -1 * float(ParaSample['phi2_yc'])])
    H_g_y = np.array([0, -0.5 * float(ParaSample['beta_yc']), -0.5 * float(ParaSample['beta_yc'])])
    H_z_y = np.array([0, -0.5 * float(ParaSample['beta_yc']), -0.5 * float(ParaSample['beta_yc'])])


    H_yt_p = np.array([0, -float(ParaSample['beta_p']), 0])
    H_g_p = np.zeros(3)
    H_z_p = np.zeros(3)

    H = np.vstack([np.hstack([H_yt_y, H_g_y, H_z_y]),np.hstack([H_yt_p, H_g_p, H_z_p])])

    # Define relation matrix (L)
    L_y_y = np.array([ParaSample['phi1_yc'], ParaSample['phi2_yc']]).squeeze()
    L_r_y = np.array([0.5 * ParaSample['beta_yc'], 0.5 * ParaSample['beta_yc']]).squeeze()
    L_p_y = np.zeros(4).squeeze()

    L_y_p = np.array([float(ParaSample['beta_p']), 0]).squeeze()
    L_r_p = np.zeros(2).squeeze()
    L_p_p = np.array([ParaSample['phi_p'], (1 - ParaSample['phi_p']) / 3, (1 - ParaSample['phi_p']) / 3, (1 - ParaSample['phi_p']) / 3]).squeeze()


    L = np.vstack([np.hstack([L_y_y, L_r_y, L_p_y]),np.hstack([L_y_p, L_r_p, L_p_p])]) 

    # Define covariance matrix (R)
    Sigma_measurement = np.diag([ParaSample['sig2_yc'], ParaSample['sig2_p']])
    R = Sigma_measurement 

    # Generate errors
    Eg = np.zeros((SizeInformation['N_T'], 3)) # N_T X 3
    Ug = np.zeros((SizeInformation['N_T'], 2)) # N_T X 2

    for t in range(SizeInformation['N_T']):
        Eg[t, :] = (np.linalg.cholesky(Sigma_transition).T @ np.random.randn(3, 1)).T
        Ug[t, :] = (np.linalg.cholesky(R).T @ np.random.randn(2, 1)).T

    # Unconditional mean and variance for initial period
    b_ll = PriorLatent['mean']
    p_ll = PriorLatent['var']
    b0g = b_ll + np.linalg.cholesky(p_ll) @ np.random.randn(SizeInformation['N_Latent'])

    # Generate data
    LVg = np.zeros((SizeInformation['N_T'] + 1, SizeInformation['N_Latent']))
    Yg = np.zeros((SizeInformation['N_T'], 2))

    b_l = b0g.reshape(-1,1)
    LVg[0, :] = b0g
    
    for t in range(SizeInformation['N_T']):
        # Generate state
        tmp1 = M + F @ b_l + G @ (Eg[t, :].reshape(-1,1))
        LVg[t+1, :] = tmp1.flatten()
        # Generate observed variable
        Yg[t, :] = (L @ X[t, :].T + H @ LVg[t+1, :].T + Ug[t, :].T) 
        # Update state
        b_l = LVg[t+1, :].reshape(-1,1)
        
    return Yg, LVg, Eg


def GenerateLatentVariables_HLW(Y, X, ParaSample, PriorLatent, SizeInformation):
    """
    잠재 변수의 posterior 샘플을 생성하는 함수
    
    Parameters
    ----------
    Y : ndarray
        관측 데이터
    X : ndarray
        설명 변수 데이터
    ParaSample : dict or array-like
                  파라미터 샘플
    PriorLatent : dict or array-like
                   잠재 변수의 prior
    SizeInformation : dict or array-like
                       데이터의 크기 정보
    
    Returns
    -------
    bt_draw : ndarray
              잠재 변수의 posterior 샘플
    """
    Yg, LVg, Eg = GenerateData_HLW(X, ParaSample, PriorLatent, SizeInformation)
    
    lnpsty, b_tt_mat, p_tt_mat, b_tl_mat, p_tl_mat = Kalman_filter_HLW(Y, X, ParaSample, PriorLatent, SizeInformation)
    lnpsty, bg_tt_mat, pg_tt_mat, bg_tl_mat, pg_tl_mat = Kalman_filter_HLW(Yg, X, ParaSample, PriorLatent, SizeInformation)

    bt_mat, et_mat = Kalman_smoothing_HLW(b_tt_mat, p_tt_mat, b_tl_mat, p_tl_mat, ParaSample, SizeInformation)
    bgt_mat, egt_mat = Kalman_smoothing_HLW(bg_tt_mat, pg_tt_mat, bg_tl_mat, pg_tl_mat, ParaSample, SizeInformation)
    
    # 후행 샘플 복원
    bt_draw = bt_mat + (LVg - bgt_mat)
    
    return bt_draw


def Kalman_filter_HLW(Y, X, ParaSample, PriorLatent, SizeInformation):
    """
    Kalman filter를 사용하여 상태 공간 모델의 로그 우도와 잠재 변수의 추정치를 계산합니다.

    Parameters
    ----------
    Y : ndarray
        관측된 데이터 행렬 [output, inflation rate, real interest rate].
    X : ndarray
        독립 변수 데이터 행렬.
    ParaSample : dict
        모델 파라미터 값.
    PriorLatent : dict
        초기 잠재 변수의 사전 정보 (평균과 분산).
    SizeInformation : dict
        샘플 크기, 잠재 변수의 차원 등 크기 정보.

    Returns
    -------
    lnpsty : float
        로그 우도 값.
    b_tt_mat : ndarray
        잠재 변수의 필터링된 평균 벡터.
    p_tt_mat : ndarray
        잠재 변수의 필터링된 공분산 행렬.
    b_tl_mat : ndarray
        잠재 변수의 예측 평균 벡터.
    p_tl_mat : ndarray
        잠재 변수의 예측 공분산 행렬.
    """
    
    # Intercept 행렬 (M) 정의
    M_yt = np.array([[0], [0], [0]])
    M_g = np.array([[0], [0], [0]])
    M_z = np.array([[ParaSample['delta_z']], [0], [0]])
    M = np.vstack((M_yt, M_g, M_z))

    # Transition 행렬 (F) 정의
    F_yt_yt = np.array([[1, 0, 0],
                        [1, 0, 0],
                        [0, 1, 0]])  # output trend (yt) dynamics
    F_g_yt = np.array([[1, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0]])  # effect of output growth (g) on yt
    F_z_yt = np.zeros((3, 3))          # effect of z on yt

    F_yt_g = np.zeros((3, 3))          # effect of yt on g
    F_g_g = np.array([[1, 0, 0],
                      [1, 0, 0],
                      [0, 1, 0]])      # g dynamics
    F_z_g = np.zeros((3, 3))           # effect of z on g

    F_yt_z = np.zeros((3, 3))          # effect of yt on z
    F_g_z = np.zeros((3, 3))           # effect of g on z
    F_z_z = np.array([[ParaSample['phi_z'], 0, 0],
                      [1, 0, 0],
                      [0, 1, 0]])      # z dynamics

    F_row1 = np.hstack((F_yt_yt, F_g_yt, F_z_yt))
    F_row2 = np.hstack((F_yt_g, F_g_g, F_z_g))
    F_row3 = np.hstack((F_yt_z, F_g_z, F_z_z))
    F = np.vstack((F_row1, F_row2, F_row3))

    # Selection 행렬 (G) 정의
    G_yt = np.array([[1, 0, 0],
                     [0, 0, 0],
                     [0, 0, 0]])
    G_g = np.array([[0, 1, 0],
                    [0, 0, 0],
                    [0, 0, 0]])
    G_z = np.array([[0, 0, 1],
                    [0, 0, 0],
                    [0, 0, 0]])
    G = np.vstack((G_yt, G_g, G_z))

    # 상태 전이 방정식의 공분산 행렬 (Q) 정의
    Sigma_transition = np.diag([ParaSample['sig2_yt'], ParaSample['sig2_g'], ParaSample['sig2_z']])

    # 관측 방정식의 잠재 변수 관계 행렬 (H) 정의
    H_yt_y = np.array([1, -float(ParaSample['phi1_yc']), -float(ParaSample['phi2_yc'])])
    H_g_y = np.array([0, -0.5 * float(ParaSample['beta_yc']), -0.5 * float(ParaSample['beta_yc'])])
    H_z_y = np.array([0, -0.5 * float(ParaSample['beta_yc']), -0.5 * float(ParaSample['beta_yc'])])

    H_yt_p = np.array([0, -float(ParaSample['beta_p']), 0])
    H_g_p = np.array([0, 0, 0])
    H_z_p = np.array([0, 0, 0])

    H_row1 = np.concatenate((H_yt_y, H_g_y, H_z_y))
    H_row2 = np.concatenate((H_yt_p, H_g_p, H_z_p))
    H = np.vstack((H_row1, H_row2))

    # 관측 방정식의 관측 변수 관계 행렬 (L) 정의
    L_y_y = np.array([float(ParaSample['phi1_yc']), float(ParaSample['phi2_yc'])])
    L_r_y = np.array([0.5 * float(ParaSample['beta_yc']), 0.5 * float(ParaSample['beta_yc'])])
    L_p_y = np.zeros(4)

    L_y_p = np.array([float(ParaSample['beta_p']), 0])
    L_r_p = np.array([0, 0])
    L_p_p = np.array([float(ParaSample['phi_p'])] + [(1 - float(ParaSample['phi_p'])) / 3] * 3)

    L_row1 = np.concatenate((L_y_y, L_r_y, L_p_y))
    L_row2 = np.concatenate((L_y_p, L_r_p, L_p_p))
    L = np.vstack((L_row1, L_row2))

    # 관측 방정식의 공분산 행렬 (R) 정의
    Sigma_measurement = np.diag([float(ParaSample['sig2_yc']), float(ParaSample['sig2_p'])])

    # R과 Q 행렬 정의
    Q = G @ Sigma_transition @ G.T
    R = Sigma_measurement

    # 초기 상태의 평균과 분산
    b_ll = np.array(PriorLatent['mean']).reshape(-1, 1)
    p_ll = np.array(PriorLatent['var'])

    # 저장 공간 할당
    N_T = SizeInformation['N_T']
    N_Latent = SizeInformation['N_Latent']
    b_tt_mat = np.zeros((N_T + 1, N_Latent))
    p_tt_mat = np.zeros((N_T + 1, N_Latent ** 2))
    b_tl_mat = np.zeros((N_T + 1, N_Latent))
    p_tl_mat = np.zeros((N_T + 1, N_Latent ** 2))

    b_tt_mat[0, :] = b_ll.flatten()
    p_tt_mat[0, :] = p_ll.flatten()

    # Forward iteration
    lnpsty = 0
    for t in range(N_T):
        # 잠재 변수의 예측
        b_tl = M + F @ b_ll
        p_tl = F @ p_ll @ F.T + Q

        # 관측값의 예측
        X_t = X[t, :].reshape(-1, 1)
        Y_tl = L @ X_t + H @ b_tl
        Y_t = Y[t, :].reshape(-1, 1)
        e_tl = Y_t - Y_tl
        f_tl = H @ p_tl @ H.T + R

        # NOTE: 2x2 행렬 f_tl에 대해 직접 역행렬과 logdet 계산
        a, b = f_tl[0, 0], f_tl[0, 1]
        c, d = f_tl[1, 0], f_tl[1, 1]
        det = a * d - b * c
        if det <= 0:
            raise ValueError("공분산 행렬이 양의 정부호가 아닙니다.")
        logdet = np.log(det)
        inv_f_tl = np.array([[d, -b], [-c, a]]) / det

        # 우도 함수 평가 (위의 내용 바탕으로)
        solution = inv_f_tl @ e_tl
        lnlik_t = -0.5 * (np.log(2 * np.pi) * len(Y_t) + logdet + (e_tl.T @ solution))
        lnpsty += lnlik_t.item()

        # 상태 업데이트
        K = p_tl @ H.T @ inv_f_tl
        b_tt = b_tl + K @ e_tl
        p_tt = p_tl - K @ H @ p_tl

        ################

        # 다음 단계 준비
        b_ll = b_tt
        p_ll = p_tt

        # 결과 저장
        b_tl_mat[t + 1, :] = b_tl.ravel()
        p_tl_mat[t + 1, :] = p_tl.ravel()
        b_tt_mat[t + 1, :] = b_tt.ravel()
        p_tt_mat[t + 1, :] = p_tt.ravel()
    

    return lnpsty, b_tt_mat, p_tt_mat, b_tl_mat, p_tl_mat
 

def Kalman_smoothing_HLW(b_tt_mat, p_tt_mat, b_tl_mat, p_tl_mat, ParaSample, SizeInformation):
    """
    Kalman smoothing을 사용하여 잠재 변수의 평활화된 추정치를 계산하고 오차의 추정치를 생성합니다.

    Parameters
    ----------
    b_tt_mat : ndarray
        잠재 변수의 필터링된 평균 벡터.
    p_tt_mat : ndarray
        잠재 변수의 필터링된 공분산 행렬.
    b_tl_mat : ndarray
        잠재 변수의 예측 평균 벡터.
    p_tl_mat : ndarray
        잠재 변수의 예측 공분산 행렬.
    ParaSample : dict
        모델 파라미터 값.
    SizeInformation : dict
        샘플 크기, 잠재 변수의 차원 등 크기 정보.

    Returns
    -------
    bt_mat : ndarray
        잠재 변수의 평활화된 평균 벡터.
    et_mat : ndarray
        오차의 평활화된 평균 벡터.
    """
    # Intercept 행렬 (M) 정의
    M_yt = np.array([[0], [0], [0]])
    M_g = np.array([[0], [0], [0]])
    M_z = np.array([[ParaSample['delta_z']], [0], [0]])
    M = np.vstack((M_yt, M_g, M_z))

    # Transition 행렬 (F) 정의
    F_yt_yt = np.array([[1, 0, 0],
                        [1, 0, 0],
                        [0, 1, 0]])
    F_g_yt = np.array([[1, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0]])
    F_z_yt = np.zeros((3, 3))
    F_g_g = np.array([[1, 0, 0],
                      [1, 0, 0],
                      [0, 1, 0]])
    F_yt_g = np.zeros((3, 3))
    F_z_g = np.zeros((3, 3))
    F_z_z = np.array([[ParaSample['phi_z'], 0, 0],
                      [1, 0, 0],
                      [0, 1, 0]])
    F_yt_z = np.zeros((3, 3))
    F_g_z = np.zeros((3, 3))
    F_row1 = np.hstack((F_yt_yt, F_g_yt, F_z_yt))
    F_row2 = np.hstack((F_yt_g, F_g_g, F_z_g))
    F_row3 = np.hstack((F_yt_z, F_g_z, F_z_z))
    F = np.vstack((F_row1, F_row2, F_row3))

    # Selection 행렬 (G) 정의
    G_yt = np.array([[1, 0, 0],
                     [0, 0, 0],
                     [0, 0, 0]])
    G_g = np.array([[0, 1, 0],
                    [0, 0, 0],
                    [0, 0, 0]])
    G_z = np.array([[0, 0, 1],
                    [0, 0, 0],
                    [0, 0, 0]])
    G = np.vstack((G_yt, G_g, G_z))

    # 상태 전이 방정식의 공분산 행렬 (Q) 정의
    Sigma_transition = np.diag([float(ParaSample['sig2_yt']), float(ParaSample['sig2_g']), float(ParaSample['sig2_z'])])

    # 관측 방정식의 잠재 변수 관계 행렬 (H) 정의
    H_yt_y = np.array([1, -float(ParaSample['phi1_yc']), -float(ParaSample['phi2_yc'])])
    H_g_y = np.array([0, -0.5 * float(ParaSample['beta_yc']), -0.5 * float(ParaSample['beta_yc'])])
    H_z_y = np.array([0, -0.5 * float(ParaSample['beta_yc']), -0.5 * float(ParaSample['beta_yc'])])

    H_yt_p = np.array([0, -float(ParaSample['beta_p']), 0])
    H_g_p = np.array([0, 0, 0])
    H_z_p = np.array([0, 0, 0])

    H_row1 = np.concatenate((H_yt_y, H_g_y, H_z_y))
    H_row2 = np.concatenate((H_yt_p, H_g_p, H_z_p))
    H = np.vstack((H_row1, H_row2))

    # 관측 방정식의 관측 변수 관계 행렬 (L) 정의
    L_y_y = np.array([float(ParaSample['phi1_yc']), float(ParaSample['phi2_yc'])])
    L_r_y = np.array([0.5 * float(ParaSample['beta_yc']), 0.5 * float(ParaSample['beta_yc'])])
    L_p_y = np.zeros(4)

    L_y_p = np.array([float(ParaSample['beta_p']), 0])
    L_r_p = np.array([0, 0])
    L_p_p = np.array([float(ParaSample['phi_p'])] + [(1 - float(ParaSample['phi_p'])) / 3] * 3)

    L_row1 = np.concatenate((L_y_y, L_r_y, L_p_y))
    L_row2 = np.concatenate((L_y_p, L_r_p, L_p_p))
    L = np.vstack((L_row1, L_row2))

    # 관측 방정식의 공분산 행렬 (R) 정의
    Sigma_measurement = np.diag([float(ParaSample['sig2_yc']), float(ParaSample['sig2_p'])])

    # R과 Q 행렬 정의
    Q = G @ Sigma_transition @ G.T
    R = Sigma_measurement

    # Anderson and Moore (1979)에 따른 상태의 평활화
    N_T = SizeInformation['N_T']
    N_Latent = SizeInformation['N_Latent']
    z_size = N_Latent

    bt_mat = np.zeros((N_T + 1, z_size))

    bt_mat[N_T, :] = b_tt_mat[-1, :]

    for t in range(N_T - 1, -1, -1):
        # 필터링된 상태 및 공분산
        b_tt = b_tt_mat[t, :].reshape(-1, 1)
        p_tt = p_tt_mat[t, :].reshape(z_size, z_size)
        b_ft = b_tl_mat[t + 1, :].reshape(-1, 1)
        p_ft = p_tl_mat[t + 1, :].reshape(z_size, z_size)

        # 예측 오차
        b_ft_error = bt_mat[t + 1, :].reshape(-1, 1) - b_ft
        inv_p_ft = np.linalg.inv(p_ft)
        # NOTE: p_ft는 9x9라 위처럼 단순화는 어려움

        # 조건부 평균 업데이트
        gain = p_tt @ F.T @ inv_p_ft
        b_tt_s = b_tt + gain @ b_ft_error
        bt_mat[t, :] = b_tt_s.flatten()

    # 오차의 평균 벡터 생성
    et_mat = np.zeros((N_T, 3))
    indices = [0, 3, 6]  # MATLAB에서의 [1 4 7]에 해당하는 Python 인덱스
    for t in range(1, N_T + 1):
        # 평활화된 상태
        b_tt = bt_mat[t, :].reshape(-1, 1)
        b_tl = bt_mat[t - 1, :].reshape(-1, 1)

        e_tt = b_tt - M - F @ b_tl
        et_mat[t - 1, :] = e_tt[indices, 0]

    return bt_mat, et_mat
 

#============================================================
# HLW FC
#============================================================

def GenerateData_FC(X, ParaSample, PriorLatent, SizeInformation):
    """
    주어진 매개변수와 사전 정보를 기반으로 데이터를 생성
    
    Parameters
    ----------
    X : array-like
        설명 변수 데이터
    ParaSample : dict
                 매개변수 샘플을 포함하는 딕셔너리
    PriorLatent : dict
                  잠재 변수의 사전 정보를 포함하는 딕셔너리
    SizeInformation : dict
                      사이즈 정보를 포함하는 딕셔너리


    Returns
    -------
    Yg : array-like of shape (N_T, 3)
         생성된 데이터
    LVg : array-like of shape (N_T + 1, N_latetn)
          생성된 잠재 변수
    Eg : array-like (N_T, 3)
         생성된 오류
    """
    
    # Define intercept matrix (M) for the transition equation
    M_yt = np.array([[0], [0], [0]])
    M_g = np.array([[0], [0], [0]])
    M_z = np.array([[ParaSample['delta_z']], [0], [0]])
    M = np.vstack([M_yt, M_g, M_z])  

    # Define transition matrix (F)
    F_yt_yt = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0]])
    F_g_yt = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    F_z_yt = np.zeros((3, 3))
    
    F_g_g = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0]])
    F_yt_g = np.zeros((3, 3))
    F_z_g = np.zeros((3, 3))
    
    F_z_z = np.array([[ParaSample['phi_z'], 0, 0], [1, 0, 0], [0, 1, 0]])
    F_yt_z = np.zeros((3, 3))
    F_g_z = np.zeros((3, 3))
    F = np.block([
        [F_yt_yt, F_g_yt, F_z_yt],
        [F_yt_g, F_g_g, F_z_g],
        [F_yt_z, F_g_z, F_z_z]
    ])

    # Define selection matrix (G)
    G_yt = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    G_g = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
    G_z = np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]])
    G = np.vstack([G_yt, G_g, G_z])

    # Define covariance matrix (Q)
    Sigma_transition = np.diag([ParaSample['sig2_yt'], ParaSample['sig2_g'], ParaSample['sig2_z']])
    Q = G @ Sigma_transition @ G.T

    # Define intercept matrix (W) for the transition equation
    W = np.array([[0], [0], [float(ParaSample['delta_f'])]])

    # Define relation matrix (H)
    H_yt_y = np.array([1, -float(ParaSample['phi1_yc']), -float(ParaSample['phi2_yc'])])
    H_g_y = np.array([0, -0.5 * float(ParaSample['beta_yc']), -0.5 * float(ParaSample['beta_yc'])])
    H_z_y = np.array([0, -0.5 * float(ParaSample['beta_yc']), -0.5 * float(ParaSample['beta_yc'])])
    # test
    H_yt_p = np.array([0, -float(ParaSample['beta_p']), 0])
    H_g_p = np.zeros(3)
    H_z_p = np.zeros(3)
    
    H_yt_f = np.zeros(3)
    H_g_f = np.array([0, -0.5 * float(ParaSample['beta_f']), -0.5 * float(ParaSample['beta_f'])])
    H_z_f = np.array([0, -0.5 * float(ParaSample['beta_f']), -0.5 * float(ParaSample['beta_f'])])
    H = np.vstack([np.hstack([H_yt_y, H_g_y, H_z_y]), np.hstack([H_yt_p, H_g_p, H_z_p]), np.hstack([H_yt_f, H_g_f, H_z_f])])

    
    # Define relation matrix (L)
    L_y_y = np.array([float(ParaSample['phi1_yc']), float(ParaSample['phi2_yc'])])
    L_r_y = np.array([0.5 * float(ParaSample['beta_yc']), 0.5 * float(ParaSample['beta_yc'])])
    L_p_y = np.zeros(4)
    L_f_y = np.zeros(2)
    
    L_y_p = np.array([float(ParaSample['beta_p']), 0])
    L_r_p = np.zeros(2)
    L_p_p = np.array([float(ParaSample['phi_p']), (1 - float(ParaSample['phi_p'])) / 3, (1 - float(ParaSample['phi_p'])) / 3, (1 - float(ParaSample['phi_p'])) / 3])
    L_f_p = np.zeros(2)
    
    L_y_f = np.zeros(2)
    L_r_f = np.array([0.5 * float(ParaSample['beta_f']), 0.5 * float(ParaSample['beta_f'])])
    L_p_f = np.zeros(4)
    L_f_f = np.array([float(ParaSample['phi1_f']), float(ParaSample['phi2_f'])])
    L = np.block([
        [L_y_y, L_r_y, L_p_y, L_f_y],
        [L_y_p, L_r_p, L_p_p, L_f_p],
        [L_y_f, L_r_f, L_p_f, L_f_f]
    ])
    
    # Define covariance matrix (R)
    Sigma_measurement = np.diag([float(ParaSample['sig2_yc']), float(ParaSample['sig2_p']), float(ParaSample['sig2_f'])])
    R = Sigma_measurement

    # Generate errors
    Eg = np.zeros((SizeInformation['N_T'], 3)) # N_T X 3
    Ug = np.zeros((SizeInformation['N_T'], 3)) # N_T X 3

    for t in range(SizeInformation['N_T']):
        Eg[t, :] = (np.linalg.cholesky(Sigma_transition).T @ np.random.randn(3, 1)).T
        Ug[t, :] = (np.linalg.cholesky(R).T @ np.random.randn(3, 1)).T

    # Unconditional mean and variance for initial period
    b_ll = PriorLatent['mean']
    p_ll = PriorLatent['var']
    b0g = b_ll + np.linalg.cholesky(p_ll) @ np.random.randn(SizeInformation['N_Latent'])

    # Generate data
    LVg = np.zeros((SizeInformation['N_T'] + 1, SizeInformation['N_Latent'])) # 11X9
    Yg = np.zeros((SizeInformation['N_T'], 3)) # 10X3

    b_l = b0g.reshape(-1,1)
    LVg[0, :] = b0g


    for t in range(SizeInformation['N_T']):
        # Generate state
        tmp1 = M + F @ b_l + G @ (Eg[t, :].reshape(-1,1))
        LVg[t+1, :] = tmp1.flatten()
        # Generate observed variable
        Yg[t, :] = (W.reshape(-1) + L @ X[t, :].T + H @ LVg[t+1, :].T + Ug[t, :].T) 
        # Update state
        b_l = LVg[t+1, :].reshape(-1,1)
        
    return Yg, LVg, Eg
 

def GenerateLatentVariables_FC(Y, X, ParaSample, PriorLatent, SizeInformation):
    """
    잠재 변수의 posterior 샘플을 생성하는 함수
    
    Parameters
    ----------
    Y : ndarray
        관측 데이터
    X : ndarray
        설명 변수 데이터
    ParaSample : dict or array-like
                  파라미터 샘플
    PriorLatent : dict or array-like
                   잠재 변수의 prior
    SizeInformation : dict or array-like
                       데이터의 크기 정보
    
    Returns
    -------
    bt_draw : ndarray
              잠재 변수의 posterior 샘플
    """
    Yg, LVg, Eg = GenerateData_FC(X, ParaSample, PriorLatent, SizeInformation)
    
    lnpsty, b_tt_mat, p_tt_mat, b_tl_mat, p_tl_mat = Kalman_filter_FC(Y, X, ParaSample, PriorLatent, SizeInformation)
    lnpsty, bg_tt_mat, pg_tt_mat, bg_tl_mat, pg_tl_mat = Kalman_filter_FC(Yg, X, ParaSample, PriorLatent, SizeInformation)
    
    bt_mat, et_mat = Kalman_smoothing_FC(b_tt_mat, p_tt_mat, b_tl_mat, p_tl_mat, ParaSample, SizeInformation)
    bgt_mat, egt_mat = Kalman_smoothing_FC(bg_tt_mat, pg_tt_mat, bg_tl_mat, pg_tl_mat, ParaSample, SizeInformation)
    
    # 후행 샘플 복원
    bt_draw = bt_mat + (LVg - bgt_mat)
    
    return bt_draw
 

def Kalman_filter_FC(Y, X, ParaSample, PriorLatent, SizeInformation):
    """
    Kalman filter를 사용하여 상태 공간 모델의 로그 우도와 잠재 변수의 추정치를 계산합니다.

    Parameters
    ----------
    Y : ndarray
        관측된 데이터 행렬 [output, inflation rate, real interest rate].
    X : ndarray
        독립 변수 데이터 행렬.
    ParaSample : dict
        모델 파라미터 값.
    PriorLatent : dict
        초기 잠재 변수의 사전 정보 (평균과 분산).
    SizeInformation : dict
        샘플 크기, 잠재 변수의 차원 등 크기 정보.

    Returns
    -------
    lnpsty : float
        로그 우도 값.
    b_tt_mat : ndarray
        잠재 변수의 필터링된 평균 벡터.
    p_tt_mat : ndarray
        잠재 변수의 필터링된 공분산 행렬.
    b_tl_mat : ndarray
        잠재 변수의 예측 평균 벡터.
    p_tl_mat : ndarray
        잠재 변수의 예측 공분산 행렬.
    """
    
    # define intercept matrix (M) for the transition equation
    M_yt = np.array([[0], [0], [0]])
    M_g = np.array([[0], [0], [0]])
    M_z = np.array([[ParaSample['delta_z']], [0], [0]])
    M = np.vstack((M_yt, M_g, M_z))

    # define transition matrix (F) for the transition equation
    F_yt_yt = np.array([[1, 0, 0],
                        [1, 0, 0],
                        [0, 1, 0]])  # block for output trend (yt) dynamics
    F_g_yt = np.array([[1, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0]])  # effect of output growth (g) on yt
    F_z_yt = np.zeros((3, 3))  # effect of z on yt

    F_yt_g = np.zeros((3, 3))  # effect of yt on g
    F_g_g = np.array([[1, 0, 0],
                      [1, 0, 0],
                      [0, 1, 0]])  # g dynamics
    F_z_g = np.zeros((3, 3))  # effect of z on g

    F_yt_z = np.zeros((3, 3))  # effect of yt on z
    F_g_z = np.zeros((3, 3))  # effect of g on z
    F_z_z = np.array([[ParaSample['phi_z'], 0, 0],
                      [1, 0, 0],
                      [0, 1, 0]])  # z dynamics

    F_row1 = np.hstack((F_yt_yt, F_g_yt, F_z_yt))
    F_row2 = np.hstack((F_yt_g, F_g_g, F_z_g))
    F_row3 = np.hstack((F_yt_z, F_g_z, F_z_z))
    F = np.vstack((F_row1, F_row2, F_row3))

    # define selection matrix (G) for the transition equation
    G_yt = np.array([[1, 0, 0],
                     [0, 0, 0],
                     [0, 0, 0]])
    G_g = np.array([[0, 1, 0],
                    [0, 0, 0],
                    [0, 0, 0]])
    G_z = np.array([[0, 0, 1],
                    [0, 0, 0],
                    [0, 0, 0]])
    G = np.vstack((G_yt, G_g, G_z))

    # define the covariance matrix (Q) of the transition equation
    Sigma_transition = np.diag([ParaSample['sig2_yt'], ParaSample['sig2_g'], ParaSample['sig2_z']])

    # define intercept matrix (W) for the measurement equation
    W = np.array([[0],
                  [0],
                  [float(ParaSample['delta_f'])]])

    # define relation matrix (H) of latent variables for the measurement equation
    H_yt_y = np.array([1, -float(ParaSample['phi1_yc']), -float(ParaSample['phi2_yc'])])
    H_g_y = np.array([0, -0.5 * float(ParaSample['beta_yc']), -0.5 * float(ParaSample['beta_yc'])])
    H_z_y = np.array([0, -0.5 * float(ParaSample['beta_yc']), -0.5 * float(ParaSample['beta_yc'])])

    H_yt_p = np.array([0, -float(ParaSample['beta_p']), 0])
    H_g_p = np.array([0, 0, 0])
    H_z_p = np.array([0, 0, 0])

    H_yt_f = np.array([0, 0, 0])
    H_g_f = np.array([0, -0.5 * float(ParaSample['beta_f']), -0.5 * float(ParaSample['beta_f'])])
    H_z_f = np.array([0, -0.5 * float(ParaSample['beta_f']), -0.5 * float(ParaSample['beta_f'])])

    H_row1 = np.concatenate((H_yt_y, H_g_y, H_z_y))
    H_row2 = np.concatenate((H_yt_p, H_g_p, H_z_p))
    H_row3 = np.concatenate((H_yt_f, H_g_f, H_z_f))
    H = np.vstack((H_row1, H_row2, H_row3))

    # define relation matrix (L) of observed variables for the measurement equation
    L_y_y = np.array([ParaSample['phi1_yc'], ParaSample['phi2_yc']]).squeeze()
    L_r_y = np.array([0.5 * ParaSample['beta_yc'], 0.5 * ParaSample['beta_yc']]).squeeze()
    L_p_y = np.zeros(4)
    L_f_y = np.zeros(2)

    L_y_p = np.array([float(ParaSample['beta_p']), 0]).squeeze()
    L_r_p = np.zeros(2)
    L_p_p = np.array([float(ParaSample['phi_p'])] + [(1 - float(ParaSample['phi_p'])) / 3] * 3).squeeze()
    L_f_p = np.zeros(2)

    L_y_f = np.zeros(2)
    L_r_f = np.array([0.5 * float(ParaSample['beta_f']), 0.5 * float(ParaSample['beta_f'])]).squeeze()
    L_p_f = np.zeros(4)
    L_f_f = np.array([ParaSample['phi1_f'], ParaSample['phi2_f']]).squeeze()

    L_row1 = np.concatenate((L_y_y, L_r_y, L_p_y, L_f_y))
    L_row2 = np.concatenate((L_y_p, L_r_p, L_p_p, L_f_p))
    L_row3 = np.concatenate((L_y_f, L_r_f, L_p_f, L_f_f))
    L = np.vstack((L_row1, L_row2, L_row3))

    # define the covariance matrix (Sigma) of the measurement equation
    Sigma_measurement = np.diag([ParaSample['sig2_yc'], ParaSample['sig2_p'], ParaSample['sig2_f']])

    # define R and Q matrices
    Q = G @ Sigma_transition @ G.T
    R = Sigma_measurement

    # Mean and variance for initial period
    b_ll = np.array(PriorLatent['mean']).reshape(-1, 1)
    p_ll = np.array(PriorLatent['var'])

    # Saving space
    N_T = SizeInformation['N_T']
    N_Latent = SizeInformation['N_Latent']
    b_tt_mat = np.zeros((N_T + 1, N_Latent))
    p_tt_mat = np.zeros((N_T + 1, N_Latent ** 2))
    b_tl_mat = np.zeros((N_T + 1, N_Latent))
    p_tl_mat = np.zeros((N_T + 1, N_Latent ** 2))

    b_tt_mat[0, :] = b_ll.flatten()
    p_tt_mat[0, :] = p_ll.flatten()

    # Forward iteration
    lnpsty = 0
    for t in range(N_T):
        # prediction for latent variable
        b_tl = M + F @ b_ll
        p_tl = F @ p_ll @ F.T + Q

        # prediction for observed return
        X_t = X[t, :].reshape(-1, 1)
        Y_tl = W + L @ X_t + H @ b_tl
        Y_t = Y[t, :].reshape(-1, 1)
        e_tl = Y_t - Y_tl
        f_tl = H @ p_tl @ H.T + R

        # evaluate the likelihood function
        sign, logdet = np.linalg.slogdet(f_tl)
        if sign <= 0:
            raise ValueError("공분산 행렬이 양의 정부호가 아닙니다.")

        # f_tl의 역행렬을 한 번 계산
        inv_f_tl = np.linalg.inv(f_tl)
        solution = inv_f_tl @ e_tl

        lnlik_t = -0.5 * (np.log(2 * np.pi) * len(Y_t) + logdet + (e_tl.T @ solution))
        lnpsty += lnlik_t.item()

        # update recent observation
        K = p_tl @ H.T @ inv_f_tl
        b_tt = b_tl + K @ e_tl
        p_tt = p_tl - K @ H @ p_tl

        # update
        b_ll = b_tt
        p_ll = p_tt

        # store
        b_tl_mat[t + 1, :] = b_tl.ravel()
        p_tl_mat[t + 1, :] = p_tl.ravel()
        b_tt_mat[t + 1, :] = b_tt.ravel()
        p_tt_mat[t + 1, :] = p_tt.ravel()
    
    return lnpsty, b_tt_mat, p_tt_mat, b_tl_mat, p_tl_mat
 

def Kalman_smoothing_FC(b_tt_mat, p_tt_mat, b_tl_mat, p_tl_mat, ParaSample, SizeInformation):
    """
    Kalman smoothing을 사용하여 잠재 변수의 평활화된 추정치를 계산하고 오차의 추정치를 생성합니다.

    Parameters
    ----------
    b_tt_mat : ndarray
        잠재 변수의 필터링된 평균 벡터.
    p_tt_mat : ndarray
        잠재 변수의 필터링된 공분산 행렬.
    b_tl_mat : ndarray
        잠재 변수의 예측 평균 벡터.
    p_tl_mat : ndarray
        잠재 변수의 예측 공분산 행렬.
    ParaSample : dict
        모델 파라미터 값.
    SizeInformation : dict
        샘플 크기, 잠재 변수의 차원 등 크기 정보.

    Returns
    -------
    bt_mat : ndarray
        잠재 변수의 평활화된 평균 벡터.
    et_mat : ndarray
        오차의 평활화된 평균 벡터.
    """

    # define intercept matrix (M) for the transition equation
    M_yt = np.array([[0], [0], [0]])
    M_g = np.array([[0], [0], [0]])
    M_z = np.array([[ParaSample['delta_z']], [0], [0]])

    M = np.vstack((M_yt, M_g, M_z))

    # define transition matrix (F) for the transition equation
    F_yt_yt = np.array([[1, 0, 0],
                        [1, 0, 0],
                        [0, 1, 0]])  # output trend (yt) dynamics
    F_g_yt = np.array([[1, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0]])  # effect of output growth (g) on yt
    F_z_yt = np.zeros((3, 3))  # effect of z on yt

    F_yt_g = np.zeros((3, 3))  # effect of yt on g
    F_g_g = np.array([[1, 0, 0],
                      [1, 0, 0],
                      [0, 1, 0]])  # g dynamics
    F_z_g = np.zeros((3, 3))  # effect of z on g

    F_yt_z = np.zeros((3, 3))  # effect of yt on z
    F_g_z = np.zeros((3, 3))  # effect of g on z
    F_z_z = np.array([[ParaSample['phi_z'], 0, 0],
                      [1, 0, 0],
                      [0, 1, 0]])  # z dynamics

    F_row1 = np.hstack((F_yt_yt, F_g_yt, F_z_yt))
    F_row2 = np.hstack((F_yt_g, F_g_g, F_z_g))
    F_row3 = np.hstack((F_yt_z, F_g_z, F_z_z))
    F = np.vstack((F_row1, F_row2, F_row3))

    # define selection matrix (G) for the transition equation
    G_yt = np.array([[1, 0, 0],
                     [0, 0, 0],
                     [0, 0, 0]])
    G_g = np.array([[0, 1, 0],
                    [0, 0, 0],
                    [0, 0, 0]])
    G_z = np.array([[0, 0, 1],
                    [0, 0, 0],
                    [0, 0, 0]])

    G = np.vstack((G_yt, G_g, G_z))

    # define the covariance matrix (Q) of the transition equation
    Sigma_transition = np.diag([ParaSample['sig2_yt'], ParaSample['sig2_g'], ParaSample['sig2_z']])

    # define intercept matrix (W) for the measurement equation
    W = np.array([[0],
                  [0],
                  [float(ParaSample['delta_f'])]])

    # define relation matrix (H) of latent variables for the measurement equation
    H_yt_y = np.array([1, -float(ParaSample['phi1_yc']), -float(ParaSample['phi2_yc'])])
    H_g_y = np.array([0, -0.5 * float(ParaSample['beta_yc']), -0.5 * float(ParaSample['beta_yc'])])
    H_z_y = np.array([0, -0.5 * float(ParaSample['beta_yc']), -0.5 * float(ParaSample['beta_yc'])])

    H_yt_p = np.array([0, -float(ParaSample['beta_p']), 0])
    H_g_p = np.array([0, 0, 0])
    H_z_p = np.array([0, 0, 0])

    H_yt_f = np.array([0, 0, 0])
    H_g_f = np.array([0, -0.5 * float(ParaSample['beta_f']), -0.5 * float(ParaSample['beta_f'])])
    H_z_f = np.array([0, -0.5 * float(ParaSample['beta_f']), -0.5 * float(ParaSample['beta_f'])])

    H_row1 = np.concatenate((H_yt_y, H_g_y, H_z_y))
    H_row2 = np.concatenate((H_yt_p, H_g_p, H_z_p))
    H_row3 = np.concatenate((H_yt_f, H_g_f, H_z_f))
    H = np.vstack((H_row1, H_row2, H_row3))

    # define relation matrix (L) of observed variables for the measurement equation
    L_y_y = np.array([ParaSample['phi1_yc'], ParaSample['phi2_yc']]).squeeze()
    L_r_y = np.array([0.5 * ParaSample['beta_yc'], 0.5 * ParaSample['beta_yc']]).squeeze()
    L_p_y = np.zeros(4)
    L_f_y = np.zeros(2)

    L_y_p = np.array([float(ParaSample['beta_p']), 0]).squeeze()
    L_r_p = np.zeros(2)
    L_p_p = np.array([ParaSample['phi_p']] + [(1 - ParaSample['phi_p']) / 3] * 3).squeeze()
    L_f_p = np.zeros(2)

    L_y_f = np.zeros(2)
    L_r_f = np.array([0.5 * float(ParaSample['beta_f']), 0.5 * float(ParaSample['beta_f'])]).squeeze()
    L_p_f = np.zeros(4)
    L_f_f = np.array([ParaSample['phi1_f'], ParaSample['phi2_f']]).squeeze()

    L_row1 = np.concatenate((L_y_y, L_r_y, L_p_y, L_f_y))
    L_row2 = np.concatenate((L_y_p, L_r_p, L_p_p, L_f_p))
    L_row3 = np.concatenate((L_y_f, L_r_f, L_p_f, L_f_f))
    L = np.vstack((L_row1, L_row2, L_row3))

    # define the covariance matrix (Sigma) of the measurement equation
    Sigma_measurement = np.diag([float(ParaSample['sig2_yc']), float(ParaSample['sig2_p']), float(ParaSample['sig2_f'])])

    # define R and Q matrices
    Q = G @ Sigma_transition @ G.T
    R = Sigma_measurement

    # Smooth moments of state
    # Anderson and Moore (1979)

    # storage space
    z_size = SizeInformation['N_Latent']
    N_T = SizeInformation['N_T']
    bt_mat = np.zeros((N_T + 1, z_size))

    # Terminal period
    bt_mat[N_T, :] = b_tt_mat[-1, :]

    for t in range(N_T - 1, -1, -1):
        # Smoothed means and variances
        b_tt = b_tt_mat[t, :].reshape(-1, 1)
        p_tt = p_tt_mat[t, :].reshape(z_size, z_size)
        b_ft = b_tl_mat[t + 1, :].reshape(-1, 1)
        p_ft = p_tl_mat[t + 1, :].reshape(z_size, z_size)

        # prediction error
        b_ft_error = bt_mat[t + 1, :].reshape(-1, 1) - b_ft
        inv_p_ft = np.linalg.inv(p_ft)

        # update conditional mean
        b_tt_s = b_tt + p_tt @ F.T @ inv_p_ft @ b_ft_error
        bt_mat[t, :] = b_tt_s.flatten()

    # Generate means of disturbances
    et_mat = np.zeros((N_T, 3))
    indices = [0, 3, 6]  # MATLAB indices [1, 4, 7] correspond to Python indices [0, 3, 6]
    for t in range(1, N_T + 1):
        b_tt = bt_mat[t, :].reshape(-1, 1)
        b_tl = bt_mat[t - 1, :].reshape(-1, 1)

        e_tt = b_tt - M - F @ b_tl
        et_mat[t - 1, :] = e_tt[indices, 0]

    return bt_mat, et_mat



#============================================================
# HLW Covid
#============================================================


def GenerateData_Covid(X, s_data, ParaSample, PriorLatent, SizeInformation):
    """
    COVID 데이터를 기반으로 데이터를 생성하는 함수

    Parameters
    ----------
    X : ndarray
        설명 변수 데이터
    s_data : ndarray
        시간 가변적 변동성을 나타내는 데이터
    ParaSample : dict
                 매개변수 샘플
    PriorLatent : dict
                  잠재 변수의 사전 정보
    SizeInformation : dict
                      데이터 크기 정보

    Returns
    -------
    Yg : ndarray
         생성된 데이터
    LVg : ndarray
          생성된 잠재 변수
    Eg : ndarray
         생성된 상태 오류
    """
    # Define intercept matrix (M) for the transition equation
    M_yt = np.array([[0], [0], [0]])
    M_g = np.array([[0], [0], [0]])
    M_z = np.array([[ParaSample['delta_z']], [0], [0]])
    M = np.vstack([M_yt, M_g, M_z])  
    
    # Define transition matrix (F)
    F_yt_yt = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0]])
    F_g_yt = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    F_z_yt = np.zeros((3, 3))
    
    F_g_g = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0]])
    F_yt_g = np.zeros((3, 3))
    F_z_g = np.zeros((3, 3))
    
    F_z_z = np.array([[ParaSample['phi_z'], 0, 0], [1, 0, 0], [0, 1, 0]])
    F_yt_z = np.zeros((3, 3))
    F_g_z = np.zeros((3, 3))
    F = np.block([
        [F_yt_yt, F_g_yt, F_z_yt],
        [F_yt_g, F_g_g, F_z_g],
        [F_yt_z, F_g_z, F_z_z]
    ])

    # Define selection matrix (G)
    G_yt = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    G_g = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
    G_z = np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]])
    G = np.vstack([G_yt, G_g, G_z])

    # Define covariance matrix (Q)
    Sigma_transition = np.diag([ParaSample['sig2_yt'], ParaSample['sig2_g'], ParaSample['sig2_z']])
    Q = G @ Sigma_transition @ G.T

    # Define relation matrix (H)
    H_yt_y = np.array([1, -float(ParaSample['phi1_yc']), -float(ParaSample['phi2_yc'])])
    H_g_y = np.array([0, -0.5 * float(ParaSample['beta_yc']), -0.5 * float(ParaSample['beta_yc'])])
    H_z_y = np.array([0, -0.5 * float(ParaSample['beta_yc']), -0.5 * float(ParaSample['beta_yc'])])
    H_yt_p = np.array([0, -float(ParaSample['beta_p']), 0])
    H_g_p = np.zeros(3)
    H_z_p = np.zeros(3)
    H = np.vstack([np.hstack([H_yt_y, H_g_y, H_z_y]), np.hstack([H_yt_p, H_g_p, H_z_p])])


    # Define relation matrix (L)
    L_y_y = np.array([float(ParaSample['phi1_yc']), float(ParaSample['phi2_yc'])])
    L_r_y = np.array([0.5 * float(ParaSample['beta_yc']), 0.5 * float(ParaSample['beta_yc'])])
    L_p_y = np.zeros(4)
    L_d_y = np.array([ParaSample['gamma_yt'], -ParaSample['phi1_yc'] * ParaSample['gamma_yt'], -ParaSample['phi2_yc'] * ParaSample['gamma_yt']]).squeeze()
    
    L_y_p = np.array([float(ParaSample['beta_p']), 0])
    L_r_p = np.zeros(2)
    L_p_p = np.array([float(ParaSample['phi_p']), (1 - float(ParaSample['phi_p'])) / 3, (1 - float(ParaSample['phi_p'])) / 3, (1 - float(ParaSample['phi_p'])) / 3])
    L_d_p = np.array([0, -float(ParaSample['beta_p']) * float(ParaSample['gamma_yt']), 0]).squeeze()
  
    L = np.vstack([np.hstack([L_y_y, L_r_y, L_p_y, L_d_y]), np.hstack([L_y_p, L_r_p, L_p_p, L_d_p])])

    # Define covariance matrix (R)
    Sigma_measurement = np.diag([float(ParaSample['sig2_yc']), float(ParaSample['sig2_p'])])
    R = Sigma_measurement
    
    # Generate errors
    Eg = np.zeros((SizeInformation['N_T'], 3))
    Ug = np.zeros((SizeInformation['N_T'], 2))
    
    for t in range(SizeInformation['N_T']):
        # Time-varying volatility
        if s_data[t] == 1:
            R_t = R * float(ParaSample['kappa20'])
        elif s_data[t] == 2:
            R_t = R * float(ParaSample['kappa21'])
        elif s_data[t] == 3:
            R_t = R * float(ParaSample['kappa22'])
        else:
            R_t = R
        Eg[t, :] = (np.linalg.cholesky(Sigma_transition).T @ np.random.randn(3, 1)).T
        Ug[t, :] = (np.linalg.cholesky(R_t).T @ np.random.randn(2, 1)).T

    # Unconditional mean and variance for initial period
    b_ll = PriorLatent['mean']
    p_ll = PriorLatent['var']
    b0g = b_ll + np.linalg.cholesky(p_ll) @ np.random.randn(SizeInformation['N_Latent'])

    # Generate data
    LVg = np.zeros((SizeInformation['N_T'] + 1, SizeInformation['N_Latent']))
    Yg = np.zeros((SizeInformation['N_T'], 2))

    b_l = b0g.reshape(-1,1)
    LVg[0, :] = b0g

    for t in range(SizeInformation['N_T']):
        # Generate state
        tmp1 = M + F @ b_l + G @ (Eg[t, :].reshape(-1,1))
        LVg[t+1, :] = tmp1.flatten()
        # Generate observed variable
        Yg[t, :] = (L @ X[t, :].T + H @ LVg[t+1, :].T + Ug[t, :].T) 
        # Update state
        b_l = LVg[t+1, :].reshape(-1,1)

    return Yg, LVg, Eg


def GenerateLatentVariables_Covid(Y, X, s_data, ParaSample, PriorLatent, SizeInformation):
    """
    잠재 변수의 posterior 샘플을 생성하는 함수
    
    Parameters
    ----------
    Y : ndarray
        관측 데이터
    X : ndarray
        설명 변수 데이터
    s_data : ndarray
        시간 가변적 변동성을 나타내는 데이터
    ParaSample : dict 
                  파라미터 샘플
    PriorLatent : dict 
                   잠재 변수의 prior
    SizeInformation : dict
                       데이터의 크기 정보
    
    Returns
    -------
    bt_draw : ndarray
              잠재 변수의 posterior 샘플
    """
    Yg, LVg, Eg = GenerateData_Covid(X, s_data, ParaSample, PriorLatent, SizeInformation)
    
    lnpsty, b_tt_mat, p_tt_mat, b_tl_mat, p_tl_mat = Kalman_filter_Covid(Y, X, s_data, ParaSample, PriorLatent, SizeInformation)
    bt_mat, et_mat = Kalman_smoothing_Covid(b_tt_mat, p_tt_mat, b_tl_mat, p_tl_mat, ParaSample, SizeInformation)
    
    lnpsty, bg_tt_mat, pg_tt_mat, bg_tl_mat, pg_tl_mat = Kalman_filter_Covid(Yg, X, s_data, ParaSample, PriorLatent, SizeInformation)
    bgt_mat, egt_mat = Kalman_smoothing_Covid(bg_tt_mat, pg_tt_mat, bg_tl_mat, pg_tl_mat, ParaSample, SizeInformation)
    
    # 후행 샘플 복원
    bt_draw = bt_mat + (LVg - bgt_mat)

    # NOTE: bt_mat[0,0] value has different trend with matlab result
    
    return bt_draw


def Kalman_filter_Covid(Y, X, s_data, ParaSample, PriorLatent, SizeInformation):
    """
    Kalman filter를 사용하여 상태 공간 모델의 로그 우도와 잠재 변수의 추정치를 계산합니다.

    Parameters
    ----------
    Y : ndarray
        관측된 데이터 행렬 [output, inflation rate].
    X : ndarray
        독립 변수 데이터 행렬.
    s_data : ndarray
        시간 가변적 변동성을 나타내는 추가 데이터 (COVID 모델에서 사용).
    ParaSample : dict
        모델 파라미터 값.
    PriorLatent : dict
        초기 잠재 변수의 사전 정보 (평균과 분산).
    SizeInformation : dict
        샘플 크기, 잠재 변수의 차원 등 크기 정보.

    Returns
    -------
    lnpsty : float
        로그 우도 값.
    b_tt_mat : ndarray
        잠재 변수의 필터링된 평균 벡터.
    p_tt_mat : ndarray
        잠재 변수의 필터링된 공분산 행렬.
    b_tl_mat : ndarray
        잠재 변수의 예측 평균 벡터.
    p_tl_mat : ndarray
        잠재 변수의 예측 공분산 행렬.
    """
    # define intercept matrix (M) for the transition equation
    M_yt = np.array([[0], [0], [0]])
    M_g = np.array([[0], [0], [0]])
    M_z = np.array([[float(ParaSample['delta_z'])], [0], [0]])
    M = np.vstack((M_yt, M_g, M_z))

    # define transition matrix (F) for the transition equation
    F_yt_yt = np.array([[1, 0, 0],
                        [1, 0, 0],
                        [0, 1, 0]])  # block for output trend (yt) dynamics
    F_g_yt = np.array([[1, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0]])  # block for the effect of output growth (g) on yt
    F_z_yt = np.zeros((3, 3))  # block for the effect of z on yt

    F_yt_g = np.zeros((3, 3))  # block for the effect of yt on g
    F_g_g = np.array([[1, 0, 0],
                      [1, 0, 0],
                      [0, 1, 0]])  # block for g dynamics
    F_z_g = np.zeros((3, 3))  # block for the effect of z on g

    F_yt_z = np.zeros((3, 3))  # block for the effect of yt on z
    F_g_z = np.zeros((3, 3))  # block for the effect of g on z
    F_z_z = np.array([[float(ParaSample['phi_z']), 0, 0],
                      [1, 0, 0],
                      [0, 1, 0]])  # block for z dynamics

    F_row1 = np.hstack((F_yt_yt, F_g_yt, F_z_yt))
    F_row2 = np.hstack((F_yt_g, F_g_g, F_z_g))
    F_row3 = np.hstack((F_yt_z, F_g_z, F_z_z))
    F = np.vstack((F_row1, F_row2, F_row3))

    # define selection matrix (G) for the transition equation
    G_yt = np.array([[1, 0, 0],
                     [0, 0, 0],
                     [0, 0, 0]])
    G_g = np.array([[0, 1, 0],
                    [0, 0, 0],
                    [0, 0, 0]])
    G_z = np.array([[0, 0, 1],
                    [0, 0, 0],
                    [0, 0, 0]])
    G = np.vstack((G_yt, G_g, G_z))

    # define the covariance matrix (Q) of the transition equation
    Sigma_transition = np.diag([float(ParaSample['sig2_yt']), float(ParaSample['sig2_g']), float(ParaSample['sig2_z'])])

    # define relation matrix (H) of latent variables for the measurement equation
    H_yt_y = np.array([1, -float(ParaSample['phi1_yc']), -float(ParaSample['phi2_yc'])])
    H_g_y = np.array([0, -0.5 * float(ParaSample['beta_yc']), -0.5 * float(ParaSample['beta_yc'])])
    H_z_y = np.array([0, -0.5 * float(ParaSample['beta_yc']), -0.5 * float(ParaSample['beta_yc'])])

    H_yt_p = np.array([0, -float(ParaSample['beta_p']), 0])
    H_g_p = np.array([0, 0, 0])
    H_z_p = np.array([0, 0, 0])

    H_row1 = np.concatenate((H_yt_y, H_g_y, H_z_y))
    H_row2 = np.concatenate((H_yt_p, H_g_p, H_z_p))
    H = np.vstack((H_row1, H_row2))

    # define relation matrix (L) of observed variables for the measurement equation
    L_y_y = np.array([float(ParaSample['phi1_yc']), float(ParaSample['phi2_yc'])])
    L_r_y = np.array([0.5 * float(ParaSample['beta_yc']), 0.5 * float(ParaSample['beta_yc'])])
    L_p_y = np.zeros(4)
    L_d_y = np.array([float(ParaSample['gamma_yt']),
                      -float(ParaSample['phi1_yc']) * float(ParaSample['gamma_yt']),
                      -float(ParaSample['phi2_yc']) * float(ParaSample['gamma_yt'])])

    L_y_p = np.array([float(ParaSample['beta_p']), 0])
    L_r_p = np.zeros(2)
    phi_p = float(ParaSample['phi_p'])
    L_p_p = np.array([phi_p] + [(1 - phi_p) / 3] * 3)
    L_d_p = np.array([0, -float(ParaSample['beta_p']) * float(ParaSample['gamma_yt']), 0])

    L_row1 = np.concatenate((L_y_y, L_r_y, L_p_y, L_d_y))
    L_row2 = np.concatenate((L_y_p, L_r_p, L_p_p, L_d_p))
    L = np.vstack((L_row1, L_row2))

    # define the covariance matrix (Sigma) of the measurement equation
    Sigma_measurement = np.diag([float(ParaSample['sig2_yc']), float(ParaSample['sig2_p'])])

    # define R and Q matrices
    Q = G @ Sigma_transition @ G.T
    R = Sigma_measurement

    # Mean and variance for initial period
    b_ll = np.array(PriorLatent['mean']).reshape(-1, 1)
    p_ll = np.array(PriorLatent['var'])

    # Saving space
    N_T = SizeInformation['N_T']
    N_Latent = SizeInformation['N_Latent']
    b_tt_mat = np.zeros((N_T + 1, N_Latent))
    p_tt_mat = np.zeros((N_T + 1, N_Latent ** 2))
    b_tl_mat = np.zeros((N_T + 1, N_Latent))
    p_tl_mat = np.zeros((N_T + 1, N_Latent ** 2))

    b_tt_mat[0, :] = b_ll.flatten()
    p_tt_mat[0, :] = p_ll.flatten()

    # Forward iteration
    lnpsty = 0

    for t in range(N_T):
        # time-varying volatility
        if s_data[t] == 1:
            R_t = R * float(ParaSample['kappa20'])
        elif s_data[t] == 2:
            R_t = R * float(ParaSample['kappa21'])
        elif s_data[t] == 3:
            R_t = R * float(ParaSample['kappa22'])
        else:
            R_t = R

        # prediction for latent variable
        b_tl = M + F @ b_ll
        p_tl = F @ p_ll @ F.T + Q

        # prediction for observed variables
        X_t = X[t, :].reshape(-1, 1)
        Y_tl = L @ X_t + H @ b_tl
        Y_t = Y[t, :].reshape(-1, 1)
        e_tl = Y_t - Y_tl
        f_tl = H @ p_tl @ H.T + R_t

        # NOTE: 2x2 행렬 f_tl에 대해 직접 역행렬과 logdet 계산
        a, b = f_tl[0, 0], f_tl[0, 1]
        c, d = f_tl[1, 0], f_tl[1, 1]
        det = a * d - b * c
        if det <= 0:
            raise ValueError("공분산 행렬이 양의 정부호가 아닙니다.")
        logdet = np.log(det)
        inv_f_tl = np.array([[d, -b], [-c, a]]) / det

        # 우도 함수 평가 (위의 내용 바탕으로)
        solution = inv_f_tl @ e_tl
        lnlik_t = -0.5 * (np.log(2 * np.pi) * len(Y_t) + logdet + (e_tl.T @ solution))
        lnpsty += lnlik_t.item()

        # 상태 업데이트
        K = p_tl @ H.T @ inv_f_tl
        b_tt = b_tl + K @ e_tl
        p_tt = p_tl - K @ H @ p_tl

        # update
        b_ll = b_tt
        p_ll = p_tt

        # store
        b_tl_mat[t + 1, :] = b_tl.ravel()
        p_tl_mat[t + 1, :] = p_tl.ravel()
        b_tt_mat[t + 1, :] = b_tt.ravel()
        p_tt_mat[t + 1, :] = p_tt.ravel()

    return lnpsty, b_tt_mat, p_tt_mat, b_tl_mat, p_tl_mat


def Kalman_smoothing_Covid(b_tt_mat, p_tt_mat, b_tl_mat, p_tl_mat, ParaSample, SizeInformation):
    """
    Kalman smoothing을 사용하여 잠재 변수의 평활화된 추정치를 계산하고 오차의 추정치를 생성합니다.

    Parameters
    ----------
    b_tt_mat : ndarray
        잠재 변수의 필터링된 평균 벡터.
    p_tt_mat : ndarray
        잠재 변수의 필터링된 공분산 행렬.
    b_tl_mat : ndarray
        잠재 변수의 예측 평균 벡터.
    p_tl_mat : ndarray
        잠재 변수의 예측 공분산 행렬.
    ParaSample : dict
        모델 파라미터 값.
    SizeInformation : dict
        샘플 크기, 잠재 변수의 차원 등 크기 정보.

    Returns
    -------
    bt_mat : ndarray
        잠재 변수의 평활화된 평균 벡터.
    et_mat : ndarray
        오차의 평활화된 평균 벡터.
    """
    # define intercept matrix (M) for the transition equation
    M_yt = np.array([[0], [0], [0]])
    M_g = np.array([[0], [0], [0]])
    M_z = np.array([[float(ParaSample['delta_z'])], [0], [0]])

    M = np.vstack((M_yt, M_g, M_z))

    # define transition matrix (F) for the transition equation
    F_yt_yt = np.array([[1, 0, 0],
                        [1, 0, 0],
                        [0, 1, 0]])  # block for output trend (yt) dynamics
    F_g_yt = np.array([[1, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0]])  # block for the effect of output growth (g) on yt
    F_z_yt = np.zeros((3, 3))  # block for the effect of z on yt

    F_g_g = np.array([[1, 0, 0],
                      [1, 0, 0],
                      [0, 1, 0]])  # block for g dynamics
    F_yt_g = np.zeros((3, 3))  # block for the effect of yt on g
    F_z_g = np.zeros((3, 3))  # block for the effect of z on g

    F_z_z = np.array([[float(ParaSample['phi_z']), 0, 0],
                      [1, 0, 0],
                      [0, 1, 0]])  # block for z dynamics
    F_yt_z = np.zeros((3, 3))  # block for the effect of yt on z
    F_g_z = np.zeros((3, 3))  # block for the effect of g on z

    F_row1 = np.hstack((F_yt_yt, F_g_yt, F_z_yt))
    F_row2 = np.hstack((F_yt_g, F_g_g, F_z_g))
    F_row3 = np.hstack((F_yt_z, F_g_z, F_z_z))
    F = np.vstack((F_row1, F_row2, F_row3))

    # define selection matrix (G) for the transition equation
    G_yt = np.array([[1, 0, 0],
                     [0, 0, 0],
                     [0, 0, 0]])
    G_g = np.array([[0, 1, 0],
                    [0, 0, 0],
                    [0, 0, 0]])
    G_z = np.array([[0, 0, 1],
                    [0, 0, 0],
                    [0, 0, 0]])

    G = np.vstack((G_yt, G_g, G_z))

    # define the covariance matrix (Q) of the transition equation
    Sigma_transition = np.diag([float(ParaSample['sig2_yt']), float(ParaSample['sig2_g']), float(ParaSample['sig2_z'])])

    # define relation matrix (H) of latent variables for the measurement equation
    H_yt_y = np.array([1, -float(ParaSample['phi1_yc']), -float(ParaSample['phi2_yc'])])
    H_g_y = np.array([0, -0.5 * float(ParaSample['beta_yc']), -0.5 * float(ParaSample['beta_yc'])])
    H_z_y = np.array([0, -0.5 * float(ParaSample['beta_yc']), -0.5 * float(ParaSample['beta_yc'])])

    H_yt_p = np.array([0, -float(ParaSample['beta_p']), 0])
    H_g_p = np.array([0, 0, 0])
    H_z_p = np.array([0, 0, 0])

    H_row1 = np.concatenate((H_yt_y, H_g_y, H_z_y))
    H_row2 = np.concatenate((H_yt_p, H_g_p, H_z_p))
    H = np.vstack((H_row1, H_row2))

    # define relation matrix (L) of observed variables for the measurement equation
    L_y_y = np.array([float(ParaSample['phi1_yc']), float(ParaSample['phi2_yc'])])
    L_r_y = np.array([0.5 * float(ParaSample['beta_yc']), 0.5 * float(ParaSample['beta_yc'])])
    L_p_y = np.zeros(4)
    L_d_y = np.array([float(ParaSample['gamma_yt']),
                      -float(ParaSample['phi1_yc']) * float(ParaSample['gamma_yt']),
                      -float(ParaSample['phi2_yc']) * float(ParaSample['gamma_yt'])])

    L_y_p = np.array([float(ParaSample['beta_p']), 0])
    L_r_p = np.zeros(2)
    phi_p = float(ParaSample['phi_p'])
    L_p_p = np.array([phi_p] + [(1 - phi_p) / 3] * 3)
    L_d_p = np.array([0, -float(ParaSample['beta_p']) * float(ParaSample['gamma_yt']), 0])

    L_row1 = np.concatenate((L_y_y, L_r_y, L_p_y, L_d_y))
    L_row2 = np.concatenate((L_y_p, L_r_p, L_p_p, L_d_p))
    L = np.vstack((L_row1, L_row2))

    # define the covariance matrix (Sigma) of the measurement equation
    Sigma_measurement = np.diag([float(ParaSample['sig2_yc']), float(ParaSample['sig2_p'])])

    # define R and Q matrices
    Q = G @ Sigma_transition @ G.T
    R = Sigma_measurement

    # Smooth moments of state
    # Anderson and Moore (1979)

    # storage space
    z_size = SizeInformation['N_Latent']
    N_T = SizeInformation['N_T']
    bt_mat = np.zeros((N_T + 1, z_size))

    # Terminal period
    bt_mat[N_T, :] = b_tt_mat[-1, :]

    for t in range(N_T - 1, -1, -1):
        # Smoothed means and variances
        b_tt = b_tt_mat[t, :].reshape(-1, 1)
        p_tt = p_tt_mat[t, :].reshape(z_size, z_size)
        b_ft = b_tl_mat[t + 1, :].reshape(-1, 1)
        p_ft = p_tl_mat[t + 1, :].reshape(z_size, z_size)

        # prediction error
        b_ft_error = bt_mat[t + 1, :].reshape(-1, 1) - b_ft
        inv_p_ft = np.linalg.inv(p_ft)

        # update conditional mean
        b_tt_s = b_tt + p_tt @ F.T @ inv_p_ft @ b_ft_error
        bt_mat[t, :] = b_tt_s.flatten()

    # Generate means of disturbances
    et_mat = np.zeros((N_T, 3))
    indices = [0, 3, 6]  # MATLAB indices [1, 4, 7] correspond to Python indices [0, 3, 6]
    for t in range(1, N_T + 1):
        b_tt = bt_mat[t, :].reshape(-1, 1)
        b_tl = bt_mat[t - 1, :].reshape(-1, 1)

        e_tt = b_tt - M - F @ b_tl
        et_mat[t - 1, :] = e_tt[indices, 0]

    return bt_mat, et_mat


