import numpy as np
from tqdm import tqdm
import scipy as sc


# NOTE: statsmodels 안 쓸 수 있으면 좋을듯
from statsmodels.tools.numdiff import approx_hess

from .. import container_class



def translate_string_arg_to_int(value, category="trend"):
    """문자열로 받은 인자를 정수로 변환한다. (호환성 유지를 위한 함수)
    """
    # 숫자일 경우 그냥 반환
    if isinstance(value, int):
        return value
    
    # 문자열일 경우 변환
    if category == "trend":
        mapper = {"C": 1, "L": 2, "Q": 3}
    elif category == "hyperparameter_opt":
        mapper = {"pass": 0, "mlo": 1, "mcmc": 2}
    elif category == "optimization_method":
        mapper = {"scipy": 0, "rs": 1}

    try:
        lower_mapper = {k.lower(): v for k, v in mapper.items()}
        return lower_mapper[value.strip()]
    except KeyError:
        raise ValueError(f"> Invalid value '{value}' for argument '{category}'")



def log_pdf_normal(beta, Sigma, Y):
    """
    정규분포의 로그 확률밀도 함수를 계산합니다.

    Parameters
    ----------
    beta : float or numpy.ndarray
        정규분포의 평균 값입니다.
    Sigma : float or numpy.ndarray
        정규분포의 분산 또는 공분산 행렬입니다.
    Y : float or numpy.ndarray
        관측 데이터입니다.

    Returns
    -------
    SUM : float
        입력된 데이터 Y에 대한 정규분포의 로그 확률밀도 함수 값들의 합입니다.

    함수 동작
    --------
    이 함수는 다음의 과정을 수행합니다:
    1. 각 관측치에 대해 정규분포의 로그 확률밀도 함수를 계산합니다.
    2. 계산된 로그 확률밀도 값들을 합산하여 반환합니다.
    """
    pdfs = -0.5 * np.log(2 * np.pi * Sigma) - ((Y - beta) ** 2) @ ((2 * Sigma) ** (-1))
    SUM = np.sum(pdfs)
    return SUM


def log_pdf_gamma_1(k, theta, x):
    """
    감마분포의 로그 확률밀도 함수를 계산합니다.

    Parameters
    ----------
    k : float
        감마분포의 shape 파라미터입니다.
    theta : float
        감마분포의 scale 파라미터입니다.
    x : float or numpy.ndarray
        확률밀도를 계산할 값 또는 값들의 배열입니다.

    Returns
    -------
    log_pdf : float or numpy.ndarray
        입력된 x에 대한 감마분포의 로그 확률밀도 함수 값입니다.

    함수 동작
    --------
    이 함수는 다음의 과정을 수행합니다:
    1. 감마분포의 로그 확률밀도 함수를 계산합니다.
    2. 계산된 로그 확률밀도 값을 반환합니다.
    """
    log_pdf = -np.log(sc.special.gamma(k)) - k * np.log(theta) + (k - 1) * np.log(x) - (x / theta)
    return log_pdf


def Hessian_cal(objective_function, min_box, w_box, verbose=True):
    """
    목적 함수의 헤시안 행렬을 계산합니다.

    Parameters
    ----------
    objective_function : callable
        헤시안을 계산할 목적 함수입니다.
    min_box : numpy.ndarray
        각 반복에서의 최소 목적 함수 값들의 배열입니다.
    w_box : numpy.ndarray
        각 반복에서의 파라미터 벡터들을 저장한 배열입니다.
    verbose: bool, optional

    Returns
    -------
    hessian : numpy.ndarray
        계산된 헤시안 행렬입니다.

    함수 동작
    --------
    이 함수는 다음의 과정을 수행합니다:
    1. 목적 함수 값에 따라 파라미터 벡터들을 정렬합니다.
    2. 정렬된 파라미터 벡터들에 대해 순서대로 헤시안 행렬을 계산합니다.
    3. 양의 정부호 헤시안 행렬이 나올 때까지 시도합니다.
    4. 유효한 헤시안 행렬을 찾으면 반환합니다.

    주의사항
    --------
    헤시안 행렬이 양의 정부호가 아니거나 계산에 실패하면 다음 파라미터 벡터로 넘어갑니다.

    """
    Sort = np.argsort(min_box)
    Sort_w_box = w_box[:, Sort]
    for i in range(Sort_w_box.shape[1]):
        hessian = approx_hess(Sort_w_box[:, i], objective_function)
        hessian = (hessian.T + hessian) / 2
        try:
            np.linalg.cholesky(hessian)
            if verbose:
                print(f"> Useful hessian matrix is calculated using {i}th Minimum value")
            return hessian
        # except:
        except np.linalg.LinAlgError:
            continue
    print("> No positive definite hessian matrix found.")
    return None


def Forecast_function(Raw_Data, Parameters, Draw, verbose=True):
    """
    샘플링된 베타 계수와 시그마를 사용하여 예측을 수행하고 결과를 반환합니다.

    Parameters
    ----------
    Raw_Data : dict
        데이터 및 관련 정보를 담은 딕셔너리로, 다음의 키를 포함해야 합니다:
        - 'Y': numpy.ndarray
            종속 변수 행렬.
    Parameters : dict
        모델의 파라미터를 담은 딕셔너리로, 다음의 키를 포함해야 합니다:
        - 'nvar': int
            변수의 수.
        - 'Forecast_period': int
            예측할 기간의 수.
        - 'ndraws': int
            샘플링 횟수.
        - 'p': int
            VAR 모형의 시차 수.
        - 'Trend': int
            트렌드 유형 (1, 2 또는 3).
        - 'T': int
            데이터의 관측치 수.
        - 'pereb': float
            예측 구간의 신뢰수준 (0과 1 사이).
    Draw : dict
        샘플링 결과를 담은 딕셔너리로, 다음의 키를 포함해야 합니다:
        - 'Sigma': numpy.ndarray
            샘플링된 시그마 행렬들.
        - 'Bet': numpy.ndarray
            샘플링된 베타 행렬들.

    Returns
    -------
    Forecast_Results : dict
        예측 결과를 담은 딕셔너리로, 다음의 키를 포함합니다:
        - 'Total': numpy.ndarray
            전체 예측 결과 배열.
        - 'Mean': numpy.ndarray
            예측의 평균 값.
        - 'UP': numpy.ndarray
            예측 구간의 상한 값.
        - 'DOWN': numpy.ndarray
            예측 구간의 하한 값.

    함수 동작
    --------
    이 함수는 다음의 과정을 수행합니다:
    1. 각 샘플에 대해 예측 오차(U_forecast)를 생성합니다.
    2. 초기값을 설정하고, 각 기간에 대해 예측 값을 계산합니다.
    3. 예측 결과로부터 평균, 상한, 하한 값을 계산합니다.
    4. 예측 결과를 Forecast_Results 딕셔너리에 저장하고 반환합니다.

    """

    U_forecast = np.empty((Parameters.nvar, Parameters.Forecast_period, Parameters.ndraws))

    if verbose:
        iters = tqdm(range(Parameters.ndraws))
    else:
        iters = range(Parameters.ndraws)

    for d in iters:
        for i in range(Parameters.Forecast_period): 
            U_forecast[:, i, d] = np.random.multivariate_normal(np.zeros(Parameters.nvar), Draw.Sigma[:, :, d])
    
    Y_Forecast = np.empty((Parameters.nvar, Parameters.Forecast_period, Parameters.ndraws))
    X_Forecast_init = np.empty((Parameters.nvar, Parameters.p))
    
    if verbose:
        iters = tqdm(range(Parameters.ndraws))
    else:
        iters = range(Parameters.ndraws)

    for d in iters:
        for p in range(Parameters.p):
            X_Forecast_init[:, p] = Raw_Data.Y[-(p + 1), :]
        
        Bet = Draw.Bet[:, :, d]

        for j in range(Parameters.Forecast_period):
            if Parameters.Trend == 1:
                X_Forecast = np.insert(X_Forecast_init.reshape(-1, order='F'), 0, 1)
            elif Parameters.Trend == 2:
                X_Forecast = np.insert(X_Forecast_init.reshape(-1, order='F'), 0, Parameters.T + j)
                X_Forecast = np.insert(X_Forecast.reshape(-1, order='F'), 0, 1)
            elif Parameters.Trend == 3:
                X_Forecast = np.insert(X_Forecast_init.reshape(-1, order='F'), 0, (Parameters.T + j) ** 2)
                X_Forecast = np.insert(X_Forecast.reshape(-1, order='F'), 0, Parameters.T + j)
                X_Forecast = np.insert(X_Forecast.reshape(-1, order='F'), 0, 1)
            Y = Bet @ X_Forecast + U_forecast[:, j, d]
            Y_Forecast[:, j, d] = Y
            X_Forecast_init = np.hstack((Y.reshape(-1, 1), X_Forecast_init[:, 0:Parameters.p - 1]))

    Mean = np.empty((Parameters.nvar, Parameters.Forecast_period))
    UP = np.empty((Parameters.nvar, Parameters.Forecast_period))
    DOWN = np.empty((Parameters.nvar, Parameters.Forecast_period))
    for i in range(Parameters.nvar):
        for j in range(Parameters.Forecast_period):
            Mean[i, j] = np.mean(Y_Forecast[i, j, :])
            UP[i, j] = np.quantile(Y_Forecast[i, j, :], Parameters.pereb)
            DOWN[i, j] = np.quantile(Y_Forecast[i, j, :], 1 - Parameters.pereb)
            
    Forecast_Results = container_class.Container()
    Forecast_Results.Total = Y_Forecast
    Forecast_Results.Mean = Mean.T
    Forecast_Results.UP = UP.T
    Forecast_Results.DOWN = DOWN.T

    return Forecast_Results
