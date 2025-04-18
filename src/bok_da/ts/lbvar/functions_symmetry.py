import numpy as np
import scipy as sc
from tqdm import tqdm
from numpy.random import default_rng


from . import functions as func


def EXOv_total_maker(Parameters, Raw_Data):
    """
    주어진 파라미터에 따라 외생 변수 행렬(EXOv_total, EXOv, EXOv_AR)을 생성하고 Raw_Data 딕셔너리에 추가합니다.

    Parameters
    ----------
    Parameters : dict
        모델의 파라미터를 담은 딕셔너리로 다음과 같은 키를 포함합니다:
        - 'Trend': int
            트렌드 유형을 나타내는 정수 값. 1: 상수항만 포함, 2: 상수항과 선형 트렌드 포함, 3: 상수항, 선형 트렌드, 2차 트렌드 포함.
        - 'n': int
            전체 데이터의 관측치 수.
        - 'T': int
            사용할 데이터의 관측치 수.

    Raw_Data : dict
        데이터 및 관련 정보를 담은 딕셔너리.

    Returns
    -------
    Raw_Data : dict
        외생 변수 행렬이 추가된 Raw_Data 딕셔너리로, 다음의 키가 추가됩니다:
        - 'EXOv_total': numpy.ndarray
            전체 데이터에 대한 외생 변수 행렬.
        - 'EXOv': numpy.ndarray
            사용할 데이터에 대한 외생 변수 행렬.
        - 'EXOv_AR': numpy.ndarray
            AR 모형에 사용할 외생 변수 행렬.

    함수 동작
    --------
    이 함수는 'Trend' 파라미터에 따라 외생 변수 행렬을 생성합니다. 각 경우에 대해 다음과 같이 처리합니다:

    1. Trend == 1 (상수항만 포함)
        - EXOv_total: 전체 데이터 길이(n)에 대해 상수항(1)만으로 구성된 열 벡터 생성.
        - EXOv: 사용할 데이터 길이(T)에 대해 상수항(1)만으로 구성된 열 벡터 생성.
        - EXOv_AR: (n - 4) 길이에 대해 상수항(1)만으로 구성된 열 벡터 생성.

    2. Trend == 2 (상수항과 선형 트렌드 포함)
        - EXOv_total: 상수항과 시간 인덱스로 구성된 행렬 생성.
        - EXOv: 상수항과 시간 인덱스로 구성된 행렬 생성.
        - EXOv_AR: 상수항과 시간 인덱스로 구성된 행렬 생성.

    3. Trend == 3 (상수항, 선형 트렌드, 2차 트렌드 포함)
        - EXOv_total: 상수항, 시간 인덱스, 시간 인덱스의 제곱으로 구성된 행렬 생성.
        - EXOv: 상수항, 시간 인덱스, 시간 인덱스의 제곱으로 구성된 행렬 생성.
        - EXOv_AR: 상수항, 시간 인덱스, 시간 인덱스의 제곱으로 구성된 행렬 생성.

    생성된 외생 변수 행렬은 Raw_Data 딕셔너리에 각각 'EXOv_total', 'EXOv', 'EXOv_AR' 키로 추가됩니다.

    """
    # EXOV_total
    if Parameters.Trend == 1:
        Raw_Data.EXOv_total = np.ones((Parameters.n, 1))
    elif Parameters.Trend == 2:
        Raw_Data.EXOv_total = np.column_stack((
            np.ones((Parameters.n, 1)),
            np.arange(1, Parameters.n + 1).reshape(-1, 1)
        )) 
    elif Parameters.Trend == 3:
        Raw_Data.EXOv_total = np.column_stack((
            np.ones((Parameters.n, 1)),
            np.arange(1, Parameters.n + 1).reshape(-1, 1),
            np.power(np.arange(1, Parameters.n + 1), 2).reshape(-1, 1)
        ))
    
    # EXOV
    if Parameters.Trend == 1:
        Raw_Data.EXOv = np.ones((Parameters.T, 1))
    elif Parameters.Trend == 2:
        Raw_Data.EXOv = np.column_stack((
            np.ones((Parameters.T, 1)),
            np.arange(1, Parameters.T + 1).reshape(-1, 1)
        ))
    elif Parameters.Trend == 3:
        Raw_Data.EXOv = np.column_stack((
            np.ones((Parameters.T, 1)),
            np.arange(1, Parameters.T + 1).reshape(-1, 1),
            np.power(np.arange(1, Parameters.T + 1), 2).reshape(-1, 1)
        ))
    
    # EXOV_AR
    TT = Parameters.n - 4
    if Parameters.Trend == 1:
        Raw_Data.EXOv_AR = np.ones((TT, 1))
    elif Parameters.Trend == 2:
        Raw_Data.EXOv_AR = np.column_stack((
            np.ones((TT, 1)),
            np.arange(1, TT + 1).reshape(-1, 1)
        ))
    elif Parameters.Trend == 3:
        Raw_Data.EXOv_AR = np.column_stack((
            np.ones((TT, 1)),
            np.arange(1, TT + 1).reshape(-1, 1),
            np.power(np.arange(1, TT + 1), 2).reshape(-1, 1)
        ))

    return Raw_Data



def LBVAR_variable_maker(Raw_Data, Parameters):
    """
    LBVAR 모형을 위한 회귀 변수 행렬(Z)과 종속 변수 행렬(Y)을 생성하여 Raw_Data 딕셔너리에 추가합니다.

    Parameters
    ----------
    Raw_Data : dict
        데이터 및 관련 정보를 담은 딕셔너리로, 다음의 키를 포함해야 합니다:
        - 'Set': numpy.ndarray
            원본 데이터 행렬 (n x nvar).
        - 'EXOv': numpy.ndarray
            외생 변수 행렬 (T x c).

    Parameters : dict
        모델의 파라미터를 담은 딕셔너리로, 다음의 키를 포함해야 합니다:
        - 'T': int
            사용할 데이터의 관측치 수.
        - 'k': int
            회귀 변수의 총 개수 (nvar * p).
        - 'p': int
            VAR 모형의 시차(lag) 수.
        - 'nvar': int
            변수의 수 (VAR 모형의 종속 변수 수).

    Returns
    -------
    Raw_Data : dict
        회귀 변수 행렬(Z)과 종속 변수 행렬(Y)이 추가된 Raw_Data 딕셔너리로, 다음의 키가 추가됩니다:
        - 'Z': numpy.ndarray
            회귀 변수 행렬 (T x (k + c)).
        - 'Y': numpy.ndarray
            종속 변수 행렬 (T x nvar).

    함수 동작
    --------
    이 함수는 VAR 모형의 시차를 반영하여 회귀 변수 행렬 Z를 생성하고, 종속 변수 행렬 Y를 생성합니다.

    1. Z 생성:
        - 각 시차에 대해 데이터의 적절한 부분을 슬라이싱하여 회귀 변수 행렬을 구성합니다.
        - 생성된 Z에 외생 변수(EXOv)를 열로 추가합니다.

    2. Y 생성:
        - 종속 변수 행렬 Y는 원본 데이터에서 시차 p 이후의 부분을 사용합니다.

    생성된 Z와 Y는 Raw_Data 딕셔너리에 각각 'Z'와 'Y' 키로 추가됩니다.

    """
    Z = np.empty((Parameters.T, Parameters.k))
    for i in range(Parameters.p):
        start_idx = Parameters.p - (i + 1)
        end_idx = Raw_Data.Set.shape[0] - (i + 1)
        Z[:, i * Parameters.nvar:(i + 1) * Parameters.nvar] = Raw_Data.Set[start_idx:end_idx, :]
    Z = np.column_stack((Raw_Data.EXOv, Z))
    Y = Raw_Data.Set[Parameters.p:, :]
    Raw_Data.Z = Z
    Raw_Data.Y = Y

    return Raw_Data


def As_LBVAR_AR_sigma_maker(Raw_Data, Parameters, Prior):
    """
    각 변수에 대해 AR(4) 모형을 적합하여 추정된 잔차 분산(Sigma_hat)을 계산하고 Prior 딕셔너리에 추가합니다.

    Parameters
    ----------
    Raw_Data : dict
        데이터 및 관련 정보를 담은 딕셔너리로, 다음의 키를 포함해야 합니다:
        - 'Set': numpy.ndarray
            원본 데이터 행렬 (n x nvar).
        - 'EXOv_AR': numpy.ndarray
            AR 모형에 사용할 외생 변수 행렬.

    Parameters : dict
        모델의 파라미터를 담은 딕셔너리로, 다음의 키를 포함해야 합니다:
        - 'nvar': int
            변수의 수 (VAR 모형의 종속 변수 수).
        - 'n': int
            전체 데이터의 관측치 수.

    Prior : dict
        사전 분포(Prior)에 관련된 정보를 담은 딕셔너리.

    Returns
    -------
    Prior : dict
        Sigma_hat이 추가된 Prior 딕셔너리로, 다음의 키가 추가됩니다:
        - 'Sigma_hat': numpy.ndarray
            각 변수에 대한 추정된 잔차 분산 행렬 (nvar x nvar 대각 행렬).

    함수 동작
    --------
    이 함수는 각 변수에 대해 다음의 과정을 수행합니다:

    1. 각 변수 i에 대해:
        - 원본 데이터에서 변수 i의 데이터를 추출하여 Y_AR로 지정합니다.
        - AR(4) 모형을 적합하기 위해 Y_AR의 시차를 생성하여 설명 변수 행렬 X_AR를 구성합니다.
        - 외생 변수(EXOv_AR)를 X_AR에 추가합니다.
        - OLS 추정으로 베타 계수(Beta_AR)를 추정합니다.
        - 추정된 모형의 잔차의 제곱합을 계산하여 평균을 구하고, Sigma_hat_dum의 대각 원소로 저장합니다.

    2. 모든 변수에 대해 Sigma_hat_dum을 계산한 후, Prior 딕셔너리에 'Sigma_hat' 키로 추가합니다.

    """
    Sigma_hat_dum = np.zeros((Parameters.nvar, Parameters.nvar))

    for i in range(Parameters.nvar):
        Y_AR = Raw_Data.Set[:, i].reshape(-1, 1)
        X_AR = np.empty((Parameters.n - 4, 4))
        for j in range(4):
            X_AR[:, j] = Y_AR[4 - (j + 1):- (j + 1), 0]
        X_AR = np.column_stack((Raw_Data.EXOv_AR, X_AR))
        Y_AR = Y_AR[4:, :]
        Beta_AR = np.linalg.inv(X_AR.T @ X_AR) @ (X_AR.T @ Y_AR)
        residuals = Y_AR - X_AR @ Beta_AR
        Sigma_hat_dum[i, i] = np.mean(residuals ** 2)

    Prior.Sigma_hat = Sigma_hat_dum

    return Prior


def LBVAR_minnesota_Prior(Raw_Data, Parameters, Prior, hyperparameters):
    """
    Minnesota Prior를 생성하여 Prior 딕셔너리에 추가합니다.

    Parameters
    ----------
    Raw_Data : dict
        데이터 및 관련 정보를 담은 딕셔너리로, 다음의 키를 포함해야 합니다:
        - 'Set': numpy.ndarray
            원본 데이터 행렬.
        - 'EXOv_AR': numpy.ndarray
            AR 모형에 사용할 외생 변수 행렬.

    Parameters : dict
        모델의 파라미터를 담은 딕셔너리로, 다음과 같은 키를 포함해야 합니다:
        - 'nvar': int
            변수의 수 (VAR 모형의 종속 변수 수).
        - 'n': int
            전체 데이터의 관측치 수.
        - 'num_of_parameter': int
            파라미터의 총 개수 (nvar * p + c).
        - 'p': int
            VAR 모형의 시차(lag) 수.
        - 'c': int
            외생 변수의 수 (상수항 및 트렌드 등).
        - 'RV_list': list
            로그 또는 레벨 형태로 들어간 변수의 인덱스 리스트.
        - 'beta': float
            Minnesota Prior의 평균에 사용될 상수.

    Prior : dict
        사전 분포(Prior)에 관련된 정보를 담은 딕셔너리.

    hyperparameters : numpy.ndarray
        Minnesota Prior의 하이퍼파라미터를 담은 배열로, 다음과 같은 순서로 구성됩니다:
        - hyperparameters[0]: 람다1 (lambda1)
        - hyperparameters[1]: 람다2 (lambda2)
        - hyperparameters[2]: 람다3 (lambda3)
        - hyperparameters[3]: 람다4 (lambda4)
        - hyperparameters[4]: 람다5 (lambda5)

    Returns
    -------
    Prior : dict
        Minnesota Prior가 추가된 Prior 딕셔너리로, 다음의 키가 추가됩니다:
        - 'Sigma_hat': numpy.ndarray
            각 변수에 대한 추정된 잔차 분산 행렬 (대각 행렬 형태).
        - 'Minnesota_beta_mat': numpy.ndarray
            Minnesota Prior의 베타 평균 행렬.
        - 'Minnesota_V': numpy.ndarray
            Minnesota Prior의 베타 분산 행렬.

    함수 동작
    --------
    이 함수는 다음의 과정을 수행합니다:

    1. As_LBVAR_AR_sigma_maker 함수를 호출하여 각 변수에 대한 Sigma_hat을 계산하고 Prior에 추가합니다.

    2. Minnesota Prior의 베타 평균(Minnesota_beta)을 생성합니다.
        - 변수 중 RV_list에 포함된 변수들에 대해 베타 평균을 설정합니다.

    3. Minnesota Prior의 베타 분산(Minnesota_V)을 생성합니다.
        - 하이퍼파라미터와 Sigma_hat을 사용하여 베타 분산을 계산합니다.

    4. Prior 딕셔너리에 'Minnesota_beta_mat'과 'Minnesota_V'를 추가합니다.

    """
    Prior = As_LBVAR_AR_sigma_maker(Raw_Data, Parameters, Prior)

    Minnesota_beta = np.zeros((Parameters.num_of_parameter * Parameters.nvar, 1))
    Minnesota_beta_mat = np.zeros((Parameters.num_of_parameter, Parameters.nvar))

    for i in range(Parameters.nvar):
        vec_minnesota_beta_prior_dum = np.zeros((Parameters.num_of_parameter, 1))
        if i in Parameters.RV_list:
            vec_minnesota_beta_prior_dum[Parameters.c + i] = Parameters.beta
        Minnesota_beta[Parameters.num_of_parameter * i: Parameters.num_of_parameter * (i + 1)] = vec_minnesota_beta_prior_dum

    for i in range(Parameters.nvar):
        Minnesota_beta_mat[:, i] = Minnesota_beta[Parameters.num_of_parameter * i: Parameters.num_of_parameter * (i + 1)].reshape(-1)

    # Minnesota Beta Variance Setting
    for t in range(Parameters.p):
        temp_V = np.empty((Parameters.nvar, 1))
        for n in range(Parameters.nvar):
            temp_V[n] = hyperparameters[0] / (((t + 1) ** hyperparameters[1]) * Prior.Sigma_hat[n, n])
        if t == 0:
            Determin = hyperparameters[2] * np.ones((Parameters.c, 1))
            Minnesota_V = np.vstack((Determin, temp_V))
        else:
            Minnesota_V = np.vstack((Minnesota_V, temp_V))

    Minnesota_V = np.diag(Minnesota_V.reshape(-1))

    Prior.Sigma_hat = hyperparameters[4] * Prior.Sigma_hat
    Prior.Minnesota_beta_mat = Minnesota_beta_mat
    Prior.Minnesota_V = Minnesota_V

    return Prior



def Posterior_Parm_Maker(Raw_Data, Parameters, hyperparameters, Prior):
    """
    Posterior 분포의 파라미터를 계산하여 Prior 딕셔너리에 추가합니다.

    Parameters
    ----------
    Raw_Data : dict
        데이터 및 관련 정보를 담은 딕셔너리로, 다음의 키를 포함해야 합니다:
        - 'Z': numpy.ndarray
            회귀 변수 행렬.
        - 'Y': numpy.ndarray
            종속 변수 행렬.

    Parameters : dict
        모델의 파라미터를 담은 딕셔너리로, 다음의 키를 포함해야 합니다:
        - 'nvar': int
            변수의 수 (VAR 모형의 종속 변수 수).
        - 'T': int
            사용할 데이터의 관측치 수.
        - 'k': int
            회귀 변수의 총 개수 (nvar * p).
        - 'c': int
            외생 변수의 수 (상수항 및 트렌드 등).

    hyperparameters : numpy.ndarray
        Minnesota Prior의 하이퍼파라미터를 담은 배열로, 다음과 같은 순서로 구성됩니다:
        - hyperparameters[3]: nu0의 보정값.

    Prior : dict
        사전 분포(Prior)에 관련된 정보를 담은 딕셔너리로, 다음의 키를 포함해야 합니다:
        - 'Minnesota_beta_mat': numpy.ndarray
            Minnesota Prior의 베타 평균 행렬.
        - 'Minnesota_V': numpy.ndarray
            Minnesota Prior의 베타 분산 행렬.
        - 'Sigma_hat': numpy.ndarray
            각 변수에 대한 추정된 잔차 분산 행렬 (대각 행렬 형태).

    Returns
    -------
    Prior : dict
        Posterior 분포의 파라미터가 추가된 Prior 딕셔너리로, 다음의 키가 추가됩니다:
        - 'nu0': float
            Posterior 분포의 자유도.
        - 'K_A': numpy.ndarray
            Posterior 분포의 공분산 행렬.
        - 'A_hat': numpy.ndarray
            Posterior 분포의 평균 행렬.
        - 'S_hat': numpy.ndarray
            Posterior 분포의 스케일 행렬.

    함수 동작
    --------
    이 함수는 다음의 과정을 수행합니다:

    1. nu0 계산:
        - nu0 = hyperparameters[3] + nvar + 1

    2. K_A 계산:
        - K_A = inv(V_A) + Z.T @ Z
        - 여기서 V_A는 Minnesota Prior의 베타 분산 행렬(Minnesota_V)의 역행렬입니다.

    3. A_hat 계산:
        - A_hat = inv(K_A) @ (inv(V_A) @ Minnesota_beta_mat + Z.T @ Y)

    4. S_hat 계산:
        - S_hat = Sigma_hat + Minnesota_beta_mat.T @ inv(V_A) @ Minnesota_beta_mat + Y.T @ Y - A_hat.T @ K_A @ A_hat

    5. 계산된 값들을 Prior 딕셔너리에 추가합니다.

    """
    nu0 = hyperparameters[3] + Parameters.nvar + 1

    inv_V_A = np.linalg.inv(Prior.Minnesota_V)
    K_A = inv_V_A + Raw_Data.Z.T @ Raw_Data.Z
    K_A = (K_A.T + K_A) / 2  # 대칭성 보장
    inv_K_A = np.linalg.inv(K_A)
    A_hat = inv_K_A @ (inv_V_A @ Prior.Minnesota_beta_mat + Raw_Data.Z.T @ Raw_Data.Y)
    S_hat = Prior.Sigma_hat + (Prior.Minnesota_beta_mat.T @ inv_V_A @ Prior.Minnesota_beta_mat) \
            + (Raw_Data.Y.T @ Raw_Data.Y) - (A_hat.T @ K_A @ A_hat)
    S_hat = (S_hat + S_hat.T) / 2  # 대칭성 보장

    Prior.nu0 = nu0
    Prior.K_A = K_A
    Prior.A_hat = A_hat
    Prior.S_hat = S_hat

    return Prior


def Posterior_Draw(Raw_Data, Parameters, Prior, Draw, verbose=True):
    """
    Posterior 분포로부터 파라미터를 샘플링하여 Draw 딕셔너리에 추가합니다.

    Parameters
    ----------
    Raw_Data : dict
        데이터 및 관련 정보를 담은 딕셔너리로, 다음의 키를 포함해야 합니다:
        - 'Y': numpy.ndarray
            종속 변수 행렬.
        - 'Z': numpy.ndarray
            회귀 변수 행렬.

    Parameters : dict
        모델의 파라미터를 담은 딕셔너리로, 다음의 키를 포함해야 합니다:
        - 'nvar': int
            변수의 수 (VAR 모형의 종속 변수 수).
        - 'T': int
            사용할 데이터의 관측치 수.
        - 'k': int
            회귀 변수의 총 개수 (nvar * p).
        - 'c': int
            외생 변수의 수 (상수항 및 트렌드 등).
        - 'ndraws': int
            샘플링 횟수.

    Prior : dict
        Posterior 분포의 파라미터를 담은 딕셔너리로, 다음의 키를 포함해야 합니다:
        - 'nu0': float
            Posterior 분포의 자유도.
        - 'S_hat': numpy.ndarray
            Posterior 분포의 스케일 행렬.
        - 'K_A': numpy.ndarray
            Posterior 분포의 공분산 행렬.
        - 'A_hat': numpy.ndarray
            Posterior 분포의 평균 행렬.

    Draw : dict
        샘플링 결과를 담을 딕셔너리.

    verbose : bool, optional
        과정 출력 여부

    Returns
    -------
    Draw : dict
        샘플링 결과가 추가된 Draw 딕셔너리로, 다음의 키가 추가됩니다:
        - 'Sigma': numpy.ndarray
            샘플링된 Sigma 행렬들.
        - 'Bet_Prime': numpy.ndarray
            샘플링된 베타 프라임 행렬들.
        - 'Bet': numpy.ndarray
            샘플링된 베타 행렬들.
        - 'U_B': numpy.ndarray
            샘플링된 잔차 행렬들.

    함수 동작
    --------
    이 함수는 다음의 과정을 반복하여 ndraws 만큼 샘플링을 수행합니다:

    1. Sigma 샘플링:
        - Sigma ~ Inverse Wishart(df=nu0+T, scale=S_hat)

    2. 베타 프라임(Bet_Prime) 샘플링:
        - Bet_Prime = A_hat + inv(Cholesky(K_A).T) @ N(0,1) @ Cholesky(Sigma).T

    3. 베타(Bet) 및 잔차(U_B) 계산:
        - Bet = Bet_Prime.T
        - U_B = Y - Z @ Bet_Prime

    샘플링된 결과는 Draw 딕셔너리에 저장됩니다.

    """
    Sigma = np.empty((Parameters.nvar, Parameters.nvar, Parameters.ndraws))
    Bet_Prime = np.empty((Parameters.k + Parameters.c, Parameters.nvar, Parameters.ndraws))
    Bet = np.empty((Parameters.nvar, Parameters.k + Parameters.c, Parameters.ndraws))
    U_B = np.empty((Raw_Data.Y.shape[0], Raw_Data.Y.shape[1], Parameters.ndraws))


    if verbose:
        iter = tqdm(range(Parameters.ndraws), desc='Posterior Draw')
    else:
        iter = range(Parameters.ndraws)

    for i in iter:
        Sigma_candi = sc.stats.invwishart.rvs(df=Prior.nu0 + Parameters.T, scale=Prior.S_hat)
        Sigma[:, :, i] = (Sigma_candi + Sigma_candi.T) / 2
        Chol_Sigma = np.linalg.cholesky(Sigma[:, :, i])
        Chol_K_A = np.linalg.cholesky(Prior.K_A)
        Bet_Prime[:, :, i] = Prior.A_hat + np.linalg.solve(Chol_K_A.T, np.random.normal(0, 1, (Parameters.k + Parameters.c, Parameters.nvar))) @ Chol_Sigma.T
        Bet[:, :, i] = Bet_Prime[:, :, i].T
        U_B[:, :, i] = Raw_Data.Y - Raw_Data.Z @ Bet_Prime[:, :, i]

    Draw.Sigma = Sigma
    Draw.Bet_Prime = Bet_Prime
    Draw.Bet = Bet
    Draw.U_B = U_B

    return Draw


def Recursive_IRF(Parameters, Draw, verbose=False):
    """
    샘플링된 베타 계수를 사용하여 Recursive Impulse Response Function(IRF)을 계산하고 Draw 딕셔너리에 추가합니다.

    Parameters
    ----------
    Parameters : dict
        모델의 파라미터를 담은 딕셔너리로, 다음의 키를 포함해야 합니다:
        - 'nvar': int
            변수의 수 (VAR 모형의 종속 변수 수).
        - 'p': int
            VAR 모형의 시차(lag) 수.
        - 'nstep': int
            IRF를 계산할 시차의 수.
        - 'ndraws': int
            샘플링 횟수.

    Draw : dict
        샘플링 결과를 담은 딕셔너리로, 다음의 키를 포함해야 합니다:
        - 'Bet': numpy.ndarray
            샘플링된 베타 행렬들.
    
    verbose : bool, optional
        과정 출력 여부

    Returns
    -------
    Draw : dict
        IRF 결과가 추가된 Draw 딕셔너리로, 다음의 키가 추가됩니다:
        - 'Imp': numpy.ndarray
            IRF 결과 행렬들.

    함수 동작
    --------
    이 함수는 다음의 과정을 수행합니다:

    1. 각 샘플링된 베타 계수에 대해:
        - VAR 모형의 Companion 행렬(BB)을 구성합니다.
        - 각 시차에 대해 IRF를 계산하여 Imp 행렬에 저장합니다.

    2. 계산된 IRF는 Draw 딕셔너리의 'Imp' 키에 추가됩니다.

    """
    wimpu = np.empty((Draw.Bet[:, Parameters.c:, 0].shape[1], Draw.Bet[:, Parameters.c:, 0].shape[1], Parameters.nstep))
    Imp = np.empty((Parameters.nvar, Parameters.nvar, Parameters.nstep, Parameters.ndraws))
    Cat_matrix = np.column_stack((np.eye(Parameters.nvar * Parameters.p - Parameters.nvar), np.zeros((Parameters.nvar * Parameters.p - Parameters.nvar, Parameters.nvar))))

    if verbose:
        iter = tqdm(range(Parameters.ndraws))
    else:
        iter = range(Parameters.ndraws)

    for d in iter:
        BB = Draw.Bet[:, Parameters.c:, d]
        BB = np.vstack((BB, Cat_matrix))

        for j in range(Parameters.nstep):
            if j == 0:
                wimpu[:, :, j] = np.eye(Draw.Bet[:, Parameters.c:, 0].shape[1])
            elif j > 0:
                wimpu[:, :, j] = np.linalg.matrix_power(BB, j)
            Imp[:, :, j, d] = wimpu[:Parameters.nvar, :Parameters.nvar, j]

    Draw['Imp'] = Imp

    return Draw


def log_pdf_iv_gamma(alpha, beta, x):
    """
    역감마분포의 로그 확률밀도 함수를 계산합니다.

    Parameters
    ----------
    alpha : float
        역감마분포의 shape 파라미터입니다.
    beta : float
        역감마분포의 scale 파라미터입니다.
    x : float or numpy.ndarray
        확률밀도를 계산할 값 또는 값들의 배열입니다.

    Returns
    -------
    log_pdf : float or numpy.ndarray
        입력된 x에 대한 역감마분포의 로그 확률밀도 함수 값입니다.

    함수 동작
    --------
    이 함수는 다음의 과정을 수행합니다:
    1. 역감마분포의 로그 확률밀도 함수를 계산합니다.
    2. 계산된 로그 확률밀도 값을 반환합니다.
    """
    log_pdf = alpha * np.log(beta) - np.log(sc.special.gamma(alpha)) - (alpha + 1) * np.log(x) - beta / x
    return log_pdf


def log_MV_gamma_function(n, x):
    """
    다변량 감마 함수의 로그 값을 계산합니다.

    Parameters
    ----------
    n : int
        변수의 차원 수입니다.
    x : float
        다변량 감마 함수의 입력 값입니다.

    Returns
    -------
    log_MV : float
        다변량 감마 함수의 로그 값입니다.

    함수 동작
    --------
    이 함수는 다음의 과정을 수행합니다:
    1. 다변량 감마 함수의 로그 값을 계산하기 위해 각 차원에 대해 loggamma 함수를 적용합니다.
    2. 계산된 값들을 합산하고 상수를 더하여 최종 로그 값을 반환합니다.

    참고
    ----
    다변량 감마 함수는 감마 함수를 일반화한 것으로, 확률론과 통계학에서 다변량 정규분포 등의 계산에 사용됩니다.
    """
    log_MV_i = np.empty(n)
    for j in range(n):
        log_MV_i[j] = sc.special.loggamma(x + (1 - (j + 1)) / 2)
    log_MV = (n * (n - 1) / 4) * np.log(np.pi) + np.sum(log_MV_i)
    return log_MV


def Log_Kernel(Raw_Data, Parameters, Prior, candi_hyperparameters):
    """
    주어진 하이퍼파라미터 후보(candi_hyperparameters)에 대해 로그 커널 값을 계산합니다.

    Parameters
    ----------
    Raw_Data : dict
        데이터 및 관련 정보를 담은 딕셔너리.
    Parameters : dict
        모델의 파라미터를 담은 딕셔너리.
    Prior : dict
        사전 분포(Prior)에 관련된 정보를 담은 딕셔너리.
    candi_hyperparameters : numpy.ndarray
        후보 하이퍼파라미터 배열로, 다음과 같은 순서로 구성됩니다:
        - candi_hyperparameters[0]: 람다1 (lambda1)
        - candi_hyperparameters[1]: 람다2 (lambda2)
        - candi_hyperparameters[2]: 람다3 (lambda3)
        - candi_hyperparameters[3]: 람다4 (lambda4)
        - candi_hyperparameters[4]: 람다5 (lambda5)

    Returns
    -------
    ln_Kernel : float
        주어진 하이퍼파라미터에 대한 로그 커널 값.
    Raw_Data : dict
        업데이트된 Raw_Data 딕셔너리.
    Prior : dict
        업데이트된 Prior 딕셔너리.

    함수 동작
    --------
    이 함수는 다음의 과정을 수행합니다:
    1. 회귀 변수 행렬(Z)과 종속 변수 행렬(Y)을 생성하고 Raw_Data를 업데이트합니다.
    2. Minnesota Prior를 생성하고 Prior를 업데이트합니다.
    3. Posterior 분포의 파라미터를 계산하고 Prior를 업데이트합니다.
    4. 주어진 하이퍼파라미터에 대한 로그 사전 분포(pi_p)를 계산합니다.
    5. 모델의 주변우도(log_p)를 계산합니다.
    6. 로그 커널 값(ln_Kernel)을 계산하여 반환합니다.

    """
    Raw_Data = LBVAR_variable_maker(Raw_Data, Parameters)
    Prior = LBVAR_minnesota_Prior(Raw_Data, Parameters, Prior, candi_hyperparameters)
    Prior = Posterior_Parm_Maker(Raw_Data, Parameters, candi_hyperparameters, Prior)

    pi_p = func.log_pdf_gamma_1(1.1711, 0.2922, candi_hyperparameters[0]) \
            + func.log_pdf_gamma_1(1.618, 0.618, candi_hyperparameters[1]) \
                + func.log_pdf_gamma_1(101, 1, candi_hyperparameters[2]) \
                    + func.log_pdf_gamma_1(1.618, 0.618, candi_hyperparameters[3]) \
                        + func.log_pdf_gamma_1(1.618, 0.618, candi_hyperparameters[4])

    x1 = (Prior.nu0 + Parameters.T) / 2
    x2 = Prior.nu0 / 2
    # log_p is Marginal Likelihood of model
    log_p = (-Parameters.nvar * Parameters.T / 2) * np.log(np.pi) \
            - (Parameters.nvar / 2) * np.log(np.linalg.det(Prior.Minnesota_V)) \
                - (Parameters.nvar / 2) * np.log(np.linalg.det(Prior.K_A)) \
                    + log_MV_gamma_function(Parameters.nvar, x1) \
                        + (Prior.nu0 / 2) * np.log(np.linalg.det(Prior.Sigma_hat)) \
                            - log_MV_gamma_function(Parameters.nvar, x2) \
                                - ((Prior.nu0 + Parameters.T) / 2) * np.log(np.linalg.det(Prior.S_hat))
    ln_Kernel = log_p + pi_p
    return ln_Kernel, Raw_Data, Prior


def Marginal_Likelihood(Raw_Data, Parameters, Prior, candi_hyperparameters):
    """
    주어진 하이퍼파라미터 후보(candi_hyperparameters)에 대해 모델의 주변우도를 계산합니다.

    Parameters
    ----------
    Raw_Data : dict
        데이터 및 관련 정보를 담은 딕셔너리.
    Parameters : dict
        모델의 파라미터를 담은 딕셔너리.
    Prior : dict
        사전 분포(Prior)에 관련된 정보를 담은 딕셔너리.
    candi_hyperparameters : numpy.ndarray
        후보 하이퍼파라미터 배열.

    Returns
    -------
    negative_log_p : float
        주어진 하이퍼파라미터에 대한 주변우도의 음수 값.
    Raw_Data : dict
        업데이트된 Raw_Data 딕셔너리.
    Prior : dict
        업데이트된 Prior 딕셔너리.

    함수 동작
    --------
    이 함수는 다음의 과정을 수행합니다:
    1. 회귀 변수 행렬(Z)과 종속 변수 행렬(Y)을 생성하고 Raw_Data를 업데이트합니다.
    2. Minnesota Prior를 생성하고 Prior를 업데이트합니다.
    3. Posterior 분포의 파라미터를 계산하고 Prior를 업데이트합니다.
    4. 모델의 주변우도(log_p)를 계산합니다.
    5. 주변우도의 음수 값을 반환합니다.

    """
    Raw_Data = LBVAR_variable_maker(Raw_Data, Parameters)
    Prior = LBVAR_minnesota_Prior(Raw_Data, Parameters, Prior, candi_hyperparameters)
    Prior = Posterior_Parm_Maker(Raw_Data, Parameters, candi_hyperparameters, Prior)

    x1 = (Prior.nu0 + Parameters.T) / 2
    x2 = Prior.nu0 / 2
    # log_p is Marginal Likelihood of model
    log_p = (-Parameters.nvar * Parameters.T / 2) * np.log(np.pi) \
            - (Parameters.nvar / 2) * np.log(np.linalg.det(Prior.Minnesota_V)) \
                - (Parameters.nvar / 2) * np.log(np.linalg.det(Prior.K_A)) \
                    + log_MV_gamma_function(Parameters.nvar, x1) \
                        + (Prior.nu0 / 2) * np.log(np.linalg.det(Prior.Sigma_hat)) \
                            - log_MV_gamma_function(Parameters.nvar, x2) \
                                - ((Prior.nu0 + Parameters.T) / 2) * np.log(np.linalg.det(Prior.S_hat))
    negative_log_p = -log_p
    return negative_log_p, Raw_Data, Prior


def Optimization(Raw_Data, Parameters, Prior, candi_hyperparameters):
    """
    주어진 하이퍼파라미터 후보에 대해 최적화 목적 함수를 계산합니다.

    Parameters
    ----------
    Raw_Data : dict
        데이터 및 관련 정보를 담은 딕셔너리.
    Parameters : dict
        모델의 파라미터를 담은 딕셔너리.
    Prior : dict
        사전 분포(Prior)에 관련된 정보를 담은 딕셔너리.
    candi_hyperparameters : numpy.ndarray
        후보 하이퍼파라미터 배열.

    Returns
    -------
    negative_log_p_pi_p : float
        -(log_p + pi_p)의 값으로, 최적화에 사용되는 목적 함수 값.
    Raw_Data : dict
        업데이트된 Raw_Data 딕셔너리.
    Prior : dict
        업데이트된 Prior 딕셔너리.

    함수 동작
    --------
    이 함수는 다음의 과정을 수행합니다:
    1. 회귀 변수 행렬(Z)과 종속 변수 행렬(Y)을 생성하고 Raw_Data를 업데이트합니다.
    2. Minnesota Prior를 생성하고 Prior를 업데이트합니다.
    3. Posterior 분포의 파라미터를 계산하고 Prior를 업데이트합니다.
    4. 모델의 주변우도(log_p)를 계산합니다.
    5. 하이퍼파라미터에 대한 로그 사전 분포(pi_p)를 계산합니다.
    6. -(log_p + pi_p)를 계산하여 반환합니다.

    """
    Raw_Data = LBVAR_variable_maker(Raw_Data, Parameters)
    Prior = LBVAR_minnesota_Prior(Raw_Data, Parameters, Prior, candi_hyperparameters)
    Prior = Posterior_Parm_Maker(Raw_Data, Parameters, candi_hyperparameters, Prior)

    x1 = (Prior.nu0 + Parameters.T) / 2
    x2 = Prior.nu0 / 2
    # log_p is Marginal Likelihood of model
    log_p = (-Parameters.nvar * Parameters.T / 2) * np.log(np.pi) \
            - (Parameters.nvar / 2) * np.log(np.linalg.det(Prior.Minnesota_V)) \
                - (Parameters.nvar / 2) * np.log(np.linalg.det(Prior.K_A)) \
                    + log_MV_gamma_function(Parameters.nvar, x1) \
                        + (Prior.nu0 / 2) * np.log(np.linalg.det(Prior.Sigma_hat)) \
                            - log_MV_gamma_function(Parameters.nvar, x2) \
                                - ((Prior.nu0 + Parameters.T) / 2) * np.log(np.linalg.det(Prior.S_hat))

    pi_p = func.log_pdf_gamma_1(1.1711, 0.2922, candi_hyperparameters[0]) \
            + func.log_pdf_gamma_1(1.618, 0.618, candi_hyperparameters[1]) \
                + func.log_pdf_gamma_1(101, 1, candi_hyperparameters[2]) \
                    + func.log_pdf_gamma_1(1.618, 0.618, candi_hyperparameters[3]) \
                        + func.log_pdf_gamma_1(1.618, 0.618, candi_hyperparameters[4])

    negative_log_p_pi_p = -(log_p + pi_p)
    return negative_log_p_pi_p, Raw_Data, Prior



def Hyperparameter_MCMC(Raw_Data, Parameters, Prior, hyperparameters, hessian, verbose=True, n_draws=10000, n_burnin=1000):
    """
    하이퍼파라미터를 MCMC를 통해 최적화합니다.

    Parameters
    ----------
    Raw_Data : dict
        데이터 및 관련 정보를 담은 딕셔너리.
    Parameters : dict
        모델의 파라미터를 담은 딕셔너리.
    Prior : dict
        사전 분포(Prior)에 관련된 정보를 담은 딕셔너리.
    hyperparameters : numpy.ndarray
        초기 하이퍼파라미터 배열.
    hessian : numpy.ndarray
        목적 함수의 헤시안 행렬.
    verbose: bool
        출력 여부


    Returns
    -------
    Medians : numpy.ndarray
        MCMC 결과로부터 계산된 하이퍼파라미터의 중앙값 배열.
    Raw_Data : dict
        업데이트된 Raw_Data 딕셔너리.
    Prior : dict
        업데이트된 Prior 딕셔너리.

    함수 동작
    --------
    이 함수는 다음의 과정을 수행합니다:
    1. 헤시안 행렬의 역행렬을 계산하여 분산 행렬로 사용합니다.
    2. MCMC 알고리즘을 통해 하이퍼파라미터를 샘플링합니다.
    3. 각 반복에서 후보 하이퍼파라미터를 생성하고, 로그 커널 값을 계산합니다.
    4. 수락 확률에 따라 하이퍼파라미터를 업데이트합니다.
    5. MCMC 결과로부터 중앙값을 계산하여 최적의 하이퍼파라미터로 반환합니다.

    주의사항
    --------
    헤시안 행렬이 양의 정부호가 아니거나 NaN 값을 포함하는 경우, 초기 하이퍼파라미터를 그대로 반환합니다.

    """
    if np.isnan(hessian).any():
        print("> NaN detected in Hessian matrix. Exiting function.")
        print(f"> Hyperparameter is selected as {np.round(hyperparameters,3)}")
        return hyperparameters, Raw_Data, Prior

    try:
        VARIANCE = np.linalg.inv(hessian)
    except:
        print("> Hessian matrix isn't Positive Definite. Exiting function.")
        print(f"> Hyperparameter is selected as {np.round(hyperparameters,3)}")
        return hyperparameters, Raw_Data, Prior

    MCMC_hyperparameters = np.empty((5, n_draws))
    MCMC_accept = np.zeros(n_draws)
    candi_hyperparameters_ = hyperparameters.copy()

    VARIANCE = (VARIANCE.T + VARIANCE) / 2

    if verbose:
        print("> Hyperparameter Optimization Start")
    try:
        if verbose:
            iters = tqdm(range(n_draws), desc="Hyperparameter MCMC")
        else:
            iters = range(n_draws)

        for i in iters:
            Draw = False

            while not Draw:
                candi_hyperparameters = np.random.multivariate_normal(candi_hyperparameters_, VARIANCE)
                if np.all(candi_hyperparameters > 0):
                    Draw = True

            ln_Kernel_new, Raw_Data_new, Prior_new = Log_Kernel(Raw_Data.copy(), Parameters, Prior.copy(), candi_hyperparameters)
            ln_Kernel_old, _, _ = Log_Kernel(Raw_Data.copy(), Parameters, Prior.copy(), candi_hyperparameters_)

            alpha = min(ln_Kernel_new - ln_Kernel_old, 0)
            U = np.log(np.random.rand())
            if U < alpha:
                MCMC_hyperparameters[:, i] = candi_hyperparameters
                candi_hyperparameters_ = candi_hyperparameters
                Raw_Data = Raw_Data_new
                Prior = Prior_new
                MCMC_accept[i] = 1
            else:
                MCMC_hyperparameters[:, i] = candi_hyperparameters_

        MCMC_results = MCMC_hyperparameters[:, n_burnin+1:]
        Medians = np.median(MCMC_results, axis=1)

        if verbose:
            print("> Hyperparameter Optimization Clear")
            print(f"> Opt Hyperparameters are {np.round(Medians,3)}")
        return Medians, Raw_Data, Prior
    except Exception as e:
        candi_hyperparameters_ = hyperparameters
        print("> MCMC isn't clearly working")
        print(f"> Hyperparameter is selected as {np.round(hyperparameters,3)}")
        return candi_hyperparameters_, Raw_Data, Prior


def Random_Search(object_function, init, K, P, alpha, verbose=True):
    """
    랜덤 서치 알고리즘을 사용하여 최적화를 수행합니다.

    Parameters
    ----------
    object_function : callable
        최적화할 목적 함수입니다. 입력으로 파라미터 벡터를 받고 스칼라 값을 반환해야 합니다.
    init : numpy.ndarray
        초기 파라미터 벡터입니다.
    K : int
        최대 반복 횟수입니다.
    P : int
        각 반복에서 생성할 후보의 수입니다.
    alpha : float or str
        스텝 크기입니다. 'diminishing'을 입력하면 스텝 크기가 1/k로 감소합니다.
    verbose: bool
        중간 출력

    Returns
    -------
    Optimizers : numpy.ndarray
        최적화된 파라미터 벡터입니다.
    w_box : numpy.ndarray
        각 반복에서의 파라미터 벡터들을 저장한 배열입니다.
    min_box : numpy.ndarray
        각 반복에서의 최소 목적 함수 값을 저장한 배열입니다.

    함수 동작
    --------
    이 함수는 다음의 과정을 수행합니다:
    1. 초기 파라미터 벡터에서 시작하여 최대 K번의 반복을 수행합니다.
    2. 각 반복에서 P개의 후보 파라미터 벡터를 무작위로 생성합니다.
    3. 목적 함수 값을 평가하여 최소 값을 찾습니다.
    4. 최소 값이 이전보다 개선되면 파라미터를 업데이트합니다.
    5. 수렴 조건이 만족되면 조기 종료합니다.

    참고
    ----
    Watt et al. (2020), "Machine Learning Refined: Foundations, Algorithms, and Applications"의 알고리즘을 응용하였습니다.
    """
    rng = default_rng()  # seed 미사용
    w = init
    init_size = init.shape[0]
    min_value = object_function(w)

    w_box = np.zeros((init_size, K))
    min_box = np.zeros(K)
    min_temp_box = np.zeros(P)
    w_temp_box = np.zeros((init_size, P))


    if verbose:
        iters = tqdm(range(K), desc="")
    else:
        iters = range(K)

    for k in iters:
        if k == 0 or k % 10 == 0:
            print_str = f"Current Optimization {k}/{K}: {np.round(min_value, 3)}, {np.round(w, 3)}"
            if verbose:
                iters.set_description(print_str)

        if alpha == "diminishing":
            a = 1 / (k + 1)
        else:
            a = alpha

        w_box[:, k] = w
        min_box[k] = min_value

        if k % 100 == 0 and k > 199:
            if np.abs(min_value - min_box[k - 100]) < 0.01:
                if verbose:
                    print(f"> Early return: Optimization converged at iteration {k}")
                w_box = w_box[:, :k + 1]
                min_box = min_box[:k + 1]
                return w, w_box, min_box

        cal = 0
        while cal < P:
            directions = rng.normal(0, 1, init_size)
            norms = np.sqrt(np.sum(directions ** 2))
            directions = directions / norms
            ww = w + a * directions

            if any(w < 0 for w in ww):
                continue
            try:
                values = object_function(ww)
                if np.isnan(values) or np.isinf(values):
                    continue
            except Exception as e:
                continue

            w_temp_box[:, cal] = ww
            min_temp_box[cal] = values
            cal += 1

        if np.min(min_temp_box) > min_box[k]:
            w = w_box[:, k]
            min_value = min_box[k]
        else:
            w = w_temp_box[:, np.argmin(min_temp_box)]
            min_value = np.min(min_temp_box)
    Optimizers = w_box[:, np.argmin(min_box)]
    if verbose:
        print(f"> Calculated Optimization is : ${np.round(np.min(min_box),3)}, {np.round(Optimizers, 3)}")
    return Optimizers, w_box, min_box

