from typing import Union, Literal
import math as math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from . import functions as func
from .. import container_class


class HLW_FC:
    """
    이 클래스는 HLW_FC 모델을 사용하여 주어진 경제 데이터를 기반으로 중립금리를 추정합니다.
    이 모델은 기존의 HLW 모델에 금융 사이클(financial cycle)을 추가로 고려합니다.

    Attributes
    ----------
    verbose : bool
        모델 실행 중간 과정 및 결과를 출력할지 여부를 지정합니다. 기본값은 True입니다.

    numMCMC : int
        번인(burn-in) 기간 이후의 MCMC 샘플 수를 지정합니다. `numMCMC = n_draws - burnin`으로 계산됩니다.

    burnin : int
        MCMC 알고리즘에서 초기 번인 기간의 길이를 지정합니다. 기본값은 10000입니다.

    PercPara : list of float
        모델 파라미터의 백분위수 계산을 위한 하한 및 상한을 지정합니다. 예를 들어, [0.05, 0.95]로 설정하면 5%와 95% 백분위수를 계산합니다. 기본값은 [0.05, 0.95]입니다.

    PercLV : list of float
        잠재 변수(latent variable)의 백분위수 계산을 위한 하한 및 상한을 지정합니다. 예를 들어, [5, 95]로 설정하면 5%와 95% 백분위수를 계산합니다. 기본값은 [5, 95]입니다.

    Ytivs : float
        출력 추세(output trend)의 초기값에 대한 사전 분산(prior variance) 스케일 팩터입니다. 기본값은 10000입니다 (즉, 100^2).

    Ygivs : float
        출력 장기 성장률(long-term growth rate)의 초기값에 대한 사전 분산 스케일 팩터입니다. 기본값은 1입니다 (즉, 1^2).

    Izivs : float
        이자율 z 구성 요소의 초기값에 대한 사전 분산 스케일 팩터입니다. 기본값은 1입니다 (즉, 1^2).

    SSRatio : float
        잠재 변수의 초기값의 사전 분산을 계산하기 위해 사용할 데이터의 비율입니다. 기본값은 0.1입니다.

    ZSpec : str
        이자율 z 구성 요소의 모델 사양을 지정합니다. 가능한 값은 'rw', 'sar_c', 'sar'입니다. 기본값은 'rw'입니다.

    MergedMCMCAlgorithm : dict
        MCMC 알고리즘 설정을 저장하는 딕셔너리입니다. 각 키는 파라미터의 이름이며, 값은 해당 파라미터에 대한 MCMC 알고리즘 유형을 나타냅니다.
        값이 1이면 해당 파라미터에 대해 MCMC 알고리즘을 사용하고, 0이면 깁스 샘플링(Gibbs sampling)을 사용합니다.

        예시:
        ```python
        {
            'sig2_g': 1,
            'sig2_yt': 1,
            'sig2_yc': 1,
            'sig2_z': 1,
            'sig2_p': 1,
            'slope_yc': 1,
            'slope_z': 1,
            'slope_p': 1,
            'sig2_f': 1,
            'slope_f': 1
        }
        ```

    MergedPriorPara : dict
        파라미터에 대한 사전 분포(prior distribution) 정보를 저장하는 딕셔너리입니다. 각 키는 파라미터의 이름이며, 값은 해당 파라미터의 사전 분포 파라미터를 나타냅니다.

        주요 키와 값:
        - 'yc_mean': numpy.ndarray, shape (3,)
            출력 사이클 방정식의 기울기 계수에 대한 사전 평균입니다.
        - 'yc_variance': numpy.ndarray, shape (3, 3)
            출력 사이클 방정식의 기울기 계수에 대한 사전 분산 행렬입니다.
        - 'sig2_yc_shape': float
            출력 사이클의 분산에 대한 inverse-gamma 분포의 shape 파라미터입니다.
        - 'sig2_yc_scale': float
            출력 사이클의 분산에 대한 inverse-gamma 분포의 scale 파라미터입니다.
        - 'sig2_g_shape', 'sig2_g_scale', 'sig2_yt_shape', 'sig2_yt_scale', 'sig2_z_shape', 'sig2_z_scale', 'sig2_p_shape', 'sig2_p_scale', 'sig2_f_shape', 'sig2_f_scale': float
            해당 파라미터들의 inverse-gamma 분포의 shape 및 scale 파라미터입니다.
        - 'p_mean': numpy.ndarray, shape (2,)
            인플레이션 방정식의 기울기 계수에 대한 사전 평균입니다.
        - 'p_variance': numpy.ndarray, shape (2, 2)
            인플레이션 방정식의 기울기 계수에 대한 사전 분산 행렬입니다.
        - 'f_mean': numpy.ndarray, shape (4,)
            금융 사이클 방정식의 기울기 계수에 대한 사전 평균입니다.
        - 'f_variance': numpy.ndarray, shape (4, 4)
            금융 사이클 방정식의 기울기 계수에 대한 사전 분산 행렬입니다.
        - 'z_mean', 'z_variance' (ZSpec이 'sar_c' 또는 'sar'인 경우):
            이자율 z 구성 요소의 기울기 계수에 대한 사전 평균 및 분산입니다.

    MergedInitialPara : dict
        파라미터의 초기값을 저장하는 딕셔너리입니다. 각 키는 파라미터의 이름이며, 값은 초기값입니다.

        예시:
        ```python
        {
            'phi1_yc': 1.2,
            'phi2_yc': -0.5,
            'sig2_yc': 1.0,
            'beta_yc': -0.1,
            'sig2_yt': 1.0,
            'sig2_g': 0.01,
            'phi_p': 0.7,
            'beta_p': 1.0,
            'sig2_p': 1.0,
            'sig2_z': 1.0,
            'delta_z': 0.0,
            'phi_z': 1.0,
            'delta_f': 0.0,
            'phi1_f': 1.2,
            'phi2_f': -0.5,
            'beta_f': 0.1,
            'sig2_f': 1.0
        }
        ```

    dates : numpy.ndarray
        입력 데이터의 날짜 정보를 저장하는 배열입니다. `fit` 메서드에서 설정됩니다.

    ParaStorage : dict
        MCMC 샘플링을 통해 얻은 파라미터 샘플들을 저장하는 딕셔너리입니다. 각 키는 파라미터의 이름이며, 값은 샘플 배열입니다.

        예시:
        - 'phi_yc': numpy.ndarray, shape (numMCMC, 2)
            출력 사이클의 AR 계수(phi1_yc, phi2_yc)에 대한 샘플들.
        - 'beta_yc': numpy.ndarray, shape (numMCMC, 1)
            출력 사이클의 회귀 계수(beta_yc)에 대한 샘플들.
        - 'sig2_yc', 'sig2_yt', 'sig2_g', 'sig2_p', 'sig2_z', 'sig2_f': numpy.ndarray, shape (numMCMC, 1)
            해당 분산 파라미터들의 샘플들.

    SummaryParas : dict
        파라미터 샘플들의 통계 요약 정보를 저장하는 딕셔너리입니다. 'Avg', 'Med', 'PercLow', 'PercUpper'의 키를 가지며, 각각 평균, 중앙값, 하한 백분위수, 상한 백분위수를 의미합니다.

        예시:
        ```python
        {
            'Avg': {'phi_yc': ..., 'beta_yc': ..., ...},
            'Med': {'phi_yc': ..., 'beta_yc': ..., ...},
            'PercLow': {'phi_yc': ..., 'beta_yc': ..., ...},
            'PercUpper': {'phi_yc': ..., 'beta_yc': ..., ...}
        }
        ```

    LVariableStorage : dict
        MCMC 샘플링을 통해 얻은 잠재 변수(latent variables)의 샘플들을 저장하는 딕셔너리입니다. 각 키는 변수의 이름이며, 값은 샘플 배열입니다.

        예시:
        - 'yt': numpy.ndarray, shape (numMCMC, N_T)
            출력 추세의 샘플들.
        - 'yc': numpy.ndarray, shape (numMCMC, N_T)
            출력 사이클의 샘플들.
        - 'g': numpy.ndarray, shape (numMCMC, N_T)
            장기 성장률의 샘플들.
        - 'z': numpy.ndarray, shape (numMCMC, N_T)
            이자율 z 구성 요소의 샘플들.
        - 'rNatural': numpy.ndarray, shape (numMCMC, N_T)
            자연 이자율의 샘플들.
        - 'rCycle': numpy.ndarray, shape (numMCMC, N_T)
            이자율 사이클의 샘플들.

    SummaryLV : dict
        잠재 변수 샘플들의 통계 요약 정보를 저장하는 딕셔너리입니다. 'Avg', 'Med', 'PercLower', 'PercUpper'의 키를 가지며, 각각 평균, 중앙값, 하한 백분위수, 상한 백분위수를 의미합니다.

    Accept_ratio : dict
        MCMC 알고리즘에서 각 파라미터의 샘플 수락 비율(acceptance ratio)을 저장하는 딕셔너리입니다. 각 키는 파라미터의 이름이며, 값은 수락 비율입니다.

        예시:
        ```python
        {
            'sig2_g': 0.234,
            'sig2_yt': 0.251,
            'sig2_yc': 0.245,
            'sig2_z': 0.230,
            'sig2_p': 0.240,
            'slope_yc': 0.235,
            'slope_z': 0.238,
            'slope_p': 0.232,
            'sig2_f': 0.245,
            'slope_f': 0.237
        }
        ```
    """


    def __init__(self,
                n_draws=20000, burnin=10000,
                PercPara=[0.05, 0.95], PercLV=[5, 95],
                Ytivs=100**2, Ygivs=1**2, Izivs=1**2, SSRatio=0.1,
                PriorPara=None, InitialPara=None, MCMCAlgorithm=None, ZSpec='rw',
                MHSample_size=1000,
                verbose=True):
        """
        HLW_FC 모델의 인스턴스를 초기화합니다.

        Parameters
        ----------
        n_draws : int, optional
            총 MCMC 반복 횟수입니다. 기본값은 20000입니다.
        burnin : int, optional
            초기 번인(burn-in) 기간의 길이입니다. 기본값은 10000입니다.
            실제 모델 파라미터 추정을 위한 샘플 수는 `numMCMC = n_draws - burnin`으로 계산됩니다.
        PercPara : list of float, optional
            모델 파라미터의 백분위수 계산을 위한 하한 및 상한입니다. 기본값은 [0.05, 0.95]입니다.
        PercLV : list of float, optional
            잠재 변수의 백분위수 계산을 위한 하한 및 상한입니다. 기본값은 [5, 95]입니다.
        Ytivs : float, optional
            출력 추세의 초기값에 대한 사전 분산(prior variance) 스케일 팩터입니다. 기본값은 10000입니다 (100^2).
        Ygivs : float, optional
            출력 장기 성장률의 초기값에 대한 사전 분산 스케일 팩터입니다. 기본값은 1입니다 (1^2).
        Izivs : float, optional
            이자율 z 구성 요소의 초기값에 대한 사전 분산 스케일 팩터입니다. 기본값은 1입니다 (1^2).
        SSRatio : float, optional
            잠재 변수의 초기값의 사전 분산을 계산하기 위해 사용할 데이터의 비율입니다. 기본값은 0.1입니다.
        PriorPara : dict, optional
            파라미터에 대한 사전 분포(prior distribution) 정보를 포함하는 딕셔너리입니다. 사용자가 지정하지 않으면 기본값이 사용됩니다.
        InitialPara : dict, optional
            파라미터의 초기값을 포함하는 딕셔너리입니다. 사용자가 지정하지 않으면 기본값이 사용됩니다.
        MCMCAlgorithm : dict, optional
            MCMC 알고리즘 설정을 포함하는 딕셔너리입니다. 각 파라미터에 대해 MCMC 알고리즘을 사용할지 여부를 지정합니다. 사용자가 지정하지 않으면 기본값이 사용됩니다.
        ZSpec : str, optional
            이자율 z 구성 요소의 모델 사양을 지정합니다. 가능한 값은 'rw' (랜덤 워크), 'sar_c' (self-adjusting rate with constant), 'sar' (self-adjusting rate)입니다. 기본값은 'rw'입니다.
        MHSample_size : int, optional
            adaptive MH 과정에서 공분산 계산에 사용되는 샘플 크기입니다. 기본값은 1000입니다.
        verbose : bool, optional
            모델 실행 중간 과정 및 결과를 출력할지 여부를 지정합니다. 기본값은 True입니다.

        """

        # save parameters
        self.verbose = verbose

        self.numMCMC = n_draws - burnin
        self.burnin = burnin
        self.PercPara = PercPara
        self.PercLV = PercLV
        self.Ytivs = Ytivs
        self.Ygivs = Ygivs
        self.Izivs = Izivs
        self.SSRatio = SSRatio
        self.ZSpec = ZSpec
        self.is_fitted = False
        self.MHSample_size = MHSample_size

        # Default MCMC algorithms
        DefaultMCMCAlgorithm = {
            'sig2_g': 1,
            'sig2_yt': 1,
            'sig2_yc': 1,
            'sig2_z': 1,
            'sig2_p': 1,
            'slope_yc': 1,
            'slope_z': 1,
            'slope_p': 1,
            'sig2_f': 1,
            'slope_f': 1
        }

        # If some elements of 'MCMCAlgorithm' are empty, use default values
        # merge default and input MCMCAlgorithm indicator
        if not MCMCAlgorithm:
            MergedMCMCAlgorithm = DefaultMCMCAlgorithm
        else:
            MergedMCMCAlgorithm = func.mergeOptions(DefaultMCMCAlgorithm, MCMCAlgorithm)

        # define default priors
        # output equation
        # slope coefficients
        DefaultPriorPara = {
            'yc_mean': np.array([0.7, -0.5, 0]),
            'yc_variance': np.diag(np.array([0.5, 0.5, 10]) ** 2),
            'sig2_yc_shape': 5,
            'sig2_yc_scale': 4 * 2**2,
            'sig2_g_shape': 5,
            'sig2_g_scale': 4 * 0.1**2,
            'sig2_yt_shape': 5,
            'sig2_yt_scale': 4 * 2**2,
            'sig2_z_shape': 5,
            'sig2_z_scale': 4 * 0.5**2,
            'p_mean': np.array([0.7, 0]),
            'p_variance': np.diag(np.array([0.5, 10]) ** 2),
            'sig2_p_shape': 5,
            'sig2_p_scale': 4 * 0.1**2,
            'f_mean': np.array([0, 0.7, -0.5, 0]),
            'f_variance': np.diag(np.array([0.1, 0.5, 0.5, 10]) ** 2),
            'sig2_f_shape': 5,
            'sig2_f_scale': 4 * 0.1**2
        }

        # 중립금리모형에서 장기경제성장률을 제외한 latent vector 가 가지는 stochastic process 정의
        if ZSpec == 'rw':
            ParaSample = {'delta_z': 0, 'phi_z': 1}
            DefaultPriorPara['sig2_z_shape'] = 5
            DefaultPriorPara['sig2_z_scale'] = 4 * 0.5**2
        elif ZSpec == 'sar_c':
            DefaultPriorPara['z_mean'] = np.array([0, 0.7])
            DefaultPriorPara['z_variance'] = np.diag([10, 0.5]) ** 2
            DefaultPriorPara['sig2_z_shape'] = 5
            DefaultPriorPara['sig2_z_scale'] = 4 * 1**2
        elif ZSpec == 'sar':
            DefaultPriorPara['z_mean'] = np.array([0.7])
            DefaultPriorPara['z_variance'] = np.diag([0.5]) ** 2
            DefaultPriorPara['sig2_z_shape'] = 5
            DefaultPriorPara['sig2_z_scale'] = 4 * 1**2

        # If some elements of 'PriorPara' are empty, use default values
        # merge default and input prior parameters
        if not PriorPara:
            MergedPriorPara = DefaultPriorPara
        else:
            MergedPriorPara = func.mergeOptions(DefaultPriorPara, PriorPara)

        # compute other prior hyper-parameters
        MergedPriorPara['yc_logdet_variance'] = func.logdet(MergedPriorPara['yc_variance'])
        MergedPriorPara['yc_Inv_variance'] = np.linalg.inv(MergedPriorPara['yc_variance'])
        MergedPriorPara['yc_Inv_variance_mean'] = MergedPriorPara['yc_Inv_variance'] @ MergedPriorPara['yc_mean']
        MergedPriorPara['p_logdet_variance'] = func.logdet(MergedPriorPara['p_variance'])
        MergedPriorPara['p_Inv_variance'] = np.linalg.inv(MergedPriorPara['p_variance'])
        MergedPriorPara['p_Inv_variance_mean'] = MergedPriorPara['p_Inv_variance'] @ MergedPriorPara['p_mean']
        MergedPriorPara['f_logdet_variance'] = func.logdet(MergedPriorPara['f_variance'])
        MergedPriorPara['f_Inv_variance'] = np.linalg.inv(MergedPriorPara['f_variance'])
        MergedPriorPara['f_Inv_variance_mean'] = MergedPriorPara['f_Inv_variance'] @ MergedPriorPara['f_mean']

        if ZSpec in ['sar_c', 'sar']:
            MergedPriorPara['z_logdet_variance'] = func.logdet(MergedPriorPara['z_variance'])
            MergedPriorPara['z_Inv_variance'] = np.linalg.inv(MergedPriorPara['z_variance'])
            MergedPriorPara['z_Inv_variance_mean'] = MergedPriorPara['z_Inv_variance'] @ MergedPriorPara['z_mean']

        # define default values for initial model parameter values
        # model parameters (18 parameters)
        # output cycle
        DefaultInitialPara = {
            'phi1_yc': 1.2,
            'phi2_yc': -0.5,
            'sig2_yc': 1 ** 2,
            'beta_yc': -0.1,
            'sig2_yt': 1 ** 2,
            'sig2_g': 0.1 ** 2,
            'phi_p': 0.7,
            'beta_p': 1,
            'sig2_p': 1 ** 2,
            'sig2_z': 1 ** 2,
            'delta_f': 0,
            'phi1_f': 1.2,
            'phi2_f': -0.5,
            'beta_f': 0.1,
            'sig2_f': 1**2
        }

        if ZSpec == 'rw':
            DefaultInitialPara['delta_z'] = 0
            DefaultInitialPara['phi_z'] = 1
        elif ZSpec in ['sar_c', 'sar']:
            DefaultInitialPara['delta_z'] = 0
            DefaultInitialPara['phi_z'] = 0.7

        # If some elements of 'InitialPara' are empty, use default values
        # merge default and input prior parameters
        if not InitialPara:
            MergedInitialPara = DefaultInitialPara
        else:
            MergedInitialPara = func.mergeOptions(DefaultInitialPara, InitialPara)

        ## SAVE Parameter Dictionary ##
        self.MergedMCMCAlgorithm = MergedMCMCAlgorithm
        self.MergedPriorPara = MergedPriorPara
        self.MergedInitialPara = MergedInitialPara

        # PRINT: 모델 초기화 정보 출력
        if self.verbose:
            print("> Initializing Model...")
            print(f"  - Total MCMC iterations: {n_draws}")
            print(f"  - Burn-in period: {burnin}")
            print(f"  - Latent variables percentiles: {PercLV}")
            print(f"  - Parameters percentiles: {PercPara}")
            print(f"  - Interest rate z-component specification: {ZSpec}")


    def fit(self, data, dates=None):
        """
        주어진 데이터에 HLW_FC 모델을 적합시킵니다.

        Parameters
        ----------
        data : pandas.DataFrame or numpy.ndarray
            입력 데이터 프레임 또는 numpy array.
            컬럼 구성은 반드시 다음과 같아야 합니다. [ln real gdp, inflation rate, real interest rate, financial cycle]
            - 시간 기준 오름차순으로 정렬되어야 합니다

        dates : array-like, optional
            data에 매핑되는 날짜입니다. 정렬 순서와 크기가 data와 동일해야 합니다

        Returns
        -------
        None

        Notes
        -----
        워크플로:
        1. 입력 데이터에서 필요한 변수들을 추출하고 전처리합니다.
        2. 모델을 적합시키기 위해 내부 메서드 `_fit_hlw_fc`를 호출합니다.
        3. 적합 결과를 클래스 속성으로 저장합니다.
        """
        ### 데이터 처리 ###

        # PRINT: 데이터 적합 시작
        if self.verbose:
            print("> Starting model fitting...")
            print(f"  - Data shape: {data.shape}")
            if dates is not None:
                print(f"  - Dates provided with length: {len(dates)}")
            
        
        # column size check
        col_size = 4
        if data.shape[1] != col_size:
            # just warn 
            print(f"> Warning: Input data must have {col_size} columns. (Column Order: [ln real gdp, inflation rate, real interest rate, financial cycle])")


        # Extract the data columns (ln real GDP, (annualized) inflation rate, (annualized) real interest rate, financial cycle)
        if isinstance(data, pd.DataFrame):
            Data_full = data.to_numpy()
        else:
            Data_full = np.array(data)

        # 금융 사이클 데이터에서 추세를 제거하기 위해 HP 필터를 적용
        fc_trend = func.hpfilter(Data_full[:,3], 1600)  # Obtain the trend component
        Data_full[:,3] = Data_full[:,3] - fc_trend      # Subtract the trend component

        # construct dependent variable vectors
        Y = Data_full[4:, [0, 1, 3]]  # ln real GDP, inflation rate, financial cycle
        X = np.column_stack((
            Data_full[3:-1, 0], Data_full[2:-2, 0], Data_full[3:-1, 2], Data_full[2:-2, 2],
            Data_full[3:-1, 1], Data_full[2:-2, 1], Data_full[1:-3, 1], Data_full[:-4, 1],
            Data_full[3:-1, 3], Data_full[2:-2, 3]
        ))

        y_data = Y[:, 0]
        p_data = Y[:, 1]
        f_data = Y[:, 2]
        r_data = Data_full[4:, 2]
        # dates_out = dates[4:]

        # 날짜 처리
        if dates is None:
            self.dates = np.arange(len(y_data)) + 4  # 앞에서 4칸 shift (4번 인덱스부터 len+3번까지)
        else:
            dates = np.array(dates)
            self.dates = dates[4:]

        ### FIT ###
        ParaStorage, SummaryParas, LVariableStorage, SummaryLV, Accept_ratio = self._fit_hlw_fc(Data_full, Y, X, y_data, p_data, r_data, f_data)

        # save results
        self.ParaStorage = container_class.Container(ParaStorage)
        self.LVariableStorage = container_class.Container(LVariableStorage)
        self.SummaryParas = container_class.Container(SummaryParas)
        self.SummaryLV = container_class.Container(SummaryLV)
        self.Accept_ratio = container_class.Container(Accept_ratio)

        # PRINT: Done
        if self.verbose:
            print("> Model Fitted.")
        self.is_fitted = True



    def print_results(self):
        """
        모델 적합 결과를 시각화하여 출력합니다.
        잠재 변수들의 추정치를 그래프로 표시합니다.

        Returns
        -------
        None

        Notes
        -----
        예시
        ----
        ```python
        model.print_results()
        ```
        """
        n_fig = 1

        model_name = "HLW_FC"
        mapper = {"yt": "Output (Trend)",
                  "yc": "Output (Cycle)",
                  "g": "Output (Long-run Growth Rate)",
                  "z": "Natural Interest Rate (Other Factors)",
                  "rNatural": "Natural Interest Rate",
                  "rCycle": "Interest Rate Cycle"}

        # Plot posterior distributions: HLW_FC model
        fields_LVariables = self.LVariableStorage.keys()
        for fieldName in fields_LVariables:
            combinedStrings = f'{mapper[fieldName]}: {model_name}'

            plt.figure(n_fig)
            plt.fill_between(self.dates, self.SummaryLV['PercLower'][fieldName], self.SummaryLV['PercUpper'][fieldName], color=[0.9, 0.9, 0.9])
            plt.plot(self.dates, self.SummaryLV['Med'][fieldName], 'r--', linewidth=1)
            plt.title(combinedStrings)
            plt.legend(['Posterior band', 'Posterior median'], loc='lower left')
            n_fig += 1

        plt.tight_layout()
        plt.show()


    # =========================================
    # Internal Functions
    # =========================================

    def _fit_hlw_fc(self, Data_full, Y, X, y_data, p_data, r_data, f_data):
        """
        HLW_FC 모델을 사용하여 데이터에 적합하고, 파라미터 및 잠재 변수의 샘플을 생성합니다.

        Parameters
        ----------
        Data_full : numpy.ndarray, shape(N_T, 4)
            전체 입력 데이터 배열입니다.
        Y : numpy.ndarray, shape(N_T-4, 3)
            종속 변수 행렬입니다.
        X : numpy.ndarray, shape(N_T-4, 10)
            독립 변수 행렬입니다.
        y_data : numpy.ndarray, shape(N_T-4)
            출력 데이터 벡터입니다.
        p_data : numpy.ndarray, shape(N_T-4)
            인플레이션 데이터 벡터입니다.
        r_data : numpy.ndarray, shape(N_T-4)
            이자율 데이터 벡터입니다.
        f_data : numpy.ndarray, shape(N_T-4)
            금융 사이클 데이터 벡터입니다.

        Returns
        -------
        ParaStorage : dict
            파라미터 샘플 저장소입니다.
        SummaryParas : dict
            파라미터 샘플의 통계 요약 정보입니다.
        LVariableStorage : dict
            잠재 변수 샘플 저장소입니다.
        SummaryLV : dict
            잠재 변수 샘플의 통계 요약 정보입니다.
        Accept_ratio : dict
            MCMC 알고리즘의 수락률입니다.

        Notes
        -----
        이 메서드는 다음의 주요 단계를 수행합니다.
        1. 초기 설정 및 변수 초기화.
        2. MCMC 알고리즘을 통해 파라미터와 잠재 변수를 샘플링.
        3. 수집된 샘플에 대한 통계 요약 정보 계산.
        """

        #####################################################################################
        #
        # Estimation begins here. Given Data: r_data, y_data, p_data, f_data
        #
        #####################################################################################
        # define initial MCMC algorithm for first 1000 iterations
        RunningMCMCAlgorithm = {
            'sig2_g': 1,
            'sig2_yt': 1,
            'sig2_yc': 1,
            'sig2_z': 1,
            'sig2_p': 1,
            'slope_yc': 1,
            'slope_z': 1,
            'slope_p': 1,
            'sig2_f': 1,
            'slope_f': 1
        }

        # sample size
        SizeInformation = {
            'N_T': Y.shape[0],
            'N_Latent': 9
        }

        # prior for the initial values of latent variables
        PriorLatent = func.Prior_LatentVariables(Data_full, self.SSRatio, self.Ytivs, self.Ygivs, self.Izivs)

        # model parameters (18 parameters)
        # output cycle
        ParaSample = {
            'phi1_yc': self.MergedInitialPara['phi1_yc'],
            'phi2_yc': self.MergedInitialPara['phi2_yc'],
            'sig2_yc': self.MergedInitialPara['sig2_yc'],
            'beta_yc': self.MergedInitialPara['beta_yc'],
            'sig2_yt': self.MergedInitialPara['sig2_yt'],
            'sig2_g': self.MergedInitialPara['sig2_g'],
            'phi_p': self.MergedInitialPara['phi_p'],
            'beta_p': self.MergedInitialPara['beta_p'],
            'sig2_p': self.MergedInitialPara['sig2_p'],
            'sig2_z': self.MergedInitialPara['sig2_z'],
            'delta_z': self.MergedInitialPara['delta_z'],
            'phi_z': self.MergedInitialPara['phi_z'],
            'delta_f': self.MergedInitialPara['delta_f'],
            'phi1_f': self.MergedInitialPara['phi1_f'],
            'phi2_f': self.MergedInitialPara['phi2_f'],
            'beta_f': self.MergedInitialPara['beta_f'],
            'sig2_f': self.MergedInitialPara['sig2_f']
        }

        # storage space: model parameters
        ParaStorage = {
            'phi_yc': np.zeros((self.numMCMC, 2)),
            'beta_yc': np.zeros((self.numMCMC, 1)),
            'sig2_yc': np.zeros((self.numMCMC, 1)),
            'sig2_yt': np.zeros((self.numMCMC, 1)),
            'sig2_g': np.zeros((self.numMCMC, 1)),
            'phi_p': np.zeros((self.numMCMC, 1)),
            'beta_p': np.zeros((self.numMCMC, 1)),
            'sig2_p': np.zeros((self.numMCMC, 1)),
            'delta_z': np.zeros((self.numMCMC, 1)),
            'phi_z': np.zeros((self.numMCMC, 1)),
            'sig2_z': np.zeros((self.numMCMC, 1)),
            'delta_f': np.zeros((self.numMCMC,1)),
            'phi_f': np.zeros((self.numMCMC,2)),
            'beta_f': np.zeros((self.numMCMC,1)),
            'sig2_f': np.zeros((self.numMCMC,1))
        }

        # storage space: latent variables
        LVariableStorage = {
            'yt': np.zeros((self.numMCMC, SizeInformation['N_T'])),
            'yc': np.zeros((self.numMCMC, SizeInformation['N_T'])),
            'g': np.zeros((self.numMCMC, SizeInformation['N_T'])),
            'z': np.zeros((self.numMCMC, SizeInformation['N_T'])),
            'rNatural': np.zeros((self.numMCMC, SizeInformation['N_T'])),
            'rCycle': np.zeros((self.numMCMC, SizeInformation['N_T']))
        }

        # input parameter values for Adaptive MH
        ProposalType_idx = 0
        ProposalVar_0_IG = (0.05) ** 2
        ProposalVar_0_Normal = (0.05)**2

        MHSample = {
            'sig2_g': ParaSample['sig2_g'] * np.ones((self.MHSample_size, 1)),
            'sig2_yt': ParaSample['sig2_yt'] * np.ones((self.MHSample_size, 1)),
            'sig2_yc': ParaSample['sig2_yc'] * np.ones((self.MHSample_size, 1)),
            'sig2_z': ParaSample['sig2_z'] * np.ones((self.MHSample_size, 1)),
            'sig2_p': ParaSample['sig2_p'] * np.ones((self.MHSample_size, 1)),
            'slope_yc': np.tile([(ParaSample['phi1_yc'] + ParaSample['phi2_yc']), ParaSample['phi2_yc'], ParaSample['beta_yc']], (self.MHSample_size, 1)),
            'slope_p': np.tile([ParaSample['phi_p'], ParaSample['beta_p']], (self.MHSample_size, 1)),
            'sig2_f': ParaSample['sig2_f'] * np.ones((self.MHSample_size, 1)),
            'slope_f': np.tile([ParaSample['delta_f'], (ParaSample['phi1_f']+ParaSample['phi2_f']), ParaSample['phi2_f'], ParaSample['beta_f']], (self.MHSample_size, 1))
        }

        if self.ZSpec == 'sar_c':
            MHSample['slope_z'] = np.tile([ParaSample['delta_z'], ParaSample['phi_z']], (self.MHSample_size, 1))
        elif self.ZSpec == 'sar':
            MHSample['slope_z'] = ParaSample['phi_z'] * np.ones((self.MHSample_size, 1))

        # output of Adaptive MH
        Accept_idx = {
            'sig2_g': np.zeros((self.numMCMC, 1)),
            'sig2_yt': np.zeros((self.numMCMC, 1)),
            'sig2_yc': np.zeros((self.numMCMC, 1)),
            'sig2_z': np.zeros((self.numMCMC, 1)),
            'sig2_p': np.zeros((self.numMCMC, 1)),
            'slope_yc': np.zeros((self.numMCMC, 1)),
            'slope_z': np.zeros((self.numMCMC, 1)),
            'slope_p': np.zeros((self.numMCMC, 1)),
            'sig2_f': np.zeros((self.numMCMC, 1)),
            'slope_f': np.zeros((self.numMCMC, 1))
        }
        reverseStr = []

        # log likelihood storage
        last_loglik = {
            "sig2_g": None,
            "sig2_yt": None,
            "sig2_yc": None,
            "sig2_z": None,
            "sig2_p": None,
            "slope_yc": None,
            "slope_z": None,
            "slope_p": None,
            "sig2_f": None,
            "slope_f": None
        }


        # 반복문 시작
        if self.verbose:
            iterator = tqdm(range(1, self.numMCMC + self.burnin + 1), desc="Sampling")
        else:
            iterator = range(1, self.numMCMC + self.burnin + 1)

        flag = "Step 1/3"

        step_1_end_index = int(self.burnin/2)
        step_2_end_index = int(self.burnin)

        for itr in iterator:
            if self.verbose:
                iterator.set_description(f"{flag} ")
            # reverseStr = displayprogress(100 * itr / (numMCMC + burnin), reverseStr)

            # Step0: switch from Gibbs to user-specified MCMC algorithm
            if itr == 1:
                MHSample['sig2_g'] = ParaSample['sig2_g'] * np.ones((self.MHSample_size, 1)).flatten()
                MHSample['sig2_yt'] = ParaSample['sig2_yt'] * np.ones((self.MHSample_size, 1)).flatten()
                MHSample['sig2_yc'] = ParaSample['sig2_yc'] * np.ones((self.MHSample_size, 1)).flatten()
                MHSample['sig2_z'] = ParaSample['sig2_z'] * np.ones((self.MHSample_size, 1)).flatten()
                MHSample['sig2_p'] = ParaSample['sig2_p'] * np.ones((self.MHSample_size, 1)).flatten()
                MHSample['slope_yc'] = np.tile(np.array([(ParaSample['phi1_yc'] + ParaSample['phi2_yc']), ParaSample['phi2_yc'], ParaSample['beta_yc']]).reshape(1,-1), (self.MHSample_size, 1))
                MHSample['slope_p'] = np.tile(np.array([ParaSample['phi_p'], ParaSample['beta_p']]).reshape(1,-1), (self.MHSample_size, 1))
                if self.ZSpec == 'sar_c':
                    MHSample['slope_z'] = np.tile(np.array([ParaSample['delta_z'], ParaSample['phi_z']]).reshape(1,-1), (self.MHSample_size, 1))
                elif self.ZSpec == 'sar':
                    MHSample['slope_z'] = np.tile(np.array([ParaSample['phi_z']]).reshape(1,-1), (self.MHSample_size, 1))
                MHSample['sig2_f'] = ParaSample['sig2_f']*np.ones((1000,1)).flatten()
                MHSample['slope_f'] = np.tile(np.array([ParaSample['delta_f'], (ParaSample['phi1_f']+ParaSample['phi2_f']), ParaSample['phi2_f'], ParaSample['beta_f']]).reshape(1,-1),(1000,1))

            elif itr == step_1_end_index:  # NOTE: burnin/2 으로 수정
                flag = "Step 2/3"
                RunningMCMCAlgorithm = self.MergedMCMCAlgorithm
                # update input MCMC samples for Adaptive MH from Gibbs outputs

                MHSample['sig2_g'] = ParaSample['sig2_g'] * np.ones((self.MHSample_size, 1)).flatten()
                MHSample['sig2_yt'] = ParaSample['sig2_yt'] * np.ones((self.MHSample_size, 1)).flatten()
                MHSample['sig2_yc'] = ParaSample['sig2_yc'] * np.ones((self.MHSample_size, 1)).flatten()
                MHSample['sig2_z'] = ParaSample['sig2_z'] * np.ones((self.MHSample_size, 1)).flatten()
                MHSample['sig2_p'] = ParaSample['sig2_p'] * np.ones((self.MHSample_size, 1)).flatten()
                MHSample['slope_yc'] = np.tile(np.array([(ParaSample['phi1_yc'] + ParaSample['phi2_yc']), ParaSample['phi2_yc'], ParaSample['beta_yc']]).reshape(1,-1), (self.MHSample_size, 1))
                MHSample['slope_p'] = np.tile(np.array([ParaSample['phi_p'], ParaSample['beta_p']]).reshape(1,-1), (self.MHSample_size, 1))
                if self.ZSpec == 'sar_c':
                    MHSample['slope_z'] = np.tile(np.array([ParaSample['delta_z'], ParaSample['phi_z']]).reshape(1,-1), (self.MHSample_size, 1))
                elif self.ZSpec == 'sar':
                    MHSample['slope_z'] = np.tile(np.array([ParaSample['phi_z']]).reshape(1,-1), (self.MHSample_size, 1))

                MHSample['sig2_f'] = ParaSample['sig2_f']*np.ones((1000,1)).flatten()
                MHSample['slope_f'] = np.tile(np.array([ParaSample['delta_f'], (ParaSample['phi1_f']+ParaSample['phi2_f']), ParaSample['phi2_f'], ParaSample['beta_f']]).reshape(1,-1),(1000,1))
            # elif itr == 2001:
            elif itr == step_2_end_index:  # NOTE: burnin/2 으로 수정 > step2/3은 burnin 닿을 떄까지
                flag = "Step 3/3"
                ProposalType_idx = 1

            # NOTE: burnin의 수가 충분하지 않은 경우를 대응하기 위함, burnin 이 1이하일 경우 바로 step 3/3 으로 넘어감
            if itr == 1 and self.burnin <= 1:
                flag = "Step 3/3"
                ProposalType_idx = 1

                RunningMCMCAlgorithm = self.MergedMCMCAlgorithm
                # update input MCMC samples for Adaptive MH from Gibbs outputs
                MHSample['sig2_g'] = ParaSample['sig2_g'] * np.ones((self.MHSample_size, 1)).flatten()
                MHSample['sig2_yt'] = ParaSample['sig2_yt'] * np.ones((self.MHSample_size, 1)).flatten()
                MHSample['sig2_yc'] = ParaSample['sig2_yc'] * np.ones((self.MHSample_size, 1)).flatten()
                MHSample['sig2_z'] = ParaSample['sig2_z'] * np.ones((self.MHSample_size, 1)).flatten()
                MHSample['sig2_p'] = ParaSample['sig2_p'] * np.ones((self.MHSample_size, 1)).flatten()
                MHSample['slope_yc'] = np.tile(np.array([(ParaSample['phi1_yc'] + ParaSample['phi2_yc']), ParaSample['phi2_yc'], ParaSample['beta_yc']]).reshape(1,-1), (self.MHSample_size, 1))
                MHSample['slope_p'] = np.tile(np.array([ParaSample['phi_p'], ParaSample['beta_p']]).reshape(1,-1), (self.MHSample_size, 1))
                if self.ZSpec == 'sar_c':
                    MHSample['slope_z'] = np.tile(np.array([ParaSample['delta_z'], ParaSample['phi_z']]).reshape(1,-1), (self.MHSample_size, 1))
                elif self.ZSpec == 'sar':
                    MHSample['slope_z'] = ParaSample['phi_z'] * np.ones((self.MHSample_size, 1)).flatten()

                MHSample['sig2_f'] = ParaSample['sig2_f']*np.ones((1000,1)).flatten()
                MHSample['slope_f'] = np.tile(np.array([ParaSample['delta_f'], (ParaSample['phi1_f']+ParaSample['phi2_f']), ParaSample['phi2_f'], ParaSample['beta_f']]).reshape(1,-1),(1000,1))


            # Step 1: draw latent variables
            bt_draw = func.GenerateLatentVariables_FC(Y, X, ParaSample, PriorLatent, SizeInformation)

            # compute latent variables
            LVariableSample = {
                'yt': bt_draw[:, 0],  # output trend (t = 0 ~T)
                'yc': y_data - bt_draw[1:, 0],  # output cycle (t = 1 ~T)
                'g': bt_draw[:, 3],  # long-run growth (t = 0 ~T)
                'z': bt_draw[:, 6],  # natural rate z component (t = 0 ~T)
            }
            LVariableSample['rNatural'] = LVariableSample['g'] + LVariableSample['z']  # natural rate (t = 0 ~T)
            LVariableSample['rCycle'] = r_data - bt_draw[1:, 3] - bt_draw[1:, 6]  # interest rate cycle (t = 1 ~T)

            # Step 2: draw the variance of the shock to long-run growth rate (g)
            if RunningMCMCAlgorithm['sig2_g'] == 0:
                # compute data
                g_error = LVariableSample['g'][1:] - LVariableSample['g'][:-1]
                # posterior sampling
                ParaSample['sig2_g'] = func.DrawVarianceIG(g_error, self.MergedPriorPara['sig2_g_shape'], self.MergedPriorPara['sig2_g_scale'])
            elif RunningMCMCAlgorithm['sig2_g'] == 1:
                ParaSample['sig2_g'], MHSample['sig2_g'], accept_idx_sig2_g, last_loglik["sig2_g"] = func.AdaptiveMH_IG(
                    Y, X, 'sig2_g', 'FC', ProposalType_idx, MHSample['sig2_g'], ParaSample,
                    PriorLatent, self.MergedPriorPara['sig2_g_shape'], self.MergedPriorPara['sig2_g_scale'], ProposalVar_0_IG, SizeInformation,
                    last_loglik['sig2_g'],
                )

            # Step 3: draw the variance of the shock to the output trend (yt)
            if RunningMCMCAlgorithm['sig2_yt'] == 0:
                # compute data
                yt_error = (LVariableSample['yt'][1:] - LVariableSample['yt'][:-1]) - LVariableSample['g'][:-1]
                # posterior sampling
                ParaSample['sig2_yt'] = func.DrawVarianceIG(yt_error, self.MergedPriorPara['sig2_yt_shape'], self.MergedPriorPara['sig2_yt_scale'])

            elif RunningMCMCAlgorithm['sig2_yt'] == 1:
                ParaSample['sig2_yt'], MHSample['sig2_yt'], accept_idx_sig2_yt, last_loglik['sig2_yt'] = func.AdaptiveMH_IG(
                    Y, X, 'sig2_yt', 'FC', ProposalType_idx, MHSample['sig2_yt'], ParaSample,
                    PriorLatent, self.MergedPriorPara['sig2_yt_shape'], self.MergedPriorPara['sig2_yt_scale'], ProposalVar_0_IG, SizeInformation,
                    last_loglik['sig2_yt'],
                )
            
            
            # Step 4.1: draw the variance of the shock to the output cycle (yc)
            # compute data
            rCycle_mean = (LVariableSample['rCycle'][1:-1] + LVariableSample['rCycle'][:-2]) / 2
            Y_yc = LVariableSample['yc'][2:]
            X_yc = np.column_stack((LVariableSample['yc'][1:-1], (LVariableSample['yc'][:-2] - LVariableSample['yc'][1:-1]), rCycle_mean))
            yc_error = Y_yc - (X_yc @ np.array([(ParaSample['phi1_yc'] + ParaSample['phi2_yc']), ParaSample['phi2_yc'], ParaSample['beta_yc']])).squeeze()

            # posterior sampling
            if RunningMCMCAlgorithm['sig2_yc'] == 0:
                ParaSample['sig2_yc'] = func.DrawVarianceIG(yc_error, self.MergedPriorPara['sig2_yc_shape'], self.MergedPriorPara['sig2_yc_scale'])
            elif RunningMCMCAlgorithm['sig2_yc'] == 1:
                ParaSample['sig2_yc'], MHSample['sig2_yc'], accept_idx_sig2_yc, last_loglik['sig2_yc'] = func.AdaptiveMH_IG(
                    Y, X, 'sig2_yc', 'FC', ProposalType_idx, MHSample['sig2_yc'], ParaSample,
                    PriorLatent, self.MergedPriorPara['sig2_yc_shape'], self.MergedPriorPara['sig2_yc_scale'], ProposalVar_0_IG, SizeInformation,
                    last_loglik['sig2_yc'],
                )
            
            # Step 4.2: draw the slope coefficient of the output cycle equation
            # posterior sampling
            if RunningMCMCAlgorithm['slope_yc'] == 0:
                stationary_idc = 0
                while stationary_idc == 0:
                    yc_coeff_draw, Mean_post, Variance_post = func.GenerateSlopeCoefficients(Y_yc, X_yc, ParaSample['sig2_yc'], self.MergedPriorPara['yc_Inv_variance'], self.MergedPriorPara['yc_Inv_variance_mean'])
                    if all(abs(yc_coeff_draw[0]) < 0.98): 
                        stationary_idc = 1
            elif RunningMCMCAlgorithm['slope_yc'] == 1:
                yc_coeff_draw, MHSample['slope_yc'], accept_idx_slope_yc, last_loglik['slope_yc'] = func.AdaptiveMH_Normal(
                    Y, X, 'yc', 'FC', ProposalType_idx, MHSample['slope_yc'], ParaSample,
                    PriorLatent, self.MergedPriorPara['yc_mean'], self.MergedPriorPara['yc_variance'], self.MergedPriorPara['yc_logdet_variance'], ProposalVar_0_Normal, SizeInformation,
                    last_loglik['slope_yc'],
                )
            ParaSample['phi1_yc'] = yc_coeff_draw[0] - yc_coeff_draw[1]
            ParaSample['phi2_yc'] = yc_coeff_draw[1]
            ParaSample['beta_yc'] = yc_coeff_draw[2]

            # Step 5.1: draw the variance of the shock to the interest rate component z
            # compute data
            Y_z = LVariableSample['z'][1:]
            X_z = np.column_stack((np.ones_like(Y_z), LVariableSample['z'][:-1]))
            z_error = Y_z - X_z @ np.array([ParaSample['delta_z'], ParaSample['phi_z']])
            # posterior sampling
            if RunningMCMCAlgorithm['sig2_z'] == 0:
                ParaSample['sig2_z'] = func.DrawVarianceIG(z_error, self.MergedPriorPara['sig2_z_shape'], self.MergedPriorPara['sig2_z_scale'])
            elif RunningMCMCAlgorithm['sig2_z'] == 1:
                ParaSample['sig2_z'], MHSample['sig2_z'], accept_idx_sig2_z, last_loglik['sig2_z'] = func.AdaptiveMH_IG(
                    Y, X, 'sig2_z', 'FC', ProposalType_idx, MHSample['sig2_z'], ParaSample,
                    PriorLatent, self.MergedPriorPara['sig2_z_shape'], self.MergedPriorPara['sig2_z_scale'], ProposalVar_0_IG, SizeInformation,
                    last_loglik['sig2_z'],
                )

            # Step 5.2: draw the slope coefficient of the interest rate component z
            if self.ZSpec == 'rw':
                ParaSample['delta_z'] = 0
                ParaSample['phi_z'] = 1
                accept_idx_slope_z = 1
            elif self.ZSpec == 'sar_c':
                # posterior sampling
                if RunningMCMCAlgorithm['slope_z'] == 0:
                    # Gibbs sampling
                    stationary_idc = 0
                    while stationary_idc == 0:
                        z_coeff_draw, Mean_post, Variance_post = func.GenerateSlopeCoefficients(Y_z, X_z, ParaSample['sig2_z'], self.MergedPriorPara['z_Inv_variance'], self.MergedPriorPara['z_Inv_variance_mean'])
                        if abs(z_coeff_draw[1]) < 0.98:
                            stationary_idc = 1
                elif RunningMCMCAlgorithm['slope_z'] == 1:
                    # MH sampling
                    z_coeff_draw, MHSample['slope_z'], accept_idx_slope_z, last_loglik['slope_z'] = func.AdaptiveMH_Normal(
                        Y, X, 'zc', 'FC', ProposalType_idx, MHSample['slope_z'], ParaSample,
                        PriorLatent, self.MergedPriorPara['z_mean'], self.MergedPriorPara['z_variance'], self.MergedPriorPara['z_logdet_variance'], ProposalVar_0_Normal, SizeInformation,
                        last_loglik['slope_z'],
                    )
                ParaSample['delta_z'] = z_coeff_draw[0]
                ParaSample['phi_z'] = z_coeff_draw[1]
            elif self.ZSpec == 'sar':
                # posterior sampling
                if RunningMCMCAlgorithm['slope_z'] == 0:
                    # Gibbs sampling
                    stationary_idc = 0
                    while stationary_idc == 0:
                        z_coeff_draw, Mean_post, Variance_post = func.GenerateSlopeCoefficients(Y_z, X_z[:, 1], ParaSample['sig2_z'], self.MergedPriorPara['z_Inv_variance'], self.MergedPriorPara['z_Inv_variance_mean'])
                        if all(abs(z_coeff_draw) < 0.98):
                            stationary_idc = 1
                elif RunningMCMCAlgorithm['slope_z'] == 1:
                    # MH sampling
                    z_coeff_draw, MHSample['slope_z'], accept_idx_slope_z, last_loglik['slope_z'] = func.AdaptiveMH_Normal(
                        Y, X, 'z', 'FC', ProposalType_idx, MHSample['slope_z'], ParaSample,
                        PriorLatent, self.MergedPriorPara['z_mean'], self.MergedPriorPara['z_variance'], self.MergedPriorPara['z_logdet_variance'], ProposalVar_0_Normal, SizeInformation,
                        last_loglik['slope_z'],
                    )
                ParaSample['delta_z'] = 0
                ParaSample['phi_z'] = z_coeff_draw[0]  # z_coeff_draw = [value,]

            # Step 6.1: draw the variance of the shock to the inflation rate (p)
            # compute data
            psum_lag = np.mean(X[1:, 5:8], axis=1)
            Y_p = p_data[1:] - psum_lag
            X_p = np.column_stack((X[1:, 4] - psum_lag, LVariableSample['yc'][:-1]))
            p_error = Y_p - X_p @ np.array([ParaSample['phi_p'], ParaSample['beta_p']]).squeeze()

            # posterior sampling
            if RunningMCMCAlgorithm['sig2_p'] == 0:
                ParaSample['sig2_p'] = func.DrawVarianceIG(p_error, self.MergedPriorPara['sig2_p_shape'], self.MergedPriorPara['sig2_p_scale'])
            elif RunningMCMCAlgorithm['sig2_p'] == 1:
                ParaSample['sig2_p'], MHSample['sig2_p'], accept_idx_sig2_p, last_loglik['sig2_p'] = func.AdaptiveMH_IG(
                    Y, X, 'sig2_p', 'FC', ProposalType_idx, MHSample['sig2_p'], ParaSample,
                    PriorLatent, self.MergedPriorPara['sig2_p_shape'], self.MergedPriorPara['sig2_p_scale'], ProposalVar_0_IG, SizeInformation,
                    last_loglik['sig2_p'],
                )

            # Step 6.2: draw the slope coefficient of the inflation rate equation
            # posterior sampling
            if RunningMCMCAlgorithm['slope_p'] == 0:
                stationary_idc = 0
                while stationary_idc == 0:
                    p_coeff_draw, Mean_post, Variance_post = func.GenerateSlopeCoefficients(Y_p, X_p, ParaSample['sig2_p'], self.MergedPriorPara['p_Inv_variance'], self.MergedPriorPara['p_Inv_variance_mean'])
                    if all(abs(p_coeff_draw[0]) < 0.98):
                        stationary_idc = 1
            elif RunningMCMCAlgorithm['slope_p'] == 1:
                p_coeff_draw, MHSample['slope_p'], accept_idx_slope_p, last_loglik['slope_p'] = func.AdaptiveMH_Normal(
                    Y, X, 'p', 'FC', ProposalType_idx, MHSample['slope_p'], ParaSample,
                    PriorLatent, self.MergedPriorPara['p_mean'], self.MergedPriorPara['p_variance'], self.MergedPriorPara['p_logdet_variance'], ProposalVar_0_Normal, SizeInformation,
                    last_loglik['slope_p'],
                )
            ParaSample['phi_p'] = p_coeff_draw[0]
            ParaSample['beta_p'] = p_coeff_draw[1]


            # Step 7.1: draw the variance of the shock to the financial cycle (f)
            # compute data
            rCycle_mean = (LVariableSample['rCycle'][1:-1] + LVariableSample['rCycle'][:-2]) / 2
            Y_f = f_data[2:]
            X_f = np.column_stack((np.ones(Y_f.shape[0]), f_data[1:-1], f_data[:-2] - f_data[1:-1], rCycle_mean))
            f_error = Y_f - (X_f @ np.array([ParaSample['delta_f'], (ParaSample['phi1_f'] + ParaSample['phi2_f']), ParaSample['phi2_f'], ParaSample['beta_f']])).squeeze()

            # posterior sampling
            if RunningMCMCAlgorithm['sig2_f'] == 0:
                ParaSample['sig2_f'] = func.DrawVarianceIG(f_error, self.MergedPriorPara['sig2_f_shape'], self.MergedPriorPara['sig2_f_scale'])
            elif RunningMCMCAlgorithm['sig2_f'] == 1:
                ParaSample['sig2_f'], MHSample['sig2_f'], accept_idx_sig2_f, last_loglik['sig2_f'] = func.AdaptiveMH_IG(
                    Y, X, 'sig2_f', 'FC', ProposalType_idx, MHSample['sig2_f'], ParaSample,
                    PriorLatent, self.MergedPriorPara['sig2_f_shape'], self.MergedPriorPara['sig2_f_scale'], ProposalVar_0_IG, SizeInformation,
                    last_loglik['sig2_f'],
                )

            # Step 7.2: draw the slope coefficient of the financial cycle equation
            # posterior sampling
            if RunningMCMCAlgorithm['slope_f'] == 0:
                stationary_idc = 0
                while stationary_idc == 0:
                    f_coeff_draw, Mean_post, Variance_post = func.GenerateSlopeCoefficients(
                        Y_f, X_f, ParaSample['sig2_f'], self.MergedPriorPara['f_Inv_variance'], self.MergedPriorPara['f_Inv_variance_mean']
                    )
                    if all(abs(f_coeff_draw[1]) < 0.98):
                        stationary_idc = 1
            elif RunningMCMCAlgorithm['slope_f'] == 1:
                f_coeff_draw, MHSample['slope_f'], accept_idx_slope_f, last_loglik['slope_f'] = func.AdaptiveMH_Normal(
                    Y, X, 'f', 'FC', ProposalType_idx, MHSample['slope_f'], ParaSample,
                    PriorLatent, self.MergedPriorPara['f_mean'], self.MergedPriorPara['f_variance'],
                    self.MergedPriorPara['f_logdet_variance'], ProposalVar_0_Normal, SizeInformation,
                    last_loglik['slope_f'],
                )

            ParaSample['delta_f'] = f_coeff_draw[0]
            ParaSample['phi1_f'] = f_coeff_draw[1] - f_coeff_draw[2]
            ParaSample['phi2_f'] = f_coeff_draw[2]
            ParaSample['beta_f'] = f_coeff_draw[3]



            if itr > self.burnin:
                itr_idx = itr - self.burnin - 1

                # Store posterior samples of model parameters
                ParaStorage['phi_yc'][itr_idx] = np.array([ParaSample['phi1_yc'], ParaSample['phi2_yc']]).reshape(2)
                ParaStorage['beta_yc'][itr_idx] = ParaSample['beta_yc']
                ParaStorage['sig2_yc'][itr_idx] = ParaSample['sig2_yc']
                ParaStorage['sig2_yt'][itr_idx] = ParaSample['sig2_yt']
                ParaStorage['sig2_g'][itr_idx] = ParaSample['sig2_g']
                ParaStorage['phi_p'][itr_idx] = ParaSample['phi_p']
                ParaStorage['beta_p'][itr_idx] = ParaSample['beta_p']
                ParaStorage['sig2_p'][itr_idx] = ParaSample['sig2_p']
                ParaStorage['delta_z'][itr_idx] = ParaSample['delta_z']
                ParaStorage['phi_z'][itr_idx] = ParaSample['phi_z']
                ParaStorage['sig2_z'][itr_idx] = ParaSample['sig2_z']
                ParaStorage['delta_f'][itr_idx] = ParaSample['delta_f']
                ParaStorage['phi_f'][itr_idx] = np.array([ParaSample['phi1_f'], ParaSample['phi2_f']]).reshape(2)
                ParaStorage['beta_f'][itr_idx] = ParaSample['beta_f']
                ParaStorage['sig2_f'][itr_idx] = ParaSample['sig2_f']

                # Store posterior samples of latent variables
                LVariableStorage['yt'][itr_idx] = LVariableSample['yt'][1:]
                LVariableStorage['yc'][itr_idx] = LVariableSample['yc']
                LVariableStorage['g'][itr_idx] = LVariableSample['g'][1:]
                LVariableStorage['z'][itr_idx] = LVariableSample['z'][1:]
                LVariableStorage['rNatural'][itr_idx] = LVariableSample['rNatural'][1:]
                LVariableStorage['rCycle'][itr_idx] = LVariableSample['rCycle']

                # Store acceptance indices
                if RunningMCMCAlgorithm['sig2_g'] == 1:
                    Accept_idx['sig2_g'][itr_idx] = accept_idx_sig2_g
                if RunningMCMCAlgorithm['sig2_yt'] == 1:
                    Accept_idx['sig2_yt'][itr_idx] = accept_idx_sig2_yt
                if RunningMCMCAlgorithm['sig2_yc'] == 1:
                    Accept_idx['sig2_yc'][itr_idx] = accept_idx_sig2_yc
                if RunningMCMCAlgorithm['sig2_z'] == 1:
                    Accept_idx['sig2_z'][itr_idx] = accept_idx_sig2_z
                if RunningMCMCAlgorithm['sig2_p'] == 1:
                    Accept_idx['sig2_p'][itr_idx] = accept_idx_sig2_p
                if RunningMCMCAlgorithm['slope_yc'] == 1:
                    Accept_idx['slope_yc'][itr_idx] = accept_idx_slope_yc
                if RunningMCMCAlgorithm['slope_z'] == 1:
                    Accept_idx['slope_z'][itr_idx] = accept_idx_slope_z
                if RunningMCMCAlgorithm['slope_p'] == 1:
                    Accept_idx['slope_p'][itr_idx] = accept_idx_slope_p
                if RunningMCMCAlgorithm['sig2_f'] == 1:
                    Accept_idx['sig2_f'][itr_idx] = accept_idx_sig2_f
                if RunningMCMCAlgorithm['slope_f'] == 1:
                    Accept_idx['slope_f'][itr_idx] = accept_idx_slope_f


        Accept_ratio = {
                'sig2_g': np.mean(Accept_idx['sig2_g']),
                'sig2_yt': np.mean(Accept_idx['sig2_yt']),
                'sig2_yc': np.mean(Accept_idx['sig2_yc']),
                'sig2_z': np.mean(Accept_idx['sig2_z']),
                'sig2_p': np.mean(Accept_idx['sig2_p']),
                'slope_yc': np.mean(Accept_idx['slope_yc']),
                'slope_z': np.mean(Accept_idx['slope_z']),
                'slope_p': np.mean(Accept_idx['slope_p']),
                'sig2_f': np.mean(Accept_idx['sig2_f']),
                'slope_f': np.mean(Accept_idx['slope_f'])
            }

        # Post processing
        # Compute the mean, median, percentiles for model parameters
        lower_p_parameters = self.PercPara[0]
        upper_p_parameters = self.PercPara[1]

        # Initialize dictionaries to store statistics
        AvgParas = {}
        MedParas = {}
        PercLowerParas = {}
        PercUpperParas = {}

        # Get the field names from the ParaStorage dictionary
        fields_parameters = list(ParaStorage.keys())
        for fieldName in fields_parameters:
            # Access the corresponding data in ParaStorage
            posterior_sample_j = ParaStorage[fieldName]

            # Apply the sample_statistics function to the current data
            AvgParas[fieldName], MedParas[fieldName], PercLowerParas[fieldName], PercUpperParas[fieldName] = \
                func.sample_statistics(posterior_sample_j, lower_p_parameters, upper_p_parameters)

        # Summary dictionary for model parameters
        SummaryParas = {
            'Avg': AvgParas,
            'Med': MedParas,
            'PercLow': PercLowerParas,
            'PercUpper': PercUpperParas
        }

        # Compute the mean, median, percentiles for latent variables

        # Initialize dictionaries to store statistics
        AvgLV = {}
        MedLV = {}
        PercLowerLV = {}
        PercUpperLV = {}

        lower_p_LVariables = self.PercLV[0]
        upper_p_LVariables = self.PercLV[1]

        fields_LVariables = list(LVariableStorage.keys())
        for fieldName in fields_LVariables:
            # Access the corresponding data in LVariableStorage
            posterior_sample_j = LVariableStorage[fieldName]

            # Apply the sample_statistics function to the current data
            AvgLV[fieldName], MedLV[fieldName], PercLowerLV[fieldName], PercUpperLV[fieldName] = \
                func.sample_statistics(posterior_sample_j, lower_p_LVariables, upper_p_LVariables)

        # Summary dictionary for latent variables
        SummaryLV = {
            'Avg': AvgLV,
            'Med': MedLV,
            'PercLower': PercLowerLV,
            'PercUpper': PercUpperLV
        }
        return ParaStorage, SummaryParas, LVariableStorage, SummaryLV, Accept_ratio
