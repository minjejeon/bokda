import math as math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from . import functions as func
from .. import container_class


class HLW_Covid:
    """
    이 클래스는 HLW_Covid 모델을 사용하여 주어진 경제 데이터를 기반으로 중립금리를 추정합니다.
    이 모델은 기존의 HLW 모델에 Covid 를 통한 이분산성을 추가로 고려합니다.

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
            'slope_yt': 1,
            'kappa': 1
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
        - 'sig2_g_shape', 'sig2_g_scale', 'sig2_yt_shape', 'sig2_yt_scale', 'sig2_z_shape', 'sig2_z_scale', 'sig2_p_shape', 'sig2_p_scale', 'sig2_f_shape', 'sig2_f_scale', 
          'kappa20_shape', 'kappa20_scale', 'kappa21_shape', 'kappa21_scale', 'kappa22_shape', 'kappa22_scale': float
            해당 파라미터들의 inverse-gamma 분포의 shape 및 scale 파라미터입니다.
        - 'p_mean': numpy.ndarray, shape (2,)
            인플레이션 방정식의 기울기 계수에 대한 사전 평균입니다.
        - 'p_variance': numpy.ndarray, shape (2, 2)
            인플레이션 방정식의 기울기 계수에 대한 사전 분산 행렬입니다.
        - 'z_mean', 'z_variance' (ZSpec이 'sar_c' 또는 'sar'인 경우):
            이자율 z 구성 요소의 기울기 계수에 대한 사전 평균 및 분산입니다.
        - 'yt_mean': float
            실질 GDP yt 파라미터의 사전 분포의 평균입니다.
        - 'yt_variance': float
            실질 GDP yt 파라미터의 사전 분포의 분산입니다. 

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
            'gamma_yt': 0, 
            'kappa20': 10**2,
            'kappa21': 10**2,
            'kappa22': 10**2
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
            'slope_yt': 0.202,
            'kappa20': 0.588,
            'kappa21': 0.343,
            'kappa22': 0.392
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
        HLW_Covid 모델의 인스턴스를 초기화합니다.

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
            'slope_yt': 1,
            'kappa': 1
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
            'yt_mean': 0,
            'yt_variance': 10**2,
            'kappa20_shape': 5,
            'kappa20_scale': 4*(10**2),
            'kappa21_shape': 5,
            'kappa21_scale': 4*(10**2),
            'kappa22_shape': 5,
            'kappa22_scale': 4*(10**2)
        }

        # 중립금리모형에서 장기경제성장률을 제외한 latent vector 가 가지는 stochastic process 정의
        if ZSpec == 'rw':
            ParaSample = {'delta_z': 0, 'phi_z': 1}
            DefaultPriorPara['sig2_z_shape'] = 5
            DefaultPriorPara['sig2_z_scale'] = 4 * 0.1**2
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

        if ZSpec in ['sar_c', 'sar']:
            MergedPriorPara['z_logdet_variance'] = func.logdet(MergedPriorPara['z_variance'])
            MergedPriorPara['z_Inv_variance'] = np.linalg.inv(MergedPriorPara['z_variance'])
            MergedPriorPara['z_Inv_variance_mean'] = MergedPriorPara['z_Inv_variance'] @ MergedPriorPara['z_mean']

        MergedPriorPara['yt_logdet_variance'] = np.log(MergedPriorPara['yt_variance'])
        MergedPriorPara['yt_Inv_variance'] = 1/(MergedPriorPara['yt_variance'])
        MergedPriorPara['yt_Inv_variance_mean'] = MergedPriorPara['yt_Inv_variance'] * MergedPriorPara['yt_mean']

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
            'sig2_p': 0.1 ** 2,
            'sig2_z': 1 ** 2,
            'gamma_yt': 0, 
            'kappa20': 10**2,
            'kappa21': 10**2,
            'kappa22': 10**2
        }

        if ZSpec == 'rw':
            DefaultInitialPara['delta_z'] = 0
            DefaultInitialPara['phi_z'] = 1
            DefaultInitialPara['sig2_z'] = 0.1**2
        elif ZSpec in ['sar_c', 'sar']:
            DefaultInitialPara['delta_z'] = 0
            DefaultInitialPara['phi_z'] = 0.7
            DefaultInitialPara['sig2_z'] = 1**2

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
        주어진 데이터에 HLW_Covid 모델을 적합시킵니다.

        Parameters
        ----------
        data : pandas.DataFrame or numpy.ndarray
            입력 데이터 프레임 또는 numpy array.
            컬럼 구성은 반드시 다음과 같아야 합니다. [ln real gdp, inflation rate, real interest rate, stringency index, covid indicator]
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
        2. 모델을 적합시키기 위해 내부 메서드 `_fit_hlw_covid`를 호출합니다.
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
        col_size = 5
        if data.shape[1] != col_size:
            # just warn 
            print(f"> Warning: Input data must have {col_size} columns. (Column Order: [ln real gdp, inflation rate, real interest rate, stringency index, covid indicator])")


        # Extract the data columns (ln real GDP, (annualized) inflation rate, (annualized) real interest rate, financial cycle)
        if isinstance(data, pd.DataFrame):
            Data_full = data.to_numpy()
        else:
            Data_full = np.array(data)

        y_data = Data_full[:, 0]
        p_data = Data_full[:, 1]
        r_data = Data_full[:, 2]
        d_data = Data_full[:, 3]
        s_data = Data_full[:, 4]

        # Y와 X 벡터 생성
        Y = np.column_stack((y_data[4:], p_data[4:]))
        X = np.column_stack((
            y_data[3:-1], y_data[2:-2],  # y_data
            r_data[3:-1], r_data[2:-2],  # r_data
            p_data[3:-1], p_data[2:-2], p_data[1:-3], p_data[:-4],  # p_data
            d_data[4:], d_data[3:-1], d_data[2:-2]  # d_data
            ))
        
        y_data = y_data[4:]  # MATLAB의 5:end는 Python의 4:에 해당
        p_data = p_data[4:]
        r_data = r_data[4:]
        d_data = d_data[4:]
        s_data = s_data[4:]

        # 날짜 처리
        if dates is None:
            self.dates = np.arange(len(y_data)) + 4  # 앞에서 4칸 shift (4번 인덱스부터 len+3번까지)
        else:
            dates = np.array(dates)
            self.dates = dates[4:]

        ### FIT ###
        ParaStorage, SummaryParas, LVariableStorage, SummaryLV, Accept_ratio = self._fit_hlw_covid(Data_full, Y, X, y_data, p_data, r_data, d_data, s_data)

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

        model_name = "HLW_Covid"
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

    def _fit_hlw_covid(self, Data_full, Y, X, y_data, p_data, r_data, d_data, s_data):
        """
        HLW_Covid 모델을 사용하여 데이터에 적합하고, 파라미터 및 잠재 변수의 샘플을 생성합니다.

        Parameters
        ----------
        Data_full : numpy.ndarray, shape(N_T, 5)
            전체 입력 데이터 배열입니다.
        Y : numpy.ndarray, shape(N_T-4, 2)
            종속 변수 행렬입니다.
        X : numpy.ndarray, shape(N_T-4, 11)
            독립 변수 행렬입니다.
        y_data : numpy.ndarray, shape(N_T-4)
            출력 데이터 벡터입니다.
        p_data : numpy.ndarray, shape(N_T-4)
            인플레이션 데이터 벡터입니다.
        r_data : numpy.ndarray, shape(N_T-4)
            이자율 데이터 벡터입니다.
        d_data : numpy.ndarray, shape(N_T-4)
            Covid policy intensity 데이터 벡터입니다.
        s_data : numpy.ndarray, shape(N_T-4)
            volatility indicator 데이터 벡터입니다.



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
            'slope_yt': 1,
            'kappa': 1
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
            'gamma_yt': self.MergedInitialPara['gamma_yt'],
            'kappa20': self.MergedInitialPara['kappa20'],
            'kappa21': self.MergedInitialPara['kappa21'],
            'kappa22': self.MergedInitialPara['kappa22'],
        }


        # kappa_vec 초기화
        kappa_vec = np.ones(SizeInformation['N_T'])

        # 조건에 따른 인덱스 설정
        s1_idx = np.where(s_data == 1)[0]
        s2_idx = np.where(s_data == 2)[0]
        s3_idx = np.where(s_data == 3)[0]

        # kappa_vec 값 업데이트
        kappa_vec[s1_idx] = ParaSample['kappa20']
        kappa_vec[s2_idx] = ParaSample['kappa21']
        kappa_vec[s3_idx] = ParaSample['kappa22']


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
            'gamma_yt': np.zeros((self.numMCMC, 1)),
            'kappa': np.zeros((self.numMCMC, 3))
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
        ProposalVar_0_IG_small = (0.05)**2
        ProposalVar_0_IG_middle = (0.05)**2
        ProposalVar_0_IG_big = (0.05)**2
        ProposalVar_0_Normal = (0.05)**2
        
        MHSample = {
            'sig2_g': ParaSample['sig2_g'] * np.ones((self.MHSample_size, 1)),
            'sig2_yt': ParaSample['sig2_yt'] * np.ones((self.MHSample_size, 1)),
            'sig2_yc': ParaSample['sig2_yc'] * np.ones((self.MHSample_size, 1)),
            'sig2_z': ParaSample['sig2_z'] * np.ones((self.MHSample_size, 1)),
            'sig2_p': ParaSample['sig2_p'] * np.ones((self.MHSample_size, 1)),
            'slope_yc': np.tile([(ParaSample['phi1_yc'] + ParaSample['phi2_yc']), ParaSample['phi2_yc'], ParaSample['beta_yc']], (self.MHSample_size, 1)),
            'slope_p': np.tile([ParaSample['phi_p'], ParaSample['beta_p']], (self.MHSample_size, 1)),
            'kappa20': ParaSample['kappa20'] * np.ones((1000,1)),
            'kappa21': ParaSample['kappa21'] * np.ones((1000,1)),
            'kappa22': ParaSample['kappa22'] * np.ones((1000,1)),
            'slope_yt': ParaSample['gamma_yt'] * np.ones((1000,1)),
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
            'kappa20': np.zeros((self.numMCMC, 1)),
            'kappa21': np.zeros((self.numMCMC, 1)),
            'kappa22': np.zeros((self.numMCMC, 1)),
            'slope_yt': np.zeros((self.numMCMC, 1))
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
            "kappa20": None,
            "kappa21": None,
            "kappa22": None,
            "slope_yt": None
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
            # reverseStr = func.displayprogress(100 * itr / (self.numMCMC + self.burnin), reverseStr)

            # Step0: switch from Gibbs to user-specified MCMC algorithm
            if itr == 1:
                MHSample['sig2_g'] = ParaSample['sig2_g'] * np.ones((self.MHSample_size, 1)).flatten()
                MHSample['sig2_yt'] = ParaSample['sig2_yt'] * np.ones((self.MHSample_size, 1)).flatten()
                MHSample['sig2_yc'] = ParaSample['sig2_yc'] * np.ones((self.MHSample_size, 1)).flatten()
                MHSample['sig2_z'] = ParaSample['sig2_z'] * np.ones((self.MHSample_size, 1)).flatten()
                MHSample['sig2_p'] = ParaSample['sig2_p'] * np.ones((self.MHSample_size, 1)).flatten()
                MHSample['slope_yc'] = np.tile([(float(ParaSample['phi1_yc']) + float(ParaSample['phi2_yc'])), float(ParaSample['phi2_yc']), float(ParaSample['beta_yc'])], (self.MHSample_size, 1))
                MHSample['slope_p'] = np.tile([float(ParaSample['phi_p']), float(ParaSample['beta_p'])], (self.MHSample_size, 1))
                if self.ZSpec == 'sar_c':
                    MHSample['slope_z'] = np.tile([float(ParaSample['delta_z']), float(ParaSample['phi_z'])], (self.MHSample_size, 1))
                elif self.ZSpec == 'sar':
                    MHSample['slope_z'] = np.tile(np.array([ParaSample['phi_z']]).reshape(1,-1), (self.MHSample_size, 1))
                MHSample['slope_yt'] = ParaSample['gamma_yt'] * np.ones((self.MHSample_size, 1))
                MHSample['kappa'] = np.tile([float(ParaSample['kappa20']), float(ParaSample['kappa21']), float(ParaSample['kappa22'])], (self.MHSample_size, 1))

            elif itr == step_1_end_index:  # NOTE: burnin/2 으로 수정
                flag = "Step 2/3"
                RunningMCMCAlgorithm = self.MergedMCMCAlgorithm
                # update input MCMC samples for Adaptive MH from Gibbs outputs

                MHSample['sig2_g'] = ParaSample['sig2_g'] * np.ones((self.MHSample_size, 1)).flatten()
                MHSample['sig2_yt'] = ParaSample['sig2_yt'] * np.ones((self.MHSample_size, 1)).flatten()
                MHSample['sig2_yc'] = ParaSample['sig2_yc'] * np.ones((self.MHSample_size, 1)).flatten()
                MHSample['sig2_z'] = ParaSample['sig2_z'] * np.ones((self.MHSample_size, 1)).flatten()
                MHSample['sig2_p'] = ParaSample['sig2_p'] * np.ones((self.MHSample_size, 1)).flatten()
                MHSample['slope_yc'] = np.tile([(float(ParaSample['phi1_yc']) + float(ParaSample['phi2_yc'])), float(ParaSample['phi2_yc']), float(ParaSample['beta_yc'])], (self.MHSample_size, 1))
                MHSample['slope_p'] = np.tile([float(ParaSample['phi_p']), float(ParaSample['beta_p'])], (self.MHSample_size, 1))
                if self.ZSpec == 'sar_c':
                    MHSample['slope_z'] = np.tile([float(ParaSample['delta_z']), float(ParaSample['phi_z'])], (self.MHSample_size, 1))
                elif self.ZSpec == 'sar':
                    MHSample['slope_z'] = np.tile(np.array([ParaSample['phi_z']]).reshape(1,-1), (self.MHSample_size, 1))
                MHSample['slope_yt'] = ParaSample['gamma_yt'] * np.ones((self.MHSample_size, 1))
                MHSample['kappa'] = np.tile([float(ParaSample['kappa20']), float(ParaSample['kappa21']), float(ParaSample['kappa22'])], (self.MHSample_size, 1))
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
            bt_draw = func.GenerateLatentVariables_Covid(Y,X,s_data,ParaSample,PriorLatent,SizeInformation)

            # compute latent variables
            LVariableSample = {
                'yt': bt_draw[:, 0],  # output trend (t = 0 ~T)
                'yc': (y_data - bt_draw[1:, 0] - ParaSample['gamma_yt']* d_data).squeeze(),  # output cycle (t = 1 ~T)
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
                ParaSample['sig2_g'], MHSample['sig2_g'], accept_idx_sig2_g, last_loglik['sig2_g'] = func.AdaptiveMH_IG(
                    Y, X, 'sig2_g', 'Covid', ProposalType_idx, MHSample['sig2_g'], ParaSample,
                    PriorLatent, self.MergedPriorPara['sig2_g_shape'], self.MergedPriorPara['sig2_g_scale'], ProposalVar_0_IG_small, SizeInformation, 
                    s_data=s_data,
                    last_loglike=last_loglik['sig2_g']
                )

            # Step 3: draw the variance of the shock to the output trend (yt)
            if RunningMCMCAlgorithm['sig2_yt'] == 0:
                # compute data
                yt_error = (LVariableSample['yt'][1:] - LVariableSample['yt'][:-1]) - LVariableSample['g'][:-1]
                # posterior sampling
                ParaSample['sig2_yt'] = func.DrawVarianceIG(yt_error, self.MergedPriorPara['sig2_yt_shape'], self.MergedPriorPara['sig2_yt_scale'])

            elif RunningMCMCAlgorithm['sig2_yt'] == 1:
                ParaSample['sig2_yt'], MHSample['sig2_yt'], accept_idx_sig2_yt, last_loglik['sig2_yt'] = func.AdaptiveMH_IG(
                    Y, X, 'sig2_yt', 'Covid', ProposalType_idx, MHSample['sig2_yt'], ParaSample,
                    PriorLatent, self.MergedPriorPara['sig2_yt_shape'], self.MergedPriorPara['sig2_yt_scale'], ProposalVar_0_IG_middle, SizeInformation, 
                    s_data=s_data,
                    last_loglike=last_loglik['sig2_yt']
                )
            
            # Step 4.1: draw the variance of the shock to the output cycle (yc)
            # compute data
            rCycle_mean = (LVariableSample['rCycle'][1:-1] + LVariableSample['rCycle'][:-2]) / 2
            Y_yc = LVariableSample['yc'][2:]
            X_yc = np.column_stack((LVariableSample['yc'][1:-1], (LVariableSample['yc'][:-2] - LVariableSample['yc'][1:-1]), rCycle_mean))
            yc_error = Y_yc - (X_yc @ np.array([(ParaSample['phi1_yc'] + ParaSample['phi2_yc']), ParaSample['phi2_yc'], ParaSample['beta_yc']]).reshape(-1,1)).squeeze()

            # yc_error_kappa 계산 (re-scaling yc_error)
            yc_error_kappa = yc_error / np.sqrt(kappa_vec[2:])  # 3:end → 2: (Python은 0-based indexing)

            # Y_yc_kappa 계산 (re-scaling Y_yc)
            Y_yc_kappa = Y_yc / np.sqrt(kappa_vec[2:])

            # X_yc_kappa 계산 (re-scaling X_yc)
            X_yc_kappa = X_yc / np.column_stack((np.sqrt(kappa_vec[2:]), np.sqrt(kappa_vec[2:]), np.sqrt(kappa_vec[2:])))

            # posterior sampling
            if RunningMCMCAlgorithm['sig2_yc'] == 0:
                ParaSample['sig2_yc'] = func.DrawVarianceIG(yc_error_kappa, self.MergedPriorPara['sig2_yc_shape'], self.MergedPriorPara['sig2_yc_scale'])
            elif RunningMCMCAlgorithm['sig2_yc'] == 1:
                ParaSample['sig2_yc'], MHSample['sig2_yc'], accept_idx_sig2_yc, last_loglik["sig2_yc"] = func.AdaptiveMH_IG(
                    Y, X, 'sig2_yc', 'Covid', ProposalType_idx, MHSample['sig2_yc'], ParaSample,
                    PriorLatent, self.MergedPriorPara['sig2_yc_shape'], self.MergedPriorPara['sig2_yc_scale'], ProposalVar_0_IG_middle, SizeInformation, 
                    s_data=s_data,
                    last_loglike=last_loglik["sig2_yc"]
                )
            
            # Step 4.2: draw the slope coefficient of the output cycle equation
            # posterior sampling
            if RunningMCMCAlgorithm['slope_yc'] == 0:
                stationary_idc = 0
                while stationary_idc == 0:
                    yc_coeff_draw, Mean_post, Variance_post = func.GenerateSlopeCoefficients(Y_yc_kappa, X_yc_kappa, ParaSample['sig2_yc'], self.MergedPriorPara['yc_Inv_variance'], self.MergedPriorPara['yc_Inv_variance_mean'])
                    if all(abs(yc_coeff_draw[0]) < 0.98): 
                        stationary_idc = 1
            elif RunningMCMCAlgorithm['slope_yc'] == 1:
                yc_coeff_draw, MHSample['slope_yc'], accept_idx_slope_yc, last_loglik['slope_yc'] = func.AdaptiveMH_Normal(
                    Y, X, 'yc', 'Covid', ProposalType_idx, MHSample['slope_yc'], ParaSample,
                    PriorLatent, self.MergedPriorPara['yc_mean'], self.MergedPriorPara['yc_variance'], self.MergedPriorPara['yc_logdet_variance'], ProposalVar_0_Normal, SizeInformation, 
                    s_data=s_data,
                    last_loglike=last_loglik['slope_yc']
                )
            
            ParaSample['phi1_yc'] = yc_coeff_draw[0] - yc_coeff_draw[1]
            ParaSample['phi2_yc'] = yc_coeff_draw[1]
            ParaSample['beta_yc'] = yc_coeff_draw[2]

            # Step 5. draw the Covid policy effect (gamma_yt)
            if RunningMCMCAlgorithm['slope_yt'] == 0:
                # Step 1: yt_sample 및 Y_gamma 계산
                yt_sample = LVariableSample['yt'][1:]  # 2:end → 1: (Python은 0-based indexing)
                Y_gamma_temp1 = (y_data[2:] - ParaSample['phi1_yc'] * y_data[1:-1] - ParaSample['phi2_yc'] * y_data[:-2])
                Y_gamma_temp2 = (yt_sample[2:] - ParaSample['phi1_yc'] * yt_sample[1:-1] - ParaSample['phi2_yc'] * yt_sample[:-2])
                Y_gamma_temp3 = ParaSample['beta_yc'] * rCycle_mean
                Y_gamma = Y_gamma_temp1 - Y_gamma_temp2 - Y_gamma_temp3

                # Step 2: X_gamma 계산
                X_gamma = (d_data[2:] - ParaSample['phi1_yc'] * d_data[1:-1] - ParaSample['phi2_yc'] * d_data[:-2])

                # Step 3: 분산 스케일링
                Y_gamma_kappa = Y_gamma / np.sqrt(kappa_vec[2:])
                X_gamma_kappa = X_gamma / np.sqrt(kappa_vec[2:])

                # Step 4: Gibbs 샘플링
                temp_Variance_posterior = 1/(self.MergedPriorPara['yt_Inv_variance'] + (X_gamma_kappa.T @ X_gamma_kappa) / ParaSample['sig2_yc'])
                temp_Mean_posterior = (temp_Variance_posterior * (self.MergedPriorPara['yt_Inv_variance_mean'] + (X_gamma_kappa.T @ Y_gamma_kappa) / ParaSample['sig2_yc'])).reshape(-1,1)
                temp_Coefficients_draw = temp_Mean_posterior + np.sqrt(temp_Variance_posterior).T * np.random.randn(temp_Mean_posterior.shape[0],1)

                ParaSample['gamma_yt'] = temp_Coefficients_draw
                Mean_post = temp_Mean_posterior
                Variance_post = temp_Variance_posterior


            elif RunningMCMCAlgorithm['slope_yt'] == 1:
                # Step 5: MH 샘플링
                ParaSample['gamma_yt'], MHSample['slope_yt'], accept_idx_slope_yt, last_loglik['slope_yt'] = func.AdaptiveMH_Normal(
                    Y, X, 'yt', 'Covid', ProposalType_idx, MHSample['slope_yt'], ParaSample,
                    PriorLatent, self.MergedPriorPara['yt_mean'], self.MergedPriorPara['yt_variance'], self.MergedPriorPara['yt_logdet_variance'],
                    ProposalVar_0_Normal, SizeInformation, 
                    s_data=s_data,
                    last_loglike=last_loglik['slope_yt']
                )


            # Step 6.1: draw the variance of the shock to the interest rate component z
            # compute data
            Y_z = LVariableSample['z'][1:]
            X_z = np.column_stack((np.ones_like(Y_z), LVariableSample['z'][:-1]))
            z_error = Y_z - X_z @ np.array([ParaSample['delta_z'], ParaSample['phi_z']])
            # posterior sampling
            if RunningMCMCAlgorithm['sig2_z'] == 0:
                ParaSample['sig2_z'] = func.DrawVarianceIG(z_error, self.MergedPriorPara['sig2_z_shape'], self.MergedPriorPara['sig2_z_scale'])
            elif RunningMCMCAlgorithm['sig2_z'] == 1:
                ParaSample['sig2_z'], MHSample['sig2_z'], accept_idx_sig2_z, last_loglik['sig2_z'] = func.AdaptiveMH_IG(
                    Y, X, 'sig2_z', 'Covid', ProposalType_idx, MHSample['sig2_z'], ParaSample,
                    PriorLatent, self.MergedPriorPara['sig2_z_shape'], self.MergedPriorPara['sig2_z_scale'], ProposalVar_0_IG_small, SizeInformation, 
                    s_data=s_data,
                    last_loglike=last_loglik['sig2_z']
                )

            # Step 6.2: draw the slope coefficient of the interest rate component z
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
                        Y, X, 'zc', 'Covid', ProposalType_idx, MHSample['slope_z'], ParaSample,
                        PriorLatent, self.MergedPriorPara['z_mean'], self.MergedPriorPara['z_variance'], self.MergedPriorPara['z_logdet_variance'], ProposalVar_0_Normal, SizeInformation, 
                        s_data=s_data,
                        last_loglike=last_loglik['slope_z']
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
                        Y, X, 'z', 'Covid', ProposalType_idx, MHSample['slope_z'], ParaSample,
                        PriorLatent, self.MergedPriorPara['z_mean'], self.MergedPriorPara['z_variance'], self.MergedPriorPara['z_logdet_variance'], ProposalVar_0_Normal, SizeInformation, 
                        s_data=s_data,
                        last_loglike=last_loglik['slope_z']
                    )
                ParaSample['delta_z'] = 0
                ParaSample['phi_z'] = z_coeff_draw[0]  # z_coeff_draw = [value,]

            # Step 7.1: draw the variance of the shock to the inflation rate (p)
            # compute data
            psum_lag = np.mean(X[1:, 5:8], axis=1)
            Y_p = p_data[1:] - psum_lag
            X_p = np.column_stack((X[1:, 4] - psum_lag, LVariableSample['yc'][:-1]))
            p_error = Y_p - X_p @ np.array([ParaSample['phi_p'], ParaSample['beta_p']]).squeeze()

            # p_error_kappa 계산 (re-scaling p_error)
            p_error_kappa = p_error / np.sqrt(kappa_vec[1:])  # 2:end → 1: (Python은 0-based indexing)

            # Y_p_kappa 계산 (re-scaling Y_p)
            Y_p_kappa = Y_p / np.sqrt(kappa_vec[1:])

            # X_p_kappa 계산 (re-scaling X_p)
            X_p_kappa = X_p / np.sqrt(kappa_vec[1:])[:, np.newaxis]  # 열 방향 브로드캐스팅을 위해 np.newaxis 추가

            # posterior sampling
            if RunningMCMCAlgorithm['sig2_p'] == 0:
                ParaSample['sig2_p'] = func.DrawVarianceIG(p_error_kappa, self.MergedPriorPara['sig2_p_shape'], self.MergedPriorPara['sig2_p_scale'])
            elif RunningMCMCAlgorithm['sig2_p'] == 1:
                ParaSample['sig2_p'], MHSample['sig2_p'], accept_idx_sig2_p, last_loglik['sig2_p'] = func.AdaptiveMH_IG(
                    Y, X, 'sig2_p', 'Covid', ProposalType_idx, MHSample['sig2_p'], ParaSample,
                    PriorLatent, self.MergedPriorPara['sig2_p_shape'], self.MergedPriorPara['sig2_p_scale'], ProposalVar_0_IG_small, SizeInformation, 
                    s_data=s_data,
                    last_loglike=last_loglik['sig2_p']
                )

            # Step 7.2: draw the slope coefficient of the inflation rate equation
            # posterior sampling
            if RunningMCMCAlgorithm['slope_p'] == 0:
                stationary_idc = 0
                while stationary_idc == 0:
                    p_coeff_draw, Mean_post, Variance_post = func.GenerateSlopeCoefficients(Y_p_kappa, X_p_kappa, ParaSample['sig2_p'], self.MergedPriorPara['p_Inv_variance'], self.MergedPriorPara['p_Inv_variance_mean'])
                    if all(abs(p_coeff_draw[0]) < 0.98):
                        stationary_idc = 1
            elif RunningMCMCAlgorithm['slope_p'] == 1:
                p_coeff_draw, MHSample['slope_p'], accept_idx_slope_p, last_loglik['slope_p'] = func.AdaptiveMH_Normal(
                    Y, X, 'p', 'Covid', ProposalType_idx, MHSample['slope_p'], ParaSample,
                    PriorLatent, self.MergedPriorPara['p_mean'], self.MergedPriorPara['p_variance'], self.MergedPriorPara['p_logdet_variance'], ProposalVar_0_Normal, SizeInformation, 
                    s_data=s_data,
                    last_loglike=last_loglik['slope_p']
                )
            ParaSample['phi_p'] = p_coeff_draw[0]
            ParaSample['beta_p'] = p_coeff_draw[1]


            # Step 8: draw the variance scale parameters
            if RunningMCMCAlgorithm['kappa'] == 0:
                # Step 1: inflation rate error 계산 (t = 2 ~ T)
                p_error = Y_p - (X_p @ np.array([ParaSample['phi_p'], ParaSample['beta_p']])).squeeze()
                p_error_sig2 = p_error / np.sqrt(ParaSample['sig2_p'])
                p_error_sig2 = np.concatenate([np.nan * np.ones(1), p_error_sig2])  # nan 추가 (time period 맞추기)

                # Step 2: output cycle error 계산 (t = 3 ~ T)
                yc_error = Y_yc - (X_yc @ np.array([ParaSample['phi1_yc'] + ParaSample['phi2_yc'], ParaSample['phi2_yc'], ParaSample['beta_yc']])).squeeze()
                yc_error_sig2 = yc_error / np.sqrt(ParaSample['sig2_yc'])
                yc_error_sig2 = np.concatenate([np.nan * np.ones(2), yc_error_sig2])  # nan 추가 (time period 맞추기)

                # Step 3: kappa에 대한 에러 벡터 생성
                error_kappa20 = np.concatenate([yc_error_sig2[s1_idx], p_error_sig2[s1_idx]])
                error_kappa21 = np.concatenate([yc_error_sig2[s2_idx], p_error_sig2[s2_idx]])
                error_kappa22 = np.concatenate([yc_error_sig2[s3_idx], p_error_sig2[s3_idx]])

                # Step 4: posterior sampling을 위한 Variance IG 분포 샘플링
                ParaSample['kappa20'] = func.DrawVarianceIG(error_kappa20, self.MergedPriorPara['kappa20_shape'], self.MergedPriorPara['kappa20_scale'])
                ParaSample['kappa21'] = func.DrawVarianceIG(error_kappa21, self.MergedPriorPara['kappa21_shape'], self.MergedPriorPara['kappa21_scale'])
                ParaSample['kappa22'] = func.DrawVarianceIG(error_kappa22, self.MergedPriorPara['kappa22_shape'], self.MergedPriorPara['kappa22_scale'])

            elif RunningMCMCAlgorithm['kappa'] == 1:
                # Step 5: MH 샘플링
                ParaSample['kappa20'], MHSample['kappa20'], accept_idx_kappa20, last_loglik['kappa20'] = func.AdaptiveMH_IG(
                    Y, X, 'kappa20', 'Covid', ProposalType_idx, MHSample['kappa20'], ParaSample,
                    PriorLatent, self.MergedPriorPara['kappa20_shape'], self.MergedPriorPara['kappa20_scale'], ProposalVar_0_IG_big,
                    SizeInformation, s_data=s_data, last_loglike=last_loglik['kappa20'])

                ParaSample['kappa21'], MHSample['kappa21'], accept_idx_kappa21, last_loglik['kappa21'] = func.AdaptiveMH_IG(
                    Y, X, 'kappa21', 'Covid', ProposalType_idx, MHSample['kappa21'], ParaSample,
                    PriorLatent, self.MergedPriorPara['kappa21_shape'], self.MergedPriorPara['kappa21_scale'], ProposalVar_0_IG_big,
                    SizeInformation, s_data=s_data, last_loglike=last_loglik['kappa21'])

                ParaSample['kappa22'], MHSample['kappa22'], accept_idx_kappa22, last_loglik['kappa22'] = func.AdaptiveMH_IG(
                    Y, X, 'kappa22', 'Covid', ProposalType_idx, MHSample['kappa22'], ParaSample,
                    PriorLatent, self.MergedPriorPara['kappa22_shape'], self.MergedPriorPara['kappa22_scale'], ProposalVar_0_IG_big,
                    SizeInformation, s_data=s_data, last_loglike=last_loglik['kappa22'])

            # 결과 저장
            kappa_vec = np.ones(SizeInformation['N_T'])
            kappa_vec[s1_idx] = ParaSample['kappa20']
            kappa_vec[s2_idx] = ParaSample['kappa21']
            kappa_vec[s3_idx] = ParaSample['kappa22']



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
                ParaStorage['gamma_yt'][itr_idx] = ParaSample['gamma_yt']
                ParaStorage['kappa'][itr_idx] = np.array([ParaSample['kappa20'], ParaSample['kappa21'], ParaSample['kappa22']]).reshape(3)

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
                if RunningMCMCAlgorithm['slope_yt'] == 1:
                    Accept_idx['slope_yt'][itr_idx] = accept_idx_slope_yt
                if RunningMCMCAlgorithm['kappa'] == 1:
                    Accept_idx['kappa20'][itr_idx] = accept_idx_kappa20
                    Accept_idx['kappa21'][itr_idx] = accept_idx_kappa21
                    Accept_idx['kappa22'][itr_idx] = accept_idx_kappa22


        Accept_ratio = {
                'sig2_g': np.mean(Accept_idx['sig2_g']),
                'sig2_yt': np.mean(Accept_idx['sig2_yt']),
                'sig2_yc': np.mean(Accept_idx['sig2_yc']),
                'sig2_z': np.mean(Accept_idx['sig2_z']),
                'sig2_p': np.mean(Accept_idx['sig2_p']),
                'slope_yc': np.mean(Accept_idx['slope_yc']),
                'slope_z': np.mean(Accept_idx['slope_z']),
                'slope_p': np.mean(Accept_idx['slope_p']),
                'slope_yt': np.mean(Accept_idx['slope_yt']),
                'kappa20': np.mean(Accept_idx['kappa20']),
                'kappa21': np.mean(Accept_idx['kappa21']),
                'kappa22': np.mean(Accept_idx['kappa22'])
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
