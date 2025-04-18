# LBVAR Class
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, Optional
import scipy.optimize as optimize

from . import functions_adaptive as fad
from . import functions as func

from .. import container_class


class LBVAR_Adaptive:
    """
    Large Bayesian Vector Autoregression (LBVAR) 모델 클래스

    Attributes
    ----------
    trend : int
        트렌드 옵션 (1: 상수항만 포함, 2: 선형 추세 포함, 3: 이차 추세 포함)
    p : int
        시차(lag) 수
    forecast_period : int
        예측 기간
    ndraws : int
        Posterior 분포에서의 샘플 수
    burnin : int
        초기 번인(burn-in) 기간의 길이
    hyperparameters : dictionary
        하이퍼파라미터 딕셔너리
    verbose : bool
        출력 여부

    Methods
    -------
    fit(data, rv_list=[])
        모델을 데이터에 맞게 적합시킵니다.
    forecast(forecast_period=4, pereb=0.68)
        예측을 수행합니다.
    print_forecast(plot_index=None, plot_all=True, column_names=[])
        예측 결과를 출력합니다.
    """

    def __init__(self,
                trend: Union[int, str] = 1,
                p: int = 4,
                ndraws: int = 10000,
                burnin: int = 1000,
                hyperparameters: dict = {},
                verbose: bool = False):
        """
        LBVAR 모델의 초기 설정을 수행합니다.

        Parameters
        ----------

        trend : int or str, optional
            트렌드 옵션 (1: 상수항만 포함, 2: 선형 추세 포함, 3: 이차 추세 포함), 기본값은 1.
            - 1 or 'C': 상수항만 포함합니다.
            - 2 or 'L': 선형 추세를 포함합니다.
            - 3 or 'Q': 이차 추세를 포함합니다.

        p : int, optional
            시차(lag) 수, 기본값은 4.
            - 분기 데이터의 경우 일반적으로 4를 사용합니다.
            - 월간 데이터의 경우 일반적으로 12를 사용합니다.

        ndraws : int, optional
            Posterior 분포에서의 샘플 수, 기본값은 10000.
        
        burnin : int, optional
            Burn-in size, 기본값은 1000.
        
        hyperparameters : dict, optional
            하이퍼파라미터 설정값, 입력이 없을 경우 기본값 사용
            Default: {c_11: 1.0, c_21: 1 / (0.04), c_12: 1.0, c_22: 1 / (0.04)**2, 
                      d_1: 1.0, d_2: 1.0, a_hi: 0.0, v_hi: 10.0, nu_hi: 5.0, S_hi: 0.04, 
                      kappa_3: 1.0, kappa_4: 100.0}

        verbose : bool, optional
            출력 여부, 기본값은 False.
            - True로 설정하면 모델 적합 및 최적화 과정에서 자세한 정보를 출력합니다.
        """
        self.trend = func.translate_string_arg_to_int(trend, category="trend")
        self.p = p
        self.ndraws = ndraws
        self.burnin = burnin
        self.verbose = verbose


        # 클래스 내부에서 사용할 변수들 초기화
        self.Raw_Data = container_class.Container()
        self.Parameters = container_class.Container()
        self.Prior = container_class.Container()
        self.Draw = container_class.Container()
        self.Forecast_Results = container_class.Container()

        # 모델 적합 여부 플래그
        self.is_fitted = False


        # Adding keys and values one by one
        # 인자를 받지 않는경우 기본값으로 사용
        if hyperparameters == {}:
            hyperparameters["c_11"] = 1.0
            hyperparameters["c_21"] = 1 / (0.04)
            hyperparameters["c_12"] = 1.0
            hyperparameters["c_22"] = 1 / (0.04)**2
            hyperparameters["d_1"] = 1.0
            hyperparameters["d_2"] = 1.0
            hyperparameters["a_hi"] = 0.0
            hyperparameters["v_hi"] = 10.0
            hyperparameters["nu_hi"] = 5.0
            hyperparameters["S_hi"] = 0.04
            hyperparameters["kappa_3"] = 1.0
            hyperparameters["kappa_4"] = 100.0
        self.hyperparameters = hyperparameters


        # PRINT: model init
        if self.verbose:
            print("> Initializing Model...")
            print(f"  - Trend: {self.trend}")
            print(f"  - p(lag): {self.p}")
            print(f"  - Number of Draws: {self.ndraws}")
            print(f"  - Burn-in: {self.burnin}")
            print(f"  - Verbose: {self.verbose}")


    def fit(self, data: Union[np.ndarray, pd.DataFrame], rv_list: list = []):
        """
        LBVAR 모델을 데이터에 맞게 적합시킵니다.

        Parameters
        ----------
        data : np.ndarray 또는 pd.DataFrame
            모델에 사용할 입력 데이터입니다.
            - 행은 시간(t)을 나타내며, 첫 번째 행은 가장 과거 시점의 데이터이고 마지막 행은 가장 최근 시점의 데이터입니다.
            - 열은 변수를 나타내며, 각 열이 하나의 변수에 해당합니다.

        rv_list : list of int, optional
            비정상(non-stationary)적인 변수를 나타내는 변수를 지정하는 리스트입니다. 순서는 1부터 시작합니다.
            - 예를 들어, 5개의 변수를 사용하는 모형에서 2번과 4번 변수가 비정상적인 경우 rv_list는 [2, 4]로 입력해야 합니다.
            - 모든 변수가 정상적인 경우 rv_list는 빈 리스트([])로 입력합니다.
            - 기본값은 []입니다.

        Returns
        -------
        self : LBVAR
            적합된 모델 객체 자신을 반환합니다.

        """
        # 데이터가 DataFrame이면 numpy 배열로 변환
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()

        # 데이터 체크
        if not isinstance(data, np.ndarray):
            raise ValueError("data는 numpy.ndarray 또는 pandas.DataFrame이어야 합니다.")

        # 데이터 저장
        self.Raw_Data.Set = np.matrix(data)  # NOTE: np.matrix로 변경

        # 변수 순서를 담은 리스트를 index를 담은 리스트로 변환 eg. [1, 2, 3] -> [0, 1, 2]
        self.Parameters.RV_list = [r - 1 for r in rv_list] 

        # 모델 파라미터 설정
        self.Parameters.p = self.p
        self.Parameters.n = self.Raw_Data.Set.shape[0]
        self.Parameters.nvar = self.Raw_Data.Set.shape[1]
        self.Parameters.T = self.Parameters.n - self.Parameters.p
        self.Parameters.k = self.Parameters.nvar * self.Parameters.p
        self.Parameters.beta = 1
        self.Parameters.ndraws = self.ndraws

        self.Parameters.Trend = self.trend
        self.Parameters.burnin = self.burnin

        # 외생 변수 생성
        self.Raw_Data = fad.EXOv_total_maker(self.Parameters, self.Raw_Data)

        self.Parameters.c = self.Raw_Data.EXOv.shape[1]
        self.Parameters.num_of_parameter = self.Parameters.nvar * self.Parameters.p + self.Parameters.c
        
        # LBVAR 변수 생성
        self.Raw_Data = fad.LBVAR_variable_maker(self.Raw_Data, self.Parameters)

        # Prior 설정 및 Posterior 추정
        self._fit_adaptive()

        self.is_fitted = True

        return self


    def forecast(self, forecast_period: int = 4, pereb: float = 0.16):
        """
        적합된 모델을 사용하여 예측을 수행합니다.

        Parameters
        ----------
        forecast_period : int, optional
            예측 기간 (기본값: 4)
        pereb : float, optional
            예측 구간 설정값 (기본값: 0.16(68% 신용구간))

        Returns
        -------
        Forecast_Results : dict
            예측 결과를 담은 딕셔너리
        """
        if not self.is_fitted:
            raise ValueError("> 모델이 적합되지 않았습니다. 먼저 fit 메서드를 호출하세요.")

        if self.verbose:
            print("> Forecast Start")

        self.Parameters.pereb = pereb
        self.forecast_period = forecast_period
        self.Parameters.Forecast_period = self.forecast_period

        if self.verbose:
            print("> Reduced Transform")
        Draw = fad.Reduced_Transform(self.Parameters, self.Draw)
        if self.verbose:
            print("> Forecasting")
        self.Forecast_Results = fad.Forecast_function(self.Raw_Data, self.Parameters, Draw, verbose=self.verbose)

        if self.verbose:
            print("> Done")

        return self.Forecast_Results

    
    def print_forecast(self, plot_index=None, plot_all=True, column_names=[]):
        """
        eg. plot_index = [1, 2, 3]
        """
        column_names = list(column_names)
        # Plotting
        forecast_results = self.Forecast_Results
        mean = forecast_results.Mean
        variables = mean.shape[1]
        x = np.arange(mean.shape[0]) + 1 # Time (예측하는 기수 크기)

        # Extract data
        up = forecast_results.UP
        down = forecast_results.DOWN
        # Plot Confidence Interval for each variable (column)

        for var in range(variables):
            if plot_index is None or (var + 1) in plot_index:
                plt.figure()
                plt.fill_between(x, down[:, var], up[:, var], color=[0.9, 0.9, 0.9], label='Interval')
                if len(column_names) > 0:
                    plt.plot(x, mean[:, var], "r--", linewidth=1, label=f'{column_names[var]}')
                    plt.title(f'Forecast Result for {column_names[var]}')
                else:
                    plt.plot(x, mean[:, var], "r--", linewidth=1, label=f'{var + 1}')
                    plt.title(f'Forecast Result for Variable {var + 1}')
                plt.xlabel('Time')
                plt.xticks(x)
                plt.ylabel('Values')
                plt.legend()
                plt.grid()
                plt.show()

        if plot_all:
            # 같이 그리는 것
            temp_df = pd.DataFrame(mean, index=x)
            temp_df.columns = [f'{col}' for col in column_names]
            temp_df.plot(kind='line', marker='.')

            plt.title('Forecast Result for All Variables')
            plt.xlabel('Time')
            plt.xticks(x)
            plt.ylabel('Mean')
            plt.legend()
            plt.grid()
            plt.show()


    # ==============================

    def _fit_adaptive(self):
        """
        """
        if self.verbose:
            print("> Prior Making (Adaptive)")
        self.Prior = fad.Prior_Maker(self.Raw_Data, self.Parameters, self.Prior, self.hyperparameters)
        if self.verbose:
            print("> Posterior Draw (Adaptive)")
        self.Draw = fad.mcmc_mng(self.Raw_Data, self.Parameters, self.Prior, self.hyperparameters)

