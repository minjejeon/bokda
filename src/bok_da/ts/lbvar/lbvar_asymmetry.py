# LBVAR Asymmetry Class
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, Optional
import scipy.optimize as optimize

from statsmodels.tools.numdiff import approx_hess
from threadpoolctl import threadpool_limits

# Import the asymmetry functions as fas
from . import functions_asymmetry as fas
# Import the common functions as func
from . import functions as func

from .. import container_class

class LBVAR_Asymmetry:
    """
    Large Bayesian Vector Autoregression (LBVAR) 모델 클래스 - Asymmetry Version

    Attributes
    ----------
    trend : int
        트렌드 옵션 (1: 상수항만 포함, 2: 선형 추세 포함, 3: 이차 추세 포함)
    p : int
        시차(lag) 수
    ndraws : int
        Posterior 분포에서의 샘플 수
    hyperparameters : np.ndarray
        하이퍼파라미터 배열
    hyperparameter_opt : int
        하이퍼파라미터 최적화 옵션 (0: 사용자가 선택한 하이퍼파라미터 사용, 1: Marginal Likelihood Optimization, 2: MCMC를 통한 하이퍼파라미터 선택)
    optimization_method : int
        최적화 방법 (0: Scipy를 통한 최적화, 1: Random Search를 통한 최적화)
    random_search_options : dict
        Random Search 최적화에 사용할 옵션을 지정하는 딕셔너리('K': 최대 반복 횟수, 'P': 각 반복에서 생성할 후보의 수, 'alpha': 스텝 크기)
    hyperparamter_mcmc_options : dict
        하이퍼파라미터 MCMC 최적화에 사용할 옵션을 지정하는 딕셔너리('n_draws': MCMC 샘플 수, 'n_burnin': 번인 기간)
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
                hyperparameters: Optional[np.ndarray] = None,
                hyperparameter_opt: Union[int, str] = 0,
                optimization_method: Union[int, str] = 0,
                random_search_options: Optional[dict] = {'K': 1000, 'P': 1000, 'alpha': 1},
                hyperparameter_mcmc_options: Optional[dict] = {'n_draws': 10000, 'n_burnin': 1000},
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

        hyperparameters : np.ndarray, optional
            하이퍼파라미터 배열로, 분석에 사용할 하이퍼파라미터입니다.
            - np.array 형태로 입력되어야 합니다.
            - 'asymmetric' prior의 경우 기본값은 np.array([0.05, 0.005, 100])입니다.

        hyperparameter_opt : int, optional
            하이퍼파라미터 최적화 옵션, 기본값은 0.
            - 0 or 'pass': 사용자가 선택한 하이퍼파라미터를 사용합니다.
            - 1 or 'mlo': Marginal Likelihood Optimization을 수행합니다.
            - 2 or 'mcmc': MCMC를 통한 하이퍼파라미터 선택을 수행합니다.

        optimization_method : int, optional
            최적화 방법, 기본값은 0.
            - 0 or 'scipy': Scipy를 통한 최적화를 수행합니다.
            - 1 or 'rs': Random Search를 통한 최적화를 수행합니다.
            - 이 옵션은 hyperparameter_opt가 1 또는 2인 경우에만 사용됩니다.

        random_search_options : dict, optional
            Random Search 최적화에 사용할 옵션 딕셔너리입니다. 기본값은 {'K': 1000, 'P': 1000, 'alpha': 1}.
            - 'K': 최대 반복 횟수입니다.
            - 'P': 각 반복에서 생성할 후보의 수입니다.
            - 'alpha': 스텝 크기입니다. (float 또는 'diminishing')

            예시:
            random_search_options = {'K': 500, 'P': 500, 'alpha': 0.5}
        
        hyperparameter_mcmc_options : dict, optional
            hyperparameter MCMC 최적화에 사용할 옵션 딕셔너리입니다. 
            기본값은 {'n_draws': 10000, 'n_burnin': 1000}.

        verbose : bool, optional
            출력 여부, 기본값은 False.
            - True로 설정하면 모델 적합 및 최적화 과정에서 자세한 정보를 출력합니다.
        """
        self.trend = func.translate_string_arg_to_int(trend, category="trend")
        self.p = p
        self.ndraws = ndraws
        self.hyperparameter_opt = func.translate_string_arg_to_int(hyperparameter_opt, category="hyperparameter_opt")
        self.optimization_method = func.translate_string_arg_to_int(optimization_method, category="optimization_method")
        self.verbose = verbose

        if hyperparameters is not None:
            self.hyperparameters = hyperparameters
        else:
            # Default hyperparameters for asymmetry
            self.hyperparameters = np.array([0.05, 0.005, 100])

        # Random Search 옵션 설정
        self.random_search_options = random_search_options  # 딕셔너리 key의 valid 체크는 생략

        # Hyperparameter MCMC 옵션
        self.hyperparameter_mcmc_options = hyperparameter_mcmc_options  # 딕셔너리 key의 valid 체크는 생략

        # 클래스 내부에서 사용할 변수들 초기화
        self.Raw_Data = container_class.Container()
        self.Parameters = container_class.Container()
        self.Prior = container_class.Container()
        self.Draw = container_class.Container()
        self.Forecast_Results = container_class.Container()

        # 모델 적합 여부 플래그
        self.is_fitted = False

        # PRINT: model init
        if self.verbose:
            print("> Initializing Model...")
            print(f"  - Trend: {self.trend}")
            print(f"  - p(lag): {self.p}")
            print(f"  - Number of Draws: {self.ndraws}")
            print(f"  - Hyperparameter Optimization: {self.hyperparameter_opt}")
            print(f"  - Optimization Method: {self.optimization_method}")
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
        self : LBVAR_Asymmetry
            적합된 모델 객체 자신을 반환합니다.

        """
        # 데이터가 DataFrame이면 numpy 배열로 변환
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()
        
        data = np.matrix(data)  # NOTE: 하.... 이거 하나 때문에 뒷부분 연산이 다 꼬임. 왜 matrix를 사용한거지 (이로 인한 호환성 문제가 있을 수 있어 클래스 분리함)

        # 데이터 체크
        if not isinstance(data, np.ndarray):
            raise ValueError("data는 numpy.ndarray 또는 pandas.DataFrame이어야 합니다.")

        # 데이터 저장
        self.Raw_Data.Set = data

        # 변수 순서를 담은 리스트를 index를 담은 리스트로 변환 eg. [1, 2, 3] -> [0, 1, 2]
        self.Parameters.RV_list = [r - 1 for r in rv_list] 
        # self.Parameters.RV_list = rv_list

        # 모델 파라미터 설정
        self.Parameters.p = self.p
        self.Parameters.n = self.Raw_Data.Set.shape[0]
        self.Parameters.nvar = self.Raw_Data.Set.shape[1]
        self.Parameters.T = self.Parameters.n - self.Parameters.p
        self.Parameters.k = self.Parameters.nvar * self.Parameters.p
        self.Parameters.beta = 1  # In the code, 'beta' is set to 1
        self.Parameters.ndraws = self.ndraws
        self.Parameters.Trend = self.trend


        # 외생 변수 생성
        self.Raw_Data = fas.EXOv_total_maker(self.Parameters, self.Raw_Data)

        self.Parameters.c = self.Raw_Data.EXOv.shape[1]
        self.Parameters.num_of_parameter = self.Parameters.nvar * self.Parameters.p + self.Parameters.c

        # LBVAR 변수 생성
        self.Raw_Data = fas.LBVAR_variable_maker(Raw_Data=self.Raw_Data, Parameters=self.Parameters)

        # 하이퍼파라미터 최적화
        if self.hyperparameter_opt == 1 or self.hyperparameter_opt == 2:
            self._hyperparameter_optimization_asymmetric()
        else:
            if self.verbose:
                print("> 하이퍼파라미터 최적화를 수행하지 않습니다. 사용자가 입력한 하이퍼파라미터, 또는 기본값 사용")

        # Prior 설정 및 Posterior 추정
        self._fit_asymmetry()

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

        self.Forecast_Results = func.Forecast_function(self.Raw_Data, self.Parameters, self.Draw, verbose=self.verbose)

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

    def _hyperparameter_optimization_asymmetric(self):
        """
        Asymmetric Prior를 사용하는 경우의 하이퍼파라미터 최적화를 수행합니다.

        하이퍼파라미터 최적화 옵션과 최적화 방법에 따라 적절한 최적화를 수행하고,
        최적의 하이퍼파라미터를 self.hyperparameters에 저장합니다.
        """
        if self.hyperparameter_opt == 1:
            # Marginal Likelihood Optimization
            negative_log_p = fas.Marginal_Likelihood(self.Raw_Data, self.Parameters, self.Prior, self.hyperparameters)
            if np.isinf(negative_log_p) or np.isnan(negative_log_p):
                if self.verbose:
                    print("> Optimization couldn't calculate the marginal likelihood.")
                    print(f"> Hyperparameter is selected as {np.round(self.hyperparameters, 3)}")
            else:
                if self.optimization_method == 0:
                    # Use scipy optimization
                    if self.verbose:
                        print("> Marginal Likelihood will be optimized using scipy Optimization Method")
                    def objective_function(candi_hyperparameters):
                        if np.all(candi_hyperparameters > 0):
                            return fas.Marginal_Likelihood(self.Raw_Data.copy(), self.Parameters, self.Prior.copy(), candi_hyperparameters)
                        else:
                            return np.inf
                    result = optimize.minimize(objective_function, self.hyperparameters, method='Nelder-Mead')
                    self.hyperparameters = result.x
                    if self.verbose:
                        print(f"> Optimized Hyperparameters are {np.round(self.hyperparameters, 3)}")
                elif self.optimization_method == 1:
                    # Use Random Search
                    if self.verbose:
                        print("> Marginal Likelihood will be optimized using Random Search")
                    def objective_function(candi_hyperparameters):
                        if np.all(candi_hyperparameters > 0):
                            return fas.Marginal_Likelihood(self.Raw_Data.copy(), self.Parameters, self.Prior.copy(), candi_hyperparameters)
                        else:
                            return np.inf
                    with threadpool_limits(limits=1):
                        self.hyperparameters, w_box, min_box = fas.Random_Search(
                            objective_function, self.hyperparameters,
                            K=self.random_search_options['K'],
                            P=self.random_search_options['P'],
                            alpha=self.random_search_options['alpha'],
                            verbose=self.verbose
                        )
                    if self.verbose:
                        print(f"> Optimized Hyperparameters are {np.round(self.hyperparameters, 3)}")
        elif self.hyperparameter_opt == 2:
            # MCMC Hyperparameter Selection
            negative_log_p_pi_p, self.Prior = fas.Optimization(self.Raw_Data, self.Parameters, self.Prior, self.hyperparameters)
            if np.isinf(negative_log_p_pi_p) or np.isnan(negative_log_p_pi_p):
                if self.verbose:
                    print("> Optimization couldn't calculate the objective function.")
                    print(f"> Hyperparameter is selected as {np.round(self.hyperparameters, 3)}")
            else:
                if self.optimization_method == 0:
                    # Use scipy optimization
                    if self.verbose:
                        print("> Optimizing objective function using scipy Optimization Method")
                    def objective_function(candi_hyperparameters):
                        if np.all(candi_hyperparameters > 0):
                            return fas.Optimization(self.Raw_Data.copy(), self.Parameters, self.Prior.copy(), candi_hyperparameters)[0]
                        else:
                            return np.inf
                    opt_result = optimize.minimize(objective_function, self.hyperparameters, method='Nelder-Mead')
                    opt_hyperparameters = opt_result.x
                    if self.verbose:
                        print(f"> Optimized Hyperparameters are {np.round(opt_hyperparameters, 3)}")
                    # Calculate Hessian
                    hessian = approx_hess(opt_hyperparameters, objective_function)
                    hessian = (hessian + hessian.T) / 2
                    # Perform MCMC
                    self.hyperparameters = fas.Hyperparameter_MCMC(
                        self.Raw_Data, self.Parameters, self.Prior, opt_hyperparameters, hessian,
                        n_draws=self.hyperparameter_mcmc_options["n_draws"], n_burnin=self.hyperparameter_mcmc_options["n_burnin"],
                        verbose=self.verbose
                    )
                elif self.optimization_method == 1:
                    # Use Random Search
                    if self.verbose:
                        print("> Optimizing objective function using Random Search")
                    def objective_function(candi_hyperparameters):
                        if np.all(candi_hyperparameters > 0):
                            return fas.Optimization(self.Raw_Data.copy(), self.Parameters, self.Prior.copy(), candi_hyperparameters)[0]
                        else:
                            return np.inf
                    with threadpool_limits(limits=1):
                        opt_hyperparameters, w_box, min_box = fas.Random_Search(
                            objective_function, self.hyperparameters,
                            K=self.random_search_options['K'],
                            P=self.random_search_options['P'],
                            alpha=self.random_search_options['alpha'],
                            verbose=self.verbose
                        )
                    if self.verbose:
                        print(f"> Optimized Hyperparameters are {np.round(opt_hyperparameters, 3)}")
                    # Calculate Hessian
                    hessian = func.Hessian_cal(objective_function, min_box, w_box, verbose=self.verbose)
                    if hessian is not None:
                        # Perform MCMC
                        self.hyperparameters = fas.Hyperparameter_MCMC(
                            self.Raw_Data, self.Parameters, self.Prior, opt_hyperparameters, hessian,
                            n_draws=self.hyperparameter_mcmc_options["n_draws"], n_burnin=self.hyperparameter_mcmc_options["n_burnin"],
                            verbose=self.verbose
                        )
                    else:
                        # If Hessian is not positive definite, keep initial hyperparameters
                        self.hyperparameters = opt_hyperparameters
        else:
            if self.verbose:
                print("> 하이퍼파라미터 최적화를 수행하지 않습니다.")

    def _fit_asymmetry(self):
        """
        Asymmetric Prior를 사용하는 경우의 모델 적합을 수행합니다.

        이 메서드는 Prior 설정, Posterior 파라미터 계산, Posterior 샘플링을 수행합니다.
        """
        if self.verbose:
            print("> Prior Making (Asymmetric)")
        self.Prior = fas.Prior_Maker(self.Raw_Data, self.Parameters, self.Prior, self.hyperparameters)
        if self.verbose:
            print("> Posterior Draw (Asymmetric)")
        self.Draw = fas.Posterior_Draw(self.Parameters, self.Raw_Data, self.Prior)
