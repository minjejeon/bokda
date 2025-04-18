import sys
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from typing import Union, List
from scipy.stats import norm
from matplotlib import cm
from tqdm import tqdm


class ProbitRegressionBayes:

    def __init__(self, mcmcsize: int, n_chains: int = 1, method: str = 'adaptiveMCMC', thinning: int = 1,
                 prior: str = 'weaklyinformative',
                 prior_location: List[float] = [], prior_scale: List[float] = [],
                 prior_alpha: float = 0.001, prior_beta: float = 0.001,
                 jumping_rate: float = 0.1,
                 acceptance_rate: float = 0.234,
                 seed: int = None, verbose: bool = False):
        """
        Probit 회귀 모델을 초기화

        Parameters
        ----------
        mcmcsize : int
            MCMC 체인의 길이
        n_chains : int, optional
            실행할 MCMC 체인의 수 | Default: 1
        method : str, optional
            MCMC 샘플링 방법 | Default: 'adaptiveMCMC'
        thinning : int, optional
            MCMC 체인의 thin 간격 | Default: 1 (thinning 없음)
        prior : str, optional
            사용할 사전 확률 분포 | Default: 'weaklyinformative'
            NOTE: 현재 'weaklyinformative', 'normal_unknown_sigma', 'normal_known_sigma' 지원함
        prior_location : list of float, optional
            사전 확률 분포의 위치 파라미터(mu) 지정
        prior_scale : list of float, optional
            사전 확률 분포의 스케일 파라미터(sigma) 지정
        prior_alpha : float, optional
            'normal_unknown_sigma'인 경우, 사전 확률 분포의 sigma 파라미터에 설정된 inverse gamma 분포의 alpha 값 | Default: 0.001
        prior_beta : float, optional
            'normal_unknown_sigma'인 경우, 사전 확률 분포의 sigma 파라미터에 설정된 inverse gamma 분포의 beta 값 | Default: 0.001
        jumping_rate : float, optional
            MCMC 체인의 점프 규칙 설정 | Default: 0.1
        acceptance_rate : float, optional
            adaptive MCMC 체인의 수락 비율 설정 | Default: 0.234
        seed : int, optional
            랜덤 샘플링 프로세스를 위한 시드 설정 | Default: None
        verbose : bool, optional
            과정 출력을 활성화할지 여부 | Default: False
        """
        self.method = method
        self.mcmcsize = mcmcsize
        self.n_chains = n_chains
        self.thinning = thinning
        self.seed = seed
        self.columns = []
        self.verbose = verbose
        self.beta = [None] * self.n_chains  # 각 체인의 베타를 저장할 리스트
        self.accept = [None] * self.n_chains  # 각 체인의 수락율을 저장할 리스트
        self.add_const = None  # 상수항이 포함되었는지 여부 (fit에서 업데이트됨)
        self.is_fitted = False
        self.prior = prior
        self.prior_location = prior_location
        self.prior_scale = prior_scale
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.jumping_rate = jumping_rate
        self.acceptance_rate = acceptance_rate
        self.trueindex = [0] * self.n_chains  # 각 체인의 마지막 인덱스

        # PRINT: model init
        if self.verbose:
            print("> Initializing Model...")
            print(f"  - Method: {self.method}")
            print(f"  - MCMC size: {self.mcmcsize}")
            print(f"  - Number of chains: {self.n_chains}")
            print(f"  - Thinning: {self.thinning}")
            print(f"  - Prior: {self.prior}")
            print(f"  - Prior location: {self.prior_location}")
            print(f"  - Prior scale: {self.prior_scale}")
            print(f"  - Prior alpha: {self.prior_alpha}")
            print(f"  - Prior beta: {self.prior_beta}")
            print(f"  - Jumping rate: {self.jumping_rate}")
            if self.method == "adaptiveMCMC":
                print(f"  - Acceptance rate: {self.acceptance_rate}")
            print(f"  - Seed: {self.seed}")
            print(f"  - Verbose: {self.verbose}")

    ###############
    # Public Method
    ###############

    def fit(self, X: Union[np.ndarray, pd.DataFrame],
            y: Union[np.ndarray, pd.Series],
            columns: List[str] = [],
            add_const: bool = True,
            standardization: bool = True) -> 'ProbitRegressionBayes':
        """
        주어진 데이터로 베이지안 Probit 회귀 모델을 적합
        NOTE: 데이터프레임을 입력할 경우 내부적으로 numpy array로 변환하여 사용함

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features) or pd.DataFrame
            입력 데이터 행렬
        y : numpy array of shape (n_samples,) or pd.Series
            타겟 라벨(0/1) 벡터
        columns : list of str, optional
            X의 컬럼 명칭(결과 출력 시 활용), 데이터프레임인 경우 기본값은 X.columns로 사용, 컬럼명 지정이 없을 경우 x1~xn으로 자동 설정
        add_const : bool, optional
            상수항 추가 여부 | Default: True
        standardization : bool, optional
            데이터 표준화 여부 | Default: True

        Returns
        -------
        self : object
            객체 자기 자신을 반환
        """
        # 1) 컬럼명 설정
        if (not columns) and (isinstance(X, pd.DataFrame)):
            self.columns = X.columns.tolist()  # 컬럼명이 따로 정해지지 않고 데이터프레임이 입력된 경우, 데이터프레임의 컬럼명을 사용
        else:
            self.columns = columns  # 컬럼명이 따로 입력된 경우나, 기본값(빈 컬럼명)인 경우

        # 2) 입력값이 데이터프레임인 경우 numpy로 변환
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()  # numpy array로 변환

        if isinstance(y, pd.Series):
            y = y.to_numpy()

        # 3) numpy array에 문제가 없는지 확인
        self._check_input_array(X, y)

        # 4) 데이터 표준화
        if standardization:
            X = self._standardize_binary_continous(data=X)

        # 5) 상수항 추가
        self.add_const = add_const  # 상수항 포함 여부를 저장
        if add_const:
            X = np.hstack([np.ones((X.shape[0], 1)), X])  # X의 맨 앞에 상수항 추가

        # PRINT: start
        if self.verbose:
            print("> Start Fitting Model...")
            print(f"  - Input Data: {X.shape[0]} samples, {X.shape[1]} features")
            if self.add_const:
                print("  - Adding constant term to the model")
            print(f"  - Optimization Method: {self.method}")

        # 6) 적합 과정: 각 체인에 대해 별도로 적합
        for chain_idx in range(self.n_chains):
            if self.method == 'MCMC':
                if self.prior == 'weaklyinformative':
                    self._fit_MCMC_weaklyinformative(X, y, self.jumping_rate, self.prior_alpha, self.prior_beta,
                                                     chain_idx)
                elif self.prior == 'normal_unknown_sigma':
                    self._fit_MCMC_normal_unknown_sigma(X, y, self.jumping_rate, self.prior_alpha, self.prior_beta,
                                                        chain_idx)
                elif self.prior == 'normal_known_sigma':
                    self._fit_MCMC_normal_known_sigma(X, y, self.jumping_rate, chain_idx)
                else:
                    raise ValueError(f"지원하지 않는 prior입니다: '{self.prior}'")

            elif self.method == 'adaptiveMCMC':
                if self.prior == 'weaklyinformative':
                    self._fit_adaptiveMCMC_weaklyinformative(X, y, self.acceptance_rate, self.prior_alpha,
                                                             self.prior_beta, chain_idx)
                elif self.prior == 'normal_unknown_sigma':
                    self._fit_adaptiveMCMC_normal_unknown_sigma(X, y, self.acceptance_rate, self.prior_alpha,
                                                                self.prior_beta, chain_idx)
                elif self.prior == 'normal_known_sigma':
                    self._fit_adaptiveMCMC_normal_known_sigma(X, y, self.acceptance_rate, chain_idx)
                else:
                    raise ValueError(f"지원하지 않는 prior입니다: '{self.prior}'")
            else:
                raise ValueError(f"지원하지 않는 MCMC 방법입니다: '{self.method}'")

        # 통계량 계산은 모든 체인에 대해 별도로 수행
        self.X = X  # X를 저장하여 이후 통계량 계산에 사용
        self.y = y  # y를 저장하여 이후 통계량 계산에 사용
        self.is_fitted = True  # fit 완료 이후 flag 설정
        self.summary_stats = self._calculate_summary_stats(burn_in=0)  # 기본적으로 burn_in=0으로 통계량 계산

        # PRINT: fitting 완료
        if self.verbose:
            print("> Model Fitted.")

        # 반환
        return self

    ##########
    # Report Part (Public Method)

    def print_summary(self, digits: int = 4, burn_in: int = 0):
        """
        모델 적합 결과의 요약 정보를 출력

        Parameters
        ----------
        digits : int, optional
            출력 값을 소수점 아래로 몇 자리까지 출력할 지 여부 | Default: 4
        burn_in : int, optional
            burn-in의 수 | Default: 0
        """
        if not self.is_fitted:
            raise ValueError("모델을 fit 해야 통계량을 반환할 수 있습니다.")

        # burn_in을 적용하여 통계량 재계산
        summary_stats = self._calculate_summary_stats(burn_in=burn_in)

        # 계수와 관련된 통계량을 준비
        column_names = self._get_column_names()
        mean_values = summary_stats["posterior_mean"]
        var_values = summary_stats["posterior_variance"]
        p25_values = summary_stats["hpd_lower_bound"]
        p95_values = summary_stats["hpd_upper_bound"]
        ess_values = summary_stats["ESS"]
        acceptance_rate_values = summary_stats["acceptance_rate"]

        # 단일 통계량
        single_stats = {
            "BIC": summary_stats["BIC"],
        }

        # 계수 테이블
        header = ["Variable", "Mean", "Var", "2.5", "97.5", "ESS", "Acc Rate"]
        # 컬럼 너비 계산
        col_widths = [len(h) for h in header]

        # 데이터 준비 및 컬럼 너비 계산
        data_rows = []
        for i in range(len(mean_values)):
            # 값 포맷팅
            mean = f"{mean_values[i]:.{digits}f}"
            var = f"{var_values[i]:.{digits}f}"
            p25 = f"{p25_values[i]:.{digits}f}"
            p95 = f"{p95_values[i]:.{digits}f}"
            ess = f"{ess_values[i]:.{digits}f}"

            # acceptance_rate_values[i]가 실수인지 확인
            if isinstance(acceptance_rate_values, (list, np.ndarray)) and not isinstance(acceptance_rate_values[i],
                                                                                         tuple):
                acceptance_rate = f"{acceptance_rate_values[i]:.{digits}f}"
            else:
                # 오류가 발생하면 오류 메시지를 출력하고 기본값을 사용
                acceptance_rate = "N/A"
                print(
                    f"Warning: acceptance_rate_values[{i}] is not a float. It is of type {type(acceptance_rate_values[i])}.")

            # row append
            data_rows.append([column_names[i], mean, var, p25, p95, ess, acceptance_rate])

            # 컬럼 너비 업데이트
            for j, val in enumerate(data_rows[-1]):
                val = str(val)
                val_len = len(val)
                col_widths[j] = max([col_widths[j], val_len + 1])

        # 모든 컬럼 너비 동일하게 설정
        max_col_width = max(col_widths)
        col_widths = [max_col_width] * len(col_widths)

        # 헤더 너비 보정 및 정렬
        header_line = "".join(
            header[i].ljust(col_widths[i]) if i == 0 else header[i].rjust(col_widths[i])
            for i in range(len(header))
        )

        # 출력 시작
        print("-" * len(header_line))
        print("Model Summary:\n")

        # 단일 통계량 출력
        for key, value in single_stats.items():
            if isinstance(value, float):
                formatted_value = f"{value:.{digits}f}"
            else:
                formatted_value = str(value)
            print(f"{key}: {formatted_value}")
        print()

        # 헤더 출력
        print("-" * len(header_line))
        print(header_line)
        print("-" * len(header_line))

        # 데이터 행 출력
        for row in data_rows:
            line = "".join(
                str(row[i]).ljust(col_widths[i]) if i == 0 else str(row[i]).rjust(col_widths[i])
                for i in range(len(row))
            )
            print(line)

        print("-" * len(header_line))

    def traceplot(self, burn_in: int = 0, multichain: bool = False) -> None:
        """
        Traceplot을 출력하는 함수

        Parameters
        ----------
        burn_in : int, optional
            burn-in의 수 | Default: 0
        multichain : bool, optional
            여러 체인이 있는 경우, True로 설정하여 각 체인의 값을 모두 플롯함 | Default: False
        """
        if not self.is_fitted:
            raise ValueError("모델을 fit 해야 Traceplot을 그릴 수 있습니다.")

        num_beta = self.beta[0].shape[1]  # 베타의 수
        plt.figure(figsize=(12, num_beta * 3))

        for i in range(num_beta):
            plt.subplot(num_beta, 1, i + 1)
            if multichain:
                for chain_idx, chain in enumerate(self.beta):
                    if burn_in >= chain.shape[0]:
                        raise ValueError(
                            f"`burn_in` ({burn_in}) exceeds the number of available samples ({chain.shape[0]}). Chain {chain_idx + 1}")
                    sns.lineplot(data=chain[burn_in:self.trueindex[chain_idx], i], label=f'Chain {chain_idx + 1}',
                                 alpha=0.6)
            else:
                for chain_idx, chain in enumerate(self.beta):
                    if burn_in >= chain.shape[0]:
                        raise ValueError(
                            f"`burn_in` ({burn_in}) exceeds the number of available samples ({chain.shape[0]}). Chain {chain_idx + 1}")
                    sns.lineplot(data=chain[burn_in:self.trueindex[chain_idx], i],
                                 label=f'Chain {chain_idx + 1}' if self.n_chains > 1 else None)
            plt.title(f"Traceplot for Beta {i + 1}")
            plt.xlabel("Iteration")
            plt.ylabel(f"Beta {i + 1}")
        plt.tight_layout()
        if multichain and self.n_chains > 1:
            plt.legend()
        plt.show()

    def hpd_interval(self, beta_chain: Union[np.ndarray, List[np.ndarray]] = None,
                     hdi_prob: float = 0.95, burn_in: int = 0, multichain: bool = False) -> None:
        """
        각 체인별 HPD Interval을 플롯하는 함수

        Parameters
        ----------
        beta_chain : numpy array of shape (mcmcsize, n_features) or list of numpy arrays, optional
            MCMC 체인의 beta 값. 지정하지 않으면 모델의 체인을 사용함.
        hdi_prob : float, optional
            HPD Interval의 확률 값 | Default: 0.95
        burn_in : int, optional
            burn-in의 수 | Default: 0
        multichain : bool, optional
            여러 체인이 있는 경우, True로 설정하여 각 체인의 값을 모두 플롯함 | Default: False
        """
        if beta_chain is None:
            beta_chain = self.beta
        else:
            if multichain and not isinstance(beta_chain, list):
                raise ValueError("multichain이 True인 경우, beta_chain은 리스트 형태여야 합니다.")

        lower_percentile = (1 - hdi_prob) / 2 * 100
        upper_percentile = (1 + hdi_prob) / 2 * 100

        num_beta = self.beta[0].shape[-1]
        num_chains = len(beta_chain) if multichain else 1

        plt.figure(figsize=(12, num_beta * 3))

        for i in range(num_beta):
            plt.subplot(num_beta, 1, i + 1)
            handles = []

            if multichain:
                for chain_idx, chain in enumerate(beta_chain):
                    chain_samples = chain[burn_in:self.trueindex[chain_idx], i]
                    hdi_lower = np.percentile(chain_samples, lower_percentile)
                    hdi_upper = np.percentile(chain_samples, upper_percentile)

                    # 히스토그램 플롯
                    color_rgb = plt.get_cmap('tab10')(chain_idx % 10)
                    sns.histplot(chain_samples, kde=True, color=color_rgb, alpha=0.3, label=f'Chain {chain_idx + 1}')

                    plt.axvline(hdi_lower, color=color_rgb, linestyle='--')
                    plt.axvline(hdi_upper, color=color_rgb, linestyle='--')

                    handles.append(plt.Line2D([0], [0], color=color_rgb, linestyle='--',
                                              label=f'Chain {chain_idx + 1} HPD'))
            else:
                for chain_idx, chain in enumerate(beta_chain):
                    chain_samples = chain[burn_in:self.trueindex[chain_idx], i]
                    hdi_lower = np.percentile(chain_samples, lower_percentile)
                    hdi_upper = np.percentile(chain_samples, upper_percentile)

                    # 히스토그램 플롯
                    sns.histplot(chain_samples, kde=True, color='grey', alpha=0.3)

                    plt.axvline(hdi_lower, color='blue', linestyle='--')
                    plt.axvline(hdi_upper, color='blue', linestyle='--')

            plt.title(f'HPD Interval for Beta {i + 1}')
            plt.xlabel(f'Beta {i + 1}')

            if multichain and self.n_chains > 1:
                plt.legend()

        plt.tight_layout()
        plt.show()

    def acf_plot(self, beta_chain: Union[np.ndarray, List[np.ndarray]] = None, burn_in: int = 0,
                 multichain: bool = False, max_lag: int = 40) -> None:
        """
        각 체인별 ACF Plot을 출력하는 함수, 체인별로 색상을 다르게 적용

        Parameters
        ----------
        beta_chain : numpy array of shape (mcmcsize, n_features) or list of numpy arrays, optional
            MCMC 체인의 beta 값 (multichain인 경우 리스트 형태). 지정하지 않으면 모델의 체인을 사용함.
        burn_in : int, optional
            burn-in의 수 | Default: 0
        multichain : bool, optional
            여러 체인이 있는 경우, True로 설정하여 각 체인의 값을 모두 플롯함 | Default: False
        max_lag : int, optional
            ACF plot의 최대 lag (x축에 찍힐 점의 개수를 동일하게 설정하기 위해 사용) | Default: 40
        """

        if beta_chain is None:
            beta_chain = self.beta
        else:
            if multichain and not isinstance(beta_chain, list):
                raise ValueError("multichain이 True인 경우, beta_chain은 리스트 형태여야 합니다.")

        num_beta = self.beta[0].shape[-1]
        num_chains = len(beta_chain) if multichain else 1

        # multichain인 경우 체인별로 사용할 color map 생성
        cmap = plt.get_cmap('tab10')

        plt.figure(figsize=(12, num_beta * 3))

        for i in range(num_beta):
            plt.subplot(num_beta, 1, i + 1)

            if multichain:
                for chain_idx, chain in enumerate(beta_chain):
                    chain_samples = chain[burn_in:self.trueindex[chain_idx], i]
                    acf_values = [self._autocorrelation(chain_samples, lag) for lag in range(1, max_lag + 1)]
                    lags = np.arange(1, max_lag + 1)

                    color_rgb = cmap(chain_idx % 10)
                    color_hex = mcolors.to_hex(color_rgb)

                    # linefmt과 markerfmt에서 색상 코드를 제거하고 스타일만 지정
                    markerline, stemlines, baseline = plt.stem(
                        lags,
                        acf_values,
                        basefmt=" ",
                        linefmt='-',
                        markerfmt='o'
                    )
                    # stemlines과 markerline의 색상을 별도로 설정
                    plt.setp(stemlines, 'color', color_hex)
                    plt.setp(markerline, 'color', color_hex)

                    # Confidence Interval
                    conf_interval = 1.96 / np.sqrt(len(chain_samples))
                    plt.fill_between(lags, -conf_interval, conf_interval, color=color_hex, alpha=0.2)

            else:
                for chain_idx, chain in enumerate(beta_chain):
                    chain_samples = chain[burn_in:self.trueindex[chain_idx], i]
                    acf_values = [self._autocorrelation(chain_samples, lag) for lag in range(1, max_lag + 1)]
                    lags = np.arange(1, max_lag + 1)

                    plt.stem(lags, acf_values, linefmt='b-', markerfmt='bo', basefmt='r-')

                    # Confidence Interval
                    conf_interval = 1.96 / np.sqrt(len(chain_samples))
                    plt.fill_between(lags, -conf_interval, conf_interval, color='blue', alpha=0.2)

            plt.title(f'ACF for Beta {i + 1}')
            plt.xlabel('Lag')
            plt.ylabel('Autocorrelation')

        plt.tight_layout()
        if multichain and self.n_chains > 1:
            plt.legend([f'Chain {c + 1}' for c in range(self.n_chains)])
        plt.show()

    def density_plot(self, burn_in: int = 0, multichain: bool = False) -> None:
        """
        각 체인별 Density Plot을 출력하는 함수

        Parameters
        ----------
        burn_in : int, optional
            burn-in의 수 | Default: 0
        multichain : bool, optional
            여러 체인이 있는 경우, True로 설정하여 각 체인의 값을 모두 플롯함 | Default: False
        """

        if not self.is_fitted:
            raise ValueError("모델을 fit 해야 Density Plot을 그릴 수 있습니다.")

        num_beta = self.beta[0].shape[-1]
        plt.figure(figsize=(12, num_beta * 3))

        for i in range(num_beta):
            plt.subplot(num_beta, 1, i + 1)
            if multichain:
                for chain_idx, chain in enumerate(self.beta):
                    chain_samples = chain[burn_in:self.trueindex[chain_idx], i]
                    sns.kdeplot(chain_samples, fill=True, alpha=0.6, label=f'Chain {chain_idx + 1}')
            else:
                for chain_idx, chain in enumerate(self.beta):
                    chain_samples = chain[burn_in:self.trueindex[chain_idx], i]
                    sns.kdeplot(chain_samples, fill=True, alpha=0.6,
                                label=f'Chain {chain_idx + 1}' if self.n_chains > 1 else None)
            plt.title(f'Density Plot for Beta {i + 1}')
            plt.xlabel(f'Beta {i + 1}')
            if multichain and self.n_chains > 1:
                plt.legend()
        plt.tight_layout()
        plt.show()

    #################
    # Internal Method
    #################

    def _check_input_array(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        입력 데이터가 정확한지 확인

        Parameters
        ----------
        X : numpy array
            입력 데이터 행렬
        y : numpy array
            타겟 레이블 벡터

        Raises
        ------
        ValueError
            X, y가 numpy array가 아니거나, 차원이 맞지 않으면 에러가 발생
        """
        # numpy array 인지 확인
        if not isinstance(X, np.ndarray):
            raise ValueError("X는 numpy array 또는 DataFrame이어야 합니다.")

        if not isinstance(y, np.ndarray):
            raise ValueError("y는 numpy array 또는 Series여야 합니다.")

        # 차원 확인
        if X.ndim != 2:
            raise ValueError("X는 numpy 2D array여야 합니다.")

        if y.ndim != 1:
            raise ValueError("y는 numpy 1D array여야 합니다.")

        if X.shape[0] != y.shape[0]:
            raise ValueError("X와 y의 크기가 같아야 합니다.")

        # NaN 확인
        if np.isnan(X).any():
            raise ValueError("X에 NaN 값이 포함되어 있습니다.")

        if np.isnan(y).any():
            raise ValueError("y에 NaN 값이 포함되어 있습니다.")

    def _get_column_names(self) -> List[str]:
        """
        컬럼의 명칭을 반환

        Returns
        -------
        column_names : list of str
            컬럼의 명칭 리스트. 지정된 컬럼명이 없는 경우 'x1'부터 'xn'까지 자동으로 생성
        """
        # 컬럼명이 있는 경우
        if self.columns:
            if self.add_const:
                column_names = ["const"] + self.columns
            else:
                column_names = self.columns

        # 컬럼명이 없는 경우
        else:
            column_length = self.beta[0].shape[1]
            if self.add_const:
                column_length -= 1  # 상수항 있는 경우 계수 중 맨 첫번째는 상수항이므로 하나를 빼고 x1~x_n-1
                column_names = ["const"] + [f"x{i + 1}" for i in range(column_length)]
            else:
                column_names = [f"x{i + 1}" for i in range(column_length)]

        return column_names

    def _standardize_binary_continous(self, data: np.ndarray, desired_mean: float = 0,
                                      desired_standard_deviation: float = 0.5) -> np.ndarray:
        """
        입력 데이터를 정규화함 (binary/continuous 각각)
        NOTE: binary는 0/1 로 가정

        Parameters
        ----------
        data : numpy array
            입력 데이터 행렬
        desired_mean: float, optional
            원하는 평균값 | Default: 0
        desired_standard_deviation: float, optional
            원하는 표준편차값 | Default: 0.5

        Returns
        ------
        adjusted_data : ndarray
            정규화된 data
        """
        # binary 변수 식별
        binary_columns = np.all(np.isin(data, [0, 1]), axis=0)  # binary 변수
        continuous_columns = ~binary_columns  # continuous 변수

        # binary 변수 처리
        if np.any(binary_columns):
            for col_idx in np.where(binary_columns)[0]:
                p = np.mean(data[:, col_idx])  # 1의 비율
                data[:, col_idx] = data[:, col_idx] * (1 - p) - (1 - data[:, col_idx]) * p

        # continuous 변수 처리
        if np.any(continuous_columns):
            current_mean = np.mean(data[:, continuous_columns], axis=0)
            current_std = np.sqrt(np.var(data[:, continuous_columns], axis=0))

            # 데이터 표준화
            standardized_data = (data[:, continuous_columns] - current_mean) / current_std

            # 원하는 표준편차 및 평균으로 조정
            adjusted_data = standardized_data * desired_standard_deviation
            adjusted_data += desired_mean

            # 변환 결과를 원래 데이터에 반영
            data[:, continuous_columns] = adjusted_data

        return data

    ##########
    # Algorithm Part

    def _fit_MCMC_weaklyinformative(self, X: np.ndarray, y: np.ndarray, jumping_rate: float, prior_alpha: float,
                                    prior_beta: float, chain_idx: int) -> None:
        """
        weakly informative prior를 사용하는 MCMC 체인
        """
        np.random.seed(self.seed + chain_idx if self.seed is not None else None)  # 각 체인에 다른 시드 사용 가능
        N, C = X.shape
        beta = np.zeros((self.mcmcsize + 1, C))
        accept = np.zeros(C)

        if self.verbose:
            # PRINT: verbose = True일 경우에만 progress bar 출력
            iters = tqdm(range(self.mcmcsize), desc=f"Chain {chain_idx + 1}")
        else:
            iters = range(self.mcmcsize)

        for i in iters:
            for j in range(C):
                # 베타 업데이트 (weakly informative)
                nu_beta = beta[i].copy()
                nu_beta[j] = np.random.normal(beta[i, j], jumping_rate)
                prop = norm.cdf(np.dot(X, beta[i]))  # Probit link
                log_likelihood = np.sum(st.binom.logpmf(y, 1, prop))
                posterior = log_likelihood + st.cauchy.logpdf(beta[i, j], loc=0, scale=10)

                # 제안된 nu_beta에 대한 log likelihood 계산
                prop = norm.cdf(np.dot(X, nu_beta))
                nu_log_likelihood = np.sum(st.binom.logpmf(y, 1, prop))
                nu_posterior = nu_log_likelihood + st.cauchy.logpdf(nu_beta[j], loc=0, scale=10)

                # 수락 조건
                if np.log(np.random.uniform(0, 1)) < (nu_posterior - posterior):
                    beta[i + 1, j] = nu_beta[j]
                    accept[j] += 1
                else:
                    beta[i + 1, j] = beta[i, j]

        self.beta[chain_idx] = beta
        self.accept[chain_idx] = np.array(accept) / self.mcmcsize
        self.trueindex[chain_idx] = self.mcmcsize

    def _fit_MCMC_normal_unknown_sigma(self, X: np.ndarray, y: np.ndarray, jumping_rate: float, prior_alpha: float,
                                       prior_beta: float, chain_idx: int) -> None:
        """
        normal prior (unknown sigma)를 사용하는 MCMC 체인
        """
        np.random.seed(self.seed + chain_idx if self.seed is not None else None)  # 각 체인에 다른 시드 사용 가능
        N, C = X.shape
        beta = np.zeros((self.mcmcsize + 1, C))
        sigma = np.zeros((self.mcmcsize + 1, C))
        # Initialize sigma with prior_scale if provided, else with 1
        if self.prior_scale:
            if len(self.prior_scale) < C:
                # If prior_scale has fewer elements than C, extend it with 1.0
                extended_scale = self.prior_scale + [1.0] * (C - len(self.prior_scale))
                sigma[0] = np.array(extended_scale[:C]) + sys.float_info.epsilon
            else:
                sigma[0] = np.array(self.prior_scale[:C]) + sys.float_info.epsilon
        else:
            sigma[0] = 1.0
        accept = np.zeros(C)

        if self.verbose:
            # PRINT: verbose = True일 경우에만 progress bar 출력
            iters = tqdm(range(self.mcmcsize), desc=f"Chain {chain_idx + 1}")
        else:
            iters = range(self.mcmcsize)

        for i in iters:
            for j in range(C):
                # 베타 업데이트 (normal unknown sigma)
                nu_beta = beta[i].copy()
                nu_beta[j] = np.random.normal(beta[i, j], jumping_rate)
                prop = norm.cdf(np.dot(X, beta[i]))  # Probit link
                log_likelihood = np.sum(st.binom.logpmf(y, 1, prop))
                posterior = log_likelihood + st.norm.logpdf(beta[i, j],
                                                            loc=self.prior_location[j] if self.prior_location else 0,
                                                            scale=np.sqrt(sigma[i, j]) + sys.float_info.epsilon)

                # 제안된 nu_beta에 대한 log likelihood 계산
                prop = norm.cdf(np.dot(X, nu_beta))
                nu_log_likelihood = np.sum(st.binom.logpmf(y, 1, prop))
                nu_posterior = nu_log_likelihood + st.norm.logpdf(nu_beta[j],
                                                                  loc=self.prior_location[j] if self.prior_location else 0,
                                                                  scale=np.sqrt(sigma[i, j]) + sys.float_info.epsilon)

                # 수락 조건
                if np.log(np.random.uniform(0, 1)) < (nu_posterior - posterior):
                    beta[i + 1, j] = nu_beta[j]
                    accept[j] += 1
                else:
                    beta[i + 1, j] = beta[i, j]

                # sigma 업데이트
                sigma[i + 1, j] = st.invgamma.rvs(N / 2 + prior_alpha,
                                                  scale=np.sum((y - norm.ppf(prop)) ** 2) / 2 + prior_beta + sys.float_info.epsilon)

        self.beta[chain_idx] = beta
        self.sigma = getattr(self, 'sigma', None)
        self.accept[chain_idx] = np.array(accept) / self.mcmcsize
        self.trueindex[chain_idx] = self.mcmcsize

    def _fit_MCMC_normal_known_sigma(self, X: np.ndarray, y: np.ndarray, jumping_rate: float, chain_idx: int) -> None:
        """
        normal prior (known sigma)를 사용하는 MCMC 체인
        """
        np.random.seed(self.seed + chain_idx if self.seed is not None else None)  # 각 체인에 다른 시드 사용 가능
        N, C = X.shape
        beta = np.zeros((self.mcmcsize + 1, C))
        accept = np.zeros(C)

        if self.verbose:
            # PRINT: verbose = True일 경우에만 progress bar 출력
            iters = tqdm(range(self.mcmcsize), desc=f"Chain {chain_idx + 1}")
        else:
            iters = range(self.mcmcsize)

        for i in iters:
            for j in range(C):
                # 베타 업데이트 (normal known sigma)
                nu_beta = beta[i].copy()
                nu_beta[j] = np.random.normal(beta[i, j], jumping_rate)
                prop = norm.cdf(np.dot(X, beta[i]))  # Probit link
                log_likelihood = np.sum(st.binom.logpmf(y, 1, prop))
                posterior = log_likelihood + st.norm.logpdf(beta[i, j],
                                                            loc=self.prior_location[j] if self.prior_location else 0,
                                                            scale=self.prior_scale[j] if self.prior_scale else 1 + sys.float_info.epsilon)

                # 제안된 nu_beta에 대한 log likelihood 계산
                prop = norm.cdf(np.dot(X, nu_beta))
                nu_log_likelihood = np.sum(st.binom.logpmf(y, 1, prop))
                nu_posterior = nu_log_likelihood + st.norm.logpdf(nu_beta[j],
                                                                  loc=self.prior_location[j] if self.prior_location else 0,
                                                                  scale=self.prior_scale[j] if self.prior_scale else 1 + sys.float_info.epsilon)

                # 수락 조건
                if np.log(np.random.uniform(0, 1)) < (nu_posterior - posterior):
                    beta[i + 1, j] = nu_beta[j]
                    accept[j] += 1
                else:
                    beta[i + 1, j] = beta[i, j]

        self.beta[chain_idx] = beta
        self.accept[chain_idx] = np.array(accept) / self.mcmcsize
        self.trueindex[chain_idx] = self.mcmcsize

    def _fit_adaptiveMCMC_weaklyinformative(self, X: np.ndarray, y: np.ndarray, acceptance_rate: float,
                                            prior_alpha: float, prior_beta: float, chain_idx: int) -> None:
        """
        weakly informative prior를 사용하는 adaptive MCMC 체인
        """
        np.random.seed(self.seed + chain_idx if self.seed is not None else None)  # 각 체인에 다른 시드 사용 가능
        N, C = X.shape
        beta = np.zeros((self.mcmcsize + 1, C))
        accept = np.zeros(C)
        prior_sigma = np.ones(C)  # 초기 sigma 값
        prior_lambda = np.ones(C)  # 초기 lambda 값
        prior_mu = np.zeros(C)  # 초기 mu 값
        l = 0.0  # 적응 과정의 단계

        if self.verbose:
            # PRINT: verbose = True일 경우에만 progress bar 출력
            iters = tqdm(range(self.mcmcsize), desc=f"Chain {chain_idx + 1}")
        else:
            iters = range(self.mcmcsize)

        for i in iters:
            for j in range(C):
                nu_beta = beta[i].copy()
                nu_beta[j] = np.random.normal(beta[i, j],
                                              np.sqrt(np.exp(np.log(prior_sigma[j] + 1e-8) + prior_lambda[j])))  # log(0) 방지

                # 현재 beta에 대한 log likelihood 계산
                prop = norm.cdf(np.dot(X, beta[i]))
                log_likelihood = np.sum(st.binom.logpmf(y, 1, prop))
                posterior = log_likelihood + st.cauchy.logpdf(beta[i, j], loc=0, scale=10)

                # 제안된 nu_beta에 대한 log likelihood 계산
                prop = norm.cdf(np.dot(X, nu_beta))
                nu_log_likelihood = np.sum(st.binom.logpmf(y, 1, prop))
                nu_posterior = nu_log_likelihood + st.cauchy.logpdf(nu_beta[j], loc=0, scale=10)

                # Metropolis-Hastings 수락 조건
                if np.log(np.random.uniform(0, 1)) < (nu_posterior - posterior):
                    beta[i + 1, j] = nu_beta[j]
                    accept[j] += 1
                else:
                    beta[i + 1, j] = beta[i, j]

                # Adaptive MCMC를 위한 적응 과정
                prior_lambda[j] += self._gamma_func(l + 1) * (
                            min(1, np.exp(nu_posterior - posterior)) - acceptance_rate)
                temp_vec = beta[i] - prior_mu
                prior_sigma[j] = prior_sigma[j] + self._gamma_func(l + 1) * (temp_vec[j] ** 2 - prior_sigma[j])
                prior_mu[j] = prior_mu[j] + self._gamma_func(l + 1) * (nu_beta[j] - prior_mu[j])
                l += 1

        self.beta[chain_idx] = beta
        self.accept[chain_idx] = np.array(accept) / self.mcmcsize
        self.trueindex[chain_idx] = self.mcmcsize

    def _fit_adaptiveMCMC_normal_unknown_sigma(self, X: np.ndarray, y: np.ndarray, acceptance_rate: float,
                                               prior_alpha: float, prior_beta: float, chain_idx: int) -> None:
        """
        normal prior (unknown sigma)를 사용하는 adaptive MCMC 체인
        """
        np.random.seed(self.seed + chain_idx if self.seed is not None else None)  # 각 체인에 다른 시드 사용 가능
        N, C = X.shape
        beta = np.zeros((self.mcmcsize + 1, C))
        sigma = np.zeros((self.mcmcsize + 1, C))
        # Initialize sigma with prior_scale if provided, else with 1
        if self.prior_scale:
            if len(self.prior_scale) < C:
                # If prior_scale has fewer elements than C, extend it with 1.0
                extended_scale = self.prior_scale + [1.0] * (C - len(self.prior_scale))
                sigma[0] = np.array(extended_scale[:C]) + sys.float_info.epsilon
            else:
                sigma[0] = np.array(self.prior_scale[:C]) + sys.float_info.epsilon
        else:
            sigma[0] = 1.0
        accept = np.zeros(C)
        prior_sigma = np.ones(C)  # 초기 sigma 값
        prior_lambda = np.ones(C)  # 초기 lambda 값
        prior_mu = np.zeros(C)  # 초기 mu 값
        l = 0.0  # 적응 과정의 단계

        if self.verbose:
            # PRINT: verbose = True일 경우에만 progress bar 출력
            iters = tqdm(range(self.mcmcsize), desc=f"Chain {chain_idx + 1}")
        else:
            iters = range(self.mcmcsize)

        for i in iters:
            for j in range(C):
                nu_beta = beta[i].copy()
                nu_beta[j] = np.random.normal(beta[i, j],
                                              np.sqrt(np.exp(np.log(prior_sigma[j]) + prior_lambda[j])))

                # 현재 beta에 대한 log likelihood 계산
                prop = norm.cdf(np.dot(X, beta[i]))
                log_likelihood = np.sum(st.binom.logpmf(y, 1, prop))
                posterior = log_likelihood + st.norm.logpdf(beta[i, j],
                                                            loc=self.prior_location[j] if self.prior_location else 0,
                                                            scale=np.sqrt(sigma[i, j]) + sys.float_info.epsilon)

                # 제안된 nu_beta에 대한 log likelihood 계산
                prop = norm.cdf(np.dot(X, nu_beta))
                nu_log_likelihood = np.sum(st.binom.logpmf(y, 1, prop))
                nu_posterior = nu_log_likelihood + st.norm.logpdf(nu_beta[j],
                                                                  loc=self.prior_location[j] if self.prior_location else 0,
                                                                  scale=np.sqrt(sigma[i, j]) + sys.float_info.epsilon)

                # Metropolis-Hastings 수락 조건
                if np.log(np.random.uniform(0, 1)) < (nu_posterior - posterior):
                    beta[i + 1, j] = nu_beta[j]
                    accept[j] += 1
                else:
                    beta[i + 1, j] = beta[i, j]

                # Adaptive MCMC를 위한 적응 과정
                prior_lambda[j] += self._gamma_func(l + 1) * (
                            min(1, np.exp(nu_posterior - posterior)) - acceptance_rate)
                temp_vec = beta[i] - prior_mu
                prior_sigma[j] = prior_sigma[j] + self._gamma_func(l + 1) * (temp_vec[j] ** 2 - prior_sigma[j])
                prior_mu[j] = prior_mu[j] + self._gamma_func(l + 1) * (nu_beta[j] - prior_mu[j])
                l += 1

                # sigma 업데이트
                sigma[i + 1, j] = st.invgamma.rvs(N / 2 + prior_alpha,
                                                  scale=np.sum((y - norm.ppf(prop)) ** 2) / 2 + prior_beta + sys.float_info.epsilon)

        self.beta[chain_idx] = beta
        self.sigma = getattr(self, 'sigma', None)
        self.accept[chain_idx] = np.array(accept) / self.mcmcsize
        self.trueindex[chain_idx] = self.mcmcsize

    def _fit_adaptiveMCMC_normal_known_sigma(self, X: np.ndarray, y: np.ndarray, acceptance_rate: float,
                                             chain_idx: int) -> None:
        """
        normal prior (known sigma)를 사용하는 adaptive MCMC 체인
        """
        np.random.seed(self.seed + chain_idx if self.seed is not None else None)  # 각 체인에 다른 시드 사용 가능
        N, C = X.shape
        beta = np.zeros((self.mcmcsize + 1, C))
        accept = np.zeros(C)
        prior_sigma = np.ones(C)  # 초기 sigma 값
        prior_lambda = np.ones(C)  # 초기 lambda 값
        prior_mu = np.zeros(C)  # 초기 mu 값
        l = 0.0  # 적응 과정의 단계

        if self.verbose:
            # PRINT: verbose = True일 경우에만 progress bar 출력
            iters = tqdm(range(self.mcmcsize), desc=f"Chain {chain_idx + 1}")
        else:
            iters = range(self.mcmcsize)

        for i in iters:
            for j in range(C):
                nu_beta = beta[i].copy()
                nu_beta[j] = np.random.normal(beta[i, j],
                                              np.sqrt(np.exp(np.log(prior_sigma[j]) + prior_lambda[j])))

                # 현재 beta에 대한 log likelihood 계산
                prop = norm.cdf(np.dot(X, beta[i]))
                log_likelihood = np.sum(st.binom.logpmf(y, 1, prop))
                posterior = log_likelihood + st.norm.logpdf(beta[i, j],
                                                            loc=self.prior_location[j] if self.prior_location else 0,
                                                            scale=self.prior_scale[j] if self.prior_scale else 1 + sys.float_info.epsilon)

                # 제안된 nu_beta에 대한 log likelihood 계산
                prop = norm.cdf(np.dot(X, nu_beta))
                nu_log_likelihood = np.sum(st.binom.logpmf(y, 1, prop))
                nu_posterior = nu_log_likelihood + st.norm.logpdf(nu_beta[j],
                                                                  loc=self.prior_location[j] if self.prior_location else 0,
                                                                  scale=self.prior_scale[j] if self.prior_scale else 1 + sys.float_info.epsilon)

                # Metropolis-Hastings 수락 조건
                if np.log(np.random.uniform(0, 1)) < (nu_posterior - posterior):
                    beta[i + 1, j] = nu_beta[j]
                    accept[j] += 1
                else:
                    beta[i + 1, j] = beta[i, j]

                # Adaptive MCMC를 위한 적응 과정
                prior_lambda[j] += self._gamma_func(l + 1) * (
                            min(1, np.exp(nu_posterior - posterior)) - acceptance_rate)
                temp_vec = beta[i] - prior_mu
                prior_sigma[j] = prior_sigma[j] + self._gamma_func(l + 1) * (temp_vec[j] ** 2 - prior_sigma[j])
                prior_mu[j] = prior_mu[j] + self._gamma_func(l + 1) * (nu_beta[j] - prior_mu[j])
                l += 1

        self.beta[chain_idx] = beta
        self.accept[chain_idx] = np.array(accept) / self.mcmcsize
        self.trueindex[chain_idx] = self.mcmcsize

    def _gamma_func(self, l: float, alpha: float = 0.5) -> float:
        """
        감마 함수: Adaptive MCMC에서 사용되는 감소 함수

        Parameters
        ----------
        l : float
            현재 단계
        alpha : float, optional
            감소 파라미터 | Default: 0.5

        Returns
        -------
        float
            감소 함수 값
        """
        return l ** (-alpha)

    ##########
    # Report Part (Internal Method)

    def _find_first_non_positive(self, sequence):
        """
        주어진 시퀀스에서 인접한 두 요소의 합이 처음으로 0 이하가 되는 위치의 인덱스를 반환

        Parameters
        ----------
        sequence : list
            숫자로 이루어진 리스트.

        Returns
        -------
        int
            합이 0 이하로 떨어지는 최초 위치의 인덱스. 찾지 못하면 len(sequence)를 반환
        """
        for i in range(0, len(sequence) - 1, 2):
            # 짝수번째와 다음 요소의 합 계산
            total_sum = sequence[i] + sequence[i + 1]

            if total_sum <= 0:
                return i  # 짝수번째 숫자의 인덱스 반환

        return len(sequence)  # 찾지 못했을 경우 전체 인덱스의 길이 반환

    def _autocorrelation(self, samples: np.ndarray, lag: int) -> float:
        """
        주어진 시차(lag)에 대한 자기상관계수를 계산

        Parameters
        ----------
        samples : numpy array
            샘플 데이터 배열
        lag : int
            시차(lag)

        Returns
        -------
        float
            계산된 자기상관계수
        """
        n = len(samples)
        mean = np.mean(samples)

        # 자기상관 계산
        numerator = np.sum((samples[:n - lag] - mean) * (samples[lag:] - mean))
        denominator = np.sum((samples - mean) ** 2)

        return numerator / denominator

    def _effective_sample_size(self, samples: np.ndarray) -> float:
        """
        주어진 샘플의 유효 샘플 크기(ESS; Effective Sample Size)를 계산

        Parameters
        ----------
        samples : numpy array
            샘플 데이터 배열

        Returns
        -------
        float
            유효 샘플 크기
        """
        n = len(samples)
        acf = [self._autocorrelation(samples, lag) for lag in range(1, len(samples))]
        max_lag = self._find_first_non_positive(acf)
        acf_values = acf[:max_lag]
        ess = n / (1 + 2 * np.sum(acf_values))
        return ess

    def _calculate_summary_stats(self, burn_in: int = 0) -> dict:
        """
        모델 평가를 위한 통계량들을 계산하여 반환

        Parameters
        ----------
        burn_in : int, optional
            burn-in의 수 | Default: 0

        Returns
        -------
        summary_stats : dict
            계산된 통계량들을 포함한 딕셔너리
        """
        # 모든 체인의 베타를 결합
        combined_beta = np.vstack(
            [chain[burn_in:self.trueindex[chain_idx], :] for chain_idx, chain in enumerate(self.beta)])

        # posterior mean
        posterior_mean = np.mean(combined_beta, axis=0)

        # posterior variance
        posterior_var = np.var(combined_beta, axis=0)

        # hpd interval
        lower_bound = np.percentile(combined_beta, 2.5, axis=0)
        upper_bound = np.percentile(combined_beta, 97.5, axis=0)

        # BIC 계산 (단일 체인 기반)
        N, C = self.X.shape
        log_likelihood = np.sum(np.log(1 / (1 + np.exp(-self.X.dot(posterior_mean)))) * self.y)
        BIC = -2 * log_likelihood + C * np.log(N)

        # ESS 계산
        ESS = []
        for i in range(C):
            samples = combined_beta[:, i]
            ESS.append(self._effective_sample_size(samples))

        # acceptance rate: 각 파라미터별 평균 수락율
        acceptance_rate = np.mean(np.stack(self.accept, axis=0), axis=0)  # shape: (C,)

        summary_stats = {
            "posterior_mean": posterior_mean,
            "posterior_variance": posterior_var,
            "hpd_lower_bound": lower_bound,
            "hpd_upper_bound": upper_bound,
            "ESS": ESS,
            "acceptance_rate": acceptance_rate,
            "BIC": BIC
        }

        return summary_stats
