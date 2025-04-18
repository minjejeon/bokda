from typing import Union, List
import time

import numpy as np
import pandas as pd
from scipy.stats import norm


def get_p_value_significance_star(p_value):
    """
    p-value의 크기에 따른 유의미성을 별표로 표시

    Parameters
    ----------
    p_value : float
        p-value 값

    Returns
    -------
    str
        유의미성에 따른 별표 표시
    """
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return ''


class ProbitRegressionFreq:
    """
    빈도론적 Probit 회귀 모델 클래스

    Attributes
    ----------
    method : str
        최적화 방법
    tolerance : float
        계수 벡터의 수렴 기준
    max_iter : int
        최대 반복 횟수
    verbose : bool
        과정 출력 여부
    is_fitted : bool
        모델 적합 여부
    beta : np.ndarray
        회귀 계수 벡터
    columns : list of str
        변수명 리스트
    add_const : bool
        상수항 추가 여부
    summary_stats : dict
        모델 적합 결과 통계량
    """

    def __init__(self, method: str = 'newton', tolerance: float = 1e-6, max_iter: int = 300, verbose: bool = False):
        """
        Probit 회귀 모델을 초기화

        Parameters
        ----------
        method : str, optional
            최적화 방법 설정 | Default: 'newton' (NOTE: stata와 동일한 default)
        tolerance : float, optional
            계수 벡터의 수렴 기준(tolerance) | Default: 1e-6
        max_iter : int, optional
            최대 반복 횟수 | Default: 300 (NOTE: stata와 동일한 default)
        verbose : bool, optional
            과정을 출력할 지 여부 | Default: False
        """
        self.method = method
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.beta = None
        self.is_fitted = False  # initialize시 fitted flag를 False로 설정
        self.columns = []
        self.verbose = verbose
        self.add_const = None  # 상수항이 포함되었는지 여부 (fit에서 업데이트됨)

        # PRINT: model init
        if self.verbose:
            print("> Initializing Model...")
            print(f"  - Method: {self.method}")
            print(f"  - Tolerance: {self.tolerance}")
            print(f"  - Max Iterations: {self.max_iter}")
            print(f"  - Verbose: {self.verbose}")

    ###############
    # Public Method
    ###############

    def fit(self, X: Union[np.ndarray, pd.DataFrame],
            y: Union[np.ndarray, pd.Series],
            columns: List[str] = [], add_const: bool = True) -> 'ProbitRegressionFreq':
        """
        주어진 데이터로 Probit 회귀 모델을 적합
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
            for col in X.columns:
                if X[col].dtype == 'bool':  # True/False인 경우 1/0으로 변환
                    X[col] = X[col].astype(int)
                elif X[col].dtype == 'O' and set(X[col].unique()) == {"yes", "no"}:  # "yes"/"no"인 경우 1/0으로 변환
                    X[col] = (X[col] == "yes").astype(int)

            X = X.to_numpy()

        if isinstance(y, pd.Series):
            y = y.to_numpy()

        # 3) numpy array에 문제가 없는지 확인
        self._check_input_array(X, y)

        # 4) 상수항 추가
        self.add_const = add_const
        if add_const:
            X = np.hstack([np.ones((X.shape[0], 1)), X])

        # PRINT: start
        if self.verbose:
            print("> Start Fitting Model...")
            print(f"  - Input Data: {X.shape[0]} samples, {X.shape[1]} features")
            if self.add_const:
                print("  - Adding constant term to the model")
            print(f"  - Optimization Method: {self.method}")

        # 5) 적합
        if self.method == 'newton':
            # NOTE: 현재 뉴턴-랩슨 방법만 구현됨
            self._fit_newton(X, y, tolerance=self.tolerance, max_iter=self.max_iter)
        else:
            raise ValueError(f"지원하지 않는 Method 입니다: '{self.method}'")

        # 6) summary stats 계산
        self.is_fitted = True  # fit 완료 이후 flag 설정
        self._calculate_summary_stats(X, y)

        # PRINT: Done
        if self.verbose:
            print("> Model Fitted.")
            print(f"  - Final Log-Likelihood: {self.summary_stats['log_likelihood']:.4f}")

        return self

    def get_beta(self, series: bool = True):
        """
        적합이 완료된 모델의 계수를 반환 (계수만 반환)

        Parameters
        ----------
        series : bool, optional
            pandas Series로 반환할 지 여부 | Default: True

        Returns
        -------
        beta : pandas Series or dict
            적합이 완료된 모델의 계수 반환
        """
        if not self.is_fitted:
            raise ValueError("모델을 fit 해야 계수를 반환할 수 있습니다.")

        column_names = self._get_column_names()

        if series:
            beta_series = pd.Series(self.beta.tolist(), index=column_names)  # index가 컬럼명, value가 계수
            return beta_series
        else:
            beta_dict = dict(zip(column_names, self.beta.tolist()))  # key가 컬럼명, value가 계수
            return beta_dict

    def get_beta_table(self, df: bool = True):
        """
        회귀 계수 테이블을 반환
        NOTE: print_summary 시 출력되는 회귀 계수 테이블을 데이터로 반환받을 수 있는 함수

        Parameters
        ----------
        df : bool, optional
            데이터프레임으로 반환할 지 여부 | Default: True

        Returns
        -------
        table : pandas DataFrame or dict
            회귀 계수 테이블
        """
        if not self.is_fitted:
            raise ValueError("모델을 fit 해야 회귀 계수 테이블을 반환할 수 있습니다.")

        table = {}
        table["Variable"] = self._get_column_names()
        table["Beta"] = self.beta.tolist()
        table["StdErr"] = self.summary_stats["std_errors"]
        table["z-value"] = self.summary_stats["z_values"]
        table["p-value"] = self.summary_stats["p_values"]

        if df:
            return pd.DataFrame(table)
        else:
            return table

    def get_fit_stats(self, series: bool = True):
        """
        모델 적합 통계량을 반환

        Parameters
        ----------
        series : bool, optional
            pandas Series로 반환할 지 여부 | Default: True

        Returns
        -------
        stats : pandas Series or dict
            모델 적합 통계량
        """
        if not self.is_fitted:
            raise ValueError("모델을 fit 해야 통계량을 반환할 수 있습니다.")

        stats = self.summary_stats.copy()
        stats.pop("std_errors")
        stats.pop("z_values")
        stats.pop("p_values")

        if series:
            return pd.Series(stats)
        else:
            return stats

    def print_summary(self, digits: int = 4):
        """
        모델 적합 결과의 요약 정보를 출력

        Parameters
        ----------
        digits : int, optional
            출력 값을 소수점 아래로 몇 자리까지 출력할 지 여부 | Default: 4

        """
        if not self.is_fitted:
            raise ValueError("모델을 fit 해야 통계량을 반환할 수 있습니다.")

        # 계수와 관련된 통계량을 준비
        column_names = self._get_column_names()
        beta = self.beta.tolist()
        std_errors = self.summary_stats["std_errors"]
        z_values = self.summary_stats["z_values"]
        p_values = self.summary_stats["p_values"]
        significance = [get_p_value_significance_star(p) for p in p_values]

        # 단일 통계량
        single_stats = {
            "Num of Observations": self.summary_stats["num_observations"],
            "Degrees of Freedom (Residual)": self.summary_stats["dof_residual"],
            "Log-Likelihood": self.summary_stats["log_likelihood"],
            "Pseudo R-squared": self.summary_stats["pseudo_r_squared"],
            "AIC": self.summary_stats["aic"],
        }

        # 계수 테이블
        header = ["Variable", "Beta", "Std Err", "z", "p-value", ""]
        # 컬럼 너비 계산
        col_widths = [len(h) for h in header]

        # 데이터 준비 및 컬럼 너비 계산
        data_rows = []
        for i in range(len(beta)):
            # 값 포맷팅
            est = f"{beta[i]:>.{digits}f}"
            se = f"{std_errors[i]:>.{digits}f}"
            z = f"{z_values[i]:>.{digits}f}"

            # p-value 포맷팅
            p = p_values[i]
            if p < 10 ** (-digits):  # p가 0.00001과 같은 경우, 0.0000 으로 표시되도록
                p_print = f"{0:.{digits}f}"
            else:
                p_print = f"{p:>.{digits}f}"

            sig = " " + significance[i]
            data_rows.append([column_names[i], est, se, z, p_print, sig])

            # 컬럼 너비 업데이트
            for j, val in enumerate(data_rows[-1]):
                val = str(val)
                val_len = len(val)
                col_widths[j] = max(col_widths[j], val_len + 1)

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
                str(row[i]).ljust(col_widths[i]) if i == 0 or i == (len(header) - 1) else str(row[i]).rjust(
                    col_widths[i])
                for i in range(len(row))
            )
            print(line)

        print("-" * len(header_line))
        print("signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05")
        print("-" * len(header_line))

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
            column_length = len(self.beta)
            if self.add_const:
                column_length -= 1
                column_names = ["const"] + [f"x{i + 1}" for i in range(column_length)]
            else:
                column_names = [f"x{i + 1}" for i in range(column_length)]

        return column_names

    def _calculate_gradient(self, X: np.ndarray, y: np.ndarray, linear_pred: np.ndarray) -> np.ndarray:
        """
        로그 우도 함수의 기울기(gradient)를 계산

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            입력 데이터 행렬
        y : numpy array of shape (n_samples,)
            실제 타겟 레이블 벡터
        linear_pred : numpy array of shape (n_samples,)
            예측 선형 결합 벡터 X @ beta

        Returns
        -------
        gradient : numpy array of shape (n_features,)
            기울기 벡터
        """
        pdf = norm.pdf(linear_pred)  # phi(X * beta)
        cdf = norm.cdf(linear_pred)  # Phi(X * beta)

        # 그라디언트 계산
        gradient = X.T @ (pdf * ((y / cdf) - (1 - y) / (1 - cdf)))
        return gradient

    def _calculate_hessian(self, X: np.ndarray, linear_pred: np.ndarray) -> np.ndarray:
        """
        헤시안 행렬(Hessian matrix)을 계산

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            입력 데이터 행렬
        linear_pred : numpy array of shape (n_samples,)
            예측 선형 결합 벡터 X @ beta

        Returns
        -------
        hessian : numpy array of shape (n_features, n_features)
            헤시안 행렬
        """
        pdf = norm.pdf(linear_pred)  # phi(X * beta)
        cdf = norm.cdf(linear_pred)  # Phi(X * beta)

        diag = (pdf ** 2) / (cdf * (1 - cdf))
        W = np.diag(diag)  # 가중치 행렬

        # 헤시안 행렬 계산
        hessian = (X.T @ W @ X)
        return hessian

    def _fit_newton(self, X: np.ndarray, y: np.ndarray, tolerance: float, max_iter: int) -> None:
        """
        뉴턴-랩슨 방법으로 Probit 모델을 적합

        NOTE: 적합된 계수는 self.beta에 저장됨

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            입력 데이터 행렬
        y : numpy array of shape (n_samples,)
            타겟 레이블 벡터
        tolerance : float
            수렴 기준
        max_iter : int
            최대 반복 수

        Raises
        ------
        np.linalg.LinAlgError
            헤시안 행렬이 특이행렬인 경우 발생
        """
        _, n_features = X.shape

        # 파라미터 초기화
        self.beta = np.zeros(n_features)

        start_time = time.time()
        for iteration in range(max_iter):
            # 선형 결합 계산
            linear_pred = X @ self.beta

            # 그라디언트 및 헤시안 계산
            gradient = self._calculate_gradient(X, y, linear_pred)
            hessian = self._calculate_hessian(X, linear_pred)

            # 파라미터 업데이트
            try:
                delta = np.linalg.solve(hessian, gradient)  # hessian * delta = gradient의 해를 구함
                self.beta += delta

            except np.linalg.LinAlgError:
                raise np.linalg.LinAlgError("Hessian matrix가 특이행렬입니다.")

            # 수렴 여부 확인
            is_converged = np.linalg.norm(delta, ord=1) <= tolerance

            if is_converged:
                if self.verbose:
                    elapsed_time = time.time() - start_time
                    print(f"> Converged at iteration {iteration + 1} / {self.max_iter} (elapsed: {elapsed_time:.4f}s)")
                break
        else:
            print(f"Warning: 뉴턴-랩슨 방법을 통해 수렴하지 않았습니다 / max-iter: {self.max_iter}")

    def _calculate_summary_stats(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        모델 평가를 위한 통계량들을 계산하여 self.summary_stats에 저장

        Parameters
        ----------
        X : numpy array
            입력 데이터 행렬
        y : numpy array
            타겟 레이블 벡터
        """
        # 예측 확률 계산
        linear_pred = X @ self.beta
        y_pred = norm.cdf(linear_pred)

        # 로그 우도 계산
        log_likelihood = np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

        # 자유도
        n_samples, n_features = X.shape
        dof_model = n_features - 1 if self.add_const else n_features
        dof_residual = n_samples - dof_model - 1

        # AIC (Akaike Information Criterion) 계산
        aic = 2 * n_features - 2 * log_likelihood

        # Pseudo R-squared 계산 (McFadden's R-squared)
        # Null model에서 로그 우도 계산 (상수항만 있는 모델)
        y_mean = np.mean(y)
        null_log_likelihood = np.sum(y * np.log(y_mean) + (1 - y) * np.log(1 - y_mean))
        pseudo_r_squared = 1 - (log_likelihood / null_log_likelihood)

        # 각 계수의 표준 오차 계산
        hessian = self._calculate_hessian(X, linear_pred)
        covariance_matrix = np.linalg.inv(hessian)  # 공분산 행렬 계산
        std_errors = np.sqrt(np.diag(covariance_matrix))  # 표준 오차는 공분산 행렬의 대각선 값의 제곱근

        # z-값과 p-값 계산
        z_values = self.beta / std_errors
        p_values = 2 * (1 - norm.cdf(np.abs(z_values)))  # 양측 검정이므로 2배

        # 계산된 통계량을 저장
        self.summary_stats = {
            "log_likelihood": float(log_likelihood),
            "pseudo_r_squared": float(pseudo_r_squared),
            "aic": float(aic),
            "std_errors": std_errors.tolist(),
            "z_values": z_values.tolist(),
            "p_values": p_values.tolist(),
            "dof_model": int(dof_model),
            "dof_residual": int(dof_residual),
            "num_observations": int(n_samples),
        }
