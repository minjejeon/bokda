# ProbitRegressionBayes

> 베이지안 Probit 회귀 모델 클래스 사용 방법 매뉴얼

### 개요
- `ProbitRegressionBayes` 클래스는 베이지안 접근 방식을 이용한 Probit 회귀 모델을 구현합니다.
- MCMC 및 Adaptive MCMC 방법을 통해 모델의 파라미터를 추정합니다.
- Weakly Informative Prior, Normal Prior with Known/Unknown Sigma 등 다양한 사전 분포를 지원합니다.
- 모델 적합 후 다양한 진단 플롯과 요약 정보를 제공합니다.

---

### Model Initialization

| **Argument**          | **Type**                                                                 | **설명**                                                                                                                                                              | **기본값**              | **예시**                                 |
|-----------------------|--------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------|------------------------------------------|
| `mcmcsize`            | `int`                                                                    | MCMC 체인의 길이 (반복 횟수)                                                                                                                                          | -                       | `10000`                                  |
| `n_chains`            | `int`                                                                    | 실행할 MCMC 체인의 수                                                                                                                                                 | `1`                     | `2`                                      |
| `method`              | `str` (`'MCMC'` 또는 `'adaptiveMCMC'`)                                   | MCMC 샘플링 방법 선택                                                                                                                                                 | `'adaptiveMCMC'`        | `'MCMC'`                                 |
| `thinning`            | `int`                                                                    | MCMC 체인의 thinning 간격 (1이면 thinning 없음)                                                                                                                       | `1`                     | `2`                                      |
| `prior`               | `str` (`'weaklyinformative'`, `'normal_known_sigma'`, `'normal_unknown_sigma'`) | 사용할 사전 분포 설정                                                                                                                                                | `'weaklyinformative'`   | `'normal_known_sigma'`                   |
| `prior_location`      | `list[float]`                                                            | 사전 분포의 위치 파라미터 (`mu`), 각 특징량에 대해 지정                                                                                                               | 빈 리스트 (`[]`)        | `[0.0, 0.0, 0.0]`                        |
| `prior_scale`         | `list[float]`                                                            | 사전 분포의 스케일 파라미터 (`sigma`), 각 특징량에 대해 지정                                                                                                           | 빈 리스트 (`[]`)        | `[1.0, 1.0, 1.0]`                        |
| `prior_alpha`         | `float`                                                                  | `normal_unknown_sigma`인 경우, Inverse Gamma 사전 분포의 `alpha` 파라미터                                                                                             | `0.001`                 | `0.5`                                    |
| `prior_beta`          | `float`                                                                  | `normal_unknown_sigma`인 경우, Inverse Gamma 사전 분포의 `beta` 파라미터                                                                                              | `0.001`                 | `0.5`                                    |
| `jumping_rate`        | `float`                                                                  | MCMC 체인의 점프 크기 설정                                                                                                                                            | `0.1`                   | `0.05`                                   |
| `acceptance_rate`     | `float`                                                                  | `adaptiveMCMC` 방법에서의 목표 수락 비율                                                                                                                              | `0.234`                 | `0.3`                                    |
| `seed`                | `int` 또는 `None`                                                        | 랜덤 샘플링을 위한 시드 값 설정                                                                                                                                       | `None`                  | `42`                                     |
| `verbose`             | `bool`                                                                   | 과정 출력 여부                                                                                                                                                        | `False`                 | `True`                                   |

#### 예시
```python
from bok_da import ProbitRegressionBayes

# 기본 설정으로 모델 초기화
model = ProbitRegressionBayes(mcmcsize=10000)

# Adaptive MCMC 방법과 특정 사전 분포로 모델 초기화
model = ProbitRegressionBayes(
    mcmcsize=5000,
    n_chains=2,
    method='adaptiveMCMC',
    prior='normal_known_sigma',
    prior_location=[0.0, 0.0, 0.0],
    prior_scale=[1.0, 1.0, 1.0],
    verbose=True
)
```

---

### Public Methods

#### `fit()`
주어진 데이터로 베이지안 Probit 회귀 모델을 적합합니다.

| **Argument**      | **Type**                         | **설명**                                                                                         | **기본값** | **예시**                          |
|-------------------|----------------------------------|--------------------------------------------------------------------------------------------------|------------|-----------------------------------|
| `X`               | `np.ndarray` 또는 `pd.DataFrame` | 입력 데이터 행렬 (샘플 수 × 특징 수)                                                              | -          | `pd.DataFrame([[1,2],[3,4]])`      |
| `y`               | `np.ndarray` 또는 `pd.Series`    | 타겟 라벨 벡터 (0과 1로 구성)                                                                     | -          | `pd.Series([0, 1, 0, 1])`          |
| `columns`         | `list[str]`                      | 특징량의 이름 리스트 (결과 출력 시 사용)                                                          | `[]`       | `['age', 'income']`                |
| `add_const`       | `bool`                           | 상수항을 추가할지 여부                                                                            | `True`     | `False`                            |
| `standardization` | `bool`                           | 특징량을 표준화할지 여부                                                                          | `True`     | `False`                            |

##### Output
- `self`: 적합이 완료된 객체 자신을 반환합니다.

#### `print_summary()`
모델 적합 결과의 요약 정보를 출력합니다.

| **Argument** | **Type** | **설명**                                         | **기본값** | **예시** |
|--------------|----------|--------------------------------------------------|------------|----------|
| `digits`     | `int`    | 출력 값을 소수점 몇 자리까지 표시할지 지정 | `4`        | `6`      |
| `burn_in`     | `int`    | burn-in의 수 | `0`        | `1000`      |

##### Output
- 모델 요약 정보가 출력됩니다.

#### `traceplot()`
MCMC 체인의 트레이스플롯을 출력합니다.

| **Argument** | **Type**                | **설명**                                                                                      | **기본값** | **예시**     |
|--------------|-------------------------|------------------------------------------------------------------------------------------------|------------|--------------|
| `burn_in`     | `int`    | burn-in의 수 | `0`        | `1000`      |
| `multichain` | `bool`                  | 여러 체인이 있는 경우 각 체인을 모두 플롯할지 여부                                              | `False`    | `True`       |

##### Output
- 트레이스플롯이 출력됩니다.

#### `hpd_interval()`
베타 파라미터의 HPD(Highest Posterior Density) 구간을 플롯합니다.

| **Argument**  | **Type**                             | **설명**                                                                                      | **기본값** | **예시**     |
|---------------|--------------------------------------|------------------------------------------------------------------------------------------------|------------|--------------|
| `beta_chain`  | `np.ndarray`                          | MCMC 체인의 베타 값. 지정하지 않으면 모델의 체인을 사용함                                   | `None`          | `model.beta` |
| `hdi_prob`    | `float`                               | HPD 구간의 확률 값                                                                             | `0.95`     | `0.9`        |
| `burn_in`     | `int`    | burn-in의 수 | `0`        | `1000`      |
| `multichain`  | `bool`                                | 여러 체인이 있는 경우 각 체인을 모두 플롯할지 여부                                              | `False`    | `True`       |

##### Output
- HPD 구간 플롯이 출력됩니다.

#### `acf_plot()`
각 베타 파라미터의 자기상관 함수(ACF) 플롯을 출력합니다.

| **Argument**  | **Type**                                  | **설명**                                                                                      | **기본값** | **예시**     |
|---------------|-------------------------------------------|------------------------------------------------------------------------------------------------|------------|--------------|
| `beta_chain`  | `np.ndarray` 또는 `List[np.ndarray]`      | MCMC 체인의 베타 값. 지정하지 않으면 모델의 체인을 사용함                                                   | `None`         | `model.beta` |
| `burn_in`     | `int`    | burn-in의 수 | `0`        | `1000`      |
| `multichain`  | `bool`                                   | 여러 체인이 있는 경우 각 체인을 모두 플롯할지 여부                                              | `False`    | `True`       |
| `max_lag`     | `int`                                    | ACF 플롯의 최대 시차(lag)                                                                       | `40`       | `50`         |

##### Output
- ACF 플롯이 출력됩니다.

#### `density_plot()`
베타 파라미터의 밀도 플롯을 출력합니다.

| **Argument**  | **Type**                | **설명**                                                                                      | **기본값** | **예시**     |
|---------------|-------------------------|------------------------------------------------------------------------------------------------|------------|--------------|
| `burn_in`     | `int`    | burn-in의 수 | `0`        | `1000`      |
| `multichain`  | `bool`                  | 여러 체인이 있는 경우 각 체인을 모두 플롯할지 여부                                              | `False`    | `True`       |

##### Output
- 밀도 플롯이 출력됩니다.

---

### Class Variables

- `self.method` (`str`): MCMC 샘플링 방법.
- `self.mcmcsize` (`int`): MCMC 체인의 길이.
- `self.n_chains` (`int`): 실행한 MCMC 체인의 수.
- `self.thinning` (`int`): MCMC 체인의 thinning 간격.
- `self.prior` (`str`): 사전 분포의 종류.
- `self.prior_location` (`list[float]`): 사전 분포의 위치 파라미터.
- `self.prior_scale` (`list[float]`): 사전 분포의 스케일 파라미터.
- `self.prior_alpha` (`float`): Inverse Gamma 사전 분포의 `alpha` 파라미터.
- `self.prior_beta` (`float`): Inverse Gamma 사전 분포의 `beta` 파라미터.
- `self.jumping_rate` (`float`): MCMC 체인의 점프 크기.
- `self.acceptance_rate` (`float`): Adaptive MCMC의 목표 수락 비율.
- `self.seed` (`int`): 랜덤 시드 값.
- `self.verbose` (`bool`): 과정 출력 여부.
- `self.is_fitted` (`bool`): 모델 적합 여부.
- `self.columns` (`list[str]`): 특징량의 이름 리스트.
- `self.beta` (`np.ndarray`): 적합된 베타 파라미터 체인.
- `self.accept` (`np.ndarray`): 각 파라미터별 수락 비율.
- `self.add_const` (`bool`): 상수항 포함 여부.
- `self.summary_stats` (`dict`): 모델의 통계량 (posterior mean, variance 등).
- `self.sigma` (`np.ndarray`, optional): `normal_unknown_sigma` 사전 분포 사용 시, 적합된 시그마 파라미터 체인.
- `self.trueindex` (`List[int]`): 각 체인의 마지막 인덱스.
- `self.X` (`np.ndarray`): 적합에 사용된 입력 데이터.
- `self.y` (`np.ndarray`): 적합에 사용된 타겟 레이블.

---

### Usage (사용 방법)

```python
import numpy as np
import pandas as pd
from bok_da import ProbitRegressionBayes

# 데이터 생성
np.random.seed(0)
X = np.random.rand(100, 2)
y = (X[:, 0] + X[:, 1] > 1).astype(int)

# 모델 초기화 및 적합
model = ProbitRegressionBayes(
    mcmcsize=5000,
    n_chains=2,
    method='adaptiveMCMC',
    prior='weaklyinformative',
    verbose=True
)

model.fit(X, y, columns=['Feature1', 'Feature2'], add_const=True, standardization=True)

# 요약 정보 출력
model.print_summary()

# 트레이스플롯 출력
model.traceplot(burn_in=1000, multichain=True)

# HPD 구간 플롯
model.hpd_interval(hdi_prob=0.95, burn_in=1000, multichain=True)

# ACF 플롯
model.acf_plot(burn_in=1000, multichain=True)

# 밀도 플롯
model.density_plot(burn_in=1000, multichain=True)
```

---

### Reference
- Gelman, Andrew, et al. "A weakly informative default prior distribution for logistic and other regression models." (2008): 1360-1383.
