# HLW

> **HLW 모델 클래스 사용 방법 매뉴얼**

---

### 개요
- `HLW` 클래스는 HLW 모델을 사용하여 주어진 경제 데이터를 기반으로 중립금리를 추정하는 데 사용됩니다.
- MCMC(Markov Chain Monte Carlo) 알고리즘을 활용하여 모델의 파라미터와 잠재 변수를 샘플링합니다.
- 다양한 사전 분포와 초기 파라미터 설정을 지원하며, 모델 적합 후 결과를 시각화할 수 있는 기능을 제공합니다.
- 모델 적합 과정 중 수락 비율(Acceptance Ratio)을 모니터링하여 샘플링의 효율성을 평가할 수 있습니다.
- 해당 모델은 N_T 길이의 시계열 데이터에 대해서 MCMC sampling 을 이용한 추정 방법을 적용한 모델입니다. 따라서 모델의 computation time 은 N_T 가 길어질 수록 매우 길어질 수 있으니, 모델 적합 시에 데이터 크기와 iteration 수를 종합적으로 고려하여야 합니다.
---

### Model Initialization

`HLW` 클래스는 HLW 모델을 초기화합니다.

| **Argument**         | **Type**               | **설명**                                                                                                                                                     | **기본값**           | **예시**                                                                                                   |
|----------------------|------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------|------------------------------------------------------------------------------------------------------------|
| `n_draws`            | `int`                  | 총 MCMC 반복 횟수입니다.                                                                                                                                         | `20000`              | `50000`                                                                                                    |
| `burnin`             | `int`                  | 초기 번인(burn-in) 기간의 길이입니다. 실제 모델 파라미터 추정을 위한 샘플 수는 `numMCMC = n_draws - burnin`으로 계산됩니다.                                  | `10000`              | `15000`                                                                                                    |
| `PercPara`           | `list[float]`          | 모델 파라미터의 백분위수 계산을 위한 하한 및 상한입니다. 예: `[0.05, 0.95]`는 5%와 95% 백분위수를 계산합니다.                                               | `[0.05, 0.95]`       | `[0.025, 0.975]`                                                                                           |
| `PercLV`             | `list[float]`          | 잠재 변수의 백분위수 계산을 위한 하한 및 상한입니다. 예: `[5, 95]`는 5%와 95% 백분위수를 계산합니다.                                                       | `[5, 95]`            | `[2.5, 97.5]`                                                                                              |
| `Ytivs`              | `float`                | 출력 추세(output trend)의 초기값에 대한 사전 분산(prior variance) 스케일 팩터입니다. 기본값은 10000입니다 (100²).                                               | `10000`              | `5000`                                                                                                      |
| `Ygivs`              | `float`                | 출력 장기 성장률(long-term growth rate)의 초기값에 대한 사전 분산 스케일 팩터입니다. 기본값은 1입니다 (1²).                                                    | `1`                  | `0.5`                                                                                                       |
| `Izivs`              | `float`                | 이자율 z 구성 요소의 초기값에 대한 사전 분산 스케일 팩터입니다. 기본값은 1입니다 (1²).                                                                        | `1`                  | `0.2`                                                                                                       |
| `SSRatio`            | `float`                | 잠재 변수의 초기값의 사전 분산을 계산하기 위해 사용할 데이터의 비율입니다. 기본값은 0.1입니다.                                                              | `0.1`                | `0.2`                                                                                                       |
| `PriorPara`          | `dict`, optional       | 파라미터에 대한 사전 분포(prior distribution) 정보를 포함하는 딕셔너리입니다. 사용자가 지정하지 않으면 기본값이 사용됩니다.                                   | `DefaultPriorPara`               | `{'yc_mean': np.array([0.6, -0.4, 0.1]), 'yc_variance': np.diag(np.array([0.4, 0.4, 9]) ** 2), ...}` |
| `InitialPara`        | `dict`, optional       | 파라미터의 초기값을 포함하는 딕셔너리입니다. 사용자가 지정하지 않으면 기본값이 사용됩니다.                                                                  | `DefaultInitialPara`               | `{'phi1_yc': 1.0, 'phi2_yc': -0.3, ... }` |
| `MCMCAlgorithm`      | `dict`, optional       | MCMC 알고리즘 설정을 포함하는 딕셔너리입니다. 각 파라미터에 대해 MCMC 알고리즘을 사용할지 여부를 지정합니다. 사용자가 지정하지 않으면 기본값이 사용됩니다. | `DefaultMCMCAlgorithm`               | `{'sig2_g': 1, 'sig2_yt': 0,...}` |
| `ZSpec`              | `str`                  | 이자율 z 구성 요소의 모델 사양을 지정합니다. 가능한 값은 `'rw'` (랜덤 워크), `'sar_c'` (self-adjusting rate with constant), `'sar'` (self-adjusting rate)입니다. 기본값은 `'rw'`입니다. | `'rw'`               | `'sar_c'`                                                                                                   |
| `MHSample_size` | `int` | adaptive MH 과정에서 공분산 계산에 사용되는 샘플 크기입니다. 기본값은 1000입니다. | `1000` | `2000` |
| `verbose`            | `bool`                 | 모델 실행 중간 과정 및 결과를 출력할지 여부를 지정합니다. 기본값은 `True`입니다.                                                                        | `True`               | `False`                                                                                                     |

#### 예시
```python
from bok_da import HLW

# 기본 설정으로 모델 초기화
model = HLW()

# 사용자 정의 설정으로 모델 초기화
model = HLW(
    n_draws=50000,
    burnin=15000,
    PercPara=[0.025, 0.975],
    PercLV=[2.5, 97.5],
    Ytivs=5000,
    Ygivs=0.5,
    Izivs=0.2,
    SSRatio=0.2,
    PriorPara={
        'yc_mean': np.array([0.6, -0.4, 0.1]),
        'yc_variance': np.diag(np.array([0.4, 0.4, 9]) ** 2),
        # ... 기타 사전 분포 파라미터
    },
    InitialPara={
        'phi1_yc': 1.0,
        'phi2_yc': -0.3,
        # ... 기타 초기 파라미터
    },
    MCMCAlgorithm={
        'sig2_g': 1,
        'sig2_yt': 0,
        # ... 기타 파라미터
        # 0: gibbs sampler
        # 1: MH sampler
        # gibbs sampler 에는 local trap problem 이 존재할 수 있습니다.
    },
    ZSpec='sar_c',
    verbose=True
)
```

---

### Public Methods

#### `fit()`
주어진 데이터에 HLW 모델을 적합시킵니다.

| **Argument** | **Type**           | **설명**                                                                                                                                          | **기본값** | **예시**                                |
|--------------|--------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|------------|-----------------------------------------|
| `data`       | `pd.DataFrame` 또는 `np.ndarray` | 입력 데이터 프레임 또는 numpy array. 컬럼 구성은 반드시 `[ln real gdp, inflation rate, real interest rate]`이어야 하며, 시간 기준 오름차순으로 정렬되어야 합니다. | -          | `pd.DataFrame({...})` 또는 `np.array([...])` |
| `dates`      | `array-like`, optional | 데이터에 매핑되는 날짜입니다. 정렬 순서와 크기가 `data`와 동일해야 합니다.                                                                        | `None`     | `pd.date_range(start='2000-01-01', periods=100, freq='M')` |

##### Data Input
data는 반드시 아래의 column 으로 구성된 데이터 프레임 또는 numpy array 이어야 합니다.
| **Column name**| ln real gdp | inflation rate | real interest rate |
|--------------|---------------|----------------|---------------------|
| **설명**      |     실질 gpd 의 로그변환 값     |  인플레이션 증가율   |    실질이자율         |
| **Type**     |          float |   float             |       float              |
| **예시**      |     3292.672518      |      1.480963449  |         1.084134677      |

##### Output
- `None`: 모델 적합이 완료되며, 결과는 클래스의 속성에 저장됩니다.

##### 예시
```python
import pandas as pd
from bok_da import HLW

# 데이터 생성 (예시)
data = pd.DataFrame({
    'ln_real_gdp': np.log(np.random.rand(100) * 100 + 50),
    'inflation_rate': np.random.rand(100) * 5,
    'real_interest_rate': np.random.rand(100) * 3
})
dates = pd.date_range(start='2000-01-01', periods=100, freq='M')

# 모델 초기화
model = HLW(verbose=True)

# 모델 적합
model.fit(data, dates=dates)
```

#### `print_results()`
모델 적합 결과를 시각화하여 출력합니다. 잠재 변수들의 추정치를 그래프로 표시합니다.

| **Argument** | **Type** | **설명**                                        | **기본값** | **예시** |
|--------------|----------|-------------------------------------------------|------------|----------|
| 없음         | -        | -                                               | -          | `model.print_results()` |

##### Output
- 잠재 변수들의 분포를 나타내는 그래프가 출력됩니다.

##### 예시
```python
# 모델 적합 후 결과 시각화
model.print_results()
```

---

### Class Variables

모델 클래스 내의 전체 변수 목록입니다. Container Class에 대한 설명은 `Container Class.md`파일을 참고해주세요.

#### Container Class


- `ParaStorage`: MCMC 샘플링을 통해 얻은 파라미터 샘플들을 저장하는 Container 클래스 인스턴스입니다.
  - `phi_yc`: (`numMCMC, 2` 형태)
  - `beta_yc`: (`numMCMC, 1` 형태)
  - `sig2_yc`, `sig2_yt`, `sig2_g`, `sig2_p`, `sig2_z`: 각각 (`numMCMC, 1` 형태)
  - `phi_p`: (`numMCMC, 1` 형태)
  - `beta_p`: (`numMCMC, 1` 형태)

- `SummaryParas`: 파라미터 샘플들의 통계 요약 정보를 저장하는 Container 클래스 인스턴스입니다.
  - `Avg`
  - `Med`
  - `PercLow`
  - `PercUpper`

- `LVariableStorage`: MCMC 샘플링을 통해 얻은 잠재 변수 샘플들을 저장하는 Container 클래스 인스턴스입니다.
  - `yt`: (`numMCMC, N_T` 형태)
  - `yc`: (`numMCMC, N_T` 형태)
  - `g`: (`numMCMC, N_T` 형태)
  - `z`: (`numMCMC, N_T` 형태)
  - `rNatural`: (`numMCMC, N_T` 형태)
  - `rCycle`: (`numMCMC, N_T` 형태)

- `SummaryLV`: 잠재 변수 샘플들의 통계 요약 정보를 저장하는 Container 클래스 인스턴스입니다.
  - `Avg`
  - `Med`
  - `PercLower`
  - `PercUpper`

#### Variables

- `verbose` (`bool`): 모델 실행 중간 과정 및 결과를 출력할지 여부를 지정합니다.
- `numMCMC` (`int`): 번인(burn-in) 기간 이후의 MCMC 샘플 수입니다.
- `burnin` (`int`): MCMC 알고리즘에서 초기 번인 기간의 길이입니다.
- `PercPara` (`list[float]`): 모델 파라미터의 백분위수 계산을 위한 하한 및 상한입니다.
- `PercLV` (`list[float]`): 잠재 변수의 백분위수 계산을 위한 하한 및 상한입니다.
- `Ytivs` (`float`): 출력 추세의 초기값에 대한 사전 분산 스케일 팩터입니다.
- `Ygivs` (`float`): 출력 장기 성장률의 초기값에 대한 사전 분산 스케일 팩터입니다.
- `Izivs` (`float`): 이자율 z 구성 요소의 초기값에 대한 사전 분산 스케일 팩터입니다.
- `SSRatio` (`float`): 잠재 변수의 초기값의 사전 분산을 계산하기 위해 사용할 데이터의 비율입니다.
- `ZSpec` (`str`): 이자율 z 구성 요소의 모델 사양입니다 (`'rw'`, `'sar_c'`, `'sar'`).
- `MergedMCMCAlgorithm` (`dict`): MCMC 알고리즘 설정을 저장하는 딕셔너리입니다.
- `MergedPriorPara` (`dict`): 파라미터에 대한 사전 분포 정보를 저장하는 딕셔너리입니다.
- `MergedInitialPara` (`dict`): 파라미터의 초기값을 저장하는 딕셔너리입니다.
- `dates` (`np.ndarray`): 입력 데이터의 날짜 정보를 저장하는 배열입니다.
- `Accept_ratio` (`dict`): MCMC 알고리즘에서 각 파라미터의 샘플 수락 비율을 저장하는 딕셔너리입니다.
- `is_fitted` (`bool`): 모델 적합 여부를 나타냅니다.
- `MHSample_size` (`int`): adaptive MH 과정에서 공분산 계산에 사용되는 샘플 크기입니다.

---

### Usage (사용 방법)

```python
import numpy as np
import pandas as pd
from bok_da import HLW

# 데이터 생성 (예시)
np.random.seed(0)
data = pd.DataFrame({
    'ln_real_gdp': np.log(np.random.rand(100) * 100 + 50),
    'inflation_rate': np.random.rand(100) * 5,
    'real_interest_rate': np.random.rand(100) * 3
})
dates = pd.date_range(start='2000-01-01', periods=100, freq='M')

# 모델 초기화
model = HLW(
    n_draws=50000,
    burnin=15000,
    PercPara=[0.025, 0.975],
    PercLV=[2.5, 97.5],
    Ytivs=5000,
    Ygivs=0.5,
    Izivs=0.2,
    SSRatio=0.2,
    PriorPara={
        'yc_mean': np.array([0.6, -0.4, 0.1]),
        'yc_variance': np.diag(np.array([0.4, 0.4, 9]) ** 2),
        # ... 기타 사전 분포 파라미터
    },
    InitialPara={
        'phi1_yc': 1.0,
        'phi2_yc': -0.3,
        # ... 기타 초기 파라미터
    },
    MCMCAlgorithm={
        'sig2_g': 1,
        'sig2_yt': 0,
        # ... 기타 파라미터
    },
    ZSpec='sar_c',
    verbose=True
)

# 모델 적합
model.fit(data, dates=dates)

# 결과 출력
model.print_results()

# 샘플된 파라미터 접근 예시
phi_yc_samples = model.ParaStorage['phi_yc']
beta_yc_samples = model.ParaStorage['beta_yc']

# 샘플된 잠재 변수 접근 예시
yt_samples = model.LVariableStorage['yt']
yc_samples = model.LVariableStorage['yc']

# 통계 요약 정보 접근 예시
summary_params = model.SummaryParas
summary_lv = model.SummaryLV

# 수락 비율 확인
accept_ratios = model.Accept_ratio
print(accept_ratios)
```
---

### Reference
- Durbin, James and Siem Jan Koopman. 2002. “A simple and eﬃcient simulation smoother
for state space time series analysis.” Biometrika 89:603–616.
- Haario, Heikki, Eero Saksman, and Johanna Tamminen. 2001. “An adaptive Metropolis
algorithm.” Bernoulli 7:223–242.
- Holston, Kathryn, Thomas Laubach, and John C Williams. 2017. “Measuring the natural
rate of interest: International trends and determinants.” Journal of International Eco-
nomics 108:S59–S75.
- Lewis, Kurt F and Francisco Vazquez-Grande. 2019. “Measuring the natural rate of interest:
A note on transitory shocks.” Journal of Applied Econometrics 34:425–436.
