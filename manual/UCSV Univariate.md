## UnivarUCSV

> UnivarUCSV 모델 클래스 사용 방법 매뉴얼 (최신 초안)

### 개요
- `UnivarUCSV` 클래스는 Bayesian 접근 방식을 기반으로 한 univariate UCSVO 모델입니다. 
- Gibbs Sampling과 SVRW를 통해 univariate UCSVO 모형의 파라미터를 추정합니다.
- Cython 기반 효율화 버전과, Python 만을 사용하는 호환성 버전을 모두 지원합니다.
- verbose 모드를 통해 계산 과정의 출력도 지원합니다. 

---
### Basic Model (Stock and Watson (2016))
$$\pi_t=\tau_t + \epsilon_t \\ \tau_t=\tau_{t-1}+\sigma_{\Delta\tau,t} \\ \epsilon_t = \sigma_{\epsilon,t} \times s_t$$

### Parameters
|**Parameter** | **설명**                     |
|--------------|-----------------------------|
|$\pi_t$         | 관측된 시계열 데이터 (예: 물가상승률)     |
|$\tau_t$      | $\pi_t$의 추세를 설명하는 상태 변수            |
|$\sigma_{\Delta\tau,t}$| $\tau_t$의 추세 변동성           |
|$\sigma_{\epsilon,t}$  | 잔차 $\epsilon_t$의 변동성           |
|$\omega_{\epsilon,t}$| 잔차 변동성 스케일 파라미터 ($\sigma_{\epsilon,t}$의 크기 조절) (Chan (2016))|
|$\omega_{\Delta\tau,t}$| 추세 변동성 스케일 파라미터 ($\sigma_{\Delta\tau,t}$의 크기 조절) (Chan (2016))|
|$s_t$         | $\sigma_{\epsilon,t}$ 의 스케일 혼합 변수 |

### Model Initialization

| **Argument** | **Type**                            | **설명**                               | **기본값** | **예시**    |
| ------------ | ----------------------------------- | -------------------------------------- | ---------- | ----------- |
| `run_type`   | `str` (`'python'` 또는 `'cython'`)  | 실행 백엔드 선택                       | `'python'` | `'python'`  |
| `verbose`    | `bool`                              | 과정 출력 여부                         | `False`    | `True`      |
| `n_per_year`     | `int`                       | 연간 관측치 수 (예: 분기별 데이터는 4)            | `4`        | `12`                           |
| `n_draws`        | `int`                       | 총 반복 횟수                                 | `50000`    | `100000`                       |
| `thinning`   | `int`                       | 몇 번째마다 샘플을 저장할지 결정 (`k` 값)         | `10`       | `20`                           |
| `n_burn_in`      | `int`                       | 번인(Burn-in) 수                    | `5000`     | `10000`                        |

#### 예시
```python
from bok_data_lib import UnivarUCSV

model = UnivarUCSV(run_type='cython', verbose=True)  # Cython 백엔드 사용, 과정 출력
model = UnivarUCSV(verbose=True)  # 기본 Python 백엔드 사용, 과정 출력
model = UnivarUCSV()  # 모든 파라미터를 기본값으로 사용
```

---

### Public Methods

#### `fit()`
주어진 단변량 시계열 데이터로 UCSV 모델을 적합합니다.

| **Argument**     | **Type**                    | **설명**                                         | **기본값** | **예시**                       |
| ---------------- | --------------------------- | ------------------------------------------------ | ---------- | ------------------------------ |
| `y`              | `np.ndarray` 또는 `pd.Series`| 입력 단변량 시계열 데이터                         | -          | `pd.Series([1.0, 2.0, 3.0])`   |

##### Output
- `self`: 적합이 완료된 객체 자신을 반환합니다.

#### `print_summary()`

| **Argument** | **Type** | **설명**                              | **기본값** | **예시** |
| ------------ | -------- | ------------------------------------- | ---------- | -------- |
| `digits`     | `int`    | 출력 값을 소수점 몇 자리까지 표시할지 여부 | `4`        | `6`      |

##### Output
- model summary가 출력됩니다.

#### `print_traceplot_test()`

| **Argument**   | **Type**      | **설명**                       | **기본값** | **예시**        |
| -------------- | ------------- | ------------------------------ | ---------- | --------------- |
| `data_index`   | `pd.Index`    | 데이터에 대응하는 날짜 인덱스    | -          | `data.index`    |

##### Output
- traceplot이 출력됩니다

---

### Class Variables

- `self.run_type` (`str`): 실행 백엔드 명칭.
- `self.verbose` (`bool`): 과정 출력 여부. 
- `self.is_fitted` (`bool`): 모델이 적합되었는지 여부.
- `self.scale_y` (`float`): 데이터 정규화 스케일 값. 
- `self.tau_draws` (`np.ndarray`): 모델 적합 후의 $\tau_t$ 값.
- `self.sigma_dtau_draws` (`np.ndarray`): 모델 적합 후의 $\sigma_{\Delta \tau,t}$ 값. 
- `self.sigma_eps_draws` (`np.ndarray`): 모델 적합 후의 $\sigma_{\epsilon,t}$ 값. 
- `self.g_eps_draws` (`np.ndarray`): 모델 적합 후의 $\omega_{\epsilon,t}$ 값. 
- `self.g_dtau_draws` (`np.ndarray`): 모델 적합 후의 $\omega_{\Delta \tau,t}$ 값.
- `self.scl_eps_draws` (`np.ndarray`): 모델 적합 후의 $s_t$ 값.
- `self.sigma_total_eps_draws` (`np.ndarray`): 모델 적합 후의 $\sigma_{\epsilon,t} \times s_t$ 값.
- `self.ps_draws` (`np.ndarray`): 모델 적합 후의 $s_t$의 확률 값. 
- `self.summary_stats` (`dict`): 모델의 통계량.

---

### Usage (사용 방법)

```python
import numpy as np
import pandas as pd
from bok_data_lib import UnivarUCSV

# 데이터 로드
data = pd.read_csv("data/cpi_1965.csv", index_col=0, encoding='euc-kr')
data.columns = ['cpi']
data.index = pd.date_range('1965', periods=len(data), freq='QE')  # 분기별 데이터

# 데이터 전처리
data = 400 * np.log(data).diff().loc['1966':]
y = data['cpi']

# 모델 초기화 및 적합
model = UnivarUCSV(run_type='cython', n_draws=10000, n_burn_in=5000, verbose=True,)
model.fit(y)

# model summary
model.print_summary()

# traceplot
model.print_traceplot_test(data.index)
```

---
### Reference
- James H. Stock & Mark W. Watson, 2016. "Core Inflation and Trend Inflation," The Review of Economics and Statistics, MIT Press, vol. 98(4), pages 770-784, October.
- Chan, J. C. C. (2016). Specification tests for time-varying parameter models with stochastic volatility. Econometric Reviews, 37(8), 807–823. https://doi.org/10.1080/07474938.2016.1167948