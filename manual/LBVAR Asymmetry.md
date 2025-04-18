# LBVAR_Asymmetry

> **LBVAR_Asymmetry 모델 클래스 사용 방법 매뉴얼**

---

### 개요
- `LBVAR_Asymmetry` 클래스는 Large Bayesian Vector Autoregression (LBVAR) 모델의 비대칭 버전을 구현하여 주어진 경제 데이터를 기반으로 다양한 경제 변수를 분석하고 예측하는 데 사용됩니다.
- 이 모델은 대규모 데이터셋에서 변수 간의 비대칭적 상호작용을 포착하며, 비대칭적 사전 분포(Asymmetric Prior)를 사용하여 보다 유연한 추정을 가능하게 합니다.
- 켤레사전분포(Conjugate prior)을 사용하여 사후분포를 추정합니다
- 하이퍼파라미터 최적화 옵션을 제공하여 모델의 성능을 향상시킬 수 있습니다.
- 모델 적합 후 예측을 수행하고 결과를 시각화할 수 있는 기능을 제공합니다.

---

### 모델 초기화

`LBVAR_Asymmetry` 클래스는 LBVAR 모델을 초기화합니다.

| **Argument**   | **Type** | **설명**   | **기본값**   | **예시**  |
|----------------------------|--------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| `trend`    | `int` or `str`    | 트렌드 옵션을 지정합니다. <br> - `1` or `C`: 상수항만 포함 <br> - `2` or `L`: 선형 추세 포함 <br> - `3` or `Q`: 이차 추세 포함  | `1`  | `2`   |
| `p`    | `int`    | 시차(lag) 수를 지정합니다. <br> - 분기 데이터의 경우 일반적으로 `4` <br> - 월간 데이터의 경우 일반적으로 `12`   | `4`  | `12`  |
| `ndraws`   | `int`    | Posterior 분포에서의 샘플 수를 지정합니다. | `10000`  | `5000`    |
| `hyperparameters`  | `Optional[np.ndarray]`   | 하이퍼파라미터 배열입니다. <br> - 비대칭적 사전 분포의 경우 기본값은 `np.array([0.05, 0.005, 100])`  | `np.array([0.05, 0.005, 100])`   | `np.array([0.1, 0.01, 80])`  |
| `hyperparameter_opt`   | `int` or `str`    | 하이퍼파라미터 최적화 옵션을 지정합니다. <br> - `0` or `pass`: 사용자가 선택한 하이퍼파라미터 사용 <br> - `1` or `mlo`: Marginal Likelihood Optimization 수행 <br> - `2` or `mcmc`: MCMC를 통한 하이퍼파라미터 선택 수행 | `0`  | `1`   |
| `optimization_method`  | `int` or `str`    | 최적화 방법을 지정합니다. <br> - `0` or `scipy`: Scipy를 통한 최적화 <br> - `1` or `rs`: Random Search를 통한 최적화 <br> - 이 옵션은 `hyperparameter_opt`가 `1` 또는 `2`인 경우에만 사용됩니다.    | `0`  | `1`   |
| `random_search_options`    | `Optional[dict]` | Random Search 최적화에 사용할 옵션을 지정하는 딕셔너리입니다. <br> - `'K'`: 최대 반복 횟수 <br> - `'P'`: 각 반복에서 생성할 후보의 수 <br> - `'alpha'`: 스텝 크기  | `{'K': 1000, 'P': 1000, 'alpha': 1}` | `{'K': 500, 'P': 500, 'alpha': 0.5}` |
| `hyperparameter_mcmc_options` | `Optional[dict]` | 하이퍼파라미터 MCMC 최적화에 사용할 옵션을 지정하는 딕셔너리입니다. <br> - `'n_draws'`: MCMC 샘플 수 <br> - `'n_burnin'`: 번인 기간   | `{'n_draws': 10000, 'n_burnin': 1000}`  | `{'n_draws': 5000, 'n_burnin': 500}` |
| `verbose`  | `bool`   | 모델 실행 중간 과정 및 결과를 출력할지 여부를 지정합니다.   | `False`  | `True` |

#### 예시
```python
from bok_da import LBVAR_Asymmetry

# 기본 설정으로 모델 초기화
model = LBVAR_Asymmetry()

# 사용자 정의 설정으로 모델 초기화
model = LBVAR_Asymmetry(
    trend=2,
    p=12,
    ndraws=50000,
    hyperparameters=np.array([0.1, 0.005, 80]),
    hyperparameter_opt=1,
    optimization_method=0,
    random_search_options={'K': 500, 'P': 500, 'alpha': 0.5},
    hyperparameter_mcmc_options={'n_draws': 15000, 'n_burnin': 500},
    verbose=True
)
```

---

### Public Methods

#### `fit(data, rv_list=[])`
주어진 데이터에 LBVAR_Asymmetry 모델을 적합시킵니다.

| **Argument** | **Type**   | **설명**    | **기본값** | **예시**   |
|--------------|----------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|--------------------------------------------|
| `data`   | `np.ndarray` 또는 `pd.DataFrame` | 모델에 사용할 입력 데이터입니다. <br> - 행: 시간(t), 첫 번째 행은 가장 과거 시점 <br> - 열: 변수, 각 열이 하나의 변수에 해당 | -  | `pd.DataFrame({...})` 또는 `np.array([...])` |
| `rv_list`    | `list` | 비정상(non-stationary)적인 변수를 나타내는 변수의 위치(index) 리스트입니다. <br> - 예: `[2, 4]` <br> - 모든 변수가 정상적인 경우 `[]`    | `[]`   | `[1, 3]`   |

##### Output
- `self` : 적합된 모델 객체 자신을 반환합니다.

##### 예시
```python
import pandas as pd
from bok_da import LBVAR_Asymmetry

# 데이터 생성 (예시)
data = pd.DataFrame({
    'Var1': np.random.randn(100),
    'Var2': np.random.randn(100),
    'Var3': np.random.randn(100),
    'Var4': np.random.randn(100)
})
rv_list = [2, 4]  # 예: 2번과 4번 변수가 비정상적인 경우

# 모델 초기화
model = LBVAR_Asymmetry(verbose=True)

# 모델 적합
model.fit(data, rv_list=rv_list)
```

#### `forecast(forecast_period=4, pereb = 0.16)`
적합된 모델을 사용하여 예측을 수행합니다.

| **Argument**      | **Type** | **설명**                                                                                      | **기본값** | **예시**                        |
|-------------------|----------|-----------------------------------------------------------------------------------------------|------------|---------------------------------|
| `forecast_period` | `int`    | 예측 기간을 지정합니다.                                                                         | `4`        | `8`                             |
| `pereb`       | `float`   | 예측구간 설정값                                    | `0.16`     | `0.16`                         |

##### Output
- `Forecast_Results` : 예측 결과를 담은 딕셔너리를 반환합니다.

##### 예시
```python
# 모델 적합 후 예측 수행
forecast_results = model.forecast(forecast_period=4, pereb = 0.16)

# 예측 결과 접근 예시
mean_forecast = forecast_results["Mean"]
upper_forecast = forecast_results["UP"]
lower_forecast = forecast_results["DOWN"]

print(mean_forecast)
```

#### `print_forecast(plot_index = None, plot_all = True, column_names = [])`
예측 결과를 시각화합니다. 

| **Argument**      | **Type** | **설명**                                                                | **기본값**   | **예시**                        |
|-------------------|----------|------------------------------------------------------------------------|------------|--------------------------------|
| `plot_index`       | `bool`   | 개별 그래프를 출력할 변수의 인덱스를 지정합니다. `None`인 경우 모든 변수를 시각화합니다.  | `None`     | `[1, 2, 3]`                    |
| `plot_all`         | `bool`   | 모든 변수를 하나의 그래프로 출력할지 여부를 지정합니다.                             | `True`     | `False`                        |
| `column_names`     | `list`   | 변수명 리스트를 입력합니다. (입력하지 않을 경우 인덱스 사용)                         | `[]`       | `["var1", "var2", ...]`        |


##### 예시
```python
# 예측 수행 후 예측 시각화
# 모든 변수에 대해 개별 그래프를 그리고, 모든 변수를 하나의 그래프로도 시각화
model.print_forecast(column_names = df.columns)

# 1, 2, 3번 변수에 대한 개별 그래프만 그리는 경우
model.print_forecast(plot_index = [1, 2, 3], plot_all = False, column_names = df.columns)
```


---

### Class Variables

모델 클래스 내의 전체 변수 목록입니다. Container Class에 대한 설명은 `Container Class.md`파일을 참고해주세요.

#### Container Class

- `Raw_Data`: 원본 데이터와 모델링에 필요한 변환된 변수들을 저장하는 Container 클래스 인스턴스입니다.
  - `Set`: 원본 입력 데이터
  - `EXOv_total`: 전체 기간에 대한 외생변수 행렬
  - `EXOv`: 분석 기간에 대한 외생변수 행렬
  - `EXOv_AR`: AR 모형 추정을 위한 외생변수 행렬
  - `Z`: 회귀분석의 독립변수 행렬
  - `Y`: 종속변수 행렬

- `Parameters`: 모델 파라미터를 저장하는 Container 클래스 인스턴스입니다.
  - `RV_list`: 비정상 변수 인덱스 목록
  - `p`: 시차(lag) 수
  - `n`: 전체 관측치 수
  - `nvar`: 변수 수
  - `T`: 분석에 사용되는 관측치 수
  - `k`: VAR 모델의 총 변수 개수 (nvar x p)
  - `beta`: 레벨 변수에 대한 사전 평균 값
  - `ndraws`: Posterior 분포에서의 샘플 수
  - `Trend`: 트렌드 옵션
  - `c`: 외생변수 수
  - `num_of_parameter`: 총 파라미터 수
  - `pereb`: 예측구간 설정값
  - `Forecast_period`: 예측 기간

- `Prior`: 사전 분포 정보를 저장하는 Container 클래스 인스턴스입니다.
  - `Sigma_hat`: 오차분산 추정치 행렬
  - `nu`: 자유도 파라미터
  - `S`: 스케일 파라미터
  - `mbeta`: 베타의 사전 평균
  - `Vbeta_reduced`: Reduced Form에서의 베타 분산
  - `Vbeta`: 베타의 사전 분산
  - `ma`: a의 사전 평균
  - `maV`: a의 사전 분산

- `Draw`: MCMC 샘플링을 통해 얻은 Posterior 샘플들을 저장하는 Container 클래스 인스턴스입니다.
  - `theta_is`: 각 변수별 파라미터 샘플
  - `Sigma_i_sq`: 오차 분산의 샘플
  - `A_matrix`: A 행렬의 샘플
  - `Bet`: 베타의 샘플
  - `Sigma_struct`: 구조적 오차 분산 행렬의 샘플
  - `Sigma`: 오차 분산 행렬의 샘플

- `Forecast_Results`: 모델 예측 결과를 저장하는 Container 클래스 인스턴스입니다.
  - `Total`: 개별 예측 샘플
  - `Mean`: 예측 평균
  - `UP`: 예측 상한
  - `DOWN`: 예측 하한

#### Variables

- `trend` (`int`): 트렌드 옵션 (1: 상수항만 포함, 2: 선형 추세 포함, 3: 이차 추세 포함).
- `p` (`int`): 시차(lag) 수.
- `ndraws` (`int`): Posterior 분포에서의 샘플 수.
- `hyperparameter_opt` (`int`): 하이퍼파라미터 최적화 옵션.
- `optimization_method` (`int`): 최적화 방법.
- `verbose` (`bool`): 출력 여부.
- `hyperparameters` (`np.ndarray`): 하이퍼파라미터 배열.
- `random_search_options` (`dict`): Random Search 최적화 옵션.
- `hyperparameter_mcmc_options` (`dict`): Hyperparameter MCMC 최적화 옵션.
- `is_fitted` (`bool`): 모델 적합 여부.
- `forecast_period` (`int`): 예측 기간.


---

### Usage (사용 방법)

```python
import numpy as np
import pandas as pd
from bok_da import LBVAR_Asymmetry

# 데이터 생성 (예시)
np.random.seed(0)
data = pd.DataFrame({
    'Var1': np.random.randn(100),
    'Var2': np.random.randn(100),
    'Var3': np.random.randn(100),
    'Var4': np.random.randn(100)
})
rv_list = [2, 4]  # 비정상적인 변수의 인덱스

# 모델 초기화
model = LBVAR_Asymmetry(
    trend=2,
    p=12,
    ndraws=50000,
    hyperparameters=np.array([0.1, 0.005, 80]),
    hyperparameter_opt=1,
    optimization_method=0,
    random_search_options={'K': 500, 'P': 500, 'alpha': 0.5},
    hyperparameter_mcmc_options={'n_draws': 15000, 'n_burnin': 500},
    verbose=True
)

# 모델 적합
model.fit(data, rv_list=rv_list)

# 예측 수행
forecast_results = model.forecast(forecast_period=8, plot_each=True, plot_all=True)

# 샘플된 파라미터 접근 예시
phi_samples = model.Draw['phi']
beta_samples = model.Draw['beta']

# 예측 결과 접근 예시
mean_forecast = forecast_results["Mean"]
upper_forecast = forecast_results["UP"]
lower_forecast = forecast_results["DOWN"]
```

---

### Reference


