# LBVAR_Noninformative

> **LBVAR_Noninformative 모델 클래스 사용 방법 매뉴얼**

---

### 개요
- `LBVAR_Noninformative` 클래스는 Large Bayesian Vector Autoregression (LBVAR) 모델을 구현하여 주어진 경제 데이터를 기반으로 다양한 경제 변수 간의 구조적 관계를 분석하고 예측하는 데 사용됩니다. 
- 이 모델은 비정보적 사전분포(Noninformative Prior)를 사용하여 사전 정보에 의존하지 않고 데이터로부터 사후분포를 추정합니다.
- 변수가 너무 많을 경우 추정이 불안정해지고 구조적 관계 파악이 어려워질 수 있으므로 변수의 개수를 적게 선택하는 것이 바람직합니다. 
- 예측보다는 구조적 관계 분석에 사용하는 것이 적합합니다. 

---

### 모델 초기화

`LBVAR_Noninformative` 클래스는 LBVAR 모델을 초기화합니다.

| **Argument**   | **Type**     |**설명**   | **기본값**    | **예시**   |
|--------------- |--------------|----------|-------------|-----------|
| `trend`    | `int` or `str`    | 트렌드 옵션을 지정합니다. <br> - `1` or `C`: 상수항만 포함 <br> - `2` or `L`: 선형 추세 포함 <br> - `3` or `Q`: 이차 추세 포함  | `1`  | `2`   |
| `p`            | `int`        | 시차(lag) 수를 지정합니다. <br> - 분기 데이터의 경우 일반적으로 `4` <br> - 월간 데이터의 경우 일반적으로 `12`   | `4`   | `12` |
| `ndraws`      | `int`         | Posterior 분포에서의 샘플 수를 지정합니다.                                                          | `10000`  | `5000` |
| `verbose`     | `bool`        | 모델 실행 중간 과정 및 결과를 출력할지 여부를 지정합니다.                                                 | `False`  | `True`    |

#### 예시
```python
from bok_da import LBVAR_Noninformative

# 기본 설정으로 모델 초기화
model = LBVAR_Noninformative()

# 사용자 정의 설정으로 모델 초기화
model = LBVAR_Noninformative(
    trend=2,
    p=12,
    ndraws=50000,
    verbose=True
)
```

---

### Public Methods

#### `fit(data)`
주어진 데이터에 LBVAR_Noninformative 모델을 적합시킵니다.

| **Argument** | **Type**                     | **설명**                                                                                                                                                                    | **기본값** | **예시**                                   |
|--------------|------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|--------------------------------------------|
| `data`       | `np.ndarray` 또는 `pd.DataFrame` | 모델에 사용할 입력 데이터입니다. <br> - 행: 시간(t), 첫 번째 행은 가장 과거 시점 <br> - 열: 변수, 각 열이 하나의 변수에 해당 <br> 필수 컬럼: `[변수1, 변수2, ...]`                     | -          | `pd.DataFrame({...})` 또는 `np.array([...])` |

##### Output
- `self` : 적합된 모델 객체 자신을 반환합니다.

##### 예시
```python
import pandas as pd
from bok_da import LBVAR_Noninformative

# 데이터 생성 (예시)
data = pd.DataFrame({
    'Var1': np.random.randn(100),
    'Var2': np.random.randn(100),
    'Var3': np.random.randn(100),
    'Var4': np.random.randn(100)
})

# 모델 초기화
model = LBVAR_Noninformative(verbose=True)

# 모델 적합
model.fit(data)
```

#### `recursive_irf`

적합된 모델을 사용하여 충격반응함수, 예측오차분산분해, 역사분해 등을 수행합니다. 

| **Argument**      | **Type** | **설명**                                    | **기본값** | **예시**                        |
|-------------------|----------|--------------------------------------------|----------|--------------------------------|
| `nstep`           | `int`    | 충격이 시스템에 미치는 영향을 분석하기 위한 예측 기간   | `21`     | `10`                           |
| `pereb`           | `float`  | 예측구간 설정값                                | `0.16`   | `0.16`                         |



##### Output
- `Draw` : 분해 결과를 담은 딕셔너리를 반환합니다. 

##### 예시
```python
# 모델 적합 후 충격반응함수, 예측오차분산분해, 역사분해 등을 계산
model.recursive_irf()
```

#### `plot_irf_shock_series`
각 변수의 충격데이터를 시각화

| **Argument**      | **Type** | **설명**                                           | **기본값** | **예시**                         |
|-------------------|----------|---------------------------------------------------|----------|---------------------------------|
| `column_names`    | `list`   | 변수명 리스트를 입력합니다. (입력하지 않을 경우 인덱스 사용)    | `[]`     | `["var1", "var2", ...]`         |


##### 예시
```python
model.plot_irf_shock_series(df.columns)
```

#### `plot_irf_impulse_response`
충격반응함수를 시각화

| **Argument**      | **Type** | **설명**                                           | **기본값** | **예시**                         |
|-------------------|----------|---------------------------------------------------|----------|---------------------------------|
| `column_names`    | `list`   | 변수명 리스트를 입력합니다. (입력하지 않을 경우 인덱스 사용)    | `[]`     | `["var1", "var2", ...]`         |


##### 예시
```python
model.plot_irf_impulse_response(df.columns)
```

#### `plot_irf_FEVB`
예측오차분산분석을 시각화

| **Argument**      | **Type** | **설명**                                           | **기본값** | **예시**                         |
|-------------------|----------|---------------------------------------------------|----------|---------------------------------|
| `column_names`    | `list`   | 변수명 리스트를 입력합니다. (입력하지 않을 경우 인덱스 사용)    | `[]`     | `["var1", "var2", ...]`         |


##### 예시
```python
model.plot_irf_FEVB(df.columns)
```

#### `plot_irf_historical_decomposition`
역사분해를 시각화

| **Argument**      | **Type** | **설명**                                           | **기본값** | **예시**                         |
|-------------------|----------|---------------------------------------------------|----------|---------------------------------|
| `column_names`    | `list`   | 변수명 리스트를 입력합니다. (입력하지 않을 경우 인덱스 사용)    | `[]`     | `["var1", "var2", ...]`         |


##### 예시
```python
model.plot_irf_historical_decomposition(df.columns)
```
---

### Class Variables

모델 클래스 내의 전체 변수 목록입니다. Container Class에 대한 설명은 `Container Class.md`파일을 참고해주세요.

#### Container Class

- `Raw_Data`: 원본 데이터와 모델링에 필요한 변환된 변수들을 저장하는 Container 클래스 인스턴스입니다.
  - `Set`: 원본 입력 데이터
  - `EXOv_total`: 전체 기간에 대한 외생변수(Exogenous Variables) 행렬
  - `EXOv`: 분석 기간에 대한 외생변수 행렬
  - `EXOv_AR`: AR 모형 추정을 위한 외생변수 행렬
  - `Z`: 회귀분석의 독립변수 행렬
  - `Y`: 종속변수 행렬

- `Parameters`: 모델 파라미터를 저장하는 Container 클래스 인스턴스입니다.
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
  - `nstep`: IRF 분석을 위한 스텝 수

- `Prior`: 사전 분포 정보를 저장하는 Container 클래스 인스턴스입니다.
  - 비정보적 사전분포를 사용하므로 별도의 주요 필드가 저장되지 않습니다.

- `Draw`: MCMC 샘플링을 통해 얻은 Posterior 샘플들을 저장하는 Container 클래스 인스턴스입니다.
  - `Sigma`: Sigma 행렬 샘플
  - `Bet`: Bet 행렬 샘플
  - `Bet_Prime`: Bet_Prime 행렬 샘플
  - `U_B`: 잔차 행렬 샘플
  - `BB`: 모형 계수 행렬 추정치
  - `Imp`: Impulse 행렬
  - `Impulse`: 충격반응함수 샘플
  - `Impres`: 충격반응함수 요약 통계량
  - `FEVD`: 예측오차분산분해 결과
  - `Shock`: 충격 시리즈 샘플
  - `Shock_inf`: 충격 시리즈 요약 통계량
  - `HD_shock`: 역사분해 샘플
  - `med_HD_shock`: 역사분해 중앙값

- `Forecast_Results`: 모델 예측 결과를 저장하는 Container 클래스 인스턴스입니다.
  - `Total`: 개별 예측 샘플
  - `Mean`: 예측 평균
  - `UP`: 예측 상한
  - `DOWN`: 예측 하한

#### Variables

- `trend` (`int`): 트렌드 옵션 (1: 상수항만 포함, 2: 선형 추세 포함, 3: 이차 추세 포함).
- `p` (`int`): 시차(lag) 수.
- `ndraws` (`int`): Posterior 분포에서의 샘플 수.
- `verbose` (`bool`): 출력 여부.
- `is_fitted` (`bool`): 모델 적합 여부.


---

### Usage (사용 방법)

```python
import numpy as np
import pandas as pd
from bok_da import LBVAR_Noninformative

# 데이터 생성 (예시)
np.random.seed(0)
data = pd.DataFrame({
    'Var1': np.random.randn(100),
    'Var2': np.random.randn(100),
    'Var3': np.random.randn(100),
    'Var4': np.random.randn(100)
})

# 모델 초기화
model = LBVAR_Noninformative(
    trend=2,
    p=12,
    ndraws=50000,
    verbose=True
)

# 모델 적합
model.fit(data)

# 분해 수행 및 시각화
model.recursive_irf()
model.plot_irf_shock_series(df.columns)
model.plot_irf_impulse_response(df.columns)
model.plot_irf_FEVD(df.columns)
model.plot_irf_historical_decomposition(df.columns)

# 샘플된 파라미터 접근 예시
phi_samples = model.Draw['phi']
beta_samples = model.Draw['beta']

# 분해 결과 접근 예시
shock_series = model.Draw["Shock_inf"]
impulse_response = model.Draw["Impres"]
fevb = model.Draw["FEVD"]
historical_decomposition = model.Draw["med_HD_shock"]
```
---

### Reference

