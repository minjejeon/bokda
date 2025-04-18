# ProbitRegressionFreq

> ProbitRegressionFreq 모델 클래스 사용 방법 매뉴얼

---

### 개요
- `ProbitRegressionFreq` 클래스는 빈도론적 접근 방식을 기반으로 한 Probit 회귀 모델입니다.
- 뉴턴-랩슨(Newton-Raphson) 최적화 방법을 사용하여 모델의 파라미터를 추정합니다.
- 상수항 추가 여부, 컬럼 명칭 등을 사용자가 설정할 수 있으며, `verbose` 모드를 통해 계산 과정의 출력도 지원합니다.
- 모델 적합 후 다양한 통계량과 요약 정보를 제공합니다.

---

### Model Initialization

| **Argument**   | **Type** | **설명**                                   | **기본값** | **예시**      |
| -------------- | -------- | ------------------------------------------ | ---------- | ------------- |
| `method`       | `str`    | 최적화 방법 선택 | `'newton'` | `'newton'`    |
| `tolerance`    | `float`  | 계수 벡터의 수렴 기준                      | `1e-6`     | `1e-5`        |
| `max_iter`     | `int`    | 최대 반복 횟수                             | `300`      | `500`         |
| `verbose`      | `bool`   | 과정 출력 여부                             | `False`    | `True`        |

#### 예시
```python
from bok_da import ProbitRegressionFreq

# 기본 설정으로 모델 초기화
model = ProbitRegressionFreq()

# 뉴턴-랩슨 방법과 특정 수렴 기준으로 모델 초기화
model = ProbitRegressionFreq(
    method='newton',
    tolerance=1e-5,
    max_iter=500,
    verbose=True
)
```

---

### Public Methods

#### `fit()`
주어진 데이터로 빈도주의 Probit 회귀 모델을 적합합니다.

| **Argument** | **Type** | **설명** | **기본값** | **예시** |
|--------------|----------|----------|------------|----------|
| `X` | `np.ndarray` 또는 `pd.DataFrame` | 입력 데이터 행렬 (샘플 수 × 특징 수) | - | `pd.DataFrame([[1,2],[3,4]])` |
| `y` | `np.ndarray` 또는 `pd.Series` | 타겟 라벨 벡터 (0과 1로 구성) | - | `pd.Series([0, 1, 0, 1])` |
| `columns` | `list[str]` | 특징량의 이름 리스트 (결과 출력 시 사용) | `[]` | `['age', 'income']` |
| `add_const` | `bool` | 상수항을 추가할지 여부 | `True` | `False` |

##### Output
- `self`: 적합이 완료된 객체 자신을 반환합니다.

<!-- #### `get_beta()`
적합이 완료된 모델의 계수를 반환합니다.

| **Argument** | **Type** | **설명** | **기본값** | **예시** |
|--------------|----------|----------|------------|----------|
| `series` | `bool` | pandas Series로 반환할 지 여부 | `True` | `False` |

##### Output
- `pd.Series` 또는 `dict`: 모델 계수 (Pandas Series 또는 딕셔너리 형태)

#### `get_beta_table()`
회귀 계수 테이블을 반환합니다. 이는 모델 요약 시 출력되는 계수 테이블의 데이터입니다.

| **Argument** | **Type** | **설명** | **기본값** | **예시** |
|--------------|----------|----------|------------|----------|
| `df` | `bool` | 결과를 DataFrame으로 반환할 지 여부 | `True` | `True` |

##### Output
- `pd.DataFrame` 또는 `dict`: 계수 테이블.

#### `get_fit_stats()`
모델 적합 통계량을 반환합니다.

| **Argument** | **Type** | **설명** | **기본값** | **예시** |
|--------------|----------|----------|------------|----------|
| `series` | `bool` | 통계량을 Pandas Series로 반환할지 여부 | `True` | `True` |

##### Output
- `pd.Series` 또는 `dict`: 적합 통계량. -->

#### `print_summary()`
모델 적합 결과의 요약을 출력합니다.

| **Argument** | **Type** | **설명** | **기본값** | **예시** |
|--------------|----------|----------|------------|----------|
| `digits` | `int` | 출력 값을 소수점 몇 자리까지 표시할지 지정 | `4` | `6` |

##### Output
- 모델 요약 정보가 출력됩니다.

---

### Class Variables

- `self.method` (`str`): 최적화 방법.
- `self.tolerance` (`float`): 계수 벡터의 수렴 기준.
- `self.max_iter` (`int`): 최대 반복 횟수.
- `self.verbose` (`bool`): 과정 출력 여부.
- `self.beta` (`np.ndarray`): 모델 적합 후의 계수 값.
- `self.is_fitted` (`bool`): 모델 적합 여부.
- `self.columns` (`list[str]`): 입력 데이터의 컬럼 명칭.
- `self.add_const` (`bool`): 상수항 추가 여부.
- `self.summary_stats` (`dict`): 모델의 통계량 (로그 우도, AIC, Pseudo R-squared 등).

---

### Usage (사용 방법)

```python
import numpy as np
import pandas as pd
from bok_da import ProbitRegressionFreq

# 데이터 생성
np.random.seed(0)
X = np.random.rand(100, 2)
y = (X[:, 0] + X[:, 1] > 1).astype(int)

# 모델 초기화 및 적합
model = ProbitRegressionFreq(
    method='newton',
    tolerance=1e-5,
    max_iter=500,
    verbose=True
)

model.fit(X, y, columns=['Feature1', 'Feature2'], add_const=True)

# 요약 정보 출력
model.print_summary()
```

