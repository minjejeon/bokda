# Container

> **Container 클래스 사용 매뉴얼**

---

### 개요
- `Container` 클래스는 다양한 변수를 편리하게 저장하고 관리하기 위한 유틸리티 클래스입니다.
- 파이썬의 딕셔너리처럼 키를 통해 변수를 자유롭게 접근할 수 있으며, 변수들의 추가 순서를 추적하여 직관적으로 관리할 수 있습니다.
- 메모리 사용량 및 변수 요약 정보를 한눈에 확인할 수 있고, `deepcopy`를 통해 손쉽게 객체 복사도 지원합니다.
- 외부에서 전달된 딕셔너리나 키워드 인자를 통해 빠르게 필드를 추가하고, `variable_summary` 메서드를 통해 변수들의 정보를 간략히 확인할 수 있습니다.

---

### 주요 기능
아래 표는 `Container` 클래스에서 제공하는 주요 기능을 요약한 것입니다.

| **구분**                | **설명**                                                                          |
|------------------------|-----------------------------------------------------------------------------------|
| **대괄호 인덱스 접근**     | `class.variable` 형태 뿐만 아니라 파이썬 딕셔너리와 호환되는 형식으로 `[변수명]` 형태로 저장된 변수를 접근하고 설정할 수 있음.              |
| **변수 순서 추적**       | 새로운 변수를 추가할 때마다 그 순서를 자동으로 기록하여, 변수의 생성 순서를 관리할 수 있음.          |
| **메모리 사용량 계산**    | 각 변수의 대략적인 메모리 사용량(KB 단위)을 계산하여, 변수별 규모를 빠르게 파악할 수 있음.           |
| **변수 요약 기능**       | `variable_summary` 호출 시, 변수 타입, 메모리 사용량, 추가 순서 등 핵심 정보를 일괄 출력함.           |
| **객체 복사**           | `copy()` 메서드를 통해 `deepcopy`로 완전히 독립적인 복사본을 생성할 수 있음.                        |
| **딕셔너리・키워드 인자 등록** | `update_var_by_dict`와 `update_var_by_args` 메서드를 통해 여러 변수를 빠르게 추가 가능.                |

---

### 클래스 초기화

```python
import pandas as pd
import numpy as np
from bok_da import Container

# 기본 생성
container = Container()

# 딕셔너리를 이용한 초기화
init_dict = {
    "a": 10,
    "b": np.array([1, 2, 3]),
    "c": pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
}

container_with_dict = Container(dictionary=init_dict)

# 변수 추가
container_with_dict.var1 = 1
container_with_dict["var2"] = 2
```

- `dictionary` 매개변수로 초기 변수를 등록할 수 있으며, 내부적으로 `update_var_by_dict`가 호출됩니다.

---

### 주요 메서드

#### `keys()`
저장된 변수의 이름을 리스트로 반환합니다.  
```python
variables = container_with_dict.keys()
print(variables)  # ['a', 'b', 'c']
```

#### `copy()`
깊은 복사를 통해 별도의 `Container` 객체를 생성합니다.  
```python
container_copy = container_with_dict.copy()
```

#### `variable_summary(sort='added', ascending=True)`
컨테이너 내부에 저장된 변수를 요약 정보(타입, 메모리 사용량, 변수 추가 순서 등)와 함께 출력합니다.
```python
container_with_dict.variable_summary(sort='size', ascending=False)
```
- `sort` 인자는 정렬 기준을 설정합니다. 기본값은 `'added'` (추가된 순서)이며, `'variable'`, `'type'`, `'size'`, `'info'` 등을 지정할 수도 있습니다.
- `ascending` 인자를 통해 오름차순/내림차순 정렬을 결정할 수 있습니다.

#### `update_var_by_args(**kwargs)`
키워드 인자로 전달된 모든 변수를 `Container`에 저장합니다.  
```python
container.update_var_by_args(x=123, y="Hello", z=[9, 8, 7])
```
- 이후 `container.x`, `container.y`, `container.z`와 같이 접근할 수 있습니다.

#### `update_var_by_dict(var_dict)`
딕셔너리를 통해 여러 변수를 한꺼번에 추가합니다.
```python
more_vars = {"p": 3.14, "q": np.array([10, 20])}
container.update_var_by_dict(more_vars)
```

---

### 사용 예시

```python
from bok_da import Container
import numpy as np

data_dict = {
    "numbers": np.arange(5),
    "message": "sample",
    "scale": 3.14159
}

# 초기화
my_container = Container(dictionary=data_dict)

# 변수 접근
print(my_container["numbers"])  # [0 1 2 3 4]
print(my_container.message)      # "sample"

# 변수 추가
my_container["new_var"] = [10, 20, 30]
my_container.update_var_by_args(test_var=999)

# 요약 출력
my_container.variable_summary()

# 복사
copied_container = my_container.copy()
```
