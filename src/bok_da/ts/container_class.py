import sys
import copy

import numpy as np
import pandas as pd


class Container:
    """
    변수를 담기 위한 컨테이너 클래스
    variable_summary 메서드를 이용해 저장된 변수를 간략히 요약할 수 있음
    """

    def __init__(self, dictionary=None):
        """
        컨테이너 초기화
        """
        # 변수 추가 순서를 기록하기 위한 리스트
        self.__variables_order = []

        # 초기화 시 딕셔너리를 받으면 변수로 추가
        if isinstance(dictionary, dict):
            self.update_var_by_dict(dictionary)


    def __setattr__(self, name, value):
        """
        변수 추가 시 순서를 기록.
        """
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            if name not in self.__dict__:
                self.__variables_order.append(name)
            super().__setattr__(name, value)


    # 딕셔너리와 비슷한 방식으로 변수에 접근할 수 있음
    def __getitem__(self, key):
        """
        NOTE: 딕셔너리 호환성 유지를 위한 기능
        """
        return getattr(self, key)

    def __setitem__(self, key, value):
        """
        NOTE: 딕셔너리 호환성 유지를 위한 기능
        """
        setattr(self, key, value)


    def _truncate_value(self, value, max_length=30):
        """
        긴 값을 잘라내고 '...'으로 표시.
        """
        value_str = str(value)
        return value_str if len(value_str) <= max_length else value_str[:max_length - 3] + "..."


    def _get_variable_info(self, var):
        """
        변수 타입에 따라 요약 정보를 반환.
        """
        if isinstance(var, (int, float, str)):
            string = f"{var}"
            return self._truncate_value(string)
        elif isinstance(var, np.ndarray):
            return f"shape={var.shape}"
        elif isinstance(var, pd.DataFrame):
            return f"shape={var.shape}, columns={list(var.columns)}"
        elif isinstance(var, pd.Series):
            return f"shape={var.shape}"
        elif isinstance(var, dict):
            return f"keys={list(var.keys())}"
        elif isinstance(var, list):
            return f"length={len(var)}"
        else:
            return ""


    def _get_variable_size_kb(self, var):
        """
        변수의 대략적인 메모리 사용량(KB)을 계산.
        """
        size_in_bytes = sys.getsizeof(var)
        return size_in_bytes / 1024.0
    

    def keys(self):
        """
        저장된 변수들의 리스트를 반환
        NOTE: 딕셔너리 호환성 유지를 위한 기능
        """
        return self.__variables_order.copy()
        

    def copy(self):
        """
        해당 컨테이너 객체를 deepcopy하여 반환함
        NOTE: 호환성 유지를 위한 기능
        """
        return copy.deepcopy(self)


    def variable_summary(self, sort='added', ascending=True):
        """
        컨테이너 내 변수를 요약하여 출력.
        """
        summary_list = []
        for idx, var_name in enumerate(self.__variables_order):
            var_value = getattr(self, var_name)
            var_type = type(var_value).__name__
            var_size = self._get_variable_size_kb(var_value)
            var_info = self._get_variable_info(var_value)

            summary_list.append({
                'added': idx + 1,
                'variable': var_name,
                'type': var_type,
                'size': var_size,
                'info': var_info
            })

        valid_sort_keys = ['added', 'variable', 'type', 'size', 'info']
        if sort not in valid_sort_keys:
            print(f"[WARNING] Invalid sort key '{sort}'. Defaulting to 'added'.")
            sort = 'added'

        summary_list.sort(key=lambda x: x[sort], reverse=not ascending)

        # Find max lengths for dynamic alignment
        max_var_len = max(len(item['variable']) for item in summary_list)
        max_type_len = max(len(item['type']) for item in summary_list)
        max_info_len = 30  # Info is truncated to 30 chars

        # Print header
        header = (
            f"{'added':>5} | {'variable':<{max_var_len}} | {'type':<{max_type_len}} | {'size(KB)':>10} | info"
        )
        print(header)
        print("-" * (len(header) + 5))

        # Print rows
        for item in summary_list:
            # print(f"{item['added']:>5} | {item['variable']:<{max_var_len}} | {item['type']:<{max_type_len}} | {item['size']:10.3f} | {item['info'][:max_info_len]}")
            print(f"{item['added']:>5} | {item['variable']:<{max_var_len}} | {item['type']:<{max_type_len}} | {item['size']:10.3f} | {item['info']}")


    def update_var_by_args(self, **kwargs):
        """
        키워드 인자로 받은 값들을 컨테이너의 속성으로 추가합니다.

        Parameters
        ----------
        **kwargs : keyword arguments
            컨테이너에 추가할 키-값 쌍들

        Examples
        --------
        >>> container = Container()
        >>> container.update_var_by_arg(a=1, b="test")
        >>> print(container.a)  # 1
        >>> print(container.b)  # "test"
        """
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        return self


    def update_var_by_dict(self, var_dict):
        """
        딕셔너리로 받은 값들을 컨테이너의 속성으로 추가합니다.

        Parameters
        ----------
        var_dict : dict
            컨테이너에 추가할 키-값 쌍들

        Examples
        --------
        >>> container = Container()
        >>> container.update_var_by_dict({"a": 1, "b": "test"})
        >>> print(container.a)
        """
        for key, value in var_dict.items():
            setattr(self, key, value)
        
        return self