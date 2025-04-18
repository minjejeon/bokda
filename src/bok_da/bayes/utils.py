# 공용으로 사용할만한 함수 추가
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def print_summary(summary_dict, single_stats, alignments={}, digits=4, footnote="", print_single_stats=True):
    """
    모델 적합 결과의 요약 정보를 출력하는 함수

    Parameters
    ----------
    summary_dict : dict
        헤더 이름을 키로 하고, 해당 컬럼의 값을 리스트로 가지는 딕셔너리
        ````
        eg. data_dict = {
            "Variable": ["Intercept", "X1", "X2"],
            "Mean": [1.2345, 0.1234, -0.5678],
            "Var": [0.01, 0.02, 0.03],
            "p25": [1.0, 0.1, -0.6],
            "p95": [1.5, 0.15, -0.5],
            "ESS": [1000, 900, 950],
            "Acc Rate": [0.2345, 0.5678, 0.8765],
        }
        ```

    single_stats : dict
        단일 통계량을 담은 딕셔너리
        `````
        eg. single_stats = {
            "BIC": 123.4567,
            "Log-Likelihood": -456.7890
        }
        ```

    alignments : dict, optional
        각 컬럼의 정렬 방향을 지정하는 딕셔너리 (예: {'Mean': 'left', 'Acc Rate': 'right'})
        - left 또는 right만 지원
        - 미 입력시 일괄 right align 적용

    digits : int, optional
        출력 값을 소수점 아래로 몇 자리까지 출력할지 결정 | Default: 4

    footnote : str, optional
        출력의 맨 끝에 추가로 입력할 참고사항, 주석
    
    print_single_stats : bool, optional
        단일 통계량을 출력하는지 여부 (빈 딕셔너리인 경우를 대응 하기 위함) | Default: True

    """
    # 헤더 추출
    headers = list(summary_dict.keys())
    
    # 각 컬럼의 데이터 길이 확인
    num_rows = len(next(iter(summary_dict.values())))
    
    # 컬럼 너비 초기화 (헤더 길이로 설정)
    col_widths = {header: len(header) for header in headers}
    
    # 데이터 행 준비 및 컬럼 너비 업데이트
    data_rows = []
    for i in range(num_rows):
        row = []
        for header in headers:
            value = summary_dict[header][i]
            if isinstance(value, float):
                formatted_value = f"{value:.{digits}f}"
            else:
                formatted_value = str(value)
            row.append(formatted_value)
            # 컬럼 너비 업데이트
            col_widths[header] = max(col_widths[header], len(formatted_value))
        data_rows.append(row)
    
    # 헤더 라인 준비
    header_line = ""
    for header in headers:
        # width = col_widths[header] + 2  # 여백 추가 (각 컬럼 좌우 한 칸씩 추가)
        width = col_widths[header]
        header_line += f" {header.center(width)} "
    
    # 구분선 생성
    separator_line = "-" * len(header_line)
    
    # 출력 시작
    if print_single_stats:
        print(separator_line)
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
    print(separator_line)
    print(header_line)
    print(separator_line)
    
    # 데이터 행 출력
    for row in data_rows:
        line = ""
        for idx, header in enumerate(headers):
            # width = col_widths[header] + 2  # 여백 추가 (각 컬럼 좌우 한 칸씩 추가)
            width = col_widths[header]
            value = row[idx]
            align = alignments.get(header, 'right')  # aligh 지정이 없을 경우 right
            if align == 'left':
                line += f" {value.ljust(width)} "
            else:
                line += f" {value.rjust(width)} "
        print(line)
    print(separator_line)

    # 부록 추가
    if footnote:
        print(footnote)
        print(separator_line)




def plot_traceplots(data_dict, multichain=False, rows=None, cols=None, figsize=(12, 8), filename=None, suptitle=None, linecolor="black", linewidth=0.5):
    """
    MCMC 샘플에 대한 traceplot을 그리는 함수

    Parameters
    ----------
    data_dict : dict
        키는 변수 이름이고 값은 샘플의 배열인 딕셔너리입니다.
            단일 체인 데이터의 경우 값은 [n_draws,] 형태의 배열입니다.
            다중 체인 데이터의 경우 값은 [n_chains, n_draws] 형태의 배열입니다.
        eg. {"Beta1": shape(n_draws), ..}
    multichain : bool, optional
        True로 설정하면 데이터를 다중 체인으로 간주하고 그에 따라 플롯합니다.
    rows : int, optional
        서브플롯 그리드의 행 수입니다.
    cols : int, optional
        서브플롯 그리드의 열 수입니다.
    figsize : tuple, optional
        Figure의 크기입니다. 기본값은 (12, 8)입니다.
    filename : str, optional
        지정하면 플롯을 해당 파일 이름으로 저장합니다.
    suptitle : str, optional
        Figure의 전체 제목입니다.
    linecolor : str, optional
        단일 traceplot인 경우 lineplot의 color | Default: "black"
    linewidth : int, optional
    """
    num_vars = len(data_dict)

    if rows is None and cols is None:
        cols = min(3, num_vars)
        rows = (num_vars + cols - 1) // cols
    elif rows is None:
        rows = (num_vars + cols - 1) // cols
    elif cols is None:
        cols = (num_vars + rows - 1) // rows

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if isinstance(axes, plt.Axes):
        axes = np.array([axes])
    axes = axes.flatten()

    for i, (var_name, values) in enumerate(data_dict.items()):
        ax = axes[i]
        if multichain:
            n_chains, n_draws = values.shape
            for chain_idx in range(n_chains):
                chain_values = values[chain_idx, :]
                sns.lineplot(x=np.arange(n_draws), y=chain_values, ax=ax, alpha=0.6, linewidth=linewidth)
        else:
            n_draws = values.shape[0]
            sns.lineplot(x=np.arange(n_draws), y=values, ax=ax, color=linecolor, linewidth=linewidth)

        # ax.set_title(f'Traceplot for {var_name}')
        ax.set_title(f'{var_name}')
        # plt.legend()
        # ax.set_xlabel('Iteration')
        # ax.set_ylabel(var_name)

    # 사용되지 않은 서브플롯 삭제
    for j in range(num_vars, len(axes)):
        fig.delaxes(axes[j])

    if suptitle:
        plt.suptitle(suptitle)
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.show()




