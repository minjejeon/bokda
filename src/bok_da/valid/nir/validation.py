# 기본 라이브러리
import pickle
import warnings

# 데이터
import pandas as pd
import numpy as np
import bok_da as bd
import papermill as pm

# 시각화
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 분석 날짜 처리
from pandas.tseries.offsets import DateOffset, MonthEnd

# 모형 검증
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.api as sm
import statsmodels.tsa.api as VAR
import statsmodels.formula.api as smf

#
import itertools
from tqdm.notebook import tqdm

import subprocess
import json
from datetime import datetime, date
import os
import time
from pathlib import Path

#
from dataclasses import dataclass

# 한글 폰트 설정
#warnings.filterwarnings(action='ignore')
#try:
#    plt.rc('font', family='Malgun Gothic')
#except Exception:
#    plt.rc('font', family='NanumGothicCoding')
#plt.rcParams['axes.unicode_minus'] = False


# Spec 클래스
class Spec:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
            
# 모형별 실행 프로그램 경로 설정
@dataclass
class ModelPaths:
    lm_r_path: str
    lm_r_scr_path: str
    fs_matlab_path: str
    fs_matlab_scr_path: str
    fs_matlab_lib_path: str
    hlw_nb_path: str
    hlw_output_path: str
    hlw_fc_nb_path: str
    hlw_fc_output_path: str
    hlw_covid_nb_path: str
    hlw_covid_output_path: str

# LM 모형 R 코드 실행 함수
def run_LM(end_date: str, 
           paths: ModelPaths
           ):
    
    """
    LM 모형 R 코드 실행 함수

    Parameters
    ----------
    end_date : str
        R 스크립트에 넘겨줄 마지막 날짜 (예: '2025-03-31')
    r_path : str
        R 실행 파일 경로
    r_script_path : str
        실행할 R 스크립트 파일 경로

    Returns
    -------
    pd.DataFrame or None
        'Date'를 인덱스로 하는 중립금리(rstar) 시계열 데이터프레임.
        오류 시 None 반환.
    """
    r_path = paths.lm_r_path
    r_script_path = paths.lm_r_scr_path
    
    # 실행 파일과 스크립트 경로 유효성 검사
    for path, name in [(r_path, 'R'), (r_script_path, 'R script')]:
        if not Path(path).is_file():
            raise FileNotFoundError(f"{name} 경로를 찾을 수 없습니다: {path}")
    
    work_dir = str(Path(r_script_path).parent) # r_script_path의 부모 디렉토리가 R 스크립트의 작업 디렉토리
    
    # R 스크립트 실행 명령어 구성
    cmd = [r_path, r_script_path, end_date, work_dir]

    try:
        # R 스크립트 실행
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check = True
        )
        # R 스크립트의 표준 출력 캡처
        output = process.stdout.strip()
        
        # JSON 파싱
        result = json.loads(output.split('\n')[-1])
        df = pd.DataFrame({'Date': result['Date'], 'rstar': result['rstar']})
        df.set_index('Date', inplace=True)
        df.index = pd.to_datetime(df.index)
        return df

    except subprocess.CalledProcessError as e:
        print(f"R 스크립트 실행 중 오류 발생: {e.stderr}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON 파싱 오류: {e}")
        print(f"R 스크립트 출력: {output}")
        return None

# FS 모형 Matlab 코드 실행 함수
def run_FS(end_date: str, 
           paths: ModelPaths,
           ):
    """
    FS 모형 Matlab 코드 실행 함수

    Parameters
    ----------
    end_date : str
        Matlab 함수에 넘겨줄 마지막 날짜 (예: '2025-03-31')
    matlab_path : str
        MATLAB 실행 파일 경로 (예: 'C:/Program Files/MATLAB/R2024a/bin/matlab.exe')
    matlab_script_path : str
        Matlab에서 추가할 경로 (예: 'C:/Users/bok/Desktop/projects/matlab/NIR/KangShin_2024/Main')

    Returns
    -------
    pd.DataFrame | None
        'Date'를 인덱스로 설정한 중립금리 시계열 데이터프레임. 오류 시 None 반환.
    """
    
    matlab_path = paths.fs_matlab_path
    matlab_script_path = paths.fs_matlab_scr_path
    matlab_lib_path = paths.fs_matlab_lib_path
    
    # 입력 경로 유효성 검사
    for path, name in [(matlab_path, 'MATLAB 실행 파일'), (matlab_script_path, 'Matlab script 경로')]:
        if not Path(path).exists():
            raise FileNotFoundError(f"{name}를 찾을 수 없습니다: {path}")
        
    try:
        matlab_command = (
            f"addpath('{matlab_script_path}');"
            f"compute_rstar_FS('{end_date}', '{matlab_lib_path}');"
        )

        cmd = [matlab_path, '-batch', matlab_command]
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='cp949', # cp949
            errors = 'replace',
            check=True
        )
        
        stdout = process.stdout.strip()
        stderr = process.stderr.strip()
        
        if stderr:
            raise subprocess.CalledProcessError(stderr)
        
        #json_file = r"C:\Users\bok\Desktop\projects\python\NIR_Validation\rstar_FS.json"
        json_file = 'rstar_FS.json'

        timeout = 30
        start_time = time.time()
        while not os.path.exists(json_file):
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Timeout: {timeout} 초 동안 {json_file} 파일이 생성되지 않았습니다.")
                return None
            time.sleep(0.5)
            
        with open(json_file, 'r') as f:
            result = json.load(f)
            
        df = pd.DataFrame(result)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        return df
    
    except subprocess.CalledProcessError as e:
        print(f"MATLAB 스크립트 실행 중 오류 발생: {e.stderr}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON 파싱 오류: {e}")
        return None
    except Exception as e:
        print(f"오류 발생: {e}")
        return None
    
# HLW 모형 파이썬 코드 실행 함수
def run_HLW(end_period, smoothing, paths: ModelPaths):

    try:        
        # Papermill로 Jupyter Notebook 실행하면서 end_date 파라미터 전달
        pm.execute_notebook(
            paths.hlw_nb_path,
            paths.hlw_output_path,
            parameters={"end_period": end_period, "smoothing": smoothing}
        )
        
        with open('rstar_HLW.json', 'r') as json_file:
            result = json.load(json_file)
            
        result_df = pd.json_normalize(result)
        result_df.set_index('Date', inplace=True)
        result_df.index = pd.to_datetime(result_df.index)
        return result_df
    
    except subprocess.CalledProcessError as e:
        print(f"Jupyter Notebook 실행 중 오류 발생: {e.stderr}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON 파싱 오류: {e}")
        return None
    except Exception as e:
        print(f"오류 발생: {e}")
        return None

# HLW-FC 모형 파이썬 코드 실행 함수
def run_HLW_FC(end_period, smoothing, paths: ModelPaths):

    try:
        # Papermill로 Jupyter Notebook 실행하면서 end_date 파라미터 전달
        pm.execute_notebook(
            paths.hlw_fc_nb_path,
            paths.hlw_fc_output_path,
            parameters={"end_period": end_period, "smoothing": smoothing}
        )
        
        with open('rstar_HLW_FC.json', 'r') as json_file:
            result = json.load(json_file)
            
        result_df = pd.json_normalize(result)
        result_df.set_index('Date', inplace=True)
        result_df.index = pd.to_datetime(result_df.index)
        return result_df
    
    except subprocess.CalledProcessError as e:
        print(f"Jupyter Notebook 실행 중 오류 발생: {e.stderr}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON 파싱 오류: {e}")
        return None
    except Exception as e:
        print(f"오류 발생: {e}")
        return None

# HLW-Covid 모형 파이썬 코드 실행 함수    
def run_HLW_Covid(end_period, smoothing, paths: ModelPaths):

    try:
        # Papermill로 Jupyter Notebook 실행하면서 end_date 파라미터 전달
        pm.execute_notebook(
            paths.hlw_covid_nb_path,
            paths.hlw_covid_output_path,
            parameters={"end_period": end_period, "smoothing": smoothing}
        )
        
        with open('rstar_HLW_Covid.json', 'r') as json_file:
            result = json.load(json_file)
            
        result_df = pd.json_normalize(result)
        result_df.set_index('Date', inplace=True)
        result_df.index = pd.to_datetime(result_df.index)
        return result_df
    
    except subprocess.CalledProcessError as e:
        print(f"Jupyter Notebook 실행 중 오류 발생: {e.stderr}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON 파싱 오류: {e}")
        return None
    except Exception as e:
        print(f"오류 발생: {e}")
        return None
    
# 함수
def gen_lagged_data(X, lag):
    
    temp = X.copy()
    
    for i in range(1, lag+1):
        lagx = temp.shift(i)
        lagx.columns = [col + '_lag' + str(i) for col in temp.columns]
        X = pd.concat([X, lagx], axis=1)
        
    return X

def gen_train_data(df, lag, q, rolling):
    data = df.copy()
    data = data.reindex(pd.date_range(data.index[0], q, freq='Q'))
    
    # normalization
    Xm = data.mean()
    Xs = data.std()
    Xn = (data - Xm) / Xs
    
    # without normalization
    #Xn = data.copy()
    
    # 결측치 보간 at head
    #Xn = Xn.fillna(method='bfill')
    
    # lag 생성
    Xlag = gen_lagged_data(Xn, lag)
    
    # rolling or recursive
    if (rolling > 0) and (len(Xlag) > rolling * 4):
        dump_months = len(Xlag) - rolling * 4
        Xlag = Xlag.iloc[dump_months:]
        
    # target
    y = df.loc[:, df.columns[0]] # df의 가장 첫번째 열이 target
    
    # feature
    Xlag = Xlag.drop(df.columns[0], axis=1)
    
    return Xlag, y

def print_rmse_mae(rmse_mae_results, rmse_mae_avg, layout='side_by_side'):
    from IPython.display import display, HTML

    # Convert rmse_mae_avg from Styler to DataFrame
    rmse_mae_avg_df = rmse_mae_avg.data

    if layout == 'side_by_side':
        # Create HTML for side-by-side tables
        html = '''
        <div style="display: flex; gap: 20px;">
            <div>
                <p style="text-align: center;"><b>RMSE & MAE by Horizon</b></p>
                {}
            </div>
            <div>
                <p style="text-align: center;"><b>Average RMSE & MAE</b></p>
                {}
            </div>
        </div>
        '''.format(rmse_mae_results.to_html(), rmse_mae_avg_df.to_html())
    else:
        # Create HTML for top-bottom tables
        html = '''
        <div>
            <div>
                <p style="text-align: left;"><b>RMSE & MAE by Horizon</b></p>
                {}
            </div>
            <div style="margin-top: 20px;">
                <p style="text-align: left;"><b>Average RMSE & MAE</b></p>
                {}
            </div>
        </div>
        '''.format(rmse_mae_results.to_html(), rmse_mae_avg_df.to_html())

    display(HTML(html))
    
def print_mda(mda_results, mda_avg, layout='side_by_side'):
    from IPython.display import display, HTML

    # Convert mda_avg from Styler to DataFrame
    mda_avg_df = mda_avg.data

    if layout == 'side_by_side':
        # Create HTML for side-by-side tables
        html = '''
        <div style="display: flex; gap: 20px;">
            <div>
                <p style="text-align: center;"><b>MDA by Horizon</b></p>
                {}
            </div>
            <div>
                <p style="text-align: center;"><b>Average MDA</b></p>
                {}
            </div>
        </div>
        '''.format(mda_results.to_html(), mda_avg_df.to_html())
    else:
        # Create HTML for top-bottom tables
        html = '''
        <div>
            <div>
                <p style="text-align: left;"><b>MDA by Horizon</b></p>
                {}
            </div>
            <div style="margin-top: 20px;">
                <p style="text-align: left;"><b>Average MDA</b></p>
                {}
            </div>
        </div>
        '''.format(mda_results.to_html(), mda_avg_df.to_html())

    display(HTML(html))
    
def make_cum_annual_gdp(df, quarters=4, label='rgdp'):
    '''
    누적상승률을 연율로 계산
    '''
    #months=3 # 분기별 데이터
    for t in df.index[quarters:]:
        df0 = df.loc[t - DateOffset(months=3*quarters):t, label]
        #cum = (1 + df0).prod() - 1
        #annualized = ((1 + cum) ** (4/quarters)) - 1
        #X = (1 + df0).prod()
        #annualized = X ** (4/quarters) - 1
        #annualized = annualized * 100
        yt = df0.iloc[0]
        yt_h = df0.iloc[-1]
        R = (yt_h) / yt
        annualized = ((R) ** (4/quarters)) - 1
        annualized = annualized * 100
        df.loc[t, f'rgdp_{quarters//4}y'] = annualized
        
    return df

def make_cum_annual_gdp_lndiff(df, quarters=4, label='rgdp'):
    '''
    로그차분으로 연율 계산
    '''
    for t in df.index[quarters:]:
        df0 = df.loc[t - DateOffset(months=3*quarters):t, label]
        qt = df0.iloc[0] # Q_t
        qt_h = df0.iloc[-1] # Q_t+h
        annualized = (4/quarters) * np.log(qt_h / qt)
        annualized = annualized * 100
        df.loc[t, f'rgdp_{quarters//4}y'] = annualized
        
    return df

def make_cum_annual_inf(df, quarters=4, label='inf1'):
    
    for t in df.index[quarters:]:
        df0 = df.loc[t - DateOffset(months=3*quarters):t, label]
        pt = df0.iloc[0] # P_t
        pt_h = df0.iloc[-1] # P_t+h
        R = (pt_h - pt) / pt # (P_t+h - P_t) / P_t
        annualized = ((1 + R) ** (4/quarters)) - 1
        annualized = annualized * 100
        df.loc[t, f'inf1_{quarters//4}y'] = annualized
        
    return df

def make_cum_annual_inf_lndiff(df, quarters=4, label='inf1'):
    
    for t in df.index[quarters:]:
        df0 = df.loc[t - DateOffset(months=3*quarters):t, label]
        pt = df0.iloc[0] # P_t
        pt_h = df0.iloc[-1] # P_t+h
        annualized = (4/quarters) * np.log(pt_h / pt)
        annualized = annualized * 100
        df.loc[t, f'inf1_{quarters//4}y'] = annualized
        
    return df
    

def get_mda1(pred, act):
    'MDA 계산1'
    act_diff = np.sign(act.diff())
    pred_diff = np.sign(pred.diff())
    mda = (act_diff == pred_diff).mean()
    
    return mda

def get_mda3(pred, act, hor, base='act', verbose=True, model =''):
    'MDA 계산3'
    targets = pred.index
    act_diff = act[targets].shift(-hor) - act[targets]
    act_sign = np.sign(act_diff.dropna())
    
    if base == 'act':
        pred_diff = pred[targets].shift(-hor) - act[targets]
    else:
        pred_diff = pred[targets].shift(-hor) - pred[targets]
        
    pred_sign = np.sign(pred_diff.dropna())
    
    matched_signs = act_sign == pred_sign
    mda = matched_signs.sum() / len(matched_signs)
    
    p1 = f'mda: {mda:.2f} ({matched_signs.sum()} / {len(matched_signs)})'
    
    if verbose:
        print(p1, end=', ')
        print(model)
        
    return mda, matched_signs, p1

def get_mda2(pred, act, hor, base='act', verbose=True, model =''):
    'MDA 계산2'
    targets = pred.index
    qts = hor//3
    act_diff = act[targets] - act[targets].shift(qts+1)
    act_sign = np.sign(act_diff.dropna())
    
    if base == 'act':
        pred_sign = np.sign((pred[targets] - act[targets].shift(qts+1)).dropna())
    else:
        pred_sign = np.sign((pred[targets] - pred[targets].shift(qts+1)).dropna())
        
    matched_signs = act_sign == pred_sign
    mda = matched_signs.sum() / len(matched_signs)
    
    p1 = f'mda: {mda:.2f} ({matched_signs.sum()} / {len(matched_signs)})'
    
    if verbose:
        print(p1, end=', ')
        print(model)
        
    return mda, matched_signs, p1

def get_mda_avg(mda_res, verbose=True):
    '모든 에측시계에 대한 MDA 평균 계산'
    mda_avg = mda_res.mean().to_frame('MDA')
    mda_avg = mda_avg.style.highlight_max(props='font-weight:bold;')
    
    if verbose:
        mda_avg
    
    return mda_avg


def nir_wfvalid_for_annualized_inf(data, horizon, targets, model_list, lag, rolling):
    
    # 예측 결과를 저장하기 위한 MultiIndex 생성
    idx = pd.MultiIndex.from_product([model_list, horizon], names=['model', 'horizon'])
    p_res = pd.DataFrame(index=targets, columns=idx, dtype='float64')
    p_res.index.name = 'Date'

    act = pd.DataFrame(index = targets)
    for hor in horizon:
        quarters = hor//3
        df0 = make_cum_annual_inf(data.copy(), quarters=quarters, label=data.columns[0])
        df0 = df0.dropna()
        act[f'inf1_{hor//12}y'] = df0.loc[targets, f'inf1_{hor//12}y']
        
        for model in model_list:
            YX = df0[[f'inf1_{hor//12}y', model, 'infe', 'rgdp']].copy()
            
            for tq in targets:
                # 검증데이터 준비
                q = tq - DateOffset(months=hor) + MonthEnd(0)
                X_train, Y_train = gen_train_data(YX, lag, q, rolling)
            
                # 추정
                reg_eqn = 'Y ~ 1 + ' + ' + '.join(X_train.columns.tolist())
                Y_train = Y_train.shift(-hor//3).to_frame('Y')
                XY_train = pd.concat([X_train, Y_train], axis=1).dropna(axis=0)
                reg = smf.ols(reg_eqn, data=XY_train).fit()
                
                # 예측
                predictor = X_train.loc[[q]]
                predicted = reg.predict(predictor)
                p_res.loc[tq, (model, hor)] = predicted.values[0]
    
    return p_res, act

def plot_pred_results(p_res, act, horizon, fig_size=(15, 12)):
    lw = {'LM': 1.5, 'FS': 1.5}
    al = {'LM': 0.5, 'FS': 0.5}
    ls = {
        'LM': {'color': 'blue', 'linestyle': '-'},
        'FS': {'color': 'green', 'linestyle': '-'}
    }

    fig, axs = plt.subplots(len(horizon), 1, figsize=fig_size)
    for i, h in enumerate(horizon):

        axs[i].plot(act.iloc[:,i], color='red', linestyle='--', linewidth=3, label='Actual')

        prd = p_res.loc[:, p_res.columns.get_level_values('horizon') == h]
        for col in prd.columns.get_level_values('model'):
            axs[i].plot(prd[col], linewidth=lw.get(col, 1), alpha=al.get(col, 1), **ls.get(col, 1), label=col)

        axs[i].set_title(f'{h}-months ahead', fontsize=12)
        axs[i].legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=4)
    plt.tight_layout()
    plt.show()


#####################################################################################################################     
class NIRValidation:
    def __init__(self, spec):
        self.spec = spec
        self.p_res = None
        self.act = None
        self.print_options()

    def print_options(self):
        print("========Validation Options========")
        print(f"Horizon: {self.spec.horizon[0], self.spec.horizon[1], self.spec.horizon[2]}-months ahead")
        print(f"Targets Range: {self.spec.targets[0].strftime('%Y-%m-%d')} to {self.spec.targets[-1].strftime('%Y-%m-%d')}")
        print(f"Model List: {self.spec.model_list}")
        print(f"Lag: {self.spec.lag}")
        print(f"Rolling: {self.spec.rolling}-years")
        print("==================================")

    def nir_wfvalid_for_annualized(self, data: pd.DataFrame, target_var: str = 'inf', type: str = 'regression', mp_stance: bool = False):

        # 예측 결과를 저장하기 위한 MultiIndex 생성
        idx = pd.MultiIndex.from_product([self.spec.model_list, self.spec.horizon], names=['model', 'horizon'])
        self.p_res = pd.DataFrame(index=self.spec.targets, columns=idx, dtype='float64')
        self.p_res.index.name = 'Date'
        
        self.act = pd.DataFrame(index=self.spec.targets)
        print(f'Target Variable: {data.columns[0]}')
        print(f'Validation type: Forecasting \033[1m{target_var}\033[0m based on \033[1m{type}\033[0m model')
        #pbar = tqdm(self.spec.horizon, desc="Processing horizons", leave=False)
        for hor in self.spec.horizon:
            quarters = hor // 3
            
            if target_var == 'inf':
                df0 = make_cum_annual_inf_lndiff(data.copy(), quarters=quarters, label=data.columns[0])
                df0 = df0.dropna()
                self.act[f'inf1_{hor // 12}y'] = df0.loc[self.spec.targets, f'inf1_{hor // 12}y']
                #vn = 'rgdp'
            else:
                df0 = make_cum_annual_gdp_lndiff(data.copy(), quarters=quarters, label=data.columns[0])
                df0 = df0.dropna()
                self.act[f'rgdp_{hor // 12}y'] = df0.loc[self.spec.targets, f'rgdp_{hor // 12}y']
                vn = 'inf1'

            #pbar = tqdm(self.spec.model_list, desc=f"Processing models")
            for model in self.spec.model_list:
                #pbar.set_description(f"For forecast horizon {hor}, current model is {model}")

                pbar = tqdm(self.spec.targets, desc=f"Processing targets")
                for tq in pbar: #self.spec.targets:
                    q = tq - DateOffset(months=hor) + MonthEnd(0)
                    pbar.set_description(f"Model: {model}, Horizon: {hor}, Target: {str(tq).split(' ')[0]}, q: {str(q).split(' ')[0]}")

                    if model == 'RW':
                        # Random Walk 예측
                        predicted = df0[f'{data.columns[0]}_{hor // 12}y'].loc[q]
                        self.p_res.loc[tq, (model, hor)] = predicted
                    elif model == 'AR':
                        # AR 모형 예측
                        y = df0[f'{data.columns[0]}_{hor // 12}y']
                        y_demeaned = y - y.mean()
                        y_lagged = y_demeaned.shift(self.spec.lag).dropna()
                        y_demeaned = y_demeaned[self.spec.lag:]
                        coef = np.linalg.lstsq(y_lagged.values.reshape(-1, 1), y_demeaned.values, rcond=None)[0]
                        predicted = (coef ** quarters) * y_lagged.loc[q] + y.mean()
                        self.p_res.loc[tq, (model, hor)] = predicted
                    elif type == 'regression':
                        if model == 'LM':
                            try:
                                rstar = pd.read_csv(f"input/rstar_LM_{str(q).split(' ')[0]}.csv", index_col = 0)
                                rstar.index = pd.to_datetime(rstar.index)
                                #print(f'Rstar for Lubik-Matthes model at time {str(q).split(" ")[0]} is loaded')
                            except:
                                #print(f'Estimating real neutral interest rate up to {str(q).split(" ")[0]} using the Lubik-Matthes model')
                                rstar = run_LM(end_date=str(q).split(' ')[0], paths=self.spec.paths)
                                rstar.to_csv(f"input/rstar_LM_{str(q).split(' ')[0]}.csv")
                            if mp_stance:
                                df0['LM_gap'] = df0['realrate'] - rstar.iloc[:, 0]
                                df0['LM'] = rstar
                            else:
                                df0['LM'] = rstar

                        elif model == 'FS':
                            try:
                                rstar = pd.read_csv(f"input/rstar_FS_{str(q).split(' ')[0]}.csv", index_col=0)
                                rstar.index = pd.to_datetime(rstar.index)
                                #print(f'Rstar for Ferreira-Shousha model at time {str(q).split(" ")[0]} is loaded')
                            except:
                                #print(f'Estimating real neutral interest rate up to {str(q).split(' ')[0]} using the Ferreira-Shousha model')
                                rstar = run_FS(str(q).split(' ')[0], paths=self.spec.paths)
                                rstar.to_csv(f"input/rstar_FS_{str(q).split(' ')[0]}.csv")
                            if mp_stance:
                                df0['FS_gap'] = df0['realrate'] - rstar.iloc[:, 0]
                                df0['FS'] = rstar.iloc[:, 0]
                            else:
                                df0['FS'] = rstar.iloc[:, 0]
                                
                            
                        elif model == 'HLW':
                            try: 
                                rstar = pd.read_csv(f"input/rstar_HLW_{str(q).split(' ')[0]}.csv", index_col=0)
                                rstar.index = pd.to_datetime(rstar.index)
                            except:
                                rstar = run_HLW(end_period=str(q).split(' ')[0], smoothing=False, paths=self.spec.paths)
                                rstar.to_csv(f"input/rstar_HLW_{str(q).split(' ')[0]}.csv")
                            if mp_stance:
                                df0['HLW_gap'] = df0['realrate'] - rstar.iloc[:, 0]
                                df0['HLW'] = rstar
                            else:
                                df0['HLW'] = rstar
                            
                        elif model == 'HLW_smoothed':
                            try:
                                rstar = pd.read_csv(f"input/rstar_HLW_smoothed_{str(q).split(' ')[0]}.csv", index_col=0)
                                rstar.index = pd.to_datetime(rstar.index)
                            except:
                                rstar = run_HLW(end_period=str(q).split(' ')[0], smoothing=True, paths=self.spec.paths)
                                rstar.to_csv(f"input/rstar_HLW_smoothed_{str(q).split(' ')[0]}.csv")
                            if mp_stance:
                                df0['HLW_smoothed_gap'] = df0['realrate'] - rstar.iloc[:, 0]
                                df0['HLW_smoothed'] = rstar
                            else:
                                df0['HLW_smoothed'] = rstar

                        elif model == 'HLW_FC':
                            try:
                                rstar = pd.read_csv(f"input/rstar_HLW_FC_{str(q).split(' ')[0]}.csv", index_col=0)
                                rstar.index = pd.to_datetime(rstar.index)
                            except:
                                rstar = run_HLW_FC(end_period=str(q).split(' ')[0], smoothing=False, paths=self.spec.paths)
                                rstar.to_csv(f"input/rstar_HLW_FC_{str(q).split(' ')[0]}.csv")
                            if mp_stance:
                                df0['HLW_FC_gap'] = df0['realrate'] - rstar.iloc[:, 0]
                                df0['HLW_FC'] = rstar
                            else:
                                df0['HLW_FC'] = rstar

                        elif model == 'HLW_FC_smoothed':
                            try:
                                rstar = pd.read_csv(f"input/rstar_HLW_FC_smoothed_{str(q).split(' ')[0]}.csv", index_col=0)
                                rstar.index = pd.to_datetime(rstar.index)
                            except:
                                rstar = run_HLW_FC(end_period=str(q).split(' ')[0], smoothing=True, paths=self.spec.paths)
                                rstar.to_csv(f"input/rstar_HLW_FC_smoothed_{str(q).split(' ')[0]}.csv")
                            if mp_stance:
                                df0['HLW_FC_smoothed_gap'] = df0['realrate'] - rstar.iloc[:, 0]
                                df0['HLW_FC_smoothed'] = rstar
                            else:
                                df0['HLW_FC_smoothed'] = rstar

                        elif model == 'HLW_Covid':
                            try:
                                rstar = pd.read_csv(f"input/rstar_HLW_Covid_{str(q).split(' ')[0]}.csv", index_col=0)
                                rstar.index = pd.to_datetime(rstar.index)
                            except:
                                rstar = run_HLW_Covid(end_period=str(q).split(' ')[0], smoothing=False, paths=self.spec.paths)
                                rstar.to_csv(f"input/rstar_HLW_Covid_{str(q).split(' ')[0]}.csv")
                            if mp_stance:
                                df0['HLW_Covid_gap'] = df0['realrate'] - rstar.iloc[:, 0]
                                df0['HLW_Covid'] = rstar
                            else:
                                df0['HLW_Covid'] = rstar

                        elif model == 'HLW_Covid_smoothed':
                            try:
                                rstar = pd.read_csv(f"input/rstar_HLW_Covid_smoothed_{str(q).split(' ')[0]}.csv", index_col=0)
                                rstar.index = pd.to_datetime(rstar.index)
                            except:
                                rstar = run_HLW_Covid(end_period=str(q).split(' ')[0], smoothing=True, paths=self.spec.paths)
                                rstar.to_csv(f"input/rstar_HLW_Covid_smoothed_{str(q).split(' ')[0]}.csv")
                            if mp_stance:
                                df0['HLW_Covid_smoothed_gap'] = df0['realrate'] - rstar.iloc[:, 0]
                                df0['HLW_Covid_smoothed'] = rstar
                            else:
                                df0['HLW_Covid_smoothed'] = rstar
                        
                        if mp_stance:
                            #YX = df0[[f'{data.columns[0]}_{hor // 12}y', model, f'{model}_gap', 'infe', vn]].copy()
                            YX = df0[[f'{data.columns[0]}_{hor // 12}y', f'{model}_gap']]
                        else:
                            #YX = df0[[f'{data.columns[0]}_{hor // 12}y', model, 'infe', vn]].copy()
                            YX = df0[[f'{data.columns[0]}_{hor//12}y', model]]
                        X_train, Y_train = gen_train_data(YX, self.spec.lag, q, self.spec.rolling)
                        
                        # 추정
                        reg_eqn = 'Y ~ 1 + ' + ' + '.join(X_train.columns.tolist())
                        Y_train = Y_train.shift(-hor // 3).to_frame('Y')
                        XY_train = pd.concat([X_train, Y_train], axis=1).dropna(axis=0)
                        reg = smf.ols(reg_eqn, data=XY_train).fit()

                        # 예측
                        predictor = X_train.loc[[q]]
                        predicted = reg.predict(predictor)[0]
                        self.p_res.loc[tq, (model, hor)] = predicted #.values[0]w
                    else:
                        pass
                    
        class Result:
            def __init__(self, p_res, act, horizon, model_list, spec):
                self.p_res = p_res
                self.act = act
                self.horizon = horizon
                self.model_list = model_list
                self.spec = spec

            def plot_pred_results(self, fig_size: tuple = (15, 12),
                                  line_width: dict = {'LM': 3, 
                                                      'FS': 3, 
                                                      'HLW': 3, 
                                                      'HLW_FC': 3, 
                                                      'HLW_Covid': 3, 
                                                      'HLW_smoothed': 3, 
                                                      'HLW_FC_smoothed': 3, 
                                                      'HLW_Covid_smoothed': 3,
                                                      'RW': 3, 
                                                      'AR': 3},
                                  alpha: dict = {'LM': 0.5, 
                                                 'FS': 0.5, 
                                                 'HLW': 0.5,  
                                                 'HLW_FC': 0.5, 'HLW_Covid': 0.5,
                                                 'HLW_smoothed': 0.5, 
                                                 'HLW_FC_smoothed': 0.5, 
                                                 'HLW_Covid_smoothed': 0.5, 
                                                 'RW': 0.5, 
                                                 'AR': 0.5},
                                  line_style: dict = {'LM': {'color': 'blue', 'linestyle': '-'},
                                                      'FS': {'color': 'green', 'linestyle': '-'},
                                                      'HLW': {'color': 'magenta', 'linestyle': '-'},
                                                      'HLW_FC': {'color': 'cyan', 'linestyle': '-'},
                                                      'HLW_Covid': {'color': 'black', 'linestyle': '-'},
                                                      'HLW_smoothed': {'color': 'gray', 'linestyle': '-'},
                                                      'HLW_FC_smoothed': {'color': 'brown', 'linestyle': '-'},
                                                      'HLW_Covid_smoothed': {'color': 'coral', 'linestyle': '-'},
                                                      'RW': {'color': 'orange', 'linestyle': '-'},
                                                      'AR': {'color': 'purple', 'linestyle': '-'}},
                                  font_size: int = 12,
                                  loc: str = 'upper center',
                                  n_col: int = 4):
                
                fig, axs = plt.subplots(len(self.horizon), 1, figsize=fig_size)
                for i, h in enumerate(self.horizon):
                    axs[i].plot(self.act.iloc[:, i], color='red', linestyle='--', linewidth=4, label='Actual')

                    prd = self.p_res.loc[:, self.p_res.columns.get_level_values('horizon') == h]
                    for col in prd.columns.get_level_values('model'):
                        axs[i].plot(prd[col], linewidth=line_width.get(col, 1), alpha=alpha.get(col, 1), **line_style.get(col, 1), label=col)

                    axs[i].set_title(f'{h}-months ahead', fontsize=12)
                    axs[i].legend(fontsize=font_size, loc=loc, ncol=n_col) # bbox_to_anchor=(0.5, 0.95)
                plt.tight_layout()
                plt.show()
                fig.savefig('fig/forecasting_performance.png', dpi=300, bbox_inches='tight')
                
            def plot_mda_results(self, base: str = 'act', verbose: bool = False, segments=None, fig_size: tuple = (8, 4), 
                                 h_color: str = 'red', h_linestyle: str = '--', h_alpha: float = 0.5):
                # mda_avg 데이터 시각화
                fig, ax = plt.subplots(figsize=fig_size)

                # Bar plot
                mda_res, mda_avg = self.get_mda_seg(base=base, verbose=verbose, segments=segments)
                mda_avg.data.plot(kind='bar', ax=ax, legend=False)

                # 수평선 추가 
                ax.axhline(y=0.5, color=h_color, linestyle=h_linestyle, alpha=h_alpha)

                # 차트 꾸미기
                ax.set_title('Average MDA')
                ax.set_xlabel('Model')
                ax.set_ylabel('MDA')
                ax.grid(True, alpha=0.3)

                # x축 레이블 회전 및 겹침 방지
                new_labels = [label.get_text().replace('_inf_smoothed', '*').replace('_smoothed', '*') for label in ax.get_xticklabels()]
                ax.set_xticklabels(new_labels, rotation=45, ha='right')

                plt.tight_layout()
                plt.show()
                fig.savefig(f'fig/MDA_performance_{list(segments.keys())[0]}_{target_var}.png', dpi=300, bbox_inches='tight')
                
            def get_rmse_mae(self):
                idx = pd.MultiIndex.from_product([self.model_list, ['RMSE', 'MAE']], names=['Model', 'Metric'])
                rmse_mae_results = pd.DataFrame(index=self.horizon, columns=idx, dtype='float64')
                rmse_mae_results.index.name = 'Horizon'
                
                for metric, model, hor in itertools.product(rmse_mae_results.columns.get_level_values('Metric'),
                                                            rmse_mae_results.columns.get_level_values('Model'),
                                                            rmse_mae_results.index):
                    if metric == 'RMSE':
                        label = self.act.columns[0]
                        mse_val = mean_squared_error(self.act.loc[:, f'{self.act.columns[0].split("_")[0]}_{hor//12}y'], self.p_res.loc[:, (model, hor)])
                        rmse_val = np.sqrt(mse_val)
                        rmse_mae_results.loc[hor, (model, metric)] = rmse_val
                    else:
                        mae_val = mean_absolute_error(self.act.loc[:, f'{self.act.columns[0].split("_")[0]}_{hor//12}y'], self.p_res.loc[:, (model, hor)])
                        rmse_mae_results.loc[hor, (model, metric)] = mae_val
                        
                rmse_mae_avg = rmse_mae_results.mean(axis=0).unstack()
                rmse_mae_avg = rmse_mae_avg[['RMSE', 'MAE']]
                rmse_mae_avg = rmse_mae_avg.style.highlight_min(axis=0, props='font-weight:bold;')
                
                return rmse_mae_results, rmse_mae_avg
            
            def get_mda(self, base='pred', verbose=True):
                # Mean Directional Accuracy(MDA) 계산
                mda_results = pd.DataFrame(index=self.horizon, columns=self.model_list, dtype='float64')
                mda_results.index.name = 'Horizon'

                for model, hor in itertools.product(mda_results.columns, mda_results.index):
                    if base == 'act':
                        mda, matched_signs, p1 = get_mda2(self.p_res.loc[:, (model, hor)], self.act.loc[:, f'{self.act.columns[0].split("_")[0]}_{hor//12}y'], hor, base='act', verbose=verbose, model=model)
                    else:
                        mda = get_mda1(self.p_res.loc[:, (model, hor)], self.act.loc[:, f'{self.act.columns[0].split("_")[0]}_{hor//12}y'])
                    mda_results.loc[hor, model] = mda

                mda_avg = get_mda_avg(mda_results)
                    
                return mda_results, mda_avg
            
            def get_mda_seg(self, base='pred', verbose=True, segments=None):
                """
                Calculate Mean Directional Accuracy (MDA).
                
                If segments is provided, it should be a dictionary where each key is a segment label and 
                each value is a tuple of (start_date, end_date) to define the subperiod over which MDA is computed.
                If segments is None, the full out‐of‐sample period is used.
                """
                # If no segmentation is provided, use the full range as a single segment
                if segments is None:
                    segments = {'full': (self.spec.targets[0], self.spec.targets[-1])}

                for seg_label, (start_date, end_date) in segments.items():
                    # Filter the actual and forecast results by the segment dates
                    mask = (self.act.index >= start_date) & (self.act.index <= end_date)
                    if mask.sum() == 0:
                        print(f"No data in segment {seg_label} for dates {start_date} to {end_date}.")
                        continue

                    act_seg = self.act.loc[mask]
                    p_res_seg = self.p_res.loc[mask]

                    mda_results = pd.DataFrame(index=self.horizon, columns=self.spec.model_list, dtype='float64')
                    mda_results.index.name = 'Horizon'

                    # Compute MDA for each combination of model and horizon for the segment
                    for model, hor in itertools.product(self.spec.model_list, self.horizon):
                        # Derive the actual variable name from the column header (assumes a specific naming convention)
                        actual_var_name = f'{self.act.columns[0].split("_")[0]}_{hor//12}y'
                        if base == 'act':
                            mda, matched_signs, p1 = get_mda2(
                                p_res_seg.loc[:, (model, hor)],
                                act_seg.loc[:, actual_var_name],
                                hor,
                                base='act',
                                verbose=verbose,
                                model=model
                            )
                        else:
                            mda = get_mda1(
                                p_res_seg.loc[:, (model, hor)],
                                act_seg.loc[:, actual_var_name]
                            )
                        mda_results.loc[hor, model] = mda

                    mda_avg = get_mda_avg(mda_results, verbose=verbose)

                return mda_results, mda_avg
        return Result(self.p_res, self.act, self.spec.horizon, self.spec.model_list, self.spec)
        #return Result(self.p_res, self.act, self.spec.horizon, self.spec.model_list)