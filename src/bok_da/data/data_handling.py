import pandas as pd
import numpy as np
import requests
#import s3fs
import os
import pickle

def get_bidas_data(req_ids, alias_nm, 
                     start_d=None, end_d=None,
                     period_trim=False):
    # API 호출
    API = "http://datahub.boknet.intra/api/v1/obs/lists"
    res = requests.post(API, data={"ids":req_ids})
    data_list = res.json()["data"][0]
    
    # API 호출로 받은 결과를 Data Frame으로 저장
    data = pd.DataFrame()
    for alias, value in zip(alias_nm, data_list):
        try:
            df = pd.DataFrame(value["observations"])
            df.set_index("period", inplace=True)
            df.index = pd.to_datetime(df.index)
            df.columns = [alias]
            data = df.copy() if not len(data) else data.join(df, how="outer")
        except:
            print(f'{alias} is not imported.')
    
    # 옵션에 따라 시작일, 종료일, Trim 적용
    if start_d:
        data = data[data.index >= start_d]
    if end_d:
        data = data[data.index <= end_d]
    if period_trim:
        data.index = data.index.to_period('M')
    return data

class FairsFissPreprocessing:
    def lag_1q(df, i, t, x):
        df = df.sort_values(by=[i,t])
        df['year'] = df[t].dt.year
        df['quarter'] = df[t].dt.quarter

        for v in [i, 'year', 'quarter', x]:
            lag = v + '_L1'
            df[lag] = df[v].shift(1)
        df['quarter_diff'] = (df['year'] - df['year_L1']) * 4 + (df['quarter'] - df['quarter_L1'])

        i_L1 = i + '_L1'
        x_L1 = x + '_L1'
        df.loc[(df[i] != df[i_L1]) | (df['quarter_diff'] != 1), x_L1] = np.nan
        return df

def get_fairsfiss_data(req_ids, alias_nm, bidas_stoken, start_d=None, end_d=None, period_trim=False):

    # API 호출
    API = "http://datahub.boknet.intra/api/v1/obs/lists"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0)'}
    res = requests.post(API, data={"ids": req_ids}, cookies={'sToken': bidas_stoken}, headers=headers)
    data_list = res.json()["data"][0]

    # API 호출로 받은 결과를 Data Frame으로 저장
    data = pd.DataFrame()
    null_dict = {'period': ['2023-12-01', '2024-01-01'], 'value': [0, 0]}
    for alias, value in zip(alias_nm, data_list):
        if not value["observations"]:
            df = pd.DataFrame(null_dict)
        else:
            df = pd.DataFrame(value["observations"])#, dtype="float")
        df.set_index("period", inplace=True)
        df.index = pd.to_datetime(df.index)
        df.columns = [alias]
        data = df.copy() if not len(data) else data.join(df, how="outer")

    for col_nm in data.columns:
        data[col_nm] = data[col_nm].astype(float)

    # 옵션에 따라 시작일, 종료일, Trim 적용
    if start_d:
        data = data[data.index >= start_d]
    if end_d:
        data = data[data.index <= end_d]
    if period_trim:
        data.index = data.index.to_period('M')
    return data

def has_s3key():
    s3_key = os.environ.get('FSSPEC_S3_KEY')
    
    if s3_key:
        return True
    else:
        return False


def upload_to_s3(local_path, s3_path):    
    if has_s3key():
        bidas_fs = s3fs.S3FileSystem(anon=False)
        _ = bidas_fs.put(local_path, s3_path)
        

def download_from_s3(s3_path, local_path):
    if has_s3key():
        bidas_fs = s3fs.S3FileSystem(anon=False)
    else:
        bidas_fs = s3fs.S3FileSystem(anon=True)

    _ = bidas_fs.get(s3_path, local_path)
        

def set_local_path():
    current_dir = os.getcwd()
    if os.access(current_dir, os.W_OK):
        local_path = current_dir
    else:
        local_path = os.getenv('HOME')
        
    return local_path

def nan_to_lag_data(df):
    # 공휴일은 아닌데 시장에 따라 거래가 없는 날(특히 주식시장)의 nan data를 그 전일 data로 대체
    df_lag = df.shift(1)
    new_df = df.copy()
    for cols in df.columns:
        nan_rows = pd.isna(df[cols])
        nan_rows_index = df[cols][nan_rows].index
        lag_data = df_lag[cols].loc[nan_rows_index].values
        for idx in range(len(nan_rows_index)):
            new_df.loc[nan_rows_index[idx], cols] = lag_data[idx]
    return new_df

# 관리물가, 관리제외물가지수 생성 함수
def gen_P_adm_eadm_v2(df, w, b_date, e_date):
    
    df1 = df.iloc[:,:35]
    df2 = df.iloc[:,35:]
    
    # 관리물가지수 생성
    w1 = w.iloc[0:35] / w.iloc[0:35].sum()
    w1 = np.asarray(w1)
    P_a = df1.dot(w1)
    c_name = 'P_adm'
    P_a = pd.DataFrame(P_a, columns=[c_name])
    #P_a = (np.log(P_a) - np.log(P_a.shift(12)))*100 # 데이터 변환
                                        
    # 관리제외물가지수 생성
    w2 = w.iloc[35:] / w.iloc[35:].sum()
    w2 = np.asarray(w2)
    P_ea = df2.dot(w2)
    c_name = 'P_adm'
    P_ea = pd.DataFrame(P_ea, columns=[c_name])
    #P_ea = (np.log(P_ea) - np.log(P_ea.shift(12)))*100 # 데이터 변환
                                        
    return P_a, P_ea