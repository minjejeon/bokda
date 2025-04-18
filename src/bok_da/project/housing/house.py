import pandas as pd
from pandas.tseries.offsets import DateOffset
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import requests

idx = pd.IndexSlice

import glob
import os
import time

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
from datetime import datetime

import zipfile
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

##################################  storage option ##################################################################
import s3fs

fs = s3fs.S3FileSystem(anon=False)
fs_anon = s3fs.S3FileSystem(anon=True)

s3_bigdatahub_path = 's3://open-data/주택_실거래가_빈티지/'
s3_newtech_path = 's3://newtech/public/housing'

##################################  plot option #####################################################################

# matplotlib.rc('xtick', labelsize=16)
# matplotlib.rc('ytick', labelsize=16)
# rcParams.update({'figure.autolayout': True})
# plt.tight_layout()

##################################  start year #####################################################################
syear = 2011


# start_date = '2022-4-15'
# last_date = '2022-12-31'

# tb_dw_rtms_at_files = sorted(glob.glob('/home/work/newtech/housing/realestate_new_data/tb_dw_rtms_at_*.csv.gz'))
# weekly_dates = [date.strftime("%Y%m%d") for date 
#                 in pd.date_range(start_date, last_date, freq='3D')]   # str <--from-- time  # date_range is end-inclusive
# weekly_files = [file for file in tb_dw_rtms_at_files if file[-15:-7] in weekly_dates]
# vintage_dates = [date for date in pd.date_range(start_date, last_date, freq='3D')
#                  if date.strftime("%Y%m%d") in [file[-15:-7] for file in weekly_files]]   # Timestamp
# monthly_dates = pd.date_range('2022-1-1', '2022-12-1', freq='MS')   # every 1st days of all months

##################################  regions ########################################################################

s11 = ['전국', '수도권', '지방', '6대광역시', '5대광역시', '8개도', '도심권', '동북권', '서북권', '서남권', '동남권']
e11 = ['N', 'M', 'NM', 'M6', 'M5', 'P8', 'C', 'EN', 'WN', 'WS', 'ES']

a17 = ['서울특별시', '부산광역시', '광주광역시', '전라남도', '제주특별자치도', '경기도', '인천광역시', '울산광역시', '경상북도', 
       '경상남도', '충청남도', '전라북도', '대구광역시', '대전광역시', '강원도', '충청북도', '세종특별자치시']
s17 = ['서울', '부산', '광주', '전남', '제주', '경기', '인천', '울산', '경북', '경남', '충남', '전북', '대구', '대전', '강원', '충북', '세종']
e17 = ['SE', 'BS', 'GJ', 'JN', 'JJ', 'GG', 'IC', 'US', 'GB', 'GN', 'CN', 'JB', 'DG', 'DJ', 'GW', 'CB', 'SJ']
NAMETABLE = {'N': 'National', 'SE': 'Seoul', 'ES': 'Seoul East-South', 'GG': 'Gyeonggi', 'M': 'Metropolitan Area', 'NM': 'Non-Metropolitan'}
regions = [e17, ['SE'], ['SE'], ['GG'], ['SE', 'GG', 'IC'], [re for re in e17 if re not in ['SE', 'GG', 'IC']], ['IC']]
rnames = ['N', 'SE', 'ES', 'GG', 'M', 'NM', 'IC']

addr1_SE_regions = ['서울특별시 강남구', '서울특별시 강동구', '서울특별시 강북구', '서울특별시 강서구', '서울특별시 관악구',
                    '서울특별시 광진구', '서울특별시 구로구', '서울특별시 금천구', '서울특별시 노원구', '서울특별시 도봉구', '서울특별시 동대문구',
                    '서울특별시 동작구', '서울특별시 마포구', '서울특별시 서대문구', '서울특별시 서초구', '서울특별시 성동구', '서울특별시 성북구',
                    '서울특별시 송파구', '서울특별시 양천구', '서울특별시 영등포구', '서울특별시 용산구', '서울특별시 은평구', '서울특별시 종로구',
                    '서울특별시 중구', '서울특별시 중랑구']

addr1_SE_rnames = ['SE_' + str(i) for i in range(len(addr1_SE_regions))]
SE_code2name = dict(zip(addr1_SE_rnames, addr1_SE_regions))

addr1_GG_regions = ['경기도 가평군', '경기도 고양시', '경기도 과천시', '경기도 광명시', '경기도 광주시', '경기도 구리시',
                    '경기도 군포시', '경기도 김포시', '경기도 남양주시', '경기도 동두천시', '경기도 부천시', '경기도 성남시',
                    '경기도 수원시', '경기도 시흥시', '경기도 안산시', '경기도 안성시', '경기도 안양시', '경기도 양주시',
                    '경기도 양평군', '경기도 여주시', '경기도 연천군', '경기도 오산시', '경기도 용인시', '경기도 의왕시',
                    '경기도 의정부시', '경기도 이천시', '경기도 파주시', '경기도 평택시', '경기도 포천시', '경기도 하남시',
                    '경기도 화성시']

addr1_GG_rnames = ['GG_' + str(i) for i in range(len(addr1_GG_regions))]
GG_code2name = dict(zip(addr1_GG_rnames, addr1_GG_regions))

regions = regions + addr1_SE_regions + addr1_GG_regions
rnames = rnames + addr1_SE_rnames + addr1_GG_rnames

rt_ticker = ['KAB-APT_REALPRICE-A1000-P-IX-M',
 'KAB-APT_REALPRICE-A2000-P-IX-M',
 'KAB-APT_REALPRICE-A2001-P-IX-M',
 'KAB-APT_REALPRICE-A3000-P-IX-M',
 'KAB-APT_REALPRICE-A5000-P-IX-M',
 'KAB-APT_REALPRICE-A6000-P-IX-M',
 'KAB-APT_REALPRICE-11000-P-IX-M',
 'KAB-APT_REALPRICE-11A11-P-IX-M',
 'KAB-APT_REALPRICE-11A12-P-IX-M',
 'KAB-APT_REALPRICE-11A13-P-IX-M',
 'KAB-APT_REALPRICE-11A14-P-IX-M',
 'KAB-APT_REALPRICE-11A15-P-IX-M',
 'KAB-APT_REALPRICE-26000-P-IX-M',
 'KAB-APT_REALPRICE-27000-P-IX-M',
 'KAB-APT_REALPRICE-28000-P-IX-M',
 'KAB-APT_REALPRICE-29000-P-IX-M',
 'KAB-APT_REALPRICE-30000-P-IX-M',
 'KAB-APT_REALPRICE-31000-P-IX-M',
 'KAB-APT_REALPRICE-36000-P-IX-M',
 'KAB-APT_REALPRICE-41000-P-IX-M',
 'KAB-APT_REALPRICE-42000-P-IX-M',
 'KAB-APT_REALPRICE-43000-P-IX-M',
 'KAB-APT_REALPRICE-44000-P-IX-M',
 'KAB-APT_REALPRICE-45000-P-IX-M',
 'KAB-APT_REALPRICE-46000-P-IX-M',
 'KAB-APT_REALPRICE-47000-P-IX-M',
 'KAB-APT_REALPRICE-48000-P-IX-M',
 'KAB-APT_REALPRICE-50000-P-IX-M']

rt_alias = ['N', 'M', 'NM', 'M6', 'M5', 'P8', 'SE', 'C', 'EN', 'WN', 'WS', 'ES',
       'BS', 'DG', 'IC', 'GJ', 'DJ', 'US', 'SJ', 'GG', 'GW', 'CB', 'CN', 'JB',
       'JN', 'GB', 'GN', 'JJ']

rs_ticker = ['KAB-HOUSEPRICE-1-S-A1000-IX-M',
 'KAB-HOUSEPRICE-1-S-A2000-IX-M',
 'KAB-HOUSEPRICE-1-S-A2001-IX-M',
 'KAB-HOUSEPRICE-1-S-A3000-IX-M',
 'KAB-HOUSEPRICE-1-S-A5000-IX-M',
 'KAB-HOUSEPRICE-1-S-A9000-IX-M',
 'KAB-HOUSEPRICE-1-S-A6000-IX-M',
 'KAB-HOUSEPRICE-1-S-11000-IX-M',
 'KAB-HOUSEPRICE-1-S-41000-IX-M',
 'KAB-HOUSEPRICE-1-S-28000-IX-M',
 'KAB-HOUSEPRICE-1-S-26000-IX-M',
 'KAB-HOUSEPRICE-1-S-27000-IX-M',
 'KAB-HOUSEPRICE-1-S-29000-IX-M',
 'KAB-HOUSEPRICE-1-S-30000-IX-M',
 'KAB-HOUSEPRICE-1-S-31000-IX-M',
 'KAB-HOUSEPRICE-1-S-36000-IX-M',
 'KAB-HOUSEPRICE-1-S-42000-IX-M',
 'KAB-HOUSEPRICE-1-S-43000-IX-M',
 'KAB-HOUSEPRICE-1-S-44000-IX-M',
 'KAB-HOUSEPRICE-1-S-45000-IX-M',
 'KAB-HOUSEPRICE-1-S-46000-IX-M',
 'KAB-HOUSEPRICE-1-S-47000-IX-M',
 'KAB-HOUSEPRICE-1-S-48000-IX-M',
 'KAB-HOUSEPRICE-1-S-50000-IX-M']

rs_alias = ['N', 'M', 'NM', 'M6', 'M5', 'P9', 'P8', 'SE', 'GG', 'IC', 'BS', 'DG',
       'GJ', 'DJ', 'US', 'SJ', 'GW', 'CB', 'CN', 'JB', 'JN', 'GB', 'GN', 'JJ']

##################################  defs ########################################################################

def get_breit_excel3(req_ids, alias_nm, 
                     start_d=None, end_d=None,
                     period_trim=False):
    # Copied from:  code_and_data_sample/01_timeseries_sample/Python_데이터ID_예제코드.ipynb
    
    # API 호출
    API = "http://datahub.boknet.intra/api/v1/obs/lists"
    res = requests.post(API, data={"ids":req_ids})
    data_list = res.json()["data"][0]
    
    # API 호출로 받은 결과를 Data Frame으로 저장
    data = pd.DataFrame()
    for alias, value in zip(alias_nm, data_list):
        df = pd.DataFrame(value["observations"], dtype="float")
        df.set_index("period", inplace=True)
        df.index = pd.to_datetime(df.index)
        df.columns = [alias]
        data = df.copy() if not len(data) else data.join(df, how="outer")
    
    # 옵션에 따라 시작일, 종료일, Trim 적용
    if start_d:
        data = data[data.index >= start_d]
    if end_d:
        data = data[data.index <= end_d]
    if period_trim:
        data.index = data.index.to_period('M')
    return data


                
def get_rone_hpi():
    ''' 현재 사용하지 않음 '''
    local_path = set_local_path() # for live-notebook
    
    rt = pd.read_excel(f'{local_path}/지역별_아파트_실거래가격지수.xlsx', index_col=0, parse_dates=['period'])
    rt_ch = rt.pct_change()*100

    rs = pd.read_excel(f'{local_path}/월간_매매가격지수_아파트.xlsx', index_col=0, parse_dates=['period'])
    rs_ch = rs.pct_change()*100
    
    return rt, rt_ch, rs, rs_ch


def get_save_rone_hpi():
    today = datetime.today().strftime('%Y-%m-%d')
    rt = get_breit_excel3(rt_ticker, rt_alias, '2000-01-01', today, False)
    rt.columns.name = '지 역'
    rt_ch = rt.pct_change()*100
    
    rs = get_breit_excel3(rs_ticker, rs_alias, '2000-01-01', today, False)
    rs.columns.name = '지 역'
    rs_ch = rs.pct_change()*100
    
    try:
        local_path = set_local_path() # for live-notebook
        
        rt.to_excel(f'{local_path}/지역별_아파트_실거래가격지수.xlsx')
        rs.to_excel(f'{local_path}/월간_매매가격지수_아파트.xlsx')
    except:
        pass
    
    return rt, rt_ch, rs, rs_ch

def cal_trade_vol(file, byear=2023):
    '''  '''

    date = file[-15:-7] 
    data = pd.read_csv(file, sep=';', low_memory=False)
    df = data.loc[data['trade_year'].ge(byear) & data['cancel_yn'].isnull()].copy()   # why copy?
    del data  # why del?

    df['Date'] = pd.to_datetime(df['trade_year'].astype(str) + '-' + df['trade_month'].astype(str) + '-' + df['trade_day'].astype(str), format="%Y-%m-%d")
    df[['City', 'addr0', 'addr3']] = df.addr1.str.split(' ', n=2, expand=True) ## n=2 추가
    df = df.drop(['area_code', 'jibun', 'trade_year', 'trade_month', 'trade_day', 'upd_dt', 'cancel_date', 'cancel_yn', 'req_gbn', 'rdealer_addr', 'regist_date', 'addr3'], errors='ignore', axis=1)
    df = df.rename(columns = dict(zip(df.columns.tolist(), ['CY', 'addr1', 'addr2', 'Name', 'Fl', 'A', 'P', 'Date', 'City', 'addr0'])))
    df = df.loc[~df['P'].isnull()]  # df['Price'].isnull().sum() # 10,307 obs.
    
    df['City'] = df['City'].replace(dict(zip(a17, e17)))
    df['City'] = df['City'].replace({'강원특별자치도':'GW'}) # 20230707 부터 강원도 -> 강원특별자치도
    df['City'] = df['City'].replace({'전북특별자치도':'JB'}) # 전라북도 -> 전북특별자치도
    df['P'] = df['P'].str.replace(',', '').astype(float)
    df['P'] = df['P'] / 1e4

    df = df.dropna(axis=0)  # 1588753 / 1600479 (-0.0074%) 
    # df.groupby(df.Date.dt.year).count()
    
    df['TM'] = pd.to_datetime(df.Date.dt.year.astype(str)  + '-' + df.Date.dt.month.astype(str), format='%Y-%m')
    
    # trade volume: all and 3 areas
    trade_volume = df.groupby(['City', 'TM'])['P'].count().unstack(level=0)
    trade_volume_N = trade_volume.sum(axis=1).to_frame('N')
    trade_volume_M = trade_volume[['SE', 'GG', 'IC']].sum(axis=1).to_frame('M')
    trade_volume_NM = trade_volume[[re for re in e17 if re not in ['SE', 'GG', 'IC', 'SJ']]].sum(axis=1).to_frame('NM')
    trade_volume_ES = df.loc[df.addr0.isin(['강남구', '송파구', '서초구', '강동구'])].groupby(['TM'])['P'].count().to_frame('ES')
    trade_volume = pd.concat([trade_volume, trade_volume_N, trade_volume_M, trade_volume_NM, trade_volume_ES], axis=1)
    
    trade_volume = trade_volume.reset_index()
    trade_volume['vintage'] = pd.Timestamp(date)
    trade_volume['offset'] = trade_volume['vintage'] - trade_volume['TM']
    
    return trade_volume


def plot_by_vintage(
    plot_rnames=['N', 'M', 'NM'], 
    months=(None, None),
    vintages=(None, None),
    v_interval=1,):
    
    matplotlib.rc('xtick', labelsize=16)
    matplotlib.rc('ytick', labelsize=16)
    rcParams.update({'figure.autolayout': True})
    plt.tight_layout()

    fig, ax = plt.subplots(1, len(plot_rnames), figsize=(24, 7));

    # [vintage_date_from, vintage_date_to] 범위내에서 IB2s가 계산된 날을 plot_vintages 리스트에 저장
    ib2s_dates = sorted(fs_anon.glob(f'{s3_newtech_path}/parquets/IB2s_*'))
    ib2s_dates = pd.to_datetime([date.split("parquets/IB2s_")[-1][:8] for date in ib2s_dates])

    ym = sorted(list(set([date.strftime("%Y-%m") for date in ib2s_dates])))
    latest_d = max([date.day for date in ib2s_dates if date.strftime("%Y-%m") == ym[-1]])

    if months[0] is None:
        if latest_d >= 15:                  # 거래일수가 15일 이상인 월까지 지수상승률 표시
            e_month = pd.Timestamp(ym[-1])
            b_month = pd.Timestamp(ym[-9])  # 과거 9개월에 대해 지수상승률 표시
        else:
            e_month = pd.Timestamp(ym[-2])
            b_month = pd.Timestamp(ym[-10]) # 과거 9개월에 대해 지수상승률 표시
    else:
        b_month = pd.Timestamp(months[0])   # 사용자가 특정한 월부터 지수상승률 표시
        e_month = pd.Timestamp(months[1])   # 사용자가 특정한 월까지 지수상승률 표시
    
    
    if vintages[1] is None:
        v_from = ib2s_dates[-7]             # 최근 거래데이터 입수일 이전 7일부터
        v_to = ib2s_dates[-1]               # 최근 거래데이터 입수일까지 계산된 지수상승률 표시

    else:
        v_from = vintages[0]                # 사용자가 특정한 기간 동안 입수된 데이터를 이용하여
        v_to = vintages[1]                  # 지수상승률 표시
        
    plot_vintages = [date for date in ib2s_dates if date in pd.date_range(v_to, v_from, freq=f'-{v_interval}D')]

    # plot_vintages 리스트에 속하는 날에 대해 저장된 IB2s을 모두 불러와 IBs와 IBsCh 생성
    IBs, IBsCh = read_IBs_update(plot_vintages)
    
    # Save IBs for download
    try:
        local_path = set_local_path() # for live-notebook
        IBs.to_excel(f'{local_path}/IBs.xlsx')
        IBsCh.to_excel(f'{local_path}/IBsCh.xlsx')
    except:
        pass
    
    # import trade volume by region
    trade_volume = pd.read_parquet("s3://"+sorted(fs_anon.glob(f'{s3_newtech_path}/parquets/trade_volume_*'))[-1], storage_options={"anon": True})
    
    ### 확인필요!!!!!!!!
    ### set IBsCh to np.nan for month and region where trade volume is less than 200
    ###for re in IBsCh.columns.get_level_values(0):
    ###    for mo in IBsCh.index:
    ###        if (trade_volume < 200).loc[mo, re]:
    ###            IBsCh.loc[mo, idx[re, :]] = np.nan
    
    # 부동산원 실거래가지수(rt), 월간매매가격지수(rs)
    rt, rt_ch, rs, rs_ch = get_save_rone_hpi()
    # rt, rt_ch, rs, rs_ch = get_rone_hpi()  # 7.15, 실거래가 5월분 업데이트 완료
    
    alphas = [(i+1)/(len(plot_vintages)+1) for i in range(len(plot_vintages))]  # 색의 흐릿한 정도 설정. 최근 것일 수록 짙게.
    y_max = 0
    y_min = 0
    
    for j, rname in enumerate(plot_rnames):    # 지역명
        v = None
        for i, v in enumerate(plot_vintages):  # 빈티지
            breit = IBsCh[rname].loc[b_month:e_month, v].replace({0.0:np.nan});
#             breitna = breit.fillna(method='ffill').copy();
#             ax[j].scatter(list(breitna.index.strftime('%-y.%-m')), breitna.values, label=v, alpha=0);  # np.nan 값을 투명도 0으로 출력
            ax[j].scatter(list(breit.index.strftime('%-y.%-m')), breit.values, label=v, s=200, c='Orange', alpha=alphas[i]);  # alpha는 색의 흐릿한 정도
            title = NAMETABLE[rname]
            if j == 1:
                title = " * The legend represents the dates when data was collected, and the index for the latest month is shown if there are a minimum of 15 trading days in that month.\n\n\n" + title
            ax[j].set_title(title, fontsize=16);
        rei = rt_ch[rname].loc[b_month:e_month]
        ax[j].scatter(list(rei.index.strftime('%-y.%-m')), rei.values, marker='x', label=v, s=200, c='b', alpha=.7);  # 지역별 아파트 실거래가 지수 (파란색)
        #ax[j].axhline(y=0, color='k', linestyle=':');  # 지역별 아파트 실거래가 지수 (파란색)
        y_max = max(y_max, ax[j].get_ylim()[1])
        y_min = min(y_min, ax[j].get_ylim()[0])
        
    for j, rname in enumerate(plot_rnames):    # 지역명
        ax[j].set_ylim(y_min - 0.1, y_max + 0.1)
    
    # legends
    legend_texts = [date.strftime('%-m.%-d') for date in plot_vintages] + ['REI index']
    fig.legend(legend_texts, ncol=10, fontsize=12, loc='lower left', fancybox=True, shadow=True, bbox_to_anchor=(0.03, -0.12))   # parametrize ncol
    
    fig.suptitle('BReiT/CS (percent change, mom)', fontsize=25)
    
    for j, rname in enumerate(plot_rnames):    # 지역명
        ax[j].axhline(y=0, color='k', linestyle=':');  # 지역별 아파트 실거래가 지수 (파란색)            
        
    print_volume = lambda x: f"{x:.1f}" if x > 1 else f"{x:.2f}"
    
    local_path = '/home/work/modelhub/housing'
    fig.savefig(f"{local_path}/plot/housing_latest_d.png", dpi='figure', bbox_inches='tight')
    
    fig, ax = plt.subplots(1, len(plot_rnames), figsize=(24, 5))
    
    vintage_trade_volume = pd.read_pickle(f'{s3_newtech_path}/parquets/vintage_trade_volume.pkl')
    vintage_trade_volume['diff'] = vintage_trade_volume['offset'] - (plot_vintages[-1] - e_month)
    vintage_trade_volume3 = vintage_trade_volume.loc[vintage_trade_volume.TM.between(e_month - DateOffset(months=3), e_month - DateOffset(months=1))]
    
    for j, rname in enumerate(plot_rnames):    # 지역명
        trades = trade_volume.loc[b_month:e_month, rname] / 1e3
        trades.index.names = [None]
        #trades.columns.names = ['Trade Volume (1,000)']
        trades.index = trades.index.strftime('%-y.%-m')
        trades.plot.bar(ax=ax[j], rot=0, ylim=(0, trades.max()*1.2), alpha=0.5)
        last_month = trades.copy() * np.nan
        last_three = trades.copy() * np.nan
        df = vintage_trade_volume3.set_index(['TM', 'diff'])[rname].unstack(level=0).fillna(method='bfill')
        df.index = df.index.days
        df = df.loc[df.index==0]
        last_month.iloc[-2:-1] = df.iloc[0, -1] / 1e3
        last_three.iloc[-4:-1] = df.mean(axis=1).iloc[0] / 1e3
        last_month.plot(ax=ax[j], lw=1.5, color='maroon', linestyle='-', marker='_', markerfacecolor='maroon', markersize=30, mew=1.5, ylim=(0, trades.max()*1.2), alpha=.5)
        last_three.plot(ax=ax[j], lw=1.5, color='maroon', linestyle='-', marker='_', markerfacecolor='maroon', markersize=30, mew=1.5, ylim=(0, trades.max()*1.2), alpha=.5)
        if j == 1:
            title = " * Number of housing transactions reported for each month up to today, and\nHorizontal lines represent the average number of transactions in the past 3 months and 1 month reported for the same period as the latest month.\n\n\n"
            ax[j].set_title(title, fontsize=16)
        ax[j].text(len(last_three)-3-0.25, last_three.iloc[-3]+trades.max()*0.02, print_volume(last_three.iloc[-3]), fontsize=16, color='maroon')
        ax[j].text(len(last_month)-2-0.25, last_month.iloc[-2]+trades.max()*0.02, print_volume(last_month.iloc[-2]), fontsize=16, color='maroon')
        ax[j].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax[j].set_ylabel('Trade Volume (1,000)', fontsize=12)
        for i, v in enumerate(trades.values):
            ax[j].text(i-0.25, v+trades.max()*0.02, print_volume(v), fontsize=16)
          
    fig.suptitle('Trade Volume', fontsize=25)
    
    local_path = '/home/work/modelhub/housing'
    fig.savefig(f"{local_path}/plot/housing_latest_v.png", dpi='figure', bbox_inches='tight')

def prep_td(file, byear=2007):
    '''  '''

    date = file[-15:-7] 
    data = pd.read_csv(file, sep=';', low_memory=False)
    df = data.loc[data.trade_year.ge(byear) & data.cancel_yn.isnull()].copy()   # why copy?
    del data  # why del?

    df['Date'] = pd.to_datetime(df['trade_year'].astype(str) + '-' + df['trade_month'].astype(str) + '-' + df['trade_day'].astype(str), format="%Y-%m-%d")
    df[['City', 'addr0', 'addr3']] = df.addr1.str.split(' ', n=2, expand=True) # 3개 이상으로 split 되는 case 발생하여 n=2로 설정
    
    df = df.drop(['area_code', 'jibun', 'trade_year', 'trade_month', 'trade_day', 'upd_dt', 'cancel_date', 'cancel_yn', 'req_gbn', 'rdealer_addr', 'regist_date', 'addr3'], errors='ignore', axis=1)

    df = df.rename(columns = dict(zip(df.columns.tolist(), ['CY', 'addr1', 'addr2', 'Name', 'Fl', 'A', 'P', 'Date', 'City', 'addr0'])))

    df = df.loc[~df['P'].isnull()]  # df['Price'].isnull().sum() # 10,307 obs.
    df = df.loc[~df['Fl'].isnull()]
    
    df['City'] = df['City'].replace(dict(zip(a17, e17)))
    df['City'] = df['City'].replace({'강원특별자치도':'GW'}) # 20230707 부터 강원도 -> 강원특별자치도
    df['City'] = df['City'].replace({'전북특별자치도':'JB'}) # 전라북도 -> 전북특별자치도
    df['CY'] = df['CY'].astype(int)
    df['Fl'] = df['Fl'].astype(int)
    df['P'] = df['P'].str.replace(',', '').astype(float)
    df['P'] = df['P'] / 1e4

    df = df.dropna(axis=0)  # 1588753 / 1600479 (-0.0074%) 
    # df.groupby(df.Date.dt.year).count()
   
    
    df['TM'] = pd.to_datetime(df.Date.dt.year.astype(str)  + '-' + df.Date.dt.month.astype(str), format='%Y-%m')
    
    # Assign code for floor group
    df['F'] = 1
    df.loc[df.Fl.between(2, 5), 'F'] = 2
    df.loc[df.Fl.between(6, 10), 'F'] = 3
    df.loc[df.Fl.between(11, 15), 'F'] = 4
    df.loc[df.Fl.between(16, 20), 'F'] = 5
    df.loc[df.Fl.between(21, 25), 'F'] = 6
    df.loc[df.Fl.between(26, 30), 'F'] = 7
    df.loc[df.Fl.ge(31), 'F'] = 8

    # df['F'] = 1
    # df.loc[df.Fl.between(2, 10), 'F'] = 2
    # df.loc[df.Fl.between(11, 20), 'F'] = 3
    # df.loc[df.Fl.ge(21), 'F'] = 4    
    
    # trade volume: all and 3 areas
    trade_volume = df.groupby(['City', 'TM'])['P'].count().unstack(level=0)
    trade_volume_N = trade_volume.sum(axis=1).to_frame('N')
    trade_volume_M = trade_volume[['SE', 'GG', 'IC']].sum(axis=1).to_frame('M')
    trade_volume_NM = trade_volume[[re for re in e17 if re not in ['SE', 'GG', 'IC', 'SJ']]].sum(axis=1).to_frame('NM')
    trade_volume_ES = df.loc[df.addr0.isin(['강남구', '송파구', '서초구', '강동구'])].groupby(['TM'])['P'].count().to_frame('ES')
    trade_volume = pd.concat([trade_volume, trade_volume_N, trade_volume_M, trade_volume_NM, trade_volume_ES], axis=1)
    
    try:
        path = set_store_path('parquets')
        df.to_parquet(f'{path}/parquets/mP_{date}.parquet.gzip', compression='gzip')
        trade_volume.to_parquet(f'{path}/parquets/trade_volume_{date}.parquet.gzip', compression='gzip')
    except:
        print(f'The preprocessed data (mP_{date}) is returned, but not saved. \nYou can import the files with pd.read_parquet(\'parquets/mP_yyyymmdd.parquet.gzip\')')
    
    return df


def gen_yXT(df):
    
    mP = df.groupby(['City', 'addr0', 'addr1', 'addr2', 'Name', 'CY', 'A', 'F', 'TM'], as_index=False)['P'].mean()
    grouped_mean = mP.groupby(['City', 'addr0', 'addr1', 'addr2', 'Name', 'CY', 'A', 'F'], as_index=False)  # changed
    mP['nTM'] = grouped_mean['TM'].shift(-1)
    mP['nP'] = grouped_mean['P'].shift(-1)
    mP = mP.dropna(axis=0)

    mP.index = range(len(mP))
    mP = mP.reset_index()

    mP.loc[:, 'v'] = (mP.loc[:, 'nP']/mP.loc[:, 'P'] - 1)*100
    HID = mP.copy()

    T = TT = mP.set_index(['index', 'City', 'addr0', 'addr1', 'addr2', 'Name', 'CY', 'A', 'F'])[['TM', 'nTM']].copy()

    mP0 = mP[['index', 'City', 'addr0', 'addr1', 'addr2', 'Name', 'CY', 'A', 'F', 'TM', 'P']]
    mP1 = mP[['index', 'City', 'addr0', 'addr1', 'addr2', 'Name', 'CY', 'A', 'F', 'nTM', 'nP']]
    mP0 = mP0.set_index(['index', 'City', 'addr0', 'addr1', 'addr2', 'Name', 'CY', 'A', 'F', 'TM']).unstack()['P']
    mP1 = mP1.set_index(['index', 'City', 'addr0', 'addr1', 'addr2', 'Name', 'CY', 'A', 'F', 'nTM']).unstack()['nP']
    mP1.columns.names = ['TM']

    periods = mP0.columns.union(mP1.columns)
    mP0 = mP0.reindex(columns=periods)
    mP1 = mP1.reindex(columns=periods)
    mP = mP0.fillna(0) + mP1.fillna(0)
    mP = mP.replace(0, np.nan)

    yX = mP.copy()
    del mP

    y = yX.iloc[:, [0]].copy()
    X = yX.iloc[:, 1:].copy()

    y.columns = ['ref']
    nan_y = y.loc[y.ref.isnull()].index

    multiply_minus_one = TT.loc[nan_y, ['TM']].set_index('TM', append=True)
    tmpX = X.stack().to_frame('P')
    tmpX.loc[multiply_minus_one.index, 'P'] = -1 * tmpX.loc[multiply_minus_one.index, 'P']
    X = tmpX.unstack()['P']

    T['C'] = 1
    T['T'] = (T['nTM'] - T['TM']).dt.days
    T = T.drop(['TM', 'nTM'], axis=1)
    
    return y, X, T, HID


def get_ibs(iy, iX, iZ, iT):

    periods = iX.columns
    
    try:
        beta = np.linalg.inv(iZ.T.dot(iX)).dot(iZ.T).dot(iy)
    except:
        iX = iX.iloc[:, :-1]
        iZ = iZ.iloc[:, :-1]
        periods = iX.columns
        beta = np.linalg.inv(iZ.T.dot(iX)).dot(iZ.T).dot(iy)

    beta = pd.Series(beta.squeeze(), index = periods)
    ib = 1/beta

    # FG2SLS
    e = iy['ref'] - iX.dot(beta)
    e2 = e ** 2
    eta = np.linalg.inv(iT.T.dot(iT)).dot(iT.T).dot(e2)
    e2hat = iT.dot(eta)

    w1 = e2hat**(-1/2)
    w1beta = np.linalg.inv(iZ.T.dot(iX.mul(w1, axis=0))).dot(iZ.T).dot(iy.mul(w1, axis=0))
    w1beta = pd.Series(w1beta.squeeze(), index=periods)
    ib1 = 1/w1beta

    w2 = 1/e2hat
    w2beta = np.linalg.inv(iZ.T.dot(iX.mul(w2, axis=0))).dot(iZ.T).dot(iy.mul(w2, axis=0))
    w2beta = pd.Series(w2beta.squeeze(), index=periods)
    ib2 = 1/w2beta
    
    return ib, ib1, ib2



def gen_vintage_IBs_update(date, y, X, T, HID, rnames, regions, thres=30, interval=1825, syear=2011):
    """
    date = '20220723' 
    """
    tic = time.time()
    IBs = pd.DataFrame()
    IB1s = pd.DataFrame()
    IB2s = pd.DataFrame()

    HID['T'] = (HID['nTM'] - HID['TM']).dt.days

    y = y.fillna(0)
    X = X.fillna(0)
    Z = np.sign(X)

    ibs = pd.DataFrame(index = X.columns)
    ib1s = pd.DataFrame(index = X.columns)
    ib2s = pd.DataFrame(index = X.columns)

    for rname, region in zip(rnames, regions):
        if (rname != 'ES') and (rname[:3] != 'SE_') and (rname[:3] != 'GG_'):
            hid = HID.loc[HID['City'].isin(region) &  ~(~HID.v.between(-thres, thres) & HID['T'].le(interval)), 'index'].values

        elif rname == 'ES':
            hid = HID.loc[HID['City'].isin(region) & HID['addr0'].isin(['강남구', '송파구', '서초구', '강동구']) 
                            &  ~(~HID.v.between(-thres, thres) & HID['T'].le(interval)), 'index'].values

        elif rname[:3] == 'SE_':
            hid = HID.loc[HID['addr1'].eq(region) &  ~(~HID.v.between(-thres, thres) & HID['T'].le(interval)), 'index'].values

        elif rname[:3] == 'GG_':
            hid = HID.loc[HID['addr1'].str.contains(region) &  ~(~HID.v.between(-thres, thres) & HID['T'].le(interval)), 'index'].values

        iy = y.loc[idx[hid], :]
        iX = X.loc[idx[hid], :]
        iZ = Z.loc[idx[hid], :]
        iT = T.loc[idx[hid], :]

        ib, ib1, ib2 = get_ibs(iy, iX, iZ, iT)

        ibs = pd.concat([ibs, ib.to_frame(rname)], axis=1)
        ib1s = pd.concat([ib1s, ib1.to_frame(rname)], axis=1)
        ib2s = pd.concat([ib2s, ib2.to_frame(rname)], axis=1)

    IBs = pd.concat([IBs, pd.concat([ibs], axis=0, keys=[date])], axis=0)
    IB1s = pd.concat([IB1s, pd.concat([ib1s], axis=0, keys=[date])], axis=0)
    IB2s = pd.concat([IB2s, pd.concat([ib2s], axis=0, keys=[date])], axis=0)
    
    IBs = IBs.unstack()
    IBs.index = pd.to_datetime(IBs.index)
    IBs.index.names = ['vintage']
    IBs = IBs.stack()
    IBs = IBs.reset_index('vintage')
    
    IB1s = IB1s.unstack()
    IB1s.index = pd.to_datetime(IB1s.index)
    IB1s.index.names = ['vintage']
    IB1s = IB1s.stack()
    IB1s = IB1s.reset_index('vintage')
    
    IB2s = IB2s.unstack()
    IB2s.index = pd.to_datetime(IB2s.index)
    IB2s.index.names = ['vintage']
    IB2s = IB2s.stack()
    IB2s = IB2s.reset_index('vintage')
    
    try:
        path = set_store_path('parquets')
        IBs.to_parquet(f"{path}/parquets/IBs_{date}.parquet.gzip", compression='gzip')
        IB1s.to_parquet(f"{path}/parquets/IB1s_{date}.parquet.gzip", compression='gzip')
        IB2s.to_parquet(f"{path}/parquets/IB2s_{date}.parquet.gzip", compression='gzip')
    except:
        print(f'The BReiT/CS indices (IBs/IB1s/IB2s_{date} and others) are returned, but not saved. \nYou can import the files with pd.read_parquet(\'parquets/IB2s_yyyymmdd.parquet.gzip\')')

    del ibs, ib1s, ib2s
    
    print(f"\nIBs, IB1s and IB2s (a.k.a. C-S) for {date} are saved in parquets folder.")

    return IBs, IB1s, IB2s
    
    

def read_IBs_update(vintages):
    """pd.date_range"""
    IB = pd.DataFrame()
    
    for date in vintages:
        try:
            ib = pd.read_parquet(f"{s3_newtech_path}/parquets/IB2s_{date.strftime('%Y%m%d')}.parquet.gzip", storage_options={"anon": True})
            ib = ib.set_index('vintage', append=True)
            ib = ib.unstack(level=1)
            IB = pd.concat([IB, ib], axis=1)
        except:
            continue

    IBCh = IB.pct_change()*100
    
    return IB, IBCh


def set_local_path():
    current_dir = os.getcwd()
    if os.access(current_dir, os.W_OK):
        local_path = current_dir
    else:
        local_path = os.getenv('HOME')
        
    return local_path


def upload_to_s3(lpath, rpath):    
    if os.environ.get('FSSPEC_S3_KEY'):
        _ = fs.put(lpath, rpath)
        

def set_store_path(sub_path=''):
    path = ""
    if os.environ.get('FSSPEC_S3_KEY'):
        path = s3_newtech_path
    else: # for live-notebook
        path = set_local_path()
        
        full_path = os.path.join(path, sub_path)
        
        if not os.path.isdir(full_path):
            os.makedirs(full_path)

    return path

def create_zip_file():
    local_path = set_local_path()
    
    code_files = ["realtime_hpi.ipynb", f"{local_path}/bok_da/project/housing/house.py"]
    data_files = [f"{local_path}/월간_매매가격지수_아파트.xlsx", f"{local_path}/지역별_아파트_실거래가격지수.xlsx",
                  f"{local_path}/IBs.xlsx", f"{local_path}/IBsCh.xlsx"]
    
    err_files = []
        
    zip_file = zipfile.ZipFile(f"{local_path}/output/housing_code_result.zip", "w")  # "w": write 모드
    for file in code_files+data_files:
        if os.path.isfile(file):
            zip_file.write(file, os.path.basename(file), compress_type=zipfile.ZIP_DEFLATED)
        else:
            err_files.append(file.split("/")[-1])
            
    zip_file.close()
    
    #upload_to_s3(f"{local_path}/output/housing_code_result.zip", f"{s3_newtech_path}/housing_code_result.zip")
    
        
    return err_files
