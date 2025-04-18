class EcosAPI:
    '''ECOS data management'''

    cols_to_disp = ['STAT_CODE',
                    'GRP_CODE', 'GRP_NAME', 'ITEM_CODE', 'ITEM_NAME', 'CYCLE',
                    'DATA_CNT', 'UNIT_NAME', 'START_TIME', 'END_TIME']

    def __init__(self, key = '', force = False):
        import pandas as pd

        self.key = key
        self.apisite = 'http://ecos.bok.or.kr/api' # https wouldn't work
        self.rawfile = 'ECOS-Tables.xlsx'
        self.tblfile = 'ECOS-Tables.pkl'
        self.statcode = None
        self.items = None
        self._prepare(force = force)
        # https://pandas.pydata.org/docs/reference/api/pandas.read_pickle.html
        self.tbl = pd.read_pickle(self.tblfile)

    def _prepare(self, raw = None, clean = None, force = False):
        from bok_da.utils import tools
        import numpy as np
        import pandas as pd

        if raw == None:
            raw = self.rawfile
        else:
            self.rawfile = raw

        if clean == None:
            clean = self.tblfile
        else:
            self.tblfile = clean

        # just return if unnecessary
        if not force and tools.is_newer_than(clean, raw): return

        print('* Rebuild table database from', raw)

        tbl = pd.read_excel(raw)
        names = tbl.columns.tolist()
        for i in reversed(range(5)):
            lvl = tbl[names[i]]
            sub = tbl[names[i+1]]
            is_lvl_na = pd.isna(lvl)
            out = lvl.ffill()
            out[is_lvl_na & pd.isna(sub)] = np.nan
            tbl[names[i]] = out

        tbl = tbl.replace(np.nan, '')
        names[:6] = [f'Cat{i+1}' for i in range(6)]
        names[6] = 'ID'
        tbl.columns = names

        # Serialize
        tbl.to_pickle(self.tblfile)
        print('* Table database saved as', self.tblfile)

    def search_tbl(self, pattern, cats = [1,2,3,4,5,6], regex = True):
        from bok_da.utils import searchdf
        import re
        import pandas as pd

        #cols = [x for x in self.tbl.columns if x.startswith('Cat')]
        if isinstance(cats,int): cats = [cats]
        cols = [f'Cat{x}' for x in cats]
        return searchdf.search(self.tbl, pattern, cols, invert=False, regex = regex)

    def search_item(self, pattern, cols = ['ITEM_NAME'], combine = 'or',
                    regex = True, simplify = True):
        from bok_da.utils import searchdf
        import re
        import warnings
        import pandas as pd

        df = self.items
        if simplify: df = df[self.cols_to_disp]

        return searchdf.search(df, pattern, cols, invert = False,
                               combine = combine, regex = regex)

    def download_items(self, stat_code, start_no=1, end_no=100000,
                       simplify = False):
        import pandas as pd
        import requests
        import json

        file_type = 'json'
        lang_type = 'kr'

        if self.key == '':
            raise ValueError('ECOS key is necessary.')

        # https wouldn't work
        svc = 'StatisticItemList'
        url = '/'.join([self.apisite, svc, self.key, file_type, lang_type,
                        f'{start_no}', f'{end_no}', stat_code])
        response = requests.get(url, verify = False)
        json = response.json()[svc]
        count = json['list_total_count']
        data = pd.DataFrame(json['row'])
        if simplify: data = data[self.cols_to_disp].copy()
        self.items = data
        self.statcode = stat_code
        return count

    def download_data(self, stat_code, items, start_date, end_date,
                      cycle = None, clean = True):
        import pandas as pd
        import requests
        import json

        svc = 'StatisticSearch'
        file_type = 'json'
        lang_type = 'kr'

        if self.key == '': raise ValueError('ECOS key is necessary.')

        if not isinstance(items, list): items = [items]
        if isinstance(start_date, int): start_date = f'{start_date}'
        if isinstance(end_date, int): end_date = f'{end_date}'

        if cycle==None:
            if 'Q' in start_date: cycle = 'Q'
            elif len(start_date)==4: cycle = 'A'
            elif len(start_date)==6: cycle = 'M'
            elif len(start_date)==8: cycle = 'D'
            else: raise ValueError('Cannot determine CYCLE.')

        def compose_url(start_no, end_no):
            return '/'.join([self.apisite, svc, self.key, file_type,
                             lang_type,f'{start_no}', f'{end_no}',
                             stat_code, cycle, start_date, end_date] +
                            items)

        url = compose_url(1, 1)
        response = requests.get(url, verify = False)
        count = response.json()[svc]['list_total_count']

        url = compose_url(1, count)
        response = requests.get(url, verify = False)
        json = response.json()[svc]
        count = json['list_total_count']
        data = pd.DataFrame(json['row'])
        data['CYCLE'] = cycle
        if clean: return clean_data(data)
        else: return (count,data)

def clean_data(data):
    import pandas as pd

    df = data[['TIME', 'DATA_VALUE']].copy()
    df.columns = ['time', 'value']
    df['value'] = df.value.astype(float)

    cycle = data.loc[0,'CYCLE']

    if cycle=='D':
        df['date'] = pd.to_datetime(df['time'], format='%Y%m%d')
    elif cycle=='M':
        df['date'] = pd.to_datetime(df['time'], format='%Y%m')
        df['year'] = df.date.dt.year.copy()
        df['month'] = df.date.dt.month.copy()
    elif cycle=='A':
        df['year'] = df.time.astype(int)
    elif cycle=='Q':
        df['date'] = pd.PeriodIndex(df.time,freq='Q').to_timestamp()
        df['year'] = df.date.dt.year.copy()
        df['quarter'] = df.date.dt.quarter.copy()

    # reindex
    v = 'value'
    cols = df.columns.tolist()
    cols.remove(v)
    cols.append(v)
    df = df.reindex(cols, axis=1)

    row0 = data.iloc[0]
    item_codes = [x for x in data.columns.tolist() if x.startswith('ITEM_CODE')]
    item_names = [x for x in data.columns.tolist() if x.startswith('ITEM_NAME')]
    info = {'code': row0['STAT_CODE'],
            'name': row0['STAT_NAME'].strip(),
            'itemcodes': row0[item_codes].tolist(),
            'itemnames': row0[item_names].tolist(),
            'cycle': row0['CYCLE'],
            'unit': row0['UNIT_NAME'].strip(),
            'wgt': row0['WGT']}

    return (info,df)
