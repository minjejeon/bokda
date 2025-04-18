class DartAPI:
    """API for Open DART"""

    def __init__(self, key = '', force = False,
                 corp_code_file = 'CORPCODE.xml',
                 api_info_file = 'DartAPI.xlsx'):
        import pandas as pd
        import os, json
        from bod_da.utils import tools

        self.key = key
        self.apisite = 'https://opendart.fss.or.kr/api'
        self.corp_src = corp_code_file
        fbase,_ = os.path.splitext(self.corp_src)
        self.corp_pkl = fbase + '.pkl'
        self.api_src = api_info_file
        fbase,_ = os.path.splitext(self.api_src)
        self.company_cache = 'cache_company'
        self.annual_cache = 'cache_annual'
        self.other_cache = 'cache_others'
        self.api_pkl = fbase + '.pkl'
        self.svc_pkl = fbase + '-svc.pkl'
        self.cod_pkl = fbase + '-codes.pkl'
        self.vocab_json = fbase + '-vocab.json'
        self._prepare_corp_info(force = force)
        self._prepare_api_info(force = force)
        self.corpcode = pd.read_pickle(self.corp_pkl)
        self.info = pd.read_pickle(self.api_pkl)
        self.svcs = pd.read_pickle(self.svc_pkl)
        # https://www.geeksforgeeks.org/convert-json-to-dictionary-in-python/
        # https://hayjo.tistory.com/75
        self.codes = pd.read_pickle(self.cod_pkl)
        self.vocab = tools.read_from_json(self.vocab_json)

    def _prepare_corp_info(self, force =False):
        from bok_da.utils import tools
        import pandas as pd
        import numpy as np

        xml = self.corp_src
        pkl = self.corp_pkl

        if not force and tools.is_newer_than(pkl, xml): return

        print(f'* {xml} -> {self.corp_pkl}')

        df = pd.read_xml(self.corp_src)
        for v in  df.columns:
            if isinstance(df[v][0],str): df[v] = df[v].str.strip()
        df['corp_code'] = ["%08d" % x for x in df['corp_code']]
        df.to_pickle(self.corp_pkl)

    def _prepare_api_info(self, force = False):
        from bok_da.utils import tools
        import numpy as np
        import pandas as pd
        import json

        xlsx = self.api_src
        pkl = self.api_pkl

        if not force and tools.is_newer_than(pkl, xlsx): return

        print(f'* {xlsx} -> {self.api_pkl}, {self.svc_pkl}, {self.cod_pkl}, {self.vocab_json}')

        df = pd.read_excel(xlsx)
        for v in df.columns[:3]: df[v] = df[v].ffill()
        vocab = {}
        for i in range(df.shape[0]):
            if pd.isna(df.loc[i,'Meaning']):
                df.loc[i,'Meaning'] = vocab[df.loc[i,'Arguments']]
            else:
                vocab[df.loc[i,'Arguments']] = df.loc[i,'Meaning']
        # https://blogboard.io/blog/knowledge/how-to-write-a-list-to-a-file-in-python/
        tools.write_to_json(vocab, self.vocab_json)

        df['Required'] = df.Required=='Y'
        df = df.replace(np.nan, '')

        df.to_pickle(self.api_pkl)
        ## print('* API Info database saved as', self.api_pkl)

        # per service
        df2 = df[~df.duplicated('URL')].iloc[:,:3].copy()
        df2['Arguments'] = ''
        df2 = df2.reset_index(drop = True)
        for i in df2.index:
            this_url = df2.loc[i,'URL']
            args = df.loc[df['URL']==this_url]['Arguments'].tolist()
            args = ', '.join(args)
            df2.loc[i,'Arguments'] = args

        df2.to_pickle(self.svc_pkl)
        ## print('* Service Info database saved as', self.svc_pkl)

        # code book
        df3 = pd.read_excel(xlsx, sheet_name = 1)
        for v in df3.columns: df3[v] = df3[v].ffill()
        df3.to_pickle(self.cod_pkl)

    def mkurl(self, service, **data):
        if not service.endswith('.xml'): service = f'{service}.json'
        s = f'{self.apisite}/{service}?crtfc_key={self.key}'
        for key,value in data.items():
            s = f'{s}&{key}={value}'
        return s

    def doclist(self, corp_code, bgn_de = None, end_de = None,
                last_only = False):
        """공시검색"""
        import requests, json
        import datetime
        import pandas as pd

        svc = 'list'

        if end_de is None:
            now = datetime.datetime.now()
            end_de = now.strftime('%Y%m%d')
        if bgn_de is None:
            bgn_de = end_de
        last_reprt_at = 'Y' if last_only else 'N'

        url = self.mkurl(svc, corp_code = corp_code, bgn_de = bgn_de,
                         end_de = end_de, last_reprt_at = last_reprt_at,
                         page_no = 1, page_count = 1)
        json = requests.get(url).json()
        npage = int(json['total_count'])
        ans = None
        for i in range(npage//100 + 1):
            url = self.mkurl(svc, corp_code = corp_code, bgn_de = bgn_de,
                             end_de = end_de, last_reprt_at = last_reprt_at,
                             page_no = i, page_count = 100)
            json = requests.get(url).json()
            a = pd.DataFrame(json['list'])
            ans = a if ans is None else pd.concat([ans, a], axis=0)
        return ans

    def _firminfo(self, corp_code, force = False):
        """기업개황"""
        from bok_da.utils import tools
        import requests, json, os

        cache = os.path.join(tools.safe_mkdir(self.company_cache),
                             corp_code+'.json')
        if force or not os.path.isfile(cache):
            print(f'* Download firm info for {corp_code}')
            url = self.mkurl('company', corp_code = corp_code)
            resp = requests.get(url)
            x = resp.json()
            tools.write_to_json(x,cache)
            return x
        else:
            return tools.read_from_json(cache)

    def firminfo(self, corpcodes, force = False, byrow = True):
        """여러 기업들의 기업개황을 받아 데이터프레임 리턴"""
        import pandas as pd
        if isinstance(corpcodes,list):
            out = {}
            for corpcode in corpcodes:
                out[corpcode] = self._firminfo(corpcode, force = force)
            orient = 'index' if byrow else 'columns'
            return pd.DataFrame.from_dict(out, orient=orient)
        else:
            return self._firminfo(corpcodes, force = force)

    def url_for_origdoc(self, rcept_no):
        """공시서류원본파일 URL"""
        url = self.mkurl('document.xml', rcept_no = rcept_no)
        return url

    def _update_cache(self, url, cache_file, cache_dir, force = False):
        import requests, json, os
        from bok_da.utils import tools
        cache = os.path.join(tools.safe_mkdir(cache_dir), cache_file)
        if force or not os.path.isfile(cache):
            print(f'* Download and save to {cache}')
            resp = requests.get(url)
            a = resp.json()
            tools.write_to_json(a, cache)
            if a['status'] != '000':
                print(f"*** Warning: the status is {a['status']}: {a['message']}")
        else:
            a = tools.read_from_json(cache)
        return a

    def _get_annual(self, svc, corp_code, year, reprt_code, force = False):
        """1개 연도 연차별 데이터 받기"""
        url = self.mkurl(svc, corp_code = corp_code,
                         bsns_year = year, reprt_code = reprt_code)
        cache = f'{svc}-{corp_code}-{year}-{reprt_code}.json'
        return self._update_cache(url, cache, self.annual_cache, force = force)

    def get_annual(self, svc, corp_code, years, reprt_code = '11011', force = False):
        """사업보고서 주요정보 데이터 받기"""
        import pandas as pd

        def _get_one_year(year):
            a = self._get_annual(svc,
                                 corp_code,
                                 year,
                                 reprt_code,
                                 force = force)
            x = pd.DataFrame(a['list'])
            x['year'] = year
            return x

        if isinstance(years,int) or isinstance(years,str):
            return _get_one_year(years)
        else:
            z = None
            for year in years:
                x = _get_one_year(year)
                z = x if z is None else pd.concat([z,x], axis=0)
            return z.reset_index(drop = True)

    def _get_annual_fs(self, corp_code, year, reprt_code, cfs = False, force = False):
        svc = 'fnlttSinglAcntAll'
        fs_div = "CFS" if cfs else "OFS"
        url = self.mkurl(svc, corp_code = corp_code,
                         bsns_year = year, reprt_code = reprt_code,
                         fs_div = fs_div)
        cache = f'{svc}-{corp_code}-{year}-{reprt_code}-{fs_div}.json'
        return self._update_cache(url, cache, self.annual_cache, force = force)

    def get_annual_fs(self, corp_code, years, reprt_code = '11011', cfs = False, force = False):
        """단일회사 전체 재무제표"""
        import pandas as pd

        def _get_one_year(year):
            a = self._get_annual_fs(corp_code,
                                    year,
                                    reprt_code,
                                    cfs = cfs,
                                    force = force)
            x = pd.DataFrame(a['list'])
            x['year'] = year
            return x

        if isinstance(years,int) or isinstance(years,str):
            return _get_one_year(years)
        else:
            z = None
            for year in years:
                x = _get_one_year(year)
                z = x if z is None else pd.concat([z,x], axis=0)
            return z.reset_index(drop = True)

    def xbrl_taxonomy(self, sj_div, force = False):
        """XBRL택사노미재무제표양식"""

        import pandas as pd

        svc = 'xbrlTaxonomy'
        url = self.mkurl(svc, sj_div = sj_div)
        cache = f'{svc}-{sj_div}.json'
        a = self._update_cache(url, cache, self.other_cache, force = force)
        return pd.DataFrame(a['list'])

    def _get_annual_idx(self, corp_code, year, reprt_code, idx_cl_code, multi = False, force = False):
        svc = 'fnlttCmpnyIndx' if multi else 'fnlttSinglIndx'
        url = self.mkurl(svc, corp_code = corp_code,
                         bsns_year = year, reprt_code = reprt_code,
                         idx_cl_code = idx_cl_code)
        cache = f'{svc}-{corp_code}-{year}-{reprt_code}-{idx_cl_code}.json'
        return self._update_cache(url, cache, self.annual_cache, force = force)

    def get_annual_idx(self, corp_code, years, reprt_code = '11011', idx_cl_code = 'M210000', multi = False, force = False):
        """단일회사 전체 재무제표"""
        import pandas as pd

        def _get_one_year(year):
            a = self._get_annual_idx(corp_code,
                                     year,
                                     reprt_code,
                                     idx_cl_code,
                                     multi = multi,
                                     force = force)
            x = pd.DataFrame(a['list'])
            x['year'] = year
            return x

        if isinstance(years,int) or isinstance(years,str):
            return _get_one_year(years)
        else:
            z = None
            for year in years:
                x = _get_one_year(year)
                z = x if z is None else pd.concat([z,x], axis=0)
            return z.reset_index(drop = True)

    def get_stock_info(self, svc, corp_code, use_cache = False):
        """지분공시 종합정보"""
        import pandas as pd
        if isinstance(svc,int): svc = ["majorstock", "elestock"][svc]
        url = self.mkurl(svc, corp_code = corp_code)
        cache = f'{svc}-{corp_code}.json'
        a = self._update_cache(url, cache, self.other_cache, force = not use_cache)
        return pd.DataFrame(a['list'])

    def get_major_info(self, svc, corp_code, bgn_de, end_de):
        """주요사항보고서 주요정보"""
        import pandas as pd
        import requests, json
        url = self.mkurl(svc, corp_code = corp_code, bgn_de = bgn_de, end_de = end_de)
        resp = requests.get(url)
        a = resp.json()
        if a['status']=='000':
            return pd.DataFrame(a['list'])
        else:
            print(f"* status is {a['status']}: {a['message']}")
            return None

    def get_rs_info(self, svc, corp_code, bgn_de, end_de):
        """증권신고서 주요정보"""
        import pandas as pd
        import requests, json
        url = self.mkurl(svc, corp_code = corp_code, bgn_de = bgn_de, end_de = end_de)
        resp = requests.get(url)
        a = resp.json()
        if a['status']=='000':
            w = {}
            for g in a['group']:
                w[g['title']] = pd.DataFrame(g['list'])
            return w
        else:
            print(f"* status is {a['status']}: {a['message']}")
            return None
