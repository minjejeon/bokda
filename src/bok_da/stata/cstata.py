import pandas as pd

def _new_dict(x):
    return {key[2:-1]: value for key,value in x.items()}

class Stata:
    def __init__(self, path, edition, splash=False):
        import stata_setup
        stata_setup.config(path, edition, splash)
        self._verbose = True

    def get_ready(self):
        import pystata
        import sfi
        self.status = pystata.config.status
        self.log_open = pystata.config.set_output_file
        self.log_close = pystata.config.close_output_file
        self.set_graph_size = pystata.config.set_graph_size

        #self.run = pystata.stata.run
        self.nparray_from_data = pystata.stata.nparray_from_data
        self.pdataframe_from_data = pystata.stata.pdataframe_from_data
        self.export_data = self.pdataframe_from_data

        self.Matrix = sfi.Matrix
        return self

    def be_quiet(self):
        self._verbose = False

    def be_noisy(self):
        self._verbose = True

    @property
    def verbose(self):
        return self._verbose

    def clear(self, all=True):
        if all:
            self.run('clear all')
        else:
            self.run('clear')

    @property
    def ereturn(self):
        import pystata.stata
        ans = _new_dict(pystata.stata.get_ereturn())
        if ans.get('V') is not None:
            ans['labels'] = self.Matrix.getColNames('e(V)')
        return ans

    @property
    def rreturn(self):
        import pystata.stata
        return _new_dict(pystata.stata.get_return())

    @property
    def sreturn(self):
        import pystata.stata
        return _new_dict(pystata.stata.get_sreturn())

    @property
    def returns(self):
        from types import SimpleNamespace
        rets = SimpleNamespace()
        rets.e = self.ereturn
        rets.r = self.rreturn
        rets.s = self.sreturn
        return rets

    def get_b(self, dim=2):
        x = 'e(b)'
        b = self.Matrix.get(x)
        nm = self.Matrix.getColNames(x)
        if dim==2:
            ans = pd.DataFrame(b, columns=nm)
        elif dim==1:
            ans = pd.Series(b[0], index=nm)
        return ans

    def get_V(self):
        x = 'e(V)'
        V = self.Matrix.get(x)
        nm = self.Matrix.getColNames(x)
        return pd.DataFrame(V, index=nm, columns=nm)

    def use(self, data, force=False, **kwargs):
        """pystata.stata -use-"""
        import pystata.stata
        if isinstance(data, str):
            if data.endswith('.dta'):
                cmd = f'use {data}'
            elif data.endswith('.csv') or data.endswith('.xlsx'):
                cmd = f'import delim using {data}'
            else:
                raise ValueError(f'Cannot import {data}')
                
            if force: cmd = cmd + ', clear'
            self.run(cmd, **kwargs)
        else:
            pystata.stata.pdataframe_to_data(data, force=force)

    def run(self, cmd, echo=True, inline=None, **kwargs):
        import pystata.stata
        quietly = kwargs.pop('qui', None)
        if quietly is None: quietly = kwargs.pop('quietly', None)
        pystata.stata.run(cmd, quietly, echo, inline)

    def describe(self):
        self.run('describe')

    def desc(self): self.describe()
    def d(self): self.describe()
