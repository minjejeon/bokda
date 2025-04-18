# from statsmodels.regression.linear_model import RegressionModel
from statsmodels.base.model import LikelihoodModel

def model_check_collinearity(self, eps=1e-10, **kwargs):
    """
    Check collinearity
    """
    #from ..more import collinearity
    from bok_da.patch import collinearity
    if self.exog is not None:
        keep,drop,_ = collinearity.check(self.exog, eps = eps)
        self.kept_index = keep
        self.kept_names = [self.data.xnames[i] for i in keep]
        self.dropped_index = drop
        self.dropped_names = [self.data.xnames[i] for i in drop]
        self.full_names = self.exog_names
        if len(drop)>0:
            quiet = kwargs.get('quiet', False)
            if not quiet:
                for i in drop:
                    print(
                        f"note: \033[1m{self.data.xnames[i]}\033[0;0m "
                        "omitted because of collinearity."
                    )
            self.data.exog = self.data.exog[:, keep]
            self.exog = self.exog[:, keep]
            #self.wexog = self.wexog[:, keep]
            for v in ['_param_names', 'xnames', '_cov_names']:
                attr = getattr(self.data, v)
                if attr is not None:
                    setattr(self.data, v, [attr[i] for i in keep])

def subclass_check_collinearity(self, *args, **kwargs):
    """
    Check collinearity
    """
    super(LikelihoodModel, self).check_collinearity(*args, **kwargs)
    df_model = self.df_model
    df_resid = self.df_resid
    self.initialize()
    self.df_model = df_model
    self.df_resid = df_resid
    if hasattr(self, 'rank'): del self.rank
