# Monkey-patch statsmodels

import sys
import importlib
import types
#from ..more import collinearity
from . import collinearity
from .sm_classes import (
    model_check_collinearity,
    subclass_check_collinearity,
)
from .sm_summary import (
    forg,
    summary_params,
)
from .sm_methods import (
    new_WLS_init,
    new_lm_fit,
    new_RegressionResults_summary,
)
from .extra_attrs import set_extra_attributes

import statsmodels

# set extra attributes
statsmodels.regression.linear_model.RegressionResults.set_extra_attributes = set_extra_attributes
statsmodels.regression.linear_model.RegressionModel.set_extra_attributes = set_extra_attributes


# functions to patch summary
statsmodels.iolib.summary.forg = forg
statsmodels.iolib.summary.summary_params = summary_params

# RegressionResults
cls = statsmodels.regression.linear_model.RegressionResults
cls.summary = new_RegressionResults_summary

# check collinearity
sys.modules['statsmodels.tools.collinearity'] = collinearity
setattr(statsmodels.tools, 'collinearity', collinearity)
statsmodels.tools.__all__.append('collinearity')

statsmodels.base.model.Model.check_collinearity = model_check_collinearity
for subclass in [
        statsmodels.discrete.discrete_model.DiscreteModel,
        statsmodels.genmod.generalized_linear_model.GLM,
        statsmodels.regression.linear_model.RegressionModel
]:
    subclass.check_collinearity = subclass_check_collinearity

# patch RegressionModel.fit for collinearity check
cls = statsmodels.regression.linear_model.RegressionModel
cls.fit = new_lm_fit

# patch WLS
from functools import wraps

def patch_wls_init(cls):
    original_init = cls.__init__
    @wraps(original_init)
    def wrapped_init(self, *args, **kwargs):
        new_WLS_init(self, *args, **kwargs)

    cls.__init__ = wrapped_init
    return cls

WLS = statsmodels.regression.linear_model.WLS
WLS = patch_wls_init(WLS)

# Prais-Winsten
from .sm_prais import Prais
statsmodels.regression.linear_model.Prais = Prais
statsmodels.regression.linear_model.__all__.append('Prais')
statsmodels.api.Prais = Prais
statsmodels.api.__all__.append('Prais')

# White heteroskedasticity test
# Breusch-Godfrey autocorrelation test
from .sm_stats_diagnostic import new_het_white, new_acorr_breusch_godfrey
statsmodels.stats.diagnostic.het_white = new_het_white
statsmodels.stats.api.het_white = statsmodels.stats.diagnostic.het_white
statsmodels.stats.diagnostic.acorr_breusch_godfrey = new_acorr_breusch_godfrey
statsmodels.stats.api.acorr_breusch_godfrey = statsmodels.stats.diagnostic.acorr_breusch_godfrey

# KPSS test
from .sm_tsa_stattools import new_kpss
statsmodels.tsa.stattools.kpss = new_kpss
statsmodels.tsa.api.kpss = statsmodels.tsa.stattools.kpss

