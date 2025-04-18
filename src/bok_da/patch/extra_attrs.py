# Monkey-patch

def set_extra_attributes(self, **kwargs):
    for key, value in kwargs.items():
        setattr(self, key, value)

def patch_extra_attributes():
    from statsmodels.regression.linear_model import (
        RegressionModel,
        RegressionResults
    )
    from linearmodels.iv.results import IVResults
    RegressionModel.set_extra_attributes = set_extra_attributes
    RegressionResults.set_extra_attributes = set_extra_attributes
    IVResults.set_extra_attributes = set_extra_attributes

