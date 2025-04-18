#from ..more import collinearity
from . import collinearity
from .extra_attrs import set_extra_attributes
from typing import Optional
from linearmodels.iv.data import IVData, IVDataLike
from linearmodels.typing import (
    ArrayLike,
    BoolArray,
    Float64Array,
    Numeric,
    OptionalNumeric,
)
from numpy import (
    all as npall,
    any as npany,
    array,
    c_,
    isscalar,
    nanmean,
    ones,
    sqrt,
)
import pandas as pd

from linearmodels.iv.results import IVResults
IVResults.set_extra_attributes = set_extra_attributes

def patched_parse(self) -> None:
    blocks = self._formula.strip().split("~")
    if len(blocks) == 2:
        dep = blocks[0].strip()
        exog = blocks[1].strip()
        endog = "0"
        instr = "0"
    elif len(blocks) == 3:
        blocks = [bl.strip() for bl in blocks]
        if "[" not in blocks[1] or "]" not in blocks[2]:
            raise ValueError(
                "formula not understood. Endogenous variables and "
                "instruments must be segregated in a block that "
                "starts with [ and ends with ]."
            )
        dep = blocks[0].strip()
        exog, endog = (bl.strip() for bl in blocks[1].split("["))
        instr, exog2 = (bl.strip() for bl in blocks[2].split("]"))
        if endog[0] == "+" or endog[-1] == "+":
            raise ValueError(
                "endogenous block must not start or end with +. This block "
                "was: {}".format(endog)
            )
        if instr[0] == "+" or instr[-1] == "+":
            raise ValueError(
                "instrument block must not start or end with +. This "
                "block was: {}".format(instr)
            )
        if exog:
            exog = exog[:-1].strip() if exog[-1] == "+" else exog
        if exog2:
            exog += exog2

        exog = exog.strip()
        if exog[0] == '+': exog = exog[1:]
        exog = "0" if not exog else "1 + " + exog
    else:
        raise ValueError("formula contains more then 2 separators (~)")
    comp = {
        "dependent": "0 + " + dep,
        "exog": exog,
        "endog": endog,
        "instruments": instr,
    }
    self._components = comp


def _split_into_two(keep,drop,k):
    keep1 = [elem for elem in keep if elem < k]
    drop1 = [elem for elem in drop if elem < k]
    keep2 = [elem-k for elem in keep if elem >= k]
    drop2 = [elem-k for elem in drop if elem >= k]
    return keep1,drop1,keep2,drop2


def _iv_collinearity_check(obj, keep, drop, text="", quiet=False):
    """
    Check collinearity for IV regression
    """
    if text: text = ' ' + text
    if len(drop)>0:
        obj.dropped_indices = drop
        obj.dropped_names = [obj._col_labels[i] for i in drop]
        obj._col_labels = [obj._col_labels[i] for i in keep]
        obj._ndarray = obj._ndarray[:, keep]
        obj._pandas = obj._pandas[obj._col_labels]
        if not quiet:
            for v in obj.dropped_names:
                fullv = f"\033[1m{v}\033[0;0m"
                print(f"note:{text} {fullv} omitted "
                      "because of collinearity.")
    return obj

def linearmodels_iv_collinearity_check(self, quiet=False):
    """
    Check collinearity for linearmodels.iv.model._IVLSModelBase
    """
    k_exog = 0 if self.exog.pandas is None else self.exog.pandas.shape[1]
    _x = c_[self.exog.ndarray, self.endog.ndarray]
    if _x is not None:
        keep,drop,_ = collinearity.check(_x)
        if len(drop)>0:
            x_keep,x_drop,Y_keep,Y_drop = _split_into_two(keep,drop,k_exog)
            self.exog = _iv_collinearity_check(
                self.exog, x_keep, x_drop, text='exog regressor',
                quiet = quiet,
            )
            self.endog = _iv_collinearity_check(
                self.endog, Y_keep, Y_drop, text='endog regressor',
                quiet = quiet,
            )

    _z = c_[self.exog.ndarray, self.instruments.ndarray]
    if _z is not None:
        k_exog = 0 if self.exog.pandas is None else self.exog.pandas.shape[1]
        keep,drop,_ = collinearity.check(_z)
        if len(drop)>0:
            x_keep,x_drop,z_keep,z_drop = _split_into_two(keep,drop,k_exog)
            self.exog = _iv_collinearity_check(
                self.exog, x_keep, x_drop, text='exog regressor',
                quiet = quiet,
            )
            self.instruments = _iv_collinearity_check(
                self.instruments, z_keep, z_drop, text='extra instrument',
                quiet = quiet,
            )

def __patched__init__(
    self,
    dependent: IVDataLike,
    exog: Optional[IVDataLike] = None,
    endog: Optional[IVDataLike] = None,
    instruments: Optional[IVDataLike] = None,
    *,
    weights: Optional[IVDataLike] = None,
    fuller: Numeric = 0,
    kappa: OptionalNumeric = None,
):
    self.dependent = IVData(dependent, var_name="dependent")
    nobs: int = self.dependent.shape[0]
    self.exog = IVData(exog, var_name="exog", nobs=nobs)
    self.endog = IVData(endog, var_name="endog", nobs=nobs)
    self.instruments = IVData(instruments, var_name="instruments", nobs=nobs)

    self.check_collinearity()

    self._original_index = self.dependent.pandas.index
    if weights is None:
        weights = ones(self.dependent.shape)
    weights = IVData(weights).ndarray
    if npany(weights <= 0):
        raise ValueError("weights must be strictly positive.")
    weights = weights / nanmean(weights)
    self.weights = IVData(weights, var_name="weights", nobs=nobs)

    self._drop_locs = self._drop_missing()
    # dependent variable
    w = sqrt(self.weights.ndarray)
    self._y = self.dependent.ndarray
    self._wy = self._y * w
    # model regressors
    self._x = c_[self.exog.ndarray, self.endog.ndarray]
    self._wx = self._x * w
    # first-stage regressors
    self._z = c_[self.exog.ndarray, self.instruments.ndarray]
    self._wz = self._z * w

    self._has_constant = False
    self._regressor_is_exog = array(
        [True] * self.exog.shape[1] + [False] * self.endog.shape[1]
    )
    self._columns = self.exog.cols + self.endog.cols
    self._instr_columns = self.exog.cols + self.instruments.cols
    self._index = self.dependent.rows

    self._validate_inputs()
    if not hasattr(self, "_method"):
        self._method = "IV-LIML"
        additional = []
        if fuller != 0:
            additional.append(f"fuller(alpha={fuller})")
        if kappa is not None:
            additional.append(f"kappa={kappa}")
        if additional:
            self._method += "(" + ", ".join(additional) + ")"

    self._kappa = kappa
    self._fuller = fuller
    if kappa is not None and not isscalar(kappa):
        raise ValueError("kappa must be None or a scalar")
    if not isscalar(fuller):
        raise ValueError("fuller must be None or a scalar")
    if kappa is not None and fuller != 0:
        warnings.warn(
            "kappa and fuller should not normally be used "
            "simultaneously.  Identical results can be computed "
            "using kappa only",
            UserWarning,
            stacklevel=2,
        )
    if endog is None and instruments is None:
        self._method = "OLS"
    self._formula = ""


def patched_IVLSModelBase_fit(self, *args, **kwargs):
    """
    Patched _IVLSModelBase.fit
    """
    self.check_collinearity()
    if 'cov_type' not in kwargs:
        kwargs['cov_type'] = 'unadjusted'    
    return self.original_fit(*args, **kwargs)


def patch_linearmodels_parse():
    """
    Patch linearmodels.iv._utility.IVFormularParser._parse
    so the intercept is included in the exogenous regressor part.
    """
    import linearmodels.iv._utility as utility
    utility.IVFormulaParser.original_parse = utility.IVFormulaParser._parse
    utility.IVFormulaParser._parse = patched_parse


def patch_IVModelBase() -> None:
    """
    Patch linearmodels.iv.model._IVModelBase
    """
    from linearmodels.iv.model import _IVModelBase as Klass
    Klass.__init__ = __patched__init__
    Klass.check_collinearity = linearmodels_iv_collinearity_check


def patch_IVLSModelBase_fit():
    """
    Patch linearmodels.iv.model._IVLSModelBase.fit
    """
    from linearmodels.iv.model import _IVLSModelBase as Klass
    Klass.original_fit = Klass.fit
    Klass.fit = patched_IVLSModelBase_fit

patch_linearmodels_parse()
patch_IVModelBase()
patch_IVLSModelBase_fit()
