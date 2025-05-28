# `statsmodels` 패키지 `summary` 메쏘드 관련 패치

소수점 등이 uninformative하므로 소수점 아래 정확도를 바꾸어 주고, 모델에 대하여 더 정확한 정보를 주도록 패치한다.

## 글로벌 함수 패치

다음 2개 모듈 레벨 함수들은 `statsmodels.iolib.summary` 모듈에 있는 `forg`와 `summary_params` 모듈 레벨 함수들을 copy & paste한 후 수정한 것이다. 자세한 내용은 패치 모듈을 참조하라.

```python
# module: .patch.sm_summary

import numpy as np
from statsmodels.compat.python import lrange, lzip
from statsmodels.iolib.summary import _getnames
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.tableformatting import fmt_params

# global function in statsmodels.iolib.summary module
def forg(x, prec=3):
    # ...

# global function in statsmodels.iolib.summary module
def summary_params(results, yname=None, xname=None, alpha=.05, use_t=True,
                   skip_header=False, title=None):
    # ...
    return parameter_table
```

패치된 `forg`와 `summary_params` 함수를 다음과 같이 적용한다.

```python
# module: .patch.statsmodels

from .sm_summary import (
    forg,
    summary_params,
)
import statsmodels.iolib.summary
statsmodels.iolib.summary.forg = forg
statsmodels.iolib.summary.summary_params = summary_params
```

## `RegressionResults` 클래스의 `summary` 메쏘드

원래 `statsmodels` 내 `RegressionResults` 클래스의 `summary` 메쏘드를 수정하여 더 정교한 정보를 제공하고 새로 만들 `Prais` 클래스를 처리하도록 패치해 준다. 자세한 내용은 모듈을 참조하라.

```python
# model: .patch.sm_methods

def new_RegressionResults_summary(
        self,
        yname: str | None = None,
        xname: Sequence[str] | None = None,
        title: str | None = None,
        alpha: float = 0.05,
        slim: bool = False,
):
    # ...
    return smry
```

위 수정된 `fit` 메쏘드는 다음과 같이 적용한다.

```python
# module: .patch.statsmodels

from .sm_methods import new_RegressionResults_summary
import statsmodels
cls = statsmodels.regression.linear_model.RegressionResults
cls.summary = new_RegressionResults_summary
```

# Collinearity

## `statsmodels.base.model.Model` 클래스에 메쏘드 추가

`bok.more.collinearity` 모듈 내 `check` function을 이용할 것이다. 이 함수를 이용하여 `statsmodels.base.model` 모듈 내 `Model` 클래스에 `check_collinearity` 메쏘드를 추가하고자 한다. `patch.sm_classes` 모듈 내에 다음 모듈 레벨 함수를 만들자. 이 함수는 `eps` (기본값은 `1e-10`)와 `quiet` (기본값은 `False`) 인자를 받아들여 공선성을 점검하고, 공선성이 존재하면 누락시킬 변수들을 `self.exog`로부터 제외시키며, `kept_index`, `kept_names`, `dropped_index`, `dropped_names`, `full_names` attributes를 지정한다.

```python
# module: .patch.sm_classes
def model_check_collinearity(self, eps=1e-10, **kwargs):
    """
    Check collinearity
    """
    from ..model import collinearity
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
```

이 모듈 레벨 함수를 이용하여 다음과 같이 `statsmodels.base.model.Model` 클래스에 `check_collinearity` 메쏘드를 추가한다.

```python
# module: .patch.statsmodels
import statsmodels
from .sm_classes import model_check_collinearity
statsmodels.base.model.Model.check_collinearity = model_check_collinearity
```
## `Model`의 하위 클래스에 메쏘드 추가

앞에서 `Model` 클래스에 `check_collinearity` 메쏘드를 추가하였다. 이제 `Model`의 서브클래스들(몇 단계를 거친 서브클래스들)에 `check_collinearity`를 오버라이드하는 메쏘드를 붙이고자 한다. 우선 다음 함수를 만든다. 아래에서 `LikelihoodModel`을 import하는 것은 이것의 super class가 `Model`이고 이것의 subclass들의 공통부모가 `LikelihoodModel`이기 때문이다.

```python
# module: .patch.sm_classes
from statsmodels.base.model import LikelihoodModel

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
```

위에서 `super` 부분에 유의하라. 위 `subclass_check_collinearity`를 3개 서브클래스에 메쏘드로 추가하고자 한다.

```python
# module: .patch.statsmodels
from .sm_classes import subclass_check_collinearity
import statsmodels
subclasses = [
    statsmodels.discrete.discrete_model.DiscreteModel,
    statsmodels.genmod.generalized_linear_model.GLM,
    statsmodels.regression.linear_model.RegressionModel
]

for subclass in subclasses:
    subclass.check_collinearity = subclass_check_collinearity
```

## `fit` 메쏘드 수정

선형회귀(linear model 모듈)에서 공선성을 점검하도록 `fit` 메쏘드들을 수정하자. 수정하는 내용은 다음과 같다.

* 디폴트 method를 "pinv"에서 "qr"로 바꾼다.
* `check_collinearity` argument (boolean = `True`), `eps` argument (기본값 = `1e-10`), `quiet` argument (boolean = `False`)을 주도록 하여, `check_collinearity`가 `True`이면 `check_collinearity` 메쏘드를 fit 시작 시 실행하도록 한다.

```python
# module: .patch.sm_methods

import numpy as np
from typing import Literal, Optional
from statsmodels.tools.tools import pinv_extended
from statsmodels.regression.linear_model import OLS, WLS, OLSResults, RegressionResults, RegressionResultsWrapper

def new_lm_fit(
        self,
        method: Literal["pinv", "qr"] = "qr",
        cov_type: Literal[
            "nonrobust",
            "fixed scale",
            "HC0",
            "HC1",
            "HC2",
            "HC3",
            "HAC",
            "hac-panel",
            "hac-groupsum",
            "cluster",
        ] = "nonrobust",
        cov_kwds=None,
        # use_t: bool | None = None,
        use_t: Optional[bool] = None,
        **kwargs
):
    """
    Full fit of the model.

    The results include an estimate of covariance matrix, (whitened)
    residuals and an estimate of scale.

    Parameters
    ----------
    method : str, optional
        Can be "pinv", "qr".  "pinv" uses the Moore-Penrose pseudoinverse
        to solve the least squares problem. "qr" uses the QR
        factorization.
    cov_type : str, optional
        See `regression.linear_model.RegressionResults` for a description
        of the available covariance estimators.
    cov_kwds : list or None, optional
        See `linear_model.RegressionResults.get_robustcov_results` for a
        description required keywords for alternative covariance
        estimators.
    use_t : bool, optional
        Flag indicating to use the Student's t distribution when computing
        p-values.  Default behavior depends on cov_type. See
        `linear_model.RegressionResults.get_robustcov_results` for
        implementation details.
    **kwargs
        Additional keyword arguments that contain information used when
        constructing a model using the formula interface.

    Returns
    -------
    RegressionResults
        The model estimation results.

    See Also
    --------
    RegressionResults
        The results container.
    RegressionResults.get_robustcov_results
        A method to change the covariance estimator used when fitting the
        model.

    Notes
    -----
    The fit method uses the pseudoinverse of the design/exogenous variables
    to solve the least squares minimization.
    """
    check_col = kwargs.get('check_collinearity', True)
    if check_col:
        self.check_collinearity(
            eps = kwargs.get('collinearity_eps', 1e-10),
            quiet = kwargs.get('quiet', False)
        )
    if method == "pinv":
        if not (hasattr(self, 'pinv_wexog') and
                hasattr(self, 'normalized_cov_params') and
                hasattr(self, 'rank')):

            self.pinv_wexog, singular_values = pinv_extended(self.wexog)
            self.normalized_cov_params = np.dot(
                self.pinv_wexog, np.transpose(self.pinv_wexog))

            # Cache these singular values for use later.
            self.wexog_singular_values = singular_values
            self.rank = np.linalg.matrix_rank(np.diag(singular_values))

        beta = np.dot(self.pinv_wexog, self.wendog)

    elif method == "qr":
        if not (hasattr(self, 'exog_Q') and
                hasattr(self, 'exog_R') and
                hasattr(self, 'normalized_cov_params') and
                hasattr(self, 'rank')):
            Q, R = np.linalg.qr(self.wexog)
            self.exog_Q, self.exog_R = Q, R
            self.normalized_cov_params = np.linalg.inv(np.dot(R.T, R))

            # Cache singular values from R.
            self.wexog_singular_values = np.linalg.svd(R, 0, 0)
            self.rank = np.linalg.matrix_rank(R)
        else:
            Q, R = self.exog_Q, self.exog_R
        # Needed for some covariance estimators, see GH #8157
        self.pinv_wexog = np.linalg.pinv(self.wexog)
        # used in ANOVA
        self.effects = effects = np.dot(Q.T, self.wendog)
        beta = np.linalg.solve(R, effects)
    else:
        raise ValueError('method has to be "pinv" or "qr"')

    if self._df_model is None:
        self._df_model = float(self.rank - self.k_constant)
    if self._df_resid is None:
        self.df_resid = self.nobs - self.rank

    if isinstance(self, OLS):
        lfit = OLSResults(
            self, beta,
            normalized_cov_params=self.normalized_cov_params,
            cov_type=cov_type, cov_kwds=cov_kwds, use_t=use_t)
    else:
        lfit = RegressionResults(
            self, beta,
            normalized_cov_params=self.normalized_cov_params,
            cov_type=cov_type, cov_kwds=cov_kwds, use_t=use_t,
            **kwargs)
    return RegressionResultsWrapper(lfit)
```

위 수정된 `fit` 메쏘드는 다음과 같이 적용한다.

```python
# module: .patch.statsmodels

from .sm_methods import new_lm_fit
import statsmodels
cls = statsmodels.regression.linear_model.RegressionModel
cls.fit = new_lm_fit
```

`RegressionModel` 이외에도 `GLM`과 `DiscreteModel`에서 공선성 점검이 필요한데, 여기서도 `fit` 메쏘드를 패치하거나, 아니면 사용자가 사용 전에 `check_collinearity`를 명시적으로 호출하면 될 것이다.

# WLS 클래스 패치

`WLS` 클래스를 패치하고자 한다. Wooldridge의 자동화된 WLS를 위한 옵션을 추가하기 위함이다. 이 옵션은 `WLS` 클래스 인스턴스 생성 시 `weights = "_auto"` 인자를 사용하도록 한다. `__init__` 메쏘드를 패치할 예정이며, 다음은 수정한 `new_WLS_init` 함수이다. 자세한 내용은 패치 모듈을 참조하라.

```python
# module: .patch.sm_methods

import numpy as np
from typing import Literal, Optional
from statsmodels.tools.tools import pinv_extended
from statsmodels.regression.linear_model import OLS, WLS, OLSResults, RegressionResults, RegressionResultsWrapper

def new_WLS_init(self, endog, exog, weights=1., missing='none', hasconst=None,
             **kwargs):
    if type(self) is WLS:
        self._check_kwargs(kwargs)

    is_auto_wgt = isinstance(weights, str) and weights=="_auto"
    if isinstance(weights, str): weights=1.
    # ...
```

이 패치는 다음 방법으로 적용한다.

```python
# module: .patch.statsmodels

from .sm_methods import new_WLS_init
import statsmodels
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
```

# Prais 클래스

Prais-Winsten 추정과 Cochrane-Orcutt 추정을 위한 `Prais` 클래스를 만들어 `statsmodels.regression.linear_model` 모듈에 추가한다. 우선 `.patch.sm_prais` 모듈에서 `Prais` 클래스를 생성한다. 그 대략적인 내용은 다음과 같고, 자세한 내용은 모듈을 참조하라.

```python
# module: .patch.sm_prais

import numpy as np
from statsmodels.regression.linear_model import GLS
from .sm_fast_lm import _fast_scalar_reg, _fast_ols

class Prais(GLS):
    __doc__ = """
    Prais-Winsten AR(1) FGLS and Cochrane-Orcutt Regression
    """

    def __init__(self, endog, exog=None, missing='none', hasconst=None,
                 rho=None, twostep=False, corc=False,
                 rhotype = 'regress', maxiter=100, **kwargs):
        self.corc = corc
        self.twostep = twostep
        self.maxiter = 1 if twostep else maxiter
        self.rho = rho
        self.rhotype = rhotype
        super(Prais, self).__init__(
            endog, exog, missing=missing, hasconst=hasconst, **kwargs
        )
        self.title = 'Prais-Winsten AR(1) regression'

    # ...
```

이 클래스를 추가하는 부분은 다음과 같다.

```python
# module: .patch.statsmodels

import statsmodels
from .sm_prais import Prais
statsmodels.regression.linear_model.Prais = Prais
statsmodels.regression.linear_model.__all__.append('Prais')
```

# 검정 관련

`statsmodels.stats.diagnostic` 모듈을 수정한다.

## White 이분산 검정 패치

White 이분산 검정 시 model degrees of freedom을 리턴해 주는 것이 좋다.

```python
# module: .patch.sm_stats_diagnostic

import numpy as np
from statsmodels.tools.validation import array_like
from statsmodels.stats.diagnostic import _check_het_test
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.compat.pandas import deprecate_kwarg
from statsmodels.tsa.tsatools import lagmat
from statsmodels.stats.diagnostic import ResultsStore

def new_het_white(resid, exog):
    # ...
    #return lm, lmpval, fval, fpval
    return lm, lmpval, fval, fpval, resols.df_model
```

다음 방식으로 패치를 적용시킨다.

```python
# module: .patch.statsmodels

from .sm_stats_diagnostic import new_het_white
statsmodels.stats.diagnostic.het_white = new_het_white
```

## Breusch-Godfrey 검정 패치

Breusch-Godfrey 자기상관 검정 시 공선성 점검 부분을 건너뛰도록 한다.

```python
# module: .patch.sm_stats_diagnostic
# continue

@deprecate_kwarg("results", "res")
def new_acorr_breusch_godfrey(res, nlags=None, store=False):
    # ...
```

다음과 같이 패치를 적용한다.

```python
# module: .patch.statsmodels

from .sm_stats_diagnostic import new_acorr_breusch_godfrey
statsmodels.stats.diagnostic.acorr_breusch_godfrey = new_acorr_breusch_godfrey
statsmodels.stats.api.acorr_breusch_godfrey = statsmodels.stats.diagnostic.acorr_breusch_godfrey
```

## KPSS 검정 패치

KPSS 검정 시 래그 개수(`nlags`)와 관측치 수(`nobs`)를 리턴해 주는 것이 좋다.

```python
# modules: .patch.sm_tsa_stattools

import numpy as np
import warnings
from typing import Literal
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.validation import (
    array_like,
    bool_like,
    dict_like,
    float_like,
    int_like,
    string_like,
)
from statsmodels.tsa.stattools import (
    _kpss_autolag,
    _sigma_est_kpss,
)
from statsmodels.tools.tools import Bunch, add_constant
from statsmodels.tsa.tsatools import add_trend, lagmat, lagmat2ds

def new_kpss(
    x,
    regression: Literal["c", "ct"] = "c",
    nlags: Literal["auto", "legacy"] | int = "auto",
    store: bool = False,
):
    # ...
    if store:
        # ...
        return kpss_stat, p_value, nlags, crit_dict, rstore
    else:
        return kpss_stat, p_value, nlags, nobs, crit_dict
```

다음과 같이 패치를 적용한다.

```python
# module: .patch.statsmodels

from .sm_tsa_stattools import new_kpss
statsmodels.tsa.stattools.kpss = new_kpss
```
