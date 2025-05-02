import bok_da as bd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union

from ...utils.operator import zeros, ones, eye, rows, cols, diag, meanc, stdc, sumc, inv, chol
from ...utils.pdf import lnpdfn
from ...utils.rng import randbeta, randn, randig, rand

class BayesLinearRegression:
    def __init__(self, hyper: dict):
        self.hyper = hyper
        
    def fit(self, data, oos=None, quantile: list=[0.05, 0.95], print_info: bool=False):
        y = data.iloc[:, 0]
        x = data.iloc[:, 1:]
        
        #if oos is not None:
        #    yt1 = oos.iloc[:, 0]
        #    xt1 = oos.iloc[:, 1:]
        #    self.results = gibbs_lin(Y=y, X=x, XT1=xt1, YT1=yt1, hyper=self.hyper, verbose=verbose, quantile=quantile)
        #else:
        self.results = bayes_linear(Y=y, X=x, hyper=self.hyper, print_info=print_info, quantile=quantile)
            
        return self.results
    
    def predict(self, data, oos, quantile: list=[0.05, 0.95], print_info: bool=False):
        y = data.iloc[:, 0]
        x = data.iloc[:, 1:]
        
        yt1 = oos.iloc[:, 0]
        xt1 = oos.iloc[:, 1:]
        self.results = bayes_linear(Y=y, X=x, XT1=xt1, YT1=yt1, hyper=self.hyper, print_info=print_info, quantile=quantile)
            
        return self.results
    
class BayesVariableSelection:
    def __init__(self, hyper: dict):
        self.hyper = hyper
        
    def fit(self, data, oos=None, quantile: list=[0.05, 0.95], print_info: bool=False):
        y = data.iloc[:, 0]
        x = data.iloc[:, 1:]
        
        #if oos is not None:
        #    yt1 = oos.iloc[:, 0]
        #    xt1 = oos.iloc[:, 1:]
        #    self.results = gibbs_vs(Y=y, X=x, XT1=xt1, YT1=yt1, hyper=self.hyper, verbose=verbose, quantile=quantile)
        #else:
        self.results = bayes_linear_vs(Y=y, X=x, hyper=self.hyper, print_info=print_info, quantile=quantile)
            
        return self.results
    
    def predict(self, data, oos=None, quantile: list=[0.05, 0.95], print_info: bool=False):
        y = data.iloc[:, 0]
        x = data.iloc[:, 1:]
        
        yt1 = oos.iloc[:, 0]
        xt1 = oos.iloc[:, 1:]
        self.results = bayes_linear_vs(Y=y, X=x, XT1=xt1, YT1=yt1, hyper=self.hyper, print_info=print_info, quantile=quantile)
            
        return self.results

class BayesLinResult:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        if hasattr(self, 'gam') and self.gam is not None:
            self._vs_method()
            
    def _vs_method(self):
        def plot_ppi(self):
            self._plot_scatter(self)
            
        self.plot_ppi = plot_ppi.__get__(self)
    
    def plot_hist(self, *params):
        if 'beta' in params and hasattr(self, 'beta'):
            self._plot_hist(self.beta, name='beta')
        if 'r2' in params and hasattr(self, 'r2'):
            self._plot_hist(self.r2, name='R2')
        if 'yf' in params and hasattr(self, 'yf'):
            self._plot_hist(self.yf, name='forecasts')
    
    def _plot_hist(self,
                   data,
                   name: str = '',
                   plot_mean: bool = True,
                   plot_median: bool = True):
        """plot the histogram(with median) of the Bayesian results

        Args:
            Data: Data to plot (can have multiple rows)
            name: name of the data
            plot_mean: (bool) True then plot the red vertical line of mean
            plot_median: (bool) True then plot the black vertical line of median
        """
        g = sns.FacetGrid(data.melt(var_name=name, value_name='Values'), col=name) # var_name=name
        g = g.map(sns.histplot, 'Values')

        medians = data.median()
        means = data.mean()

        for ax, col_name in zip(g.axes.flat, data.columns):
            if plot_median:
                median = medians[col_name]
                ax.axvline(median, color='r', linestyle='dashed', linewidth=1, label=f'Median: {median:.2f}')
            if plot_mean:
                mean = means[col_name]
                ax.axvline(mean, color='k', linestyle='dotted', linewidth=1, label=f'Mean: {mean:.2f}')
            ax.legend()

        plt.show()
        
    def _plot_scatter(self,
                      plot_05: bool = True,
                      plot_phat: bool = True):
        
        """plot the PPI scatterplot for the Data

        Args:
            Gamm: data to plot (Gamm)
            Pm: data to plot (Pm)
            name: name of the Data
        """
        means = self.gam.mean()
        means.index = self.gam.columns

        sns.scatterplot(means, legend=False)
        if plot_05: plt.axhline(0.5, color='red', linestyle='--')
        if plot_phat: plt.axhline(meanc(np.array(self.p)), color='k', linestyle='-.')
        plt.legend(['PPI', '0.5', 'phat'])
        plt.xticks(means.index)
        plt.grid()

        plt.show()
        
    def get_table(self, *params, quantile=[0.05, 0.95], print_info: bool=False):
        
        if 'beta' in params and hasattr(self, 'beta'):
            tab_beta = self._get_table(self.beta, quantile, print_info)
            return tab_beta
        if 'r2' in params and hasattr(self, 'r2'):
            tab_r2 = self._get_table(self.r2, quantile, print_info)
            return tab_r2
        if 'yf' in params and hasattr(self, 'yf'):
            tab_yf = self._get_table(self.yf, quantile, print_info)
            return tab_yf
        
    def _get_table(self,
                   resultm: Union[pd.Series, pd.DataFrame], 
                   quantile: list=[0.05, 0.95],
                   print_info: bool=True) -> pd.DataFrame:
        """display the result table

        Args:
            Resutlm: Result vector
            quantile: quantile list of [low, up], where defualt is 90%

        Display:
            Mean, Median, S.E., low% quantile, up% quantile

        Returns:
            result: result DataFrame
        """
        columns = resultm.columns
        resultm = np.array(resultm)
        T, K = resultm.shape

        low = quantile[0]
        up = quantile[1]

        q5 = np.quantile(np.array(resultm), low, axis=0)
        q50 = np.quantile(np.array(resultm), 0.5, axis=0)
        q95 = np.quantile(np.array(resultm), up, axis=0)

        #index = [f'var{i+1}' for i in range(K)]
        index = columns
        cols = ['Mean', 'Median', 'S.E.', f'{low}%', f'{up}%']

        data = np.hstack((meanc(resultm).reshape(-1, 1), 
                          np.matrix(q50).T, 
                          stdc(resultm).reshape(-1, 1), 
                          np.matrix(q5).T, 
                          np.matrix(q95).T))

        result = pd.DataFrame(data, index=index, columns=cols)

        if print_info:
            print(result)
            print("==================================================================")

        return result

def gen_hyper_para(type: str=None) -> dict:
    """베이지안 분석을 위한 하이퍼-파라미터 및 파라미터 생성

    Args:
        type: 베이지안 분석의 종류, string
            (defuault) - Bayesian Linear regression
            "VS"  - Bayesian Variable Selection

    Returns:
        hyper: dictionary of (hyper)parameters
            n: gibbs sampling size,
            n1: simulation size,
            n0: burn-in size
            sig2 ~ IG(alpha0/2, delta0/2)
            beta0_ = 각 베타의 사전평균, 스칼라
            B0_ = 각 베타의 사전분산, 스칼라

        if type is "VS":
            p ~ Beta(a0, c0): probability that each regressor is important
            b0 ~ IG(alpha00/2, delta00/2): spike prior variance (작을수록 엄격함)
            b1 ~ IG(alpha01/2, delta01/2): slab prior variance
    """

    # sampling size
    n1 = 10000
    n0 = int(n1 * 0.1)
    n = n0 + n1

    # prior of sig2
    alpha0 = 5
    delta0 = 5
    beta0_ = 0
    B0_ = 25
    sig2 = delta0/alpha0

    hyper = {
        'n': n,
        'n0': n0,
        'n1': n1,
        'beta0_': beta0_,
        'B0_': B0_,
        'alpha0': alpha0,
        'delta0': delta0,
        'sig2': sig2
    }

    if type == "VS":
        # prior of p
        a0 = 10
        c0 = 10
        p = c0 / (c0 + a0)

        # prior of b0
        alpha00 = 20
        delta00 = alpha00 * 0.0001
        b0 = delta00 / alpha00

        # prior of b1
        alpha01 = 20
        delta01 = alpha01 * 10
        b1 = delta01 / alpha01

        del hyper["beta0_"]
        del hyper["B0_"]

        hyper.update({
            'a0': a0,
            'c0': c0,
            'p': p,
            'alpha00': alpha00,
            'delta00': delta00,
            'b0': b0,
            'alpha01': alpha01,
            'delta01': delta01,
            'b1': b1
        })

    return hyper


def Bayes_sample_beta(Y: np.ndarray,
                      X: np.ndarray, 
                      sig2: float, 
                      beta0: np.ndarray,
                      B0: np.ndarray=None, 
                      b1: float=None, 
                      b0: float=None, 
                      Gam: np.ndarray=None) -> np.ndarray:
    """베이지안 분석을 위한 beta(설명변수의 계수) 샘플링

    Args:
        Y: 종속변수로 이루어진 np.ndarray
        X: 독립변수로 이루어진 np.ndarray
        sig2: 이전 단계에서 샘플링된(혹은 사전 평균) sig2
        beta0: 이전 단계에서 샘플링된(혹은 사전 평균) beta

        (optionary)
        b1, b0, Gam: "VS" 모형일 경우에만 입력, gen_hyper 참고

    Returns:
        beta(j) ~ Normal(beta1, B1)

        만약 b1, b0, Gam 이 None일 경우
        B1 = (sig2^{-1}X'X + B0^{-1})^{-1}
        beta1 = B1*(sig2^{-1}X'Y + B0^{-1}*beta0)
        
        만약 "VS" 모형일 경우 b1, b0, Gam 입력
        B1 = (sig2^{-1}X'X + Vbeta^{-1})^{-1}
        beta1 = B1*(sig2^{-1}X'Y + B0^{-1}*beta0)
        where Vbeta = diag[Gam*b1 + (1-Gam)*b0]
    """
    T = rows(X)
    K = cols(X)

    if B0 is None:
        B0 = diag(b1 * Gam + b0 * (1 - Gam))

    if type(sig2) != float:
        sig2 = float(sig2)

    B1 = inv((1/sig2) * X.T @ X + inv(B0))
    beta1 = B1 @ ((1/sig2) * X.T @ Y + inv(B0) @ beta0)

    B1 = 0.5*(B1+B1.T)
    B1chol = chol(B1)

    beta = beta1 + B1chol.T @ randn(K, 1)

    return beta


def Bayes_sample_sig2(Y: np.ndarray, 
                      X: np.ndarray, 
                      beta: np.ndarray, 
                      alpha0: float, 
                      delta0: float) -> np.ndarray:
    """베이지안 분석을 위한 sig2의 샘플링

    Args:
        Y: 종속변수로 이루어진 np.ndarray
        X: 독립변수로 이루어진 np.ndarray
        beta: 이전 단계에서 샘플링된(혹은 사전 평균) beta 값
        alpha0, delta0: gen_hyper 참고

    Returns:
        sig2 ~ IG(alpha1/2, delta1/2)

        alpha1 = alpha0 + T
        delta1 = delta0 + (Y - X*beta)'*(Y - X*beta)
    """

    T = rows(X)
    ehat = Y - X @ beta

    alpha1 = alpha0 + T
    delta1 = delta0 + ehat.T @ ehat

    if type(alpha1) != float or type(delta1) != float:
        alpha1 = float(alpha1)
        delta1 = float(delta1[0, 0])

    sig2 = randig(alpha1/2, delta1/2, 1) # input type must be the float

    return sig2


def Bayes_sample_gam(beta: np.ndarray, 
                     p: np.ndarray, 
                     b0: float, 
                     b1: float) -> np.ndarray:
    """베이지안 분석을 위한 Gamma의 샘플링

    Args:
        beta: 이전 단계에서 샘플링된(혹은 사전평균) beta 값
        p: 이전 단계에서 샘플링된(혹은 사전평균) p 값
        b0, b1: gen_hyper 참고

    Returns:
        Pr[gam_k=1] = p*Normal(beta_k|0, b1) / [p*Normal(beta_k|0, b1) + (1-p)*Normal(beta_k, 0|b0)]
        Pr[gam_k=0] = 1 - Pr[gam_k=1 | beta_k]

        u ~ Unif(0, 1) < Pr[gam_k=1 | beta_k] 일 경우 gam_k = 1 할당
    """

    K = rows(beta)
    gam = zeros(K, 1)

    for i in range(K):
        spike = np.exp(lnpdfn(beta[i], 0, b0))
        slab = np.exp(lnpdfn(beta[i], 0, b1))
        prob = slab * p / (slab * p + spike * (1-p))
        if float(rand(1, 1)[0, 0]) < float(prob[0]): 
            gam[i] = 1 
        else: 
            gam[i] = 0

    return gam


def Bayes_sample_b_(beta_gam: np.ndarray, 
                    alpha0_: float, 
                    delta0_: float) -> np.ndarray:
    """베이지안 분석을 위한 b0 혹은 b1의 샘플링

    Args:
        beta_gam: b#에 대해, gam_k=# 에 해당하는 계수
        alpha0_, delta0_: gen_hyper (alpha00, delta00, alpha01, delta01) 참고

    Returns:
        b_ ~ IG(alpha1_/2, delta1_/2)

        alpha1_ = alpha0_ + K_
        delta1_ = delta0_ + beta_gam'*beta_gam

        where K_: beta_gam 의 차원
    """

    K = rows(beta_gam)

    alpha1_ = alpha0_ + K

    if beta_gam.size == 0: 
        delta1_ = delta0_
    else: 
        delta1_ = delta0_ + beta_gam.T @ beta_gam

    if type(alpha1_) != float or type(delta1_) != float:
        alpha1_ = float(alpha1_)
        delta1_ = float(delta1_)

    b_ = randig(alpha1_/2, delta1_/2, 1, 1)

    return b_


def Bayes_result_table(Resultm: Union[pd.Series, pd.DataFrame], 
                       quantile: list=[0.05, 0.95],
                       verbose: bool=True) -> pd.DataFrame:
    """display the result table

    Args:
        Resutlm: Result vector
        quantile: quantile list of [low, up], where defualt is 90%

    Display:
        Mean, Median, S.E., low% quantile, up% quantile

    Returns:
        result: result DataFrame
    """
    columns = Resultm.columns
    Resultm = np.array(Resultm)
    T, K = Resultm.shape

    low = quantile[0]
    up = quantile[1]

    q5 = np.quantile(np.array(Resultm), low, axis=0)
    q50 = np.quantile(np.array(Resultm), 0.5, axis=0)
    q95 = np.quantile(np.array(Resultm), up, axis=0)

    #index = [f'var{i+1}' for i in range(K)]
    index = columns
    cols = ['Mean', 'Median', 'S.E.', f'{low}%', f'{up}%']

    data = np.hstack((meanc(Resultm).reshape(-1, 1), 
                      np.matrix(q50).T, 
                      stdc(Resultm).reshape(-1, 1), 
                      np.matrix(q5).T, 
                      np.matrix(q95).T))

    result = pd.DataFrame(data, index=index, columns=cols)
    
    if verbose:
        print(result)
        print("==================================================================")

    return result

def get_table(Resultm: Union[pd.Series, pd.DataFrame], 
              quantile: list=[0.05, 0.95],
              print_info: bool=True) -> pd.DataFrame:
    """display the result table

    Args:
        Resutlm: Result vector
        quantile: quantile list of [low, up], where defualt is 90%

    Display:
        Mean, Median, S.E., low% quantile, up% quantile

    Returns:
        result: result DataFrame
    """
    columns = Resultm.columns
    Resultm = np.array(Resultm)
    T, K = Resultm.shape

    low = quantile[0]
    up = quantile[1]

    q5 = np.quantile(np.array(Resultm), low, axis=0)
    q50 = np.quantile(np.array(Resultm), 0.5, axis=0)
    q95 = np.quantile(np.array(Resultm), up, axis=0)

    #index = [f'var{i+1}' for i in range(K)]
    index = columns
    cols = ['Mean', 'Median', 'S.E.', f'{low}%', f'{up}%']

    data = np.hstack((meanc(Resultm).reshape(-1, 1), 
                      np.matrix(q50).T, 
                      stdc(Resultm).reshape(-1, 1), 
                      np.matrix(q5).T, 
                      np.matrix(q95).T))

    result = pd.DataFrame(data, index=index, columns=cols)
    
    if print_info:
        print(result)
        print("==================================================================")

    return result


def Bayes_var(z: np.ndarray) -> np.ndarray:
    """calculate variance for Bayes R2

    Args:
        z: vector of data

    Returns:
        var_z: calculated variance of z
    """

    N = rows(z)
    K = cols(z)

    zbar = meanc(z).reshape(1, -1)@np.ones((1, N))
    zres = z - zbar.T
    var_z = zres.T@zres / (N-K)

    return var_z


def Bayes_R2(Y: np.ndarray, 
             X: np.ndarray, 
             beta: np.ndarray, 
             sig2: np.ndarray) -> np.ndarray:
    """calculate alternative bayesian R2 (Gelman, A. etc. 2018)

    Args:
        X: independent variables
        Y: dependent variable
        beta: sampled beta
        sig2: sampled sigma

    Returns:
        R2: calculated R2
    """

    T = rows(X)
    
    y_pred = X@beta + np.sqrt(float(sig2[0, 0])) * randn(T, 1)
    y_res = Y - y_pred

    var_fit = Bayes_var(y_pred)
    var_res = Bayes_var(y_res)

    R2 = var_fit / (var_fit + var_res)

    return R2

def bayes_linear_vs(Y: Union[pd.Series, pd.DataFrame], 
             X: Union[pd.Series, pd.DataFrame], 
             hyper: dict=gen_hyper_para("VS"), 
             XT1: Union[pd.Series, pd.DataFrame]=None, 
             YT1: Union[pd.Series, pd.DataFrame]=None, 
             print_info: bool=False, 
             quantile: list=[0.05, 0.95]) -> BayesLinResult:
    """Spike-and-Slab Variable Selection

    Args:
        Y: 종속변수로 이루어진 np.ndarray
        X: 독립변수로 이루어진 np.ndarray
        hyper: (하이퍼)파라미터로 구성된 dict (see gen_hyper)

        (optionary)
        XT1: 독립변수의 OSS np.ndarray
        YT1: 종속변수의 OSS(Out of Sample) np.ndarray 
        display: 1일 경우 결과 요약표 제시
        quantile: 결과 요약표에 제시할 CI quantile, 기본값 [0.05, 0.95] list

    Returns:
        Betam: sampled beta, 
        Sig2m: sampled sig2,
        Gamm: sampled gamma, 
        Pm: sampled p, 

        (optionary)
        Yfm: sampled predictied outcome, -> XT1 needed
        lnPPLm: log posterior predictive likelihood -> YT1 needed
    """
    if type(X) is pd.Series:
        X = X.to_frame()
        
    index = Y.index
    columns = X.columns
    
    if type(Y) is pd.Series:
        Y = np.array(Y).reshape(-1, 1)
    else:
        Y = np.array(Y)
        
    X = np.array(X)
    if XT1 is not None: XT1 = np.array(XT1).reshape(1, -1)
    if YT1 is not None:
        YT1 = YT1.to_frame()
        YT1 = np.array(YT1)
    # if display is None: display = 1
    
    K = cols(X)

    # load hyper-parameters and the parameters
    n1, n0, n = hyper['n1'], hyper['n0'], hyper['n']
    a0, c0, p = hyper['a0'], hyper['c0'], hyper['p']
    b0, alpha00, delta00 = hyper['b0'], hyper['alpha00'], hyper['delta00']
    b1, alpha01, delta01 = hyper['b1'], hyper['alpha01'], hyper['delta01']
    sig2, alpha0, delta0 = hyper['sig2'], hyper['alpha0'], hyper['delta0']
    sig2 = np.asarray(sig2).reshape(-1, 1)
    b1 = np.asarray(b1).reshape(-1, 1)
    b0 = np.asarray(b0).reshape(-1, 1)
    p = np.asarray(p)

    beta0 = zeros(K, 1)
    gam = ones(K, 1)

    # save the sampling results
    Betam = zeros(n1-1, K)
    Sig2m = zeros(n1-1, 1)
    Gamm = zeros(n1-1, K)
    Pm = zeros(n1-1, 1)
    Yfm = zeros(n1-1, 1)
    PPLm = zeros(n1-1, 1)
    R2m = zeros(n1-1, 1)

    for iter in range(n):
        # step 1: sample beta
        beta = Bayes_sample_beta(Y, X, float(sig2[0, 0]), beta0, None, float(b1[0, 0]), float(b0[0, 0]), gam)

        # step 2: sample sig2
        sig2 = Bayes_sample_sig2(Y, X, beta, alpha0, delta0)

        # step 3: sample gamma
        gam = Bayes_sample_gam(beta, p, float(b0[0, 0]), float(b1[0, 0]))

        # step 4: sample b0 and b1
        beta_gam0 = beta[gam == 0].T
        beta_gam1 = beta[gam == 1].T
        K0 = rows(beta_gam0)
        K1 = rows(beta_gam1)

        b0 = Bayes_sample_b_(beta_gam0, alpha00, delta00)
        b1 = Bayes_sample_b_(beta_gam1, alpha01, delta01)

        # step 5: sample p
        p = randbeta(c0 + K1, a0 + K0, 1, 1)

        # step 6: predictive dist.
        if XT1 is not None:
            yT1 = XT1 @ beta + np.sqrt(float(sig2[0, 0])) * randn(1, 1)

            # calculate log PPL
            if YT1 is not None:
                PPL = lnpdfn(YT1, XT1 @ beta, float(sig2[0, 0]))
            else:
                PPL = np.asarray(0).reshape(-1, 1)
        else:
            yT1 = np.asarray(0).reshape(-1, 1)
            PPL = np.asarray(0).reshape(-1, 1)

        # calculate R2
        R2 = Bayes_R2(Y, X, beta, sig2)

        # save the sampling if its over the burning size
        if iter > n0:
            Betam[iter - n0-1, :] = beta.T
            Sig2m[iter - n0-1, 0] = sig2[0, 0]
            Gamm[iter - n0-1, :] = gam.T
            Pm[iter - n0-1, 0] = p[0]
            Yfm[iter - n0-1, 0] = yT1[0, 0]
            PPLm[iter - n0-1, 0] = PPL[0, 0]
            R2m[iter - n0-1, 0] = R2[0, 0]
    
    Betam = pd.DataFrame(Betam, columns=columns)
    Sig2m = pd.DataFrame(Sig2m, columns=['Sig2'])
    Gamm = pd.DataFrame(Gamm, columns=[f'gam_{i}' for i in columns])
    Pm = pd.DataFrame(Pm, columns=['P'])
    R2m = pd.DataFrame(R2m, columns=['R2'])
    Yfm = pd.DataFrame(Yfm, columns=['yf'])
    
    if print_info:
        print("Coefficients: ")
        get_table(Betam, quantile)
        print("R2: ")
        get_table(R2m, quantile)

    if (YT1 is not None) and (XT1 is not None):
        if print_info:
            print("Forecast: ")
            get_table(Yfm, quantile)
            lnPPL = sumc(np.log(meanc(np.exp(PPLm))))
            print("log PPL: ", lnPPL)
        return BayesLinResult(beta=Betam, sig2=Sig2m, gam=Gamm, p=Pm, r2=R2m, yf=Yfm, lnppl=lnPPL)
    elif XT1 is not None:
        if print_info: 
            print("Forecast: ")
            get_table(Yfm, quantile)
        return BayesLinResult(beta=Betam, sig2=Sig2m, gam=Gamm, p=Pm, r2=R2m, yf=Yfm)
    else:
        return BayesLinResult(beta=Betam, sig2=Sig2m, gam=Gamm, p=Pm, r2=R2m)
            
            
def bayes_linear(Y: Union[pd.Series, pd.DataFrame], 
                 X: Union[pd.Series, pd.DataFrame], 
                 hyper: dict=gen_hyper_para(), 
                 XT1: Union[pd.Series, pd.DataFrame]=None, 
                 YT1: Union[pd.Series, pd.DataFrame]=None, 
                 print_info: bool=False, 
                 quantile: list=[0.05, 0.95]) -> BayesLinResult:
    """Bayesian Linear Regression

    Args:
        Y: 종속변수로 이루어진 pd.Series 또는 DataFrame
        X: 독립변수로 이루어진 pd.Series 또는 DataFrame
        hyper: (하이퍼)파라미터로 구성된 dict (see gen_hyper_para)

        (optionary)
        XT1: 독립변수의 out-of-sample pd.Series 또는 DataFrame
        YT1: 종속변수의 out-of-sample pd.Series 또는 DataFrame
        verbose: True인 경우 결과 요약표 제시
        quantile: 결과 요약표에 제시할 CI quantile, 기본값 [0.05, 0.95] list

    Returns:
        Betam: sampled beta, 
        Sig2m: sampled sig2,

        (optionary)
        Yfm: sampled predictied outcome -> XT1 needed
        lnPPL: log posterior predictive likelihood -> YT1 needed
    """
    if type(X) is pd.Series:
        X = X.to_frame()
        
    index = Y.index
    columns = X.columns
    
    if type(Y) is pd.Series:
        Y = np.array(Y).reshape(-1, 1)
    else:
        Y = np.array(Y)

    X = np.array(X)
    
    if XT1 is not None: XT1 = np.array(XT1).reshape(1, -1)
    if YT1 is not None:
        YT1 = YT1.to_frame()
        YT1 = np.array(YT1)

    T = rows(X)
    K = cols(X)

    # load hyper-parameters and the parameters
    n1, n0, n = hyper['n1'], hyper['n0'], hyper['n']
    sig2, beta0_, B0_, alpha0, delta0 = hyper['sig2'], hyper['beta0_'], hyper['B0_'], hyper['alpha0'], hyper['delta0']
    sig2 = np.asarray(sig2).reshape(-1, 1)

    beta0 = ones(K, 1)*beta0_
    B0 = eye(K)*B0_
    gam = ones(K, 1)

    # save the sampling results
    Betam = zeros(n1-1, K)
    Sig2m = zeros(n1-1, 1)
    Yfm = zeros(n1-1, 1)
    PPLm = zeros(n1-1, 1)
    R2m = zeros(n1-1, 1)

    for iter in range(n):
        # step 1: sample beta
        beta = Bayes_sample_beta(Y, X, float(sig2[0, 0]), beta0, B0)
        beta = beta.reshape(-1, 1)

        # step 2: sample sig2
        sig2 = Bayes_sample_sig2(Y, X, beta, alpha0, delta0)

        # step 3: predictive dist.
        if XT1 is not None:
            yT1 = XT1 @ beta + np.sqrt(float(sig2[0, 0])) * randn(1, 1)

            # calculate log PPL
            if YT1 is not None:
                PPL = lnpdfn(YT1, XT1 @ beta, float(sig2[0, 0]))
            else:
                PPL = np.asarray(0).reshape(-1, 1)
        else:
            yT1 = np.asarray(0).reshape(-1, 1)
            PPL = np.asarray(0).reshape(-1, 1)

        # calculate R2
        R2 = Bayes_R2(Y, X, beta, sig2)
        
        # save the sampling if its over the burning size
        if iter > n0:
            Betam[iter - n0-1, :] = beta.T
            Sig2m[iter - n0-1, 0] = sig2[0, 0]
            Yfm[iter - n0-1, 0] = yT1[0, 0]
            PPLm[iter - n0-1, 0] = PPL[0, 0]
            R2m[iter - n0-1, 0] = R2[0, 0]
    
    Betam = pd.DataFrame(Betam, columns=columns)
    Sig2m = pd.DataFrame(Sig2m, columns=['Sig2'])
    R2m = pd.DataFrame(R2m, columns=['R2'])
    Yfm = pd.DataFrame(Yfm, columns=['yf'])
    
    if print_info:
        print("Coefficients: ")
        get_table(Betam, quantile)

        print("R2: ")
        get_table(R2m, quantile)

    if (YT1 is not None) and (XT1 is not None):
        lnPPL = sumc(np.log(meanc(np.exp(PPLm))))
        if print_info:
            print("Forecast: ")
            get_table(Yfm, quantile)
            print("log PPL: ", lnPPL)
        return BayesLinResult(beta=Betam, sig2=Sig2m, r2=R2m, yf=Yfm, lnppl=lnPPL)
    elif XT1 is not None:
        if print_info:
            print("Forecast: ")
            get_table(Yfm, quantile)
        return BayesLinResult(beta=Betam, sig2=Sig2m, r2=R2m, yf=Yfm)
    else:
        return BayesLinResult(beta=Betam, sig2=Sig2m, r2=R2m)


def Bayes_histogram(Data: np.matrix, 
                    name: str, 
                    plot_mean: bool = True, 
                    plot_median: bool = True):
    """plot the histogram(with median) of the Bayesian results

    Args:
        Data: Data to plot (can have multiple rows)
        name: name of the data
        plot_mean: (bool) True then plot the red vertical line of mean
        plot_median: (bool) True then plot the black vertical line of median
    """
    
    df = pd.DataFrame(Data)
    g = sns.FacetGrid(df.melt(var_name=name, value_name='Values'), col=name)
    g = g.map(sns.histplot, 'Values')

    medians = df.median()
    means = df.mean()

    for ax, col_name in zip(g.axes.flat, df.columns):
        if plot_median:
            median = medians[col_name]
            ax.axvline(median, color='r', linestyle='dashed', linewidth=1, label=f'Median: {median:.2f}')
        if plot_mean:
            mean = means[col_name]
            ax.axvline(mean, color='k', linestyle='dotted', linewidth=1, label=f'Mean: {mean:.2f}')
        ax.legend()

    plt.show()
    
def plot_hist(data: Union[pd.Series, pd.DataFrame], 
              name: str = '',
              plot_mean: bool = True,
              plot_median: bool = True):
    """plot the histogram(with median) of the Bayesian results

    Args:
        Data: Data to plot (can have multiple rows)
        name: name of the data
        plot_mean: (bool) True then plot the red vertical line of mean
        plot_median: (bool) True then plot the black vertical line of median
    """
    
    g = sns.FacetGrid(data.melt(var_name=name, value_name='Values'), col=name) # var_name=name
    g = g.map(sns.histplot, 'Values')

    medians = data.median()
    means = data.mean()

    for ax, col_name in zip(g.axes.flat, data.columns):
        if plot_median:
            median = medians[col_name]
            ax.axvline(median, color='r', linestyle='dashed', linewidth=1, label=f'Median: {median:.2f}')
        if plot_mean:
            mean = means[col_name]
            ax.axvline(mean, color='k', linestyle='dotted', linewidth=1, label=f'Mean: {mean:.2f}')
        ax.legend()

    plt.show()


def Bayes_scatter(Gamm: np.matrix, 
                  Pm: np.matrix, 
                  name: str, 
                  plot_05: bool = True, 
                  plot_phat: bool = True):
    """plot the PPI scatterplot for the Data

    Args:
        Gamm: data to plot (Gamm)
        Pm: data to plot (Pm)
        name: name of the Data
    """
    df = pd.DataFrame(Gamm)
    K = cols(Gamm)

    means = df.mean()
    means.index = means.index + 1
    variindex = list(range(1, K+1, 1))

    sns.scatterplot(means, legend=False)
    if plot_05: plt.axhline(0.5, color='red', linestyle='--')
    if plot_phat: plt.axhline(meanc(Pm), color='k', linestyle='-.')

    plt.legend([name, '0.5', 'phat'])
    plt.xticks(variindex)
    plt.grid()

    plt.show()

def Bayes_scatter(Gamm: pd.DataFrame, 
                  Pm: pd.DataFrame, 
                  name: str, 
                  plot_05: bool = True, 
                  plot_phat: bool = True):
    """plot the PPI scatterplot for the Data

    Args:
        Gamm: data to plot (Gamm)
        Pm: data to plot (Pm)
        name: name of the Data
    """
    df = pd.DataFrame(Gamm)
    K = cols(Gamm)

    means = df.mean()
    means.index = means.index + 1
    variindex = list(range(1, K+1, 1))

    sns.scatterplot(means, legend=False)
    if plot_05: plt.axhline(0.5, color='red', linestyle='--')
    if plot_phat: plt.axhline(meanc(Pm), color='k', linestyle='-.')

    plt.legend([name, '0.5', 'phat'])
    plt.xticks(variindex)
    plt.grid()

    plt.show()