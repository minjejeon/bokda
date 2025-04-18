import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy import stats
from scipy import sparse
import statistics as stat

from tqdm import tqdm

import matplotlib.pyplot as plt


# custom modules
from . import ucsv_functions_univar as uf
from . import load_cython
from .. import utils


from typing import Literal, Union


class UnivarUCSV:
    """
    """


    def __init__(self, run_type: Literal['python', 'cython'] = 'cython', 
                 n_per_year: int = 4, n_draws: int = 50000, thinning: int = 10, n_burnin: int = 5000,
                 verbose: bool = False):
        """
        UCSV SVRW

        Parameters
        ----------
        run_type: str
            실행 방식 지정, 'python'일 경우 python의 패키지만 사용, 'cython'일 경우 cython 사용
        verbose : bool, optional
            과정을 출력할 지 여부 | Default: False

        -- fit_ucsv_svrw의 옵션 --
        n_per_year(int): 
            number of observations per year | Default: 4
        n_draws(int): 
            total number of draws | Default: 50000
        thinning(int): 
            save one from n | Default: 10
        n_burn_in(int): 
            number of burn-in | Default: 5000

        prior_a0(int): 0
        prior_b0(int): 10
        prior_Vomega(int): 1


        """
        self.run_type = run_type.lower()  # 소문자로 변환
        self.is_fitted = False  # initialize시 fitted flag를 False로 설정
        self.verbose = verbose
        self.n_per_year = n_per_year
        self.n_draws = n_draws
        self.thinning = thinning
        self.n_burnin = n_burnin

        # cython 로드
        if self.run_type == 'cython':
            if self.verbose:
                print("> Loading Cython Module...")
                self.uf_cython = load_cython.load_cython_ucsv_functions()
                print("> Cython Module Loaded.")
            else:
                self.uf_cython = load_cython.load_cython_ucsv_functions()

        # PRINT: model init
        if self.verbose:
            print("> Initializing Model...")
            print(f"  - Run Type: {self.run_type}")
            print(f"  - n_per_year: {self.n_per_year}")
            print(f"  - n_draws: {self.n_draws}")
            print(f"  - thinning: {self.thinning}")
            print(f"  - n_burnin: {self.n_burnin}")
            print(f"  - Verbose: {self.verbose}")


    ###############
    # Public Method
    ###############

    def fit(self, y: Union[np.ndarray, pd.Series], 
            # n_per_year: int = 4, n_draws: int = 50000, thinning: int = 10, n_burn_in: int = 50000
            ) -> 'UnivarUCSV':
        """
        Parameters
        ----------
        y : numay array of shape (n_samples,) or pd.Series
            입력 데이터 벡터 (단변량 데이터 입력)
        
        Returns
        -------
        self : object
            객체 자기 자신을 반환
        """
        # 1) 입력값이 데이터프레임인 경우 numpy로 변환
        if isinstance(y, pd.Series):
            self.index = y.index  # index 저장
            y = y.to_numpy()
        elif isinstance(y, np.ndarray):
            self.index = np.arange(len(y))  # numpy인 경우 인덱스를 저장

        # 3) numpy array에 문제가 없는지 확인
        # self._check_input_array(y)

        # PRINT: start
        if self.verbose:
            print("> Start Fitting Model...")
            print(f"  - Input Data: {y.shape[0]} samples")

        # 5) 적합
        if self.run_type == 'python':
            self._fit_ucsv_svrw_python(y=y, T=len(y), nper=self.n_per_year, n_draws=self.n_draws, thinning=self.thinning, n_burnin=self.n_burnin)
        elif self.run_type == 'cython':
            self._fit_ucsv_svrw_cython(y=y, T=len(y), nper=self.n_per_year, n_draws=self.n_draws, thinning=self.thinning, n_burnin=self.n_burnin)
            # self._fit_ucsv_svrw_cython(y=y, T=len(y), nper=n_per_year, n_draws=n_draws, thinning=thinning, n_burnin=n_burn_in)
        else:
            raise ValueError(f"지원하지 않는 run_type 입니다 | run_type: {self.run_type}")


        # 6) summary stats 계산
        self.is_fitted = True # fit 완료 이후 flag 설정

        # stats 계산
        if self.verbose:
            print("> Calculating Summary Statistics...")
        self._calculate_summary_stats(y)

        # PRINT: Done
        if self.verbose:
            print("> Model Fitted.")

        # 반환
        return self


    def print_summary(self, digits: int=4):
        to_print = ["g_eps", "g_dtau", "ps"]

        # 계수와 관련된 통계량을 준비
        column_names = to_print
        mean_values = [self.summary_stats["posterior_mean"][k] for k in to_print]
        var_values = [self.summary_stats["posterior_variance"][k] for k in to_print]
        p25_values = [self.summary_stats["hpd_lower_bound"][k] for k in to_print]
        p95_values = [self.summary_stats["hpd_upper_bound"][k] for k in to_print]
        ess_values = [self.summary_stats["ESS"][k] for k in to_print]

        # 단일 통계량
        single_stats = {
        }

        summary_dict = {
            "Variable": column_names,
            "Mean": mean_values,
            "Var": var_values,
            "2.5": p25_values,
            "97.5": p95_values,
            "ESS": ess_values,
        }

        alignments = {
            "Variable": "left",
            "Mean": "right",
            "Var": "right",
            "2.5": "right",
            "97.5": "right",
            "ESS": "right",
        }

        utils.print_summary(summary_dict, single_stats, alignments, digits, print_single_stats=False)


    def print_traceplot(self, color="black", linewidth=0.5, figsize=(22, 4), save_filename=None):
        """
        data_index: 데이터의 날짜 인덱스 제공 (plotting을 위해 임시로 받은것)
        """
        data_dict = {
            'eps': self.g_eps_draws,
            'dtau': self.g_dtau_draws,
            'ps': self.ps_draws,
        }
        utils.plot_traceplots(data_dict, rows=1, cols=3, figsize=figsize, filename=save_filename, linecolor=color, linewidth=linewidth)




    #################
    # Internal Method
    #################

    def _fit_ucsv_svrw_python(self, y: np.ndarray, T: int, nper: int, n_draws: int, thinning: int, n_burnin: int,
                            prior_a0: int = 0, prior_b0: int = 10, prior_Vomega: int = 1):
        """UCSV SVRW 단변량 모델을 적합한다
        
        Args:
            y(1d array): DATA
            T(int): number of observations
            nper(int): number of observations per year
            n_draws(int): total number of draws
            thinning(int): save one from `thinning`
            n_burnin(int): number of burn-in
            

            -- prior: h, h_tilde, g, g_tilde... --
            prior_a0(int): 0
            prior_b0(int): 10
            prior_Vomega(int): 1
            
        """
        # prior로 set
        a0 = prior_a0
        b0 = prior_b0
        Vomega = prior_Vomega
        
        used_draws = range(n_burnin, n_draws, thinning)  # take used_draws from range(n_draws)

        ### scale data
        scale_y = np.std(y[1:] - y[:-1])/5
        yn = y/scale_y

        # initialize the Markov chain
        h0 = np.log(stat.variance(y))/5
        g0 = np.log(stat.variance(y))/10
        tau0 = np.mean(y)
        omegah = np.sqrt(.2)
        omegag = np.sqrt(.2) / 5
        h_tilde = np.zeros(T)
        g_tilde = np.zeros(T)

        tau_draws = np.empty((n_draws, T))           # τ_t
        sigma_dtau_draws = np.empty((n_draws, T))    # σ_dτ,t = exp(0.5*g_t) = exp(0.5*(g0 +ω_g*̃g))
        sigma_eps_draws = np.empty((n_draws, T))     # σ_ε,t = exp(0.5*h_t) = exp(0.5*(h0 +ω_h*̃h))
        g_eps_draws = np.empty(n_draws)              # γ_ε or ω_ε
        g_dtau_draws = np.empty(n_draws)             # γ_dτ or ω_dτ
        scl_eps_draws = np.empty((n_draws, T))       # s_t
        sigma_total_eps_draws = np.empty((n_draws, T))# 
        ps_draws = np.empty(n_draws)                 # probability of outlier happening: 1-p

        # SET initial values for σ_et σ_τt, s_t: sigma = 1, scale_eps is unity
        sigma_eps = np.ones(T)
        sigma_dtau = np.ones(T)
        scale_eps = np.ones(T)

        ### scale mixture of epsilon component
        scl_eps_vec = np.arange(1, 11) # [1, 2, ... , 10]
        ps_prior = np.array([1-1/(4*nper), 1/(4*nper)]) * nper * 10

        # Initial value of ps
        n_scl_eps = len(scl_eps_vec)
        ps = ps_prior[0]/ps_prior.sum()
        prob_scl_eps_vec = ps*np.ones(n_scl_eps); prob_scl_eps_vec[1:] = (1-ps)/(n_scl_eps-1);


        # PRINT: tqdm
        if self.verbose:
            iters = tqdm(range(n_draws))
        else:
            iters = range(n_draws)

        for idraw in iters:
            
            # sample tau    
            sigma_eps_scl = sigma_eps*scale_eps
            tau, dtau, tau_f = uf.sample_tau(yn, sigma_dtau, sigma_eps_scl) # takes 1 ms
            
            ############ sampling tau method from sw 2016; takes 2.99 ms #############
            #     tau, dtau, tau_f = sample_tau(yn,sigma_dtau,sigma_eps_scl)
            ############ sampling tau method from chan 2018; takes 1.71 ms ###########
            #     part 1
            #     iOh = sparse.diags(1/sigma_eps_scl**2)
            #     HiOgH = H.T.dot(sparse.diags(1/sigma_dtau**2)).dot(H)
            #     Ktau =  HiOgH + iOh
            #     tau_hat = spsolve(Ktau, tau0*HiOgH.dot(ones(T)) + iOh.dot(yn))
            #     tau = tau_hat + spsolve(sparse_cholesky(Ktau).T, random.randn(T))
            #     part 2
            #     Ktau0 = 1/b0 + 1/sigma_dtau[0]**2
            #     tau0_hat = (a0/b0 + tau[0]/sigma_dtau[0]**2)/Ktau0
            #     tau0 = tau0_hat + random.randn()/sqrt(Ktau0)
            ########################################################################
            
            eps = yn - tau
            eps_scaled = eps/scale_eps    # Scaled version of epsilon
            
            # Step 1(b), 2(a), 2(b): Draw mixture indicators for log chi-squared(1)
            ln_e2 = np.log(eps_scaled**2 + 0.001)   # c = 0.001 factor from ksv, restud(1998), page 370)
            h_tilde, h0, omegah, omegah_hat, Domegah, ind_eps, sigma_eps = uf.SVRW(ln_e2, h_tilde, h0, omegah, a0, b0, Vomega)  # a0, b0, Vomega는 주어지는것
            
            ln_dtau2 = np.log((tau - np.append(tau0, tau[:-1]))**2 + 0.001);
            g_tilde, g0, omegag, omegag_hat, Domegag, ind_dtau, sigma_dtau = uf.SVRW(ln_dtau2, g_tilde, g0, omegag, a0, b0, Vomega)

            # Step 3: Draw Scale of epsilon
            scale_eps = uf.sample_scale_eps(eps, sigma_eps, ind_eps, scl_eps_vec, prob_scl_eps_vec)

            # Step 4; Draw probability of outlier;
            prob_scl_eps_vec = uf.sample_ps(scale_eps, ps_prior, n_scl_eps);

            # Save draws
            tau_draws[idraw] = tau
            sigma_dtau_draws[idraw] = sigma_dtau
            sigma_eps_draws[idraw] = sigma_eps
            g_eps_draws[idraw] = omegah
            g_dtau_draws[idraw] = omegag
            scl_eps_draws[idraw] = scale_eps
            sigma_total_eps_draws[idraw] = sigma_eps*scale_eps
            ps_draws[idraw] = prob_scl_eps_vec[0]  

        # 저장
        # info
        self.scale_y = scale_y
        self.used_draws = used_draws
        # sample
        self.tau_draws = tau_draws
        self.sigma_dtau_draws = sigma_dtau_draws
        self.sigma_eps_draws = sigma_eps_draws
        self.g_eps_draws = g_eps_draws
        self.g_dtau_draws = g_dtau_draws
        self.scl_eps_draws = scl_eps_draws
        self.sigma_total_eps_draws = sigma_total_eps_draws
        self.ps_draws = ps_draws

    
    def _fit_ucsv_svrw_cython(self, y: np.ndarray, T: int, nper: int, n_draws: int, thinning: int, n_burnin: int,
                            prior_a0: int = 0, prior_b0: int = 10, prior_Vomega: int = 1):
        """UCSV SVRW 단변량 모델을 적합한다 (cython 백엔드 사용)
        
        Args:
            y(1d array): DATA
            T(int): number of observations
            nper(int): number of observations per year
            n_draws(int): total number of draws
            thinning(int): save one from `thinning`
            n_burnin(int): number of burn-in
            

            -- prior: h, h_tilde, g, g_tilde... --
            prior_a0(int): 0
            prior_b0(int): 10
            prior_Vomega(int): 1
            
        """
        # prior로 set
        a0 = prior_a0
        b0 = prior_b0
        Vomega = prior_Vomega
        
        used_draws = range(n_burnin, n_draws, thinning)  # take used_draws from range(n_draws)

        ### scale data
        scale_y = np.std(y[1:] - y[:-1])/5
        yn = y/scale_y

        # initialize the Markov chain
        h0 = np.log(stat.variance(y))/5
        g0 = np.log(stat.variance(y))/10
        tau0 = np.mean(y)
        omegah = np.sqrt(.2)
        omegag = np.sqrt(.2) / 5
        h_tilde = np.zeros(T)
        g_tilde = np.zeros(T)

        # define a few things
        tau_draws = np.empty((n_draws, T))           # τ_t
        sigma_dtau_draws = np.empty((n_draws, T))    # σ_dτ,t = exp(0.5*g_t) = exp(0.5*(g0 +ω_g*̃g))
        sigma_eps_draws = np.empty((n_draws, T))     # σ_ε,t = exp(0.5*h_t) = exp(0.5*(h0 +ω_h*̃h))
        g_eps_draws = np.empty(n_draws)              # γ_ε or ω_ε
        g_dtau_draws = np.empty(n_draws)             # γ_dτ or ω_dτ
        scl_eps_draws = np.empty((n_draws, T))       # s_t
        sigma_total_eps_draws = np.empty((n_draws, T))# 
        ps_draws = np.empty(n_draws)                 # probability of outlier happening: 1-p

        # SET initial values for σ_et σ_τt, s_t: sigma = 1, scale_eps is unity
        sigma_eps = np.ones(T)
        sigma_dtau = np.ones(T)
        scale_eps = np.ones(T)

        ### scale mixture of epsilon component
        scl_eps_vec = np.arange(1, 11) # [1, 2, ... , 10]
        ps_prior = np.array([1-1/(4*nper), 1/(4*nper)]) * nper * 10

        # Initial value of ps
        n_scl_eps = len(scl_eps_vec)
        ps = ps_prior[0]/ps_prior.sum()
        prob_scl_eps_vec = ps*np.ones(n_scl_eps); prob_scl_eps_vec[1:] = (1-ps)/(n_scl_eps-1);

        # EDIT

        # PRINT: tqdm
        if self.verbose:
            iters = tqdm(range(n_draws))
        else:
            iters = range(n_draws)

        for idraw in iters:

        # for idraw in range(n_draws):
            
            # sample tau    
            sigma_eps_scl = sigma_eps*scale_eps
            # uf_cython에서 cython 으로 빌드한 함수 호출
            tau, dtau, tau_f = self.uf_cython.sample_tau(yn, sigma_dtau, sigma_eps_scl) # takes 1 ms
            
            ############ sampling tau method from sw 2016; takes 2.99 ms #############
            #     tau, dtau, tau_f = sample_tau(yn,sigma_dtau,sigma_eps_scl)
            ############ sampling tau method from chan 2018; takes 1.71 ms ###########
            #     part 1
            #     iOh = sparse.diags(1/sigma_eps_scl**2)
            #     HiOgH = H.T.dot(sparse.diags(1/sigma_dtau**2)).dot(H)
            #     Ktau =  HiOgH + iOh
            #     tau_hat = spsolve(Ktau, tau0*HiOgH.dot(ones(T)) + iOh.dot(yn))
            #     tau = tau_hat + spsolve(sparse_cholesky(Ktau).T, random.randn(T))
            #     part 2
            #     Ktau0 = 1/b0 + 1/sigma_dtau[0]**2
            #     tau0_hat = (a0/b0 + tau[0]/sigma_dtau[0]**2)/Ktau0
            #     tau0 = tau0_hat + random.randn()/sqrt(Ktau0)
            ########################################################################
            
            eps = yn - tau
            eps_scaled = eps/scale_eps    # Scaled version of epsilon
            
            # Step 1(b), 2(a), 2(b): Draw mixture indicators for log chi-squared(1)
            ln_e2 = np.log(eps_scaled**2 + 0.001)   # c = 0.001 factor from ksv, restud(1998), page 370)
            h_tilde, h0, omegah, omegah_hat, Domegah, ind_eps, sigma_eps = uf.SVRW(ln_e2, h_tilde, h0, omegah, a0, b0, Vomega)  # a0, b0, Vomega는 주어지는것
            # h = h0 + omegah*h_tilde
            
            ln_dtau2 = np.log((tau - np.append(tau0, tau[:-1]))**2 + 0.001);
            g_tilde, g0, omegag, omegag_hat, Domegag, ind_dtau, sigma_dtau = uf.SVRW(ln_dtau2, g_tilde, g0, omegag, a0, b0, Vomega)
            # g = g0 + omegag*g_tilde
            # NOTE: 

            # Step 3: Draw Scale of epsilon
            scale_eps = uf.sample_scale_eps(eps, sigma_eps, ind_eps, scl_eps_vec, prob_scl_eps_vec)

            # Step 4; Draw probability of outlier;
            prob_scl_eps_vec = uf.sample_ps(scale_eps, ps_prior, n_scl_eps);

            # Save draws
            tau_draws[idraw] = tau
            sigma_dtau_draws[idraw] = sigma_dtau
            sigma_eps_draws[idraw] = sigma_eps
            g_eps_draws[idraw] = omegah
            g_dtau_draws[idraw] = omegag
            scl_eps_draws[idraw] = scale_eps
            sigma_total_eps_draws[idraw] = sigma_eps*scale_eps
            ps_draws[idraw] = prob_scl_eps_vec[0]  

        # 저장
        # info
        self.scale_y = scale_y
        self.used_draws = used_draws
        # sample
        self.tau_draws = tau_draws
        self.sigma_dtau_draws = sigma_dtau_draws
        self.sigma_eps_draws = sigma_eps_draws
        self.g_eps_draws = g_eps_draws
        self.g_dtau_draws = g_dtau_draws
        self.scl_eps_draws = scl_eps_draws
        self.sigma_total_eps_draws = sigma_total_eps_draws
        self.ps_draws = ps_draws



    def _find_first_non_positive(self, sequence):
        """
        주어진 시퀀스에서 연속하는 짝수번째와 홀수번째 숫자의 합이 최초로 0 이하로 떨어지는 위치를 반환합니다.

        Parameters
        ----------
        sequence : list
        숫자로 이루어진 리스트.

        Returns
        -------
        int: 최초로 0 이하로 떨어지는 위치 (인덱스). 찾지 못하면 len(sequence)를 반환합니다.
        """
        for i in range(0, len(sequence) - 1, 2):
            # 짝수번째와 홀수번째 숫자의 합 계산
            total_sum = sequence[i] + sequence[i + 1]
            
            if total_sum <= 0:
                return i  # 짝수번째 숫자의 인덱스 반환
        
        return len(sequence)  # 찾지 못했을 경우 전체 인덱스의 길이 반환

    def _autocorrelation(self, samples, lag):
        n = len(samples)
        mean = np.mean(samples)
        
        # 자기상관 계산
        numerator = np.sum((samples[:n - lag] - mean) * (samples[lag:] - mean))
        denominator = np.sum((samples - mean) ** 2)  # NOTE: 여기도 warn / inf
        
        return numerator / denominator

    def _effective_sample_size(self, samples):
        n = len(samples)
        max_lag = self._find_first_non_positive([self._autocorrelation(samples, lag) for lag in range(1, len(samples))])
        acf_values = [self._autocorrelation(samples, lag) for lag in range(1, max_lag + 1)]
        ess = n / (1 + 2 * np.sum(acf_values))
        return ess

    
    def _calculate_summary_stats(self, y: np.ndarray) -> None:
        """
        모델 평가를 위한 통계량들을 계산하여 self.summary_stats에 저장

        Parameters
        ----------
        y : numpy array
            타겟 레이블 벡터
        """
        # 각 통계량 유형별로 딕셔너리 안에 key를 파라미터명으로, 안의 값을 np.array로 저장 (저장 시에 used_draws만 사용하도록 )
        # NOTE: 단변량이므로 단순 np.mean
        posterior_mean = {
            "g_eps": np.mean(self.g_eps_draws[self.used_draws]),
            "g_dtau": np.mean(self.g_dtau_draws[self.used_draws]),
            "ps": np.mean(self.ps_draws[self.used_draws])
        }

        posterior_variance = {
            "g_eps": np.var(self.g_eps_draws[self.used_draws]),
            "g_dtau": np.var(self.g_dtau_draws[self.used_draws]),
            "ps": np.var(self.ps_draws[self.used_draws])
        }

        hpd_lower_bound = {
            "g_eps": np.percentile(self.g_eps_draws[self.used_draws], 2.5),
            "g_dtau": np.percentile(self.g_dtau_draws[self.used_draws], 2.5),
            "ps": np.percentile(self.ps_draws[self.used_draws], 2.5)
        }

        hpd_upper_bound = {
            "g_eps": np.percentile(self.g_eps_draws[self.used_draws], 97.5),
            "g_dtau": np.percentile(self.g_dtau_draws[self.used_draws], 97.5),
            "ps": np.percentile(self.ps_draws[self.used_draws], 97.5)
        }

        ESS = {
            "g_eps": self._effective_sample_size(self.g_eps_draws[self.used_draws]),
            "g_dtau": self._effective_sample_size(self.g_dtau_draws[self.used_draws]),
            "ps": self._effective_sample_size(self.ps_draws[self.used_draws])
        }

        # 저장
        self.summary_stats = {
            "posterior_mean": posterior_mean,
            "posterior_variance": posterior_variance,
            "hpd_lower_bound": hpd_lower_bound,
            "hpd_upper_bound": hpd_upper_bound,
            "ESS": ESS,
        }