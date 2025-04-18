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
from . import ucsv_functions_multivar as ufm
from . import load_cython
from .. import utils


from typing import Literal, Union


class MultivarUCSV:
    """
    """

    def __init__(self, run_type: Literal['python', 'cython'] = 'python', 
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
                self.uf_cython_multivar = load_cython.load_cython_ucsv_multivar_functions()
                print("> Cython Module Loaded.")
            else:
                self.uf_cython_multivar = load_cython.load_cython_ucsv_multivar_functions()

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

    def fit(self, Y: Union[np.ndarray, pd.Series], columns: list[str] = [],
            # n_per_year: int = 4, n_draws: int = 50000, thinning: int = 10, n_burn_in: int = 5000
            ) -> 'MultivarUCSV':
        """

        Parameters
        ----------
        Y : numay array of shape (n_variables, n_samples) or pd.DataFrame(col: variables, row: samples)
            입력 데이터 벡터 (다변량 데이터 입력)
        columns : list of str, optional
            Y의 컬럼 명칭(결과 출력 시 활용), 데이터프레임인 경우 기본값은 Y.columns로 사용, 컬럼명 지정이 없을 경우 var1~varN으로 자동 설정

        
        Returns
        -------
        self : object
            객체 자기 자신을 반환
        """
        # 2) 입력값이 데이터프레임인 경우 numpy로 변환
        if isinstance(Y, pd.DataFrame):
            Y = Y.to_numpy().T  # (n_samples, n_variables) -> (n_variables, n_samples)

        # 1) 컬럼명 설정
        if (not columns) and (isinstance(Y, pd.DataFrame)):
            self.columns = Y.columns.tolist()  # 컬럼명이 따로 정해지지 않고 데이터프레임이 입력된 경우, 데이터프레임의 컬럼명을 사용
        elif columns:
            self.columns = columns  # 컬럼명이 입력된 경우 컬럼명 사용
        else:
            self.columns = [f"var{i+1}" for i in range(Y.shape[0])]  # 컬럼명이 없는 경우 var1~varN로 자동 생성
        self.columns = ["common"] + self.columns  # 맨 앞에 common effect 를 위한 컬럼명 추가


        # 3) numpy array에 문제가 없는지 확인
        # self._check_input_array(y)


        # PRINT: start
        if self.verbose:
            print("> Start Fitting Model...")
            print(f"  - Input Data: {Y.shape[0]} variables, {Y.shape[1]} samples")

        # 5) 적합
        if self.run_type == 'python':
            self._fit_ucsv_svrw_multivar_python(Y=Y, T=len(Y), nper=self.n_per_year, n_draws=self.n_draws, thinning=self.thinning, n_burnin=self.n_burnin)
        elif self.run_type == 'cython':
            self._fit_ucsv_svrw_multivar_cython(Y=Y, T=len(Y), nper=self.n_per_year, n_draws=self.n_draws, thinning=self.thinning, n_burnin=self.n_burnin)
            # self._fit_ucsv_svrw_multivar_cython(Y=Y, T=len(Y), nper=n_per_year, n_draws=n_draws, thinning=thinning, n_burnin=n_burn_in)
        else:
            raise ValueError(f"지원하지 않는 run_type 입니다 | run_type: {self.run_type}")


        # 6) summary stats 계산
        self.is_fitted = True # fit 완료 이후 flag 설정

        # stats 계산
        self._calculate_summary_stats(Y)

        # PRINT: Done
        if self.verbose:
            print("> Model Fitted.")

        # 반환
        return self


    def print_summary(self, digits: int=4):
        to_print = ["g_eps", "g_dtau", "ps"]

        # 계수와 관련된 통계량을 준비
        variables_names = []  # 계수명은 일단은 컬럼 하나로 합침 (g_eps (common), g_eps (var1), ... ps (common), ps (var1), ...)
        for target in to_print:
            for col in self.columns:
                name = f"{target} ({col})"  # eg. g_eps (var1)
                variables_names.append(name)

        stat_targets = ["posterior_mean", "posterior_variance", "hpd_lower_bound", "hpd_upper_bound", "ESS"]
        values = {s: [] for s in stat_targets}

        for target in to_print:
            for stst_target in stat_targets:
                for value in self.summary_stats[stst_target][target]:
                    values[stst_target].append(value)
        # 단일 통계량
        single_stats = {
        }

        summary_dict = {
            "Variable": variables_names,
            "Mean": values["posterior_mean"],
            "Var": values["posterior_variance"],
            "2.5": values["hpd_lower_bound"],
            "97.5": values["hpd_upper_bound"],
            "ESS": values["ESS"],
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
        Traceplot을 출력하는 함수

        각 변수의 MCMC 샘플에 대한 traceplot을 생성합니다.
        """
        for i, col in enumerate(self.columns):
            data_dict = {
                'eps': self.g_eps_draws[:, i],
                'dtau': self.g_dtau_draws[:, i],
                'ps': self.ps_draws[:, i],
            }
            utils.plot_traceplots(data_dict, rows=1, cols=3, figsize=figsize, suptitle=col, filename=save_filename, linecolor=color, linewidth=linewidth)


    #################
    # Internal Method
    #################


    def _fit_ucsv_svrw_multivar_python(self, Y: np.ndarray, T: int, nper: int, n_draws: int, thinning: int, n_burnin: int,
                            prior_a0: int = 0, prior_b0: int = 10, prior_Vomega: int = 1):
        """UCSV SVRW 다변량 모델을 적합한다
        
        Args:
            Y(2d array): DATA
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
        n_y = Y.shape[0]           # number of observed variables
        N = n_y + 1                     # number of observed variables + 1 (for common factor)
        self.N = N
        T = Y.shape[1]             # number of observations
        nper = 4                        # number of oservations per year

        ### scale data
        sd_ddp_median = np.median(np.std(Y[:, 1:] - Y[:, :-1], axis=1))
        scale_y = sd_ddp_median/5
        Y_scaled = Y/scale_y


        ### LOOP
        self.used_draws = range(n_burnin, n_draws, thinning)  # take used_draws from range(n_draws)


        ############################################################################
        # 10-component mixture from Omori, Chib, Shephard, and Nakajima JOE (2007)
        r_p = np.array([0.00609, 0.04775, 0.13057, 0.20674, 0.22715, 0.18842, 0.12047, 0.05591, 0.01575, 0.00115])
        r_m = np.array([1.92677, 1.34744, 0.73504, 0.02266, -0.85173, -1.97278, -3.46788, -5.55246, -8.68384, -14.65000])
        r_v = np.array([0.11265, 0.17788, 0.26768, 0.40611, 0.62699, 0.98583, 1.57469, 2.54498, 4.16591, 7.33342])
        r_s = np.sqrt(r_v)
        ############################################################################

        ### scale mixture of epsilon component
        scl_eps_vec = np.arange(1, 11) # [1, 2, ... , 10]  # 10개인건 위에것과 연결되는듯
        ps_prior = np.array([1 - 1/(4*nper), 1/(4*nper)]) * nper * 10


        ############################################################################
        ### Parameters for prior for factor loadings -- note these
        ############################################################################
        ### depend on scaling (scale_y) introduced above, so they are in scaled y
        # units
        # ... inital values of factor loadings
        omega_tau = 10/scale_y
        omega_eps = 10/scale_y
        sigma_tau = 0.4/scale_y
        sigma_eps = 0.4/scale_y
        var_alpha_tau = (omega_tau**2)*np.ones((n_y, n_y)) + (sigma_tau**2)*np.eye(n_y)
        var_alpha_eps = (omega_eps**2)*np.ones((n_y, n_y)) + (sigma_eps**2)*np.eye(n_y)
        prior_var_alpha = np.zeros((2*n_y, 2*n_y))
        prior_var_alpha[:n_y, :n_y] = var_alpha_eps
        prior_var_alpha[n_y:, n_y:] = var_alpha_tau

        # Alpha TVP parameters -- use "Number of prior obs (nu) and prior squared (s2)" as parameters
        # .. as in Del Negro and Otrok;
        nu_prior_alpha = 0.1*T             
        s2_prior_alpha = (0.25/np.sqrt(T))**2/(scale_y**2);

        # Initial values for sigma_alpha
        var_dalpha = 1/np.random.gamma(nu_prior_alpha/2, 2/(nu_prior_alpha*s2_prior_alpha), 2*n_y)
        sigma_dalpha = np.sqrt(var_dalpha)


        ############################################################################
        # Initial Values of parameters -- I save alpha values for each date, so the
        # same program can be used when TVP is allowed
        alpha_eps = ufm.nsample2(T, n_y).T
        alpha_tau = ufm.nsample2(T, n_y).T
        sigma_eps_common = ufm.nsample(T)
        sigma_eps_unique = ufm.nsample2(T, n_y).T
        sigma_dtau_common = ufm.nsample(T)
        sigma_dtau_unique = ufm.nsample2(T, n_y).T
        scale_eps = np.ones((N, T))

        # Initial Value of ps
        n_scl_eps = len(scl_eps_vec)
        ps = ps_prior[0]/ps_prior.sum()
        prob_scl_eps_vec = ps*np.ones(n_scl_eps)
        prob_scl_eps_vec[1:] = (1 - ps)/(n_scl_eps-1)
        prob_scl_eps_vec = np.repeat(prob_scl_eps_vec[None, :], N, axis=0)


        ############################################################################
        ## Parameters for h and g
        ############################################################################
        # prior: h, h_tilde, g, g_tilde...
        a0 = prior_a0
        b0 = prior_b0
        Vomega = prior_Vomega

        h0 = np.log(Y.var(axis=1)) / 5; h0 = np.concatenate(([min(h0)], h0))                  
        g0 = np.log(Y.var(axis=1)) / 10; g0 = np.concatenate(([min(g0)], g0))
        tau0 = Y.mean(axis=1); tau0 = np.concatenate(([min(tau0)], tau0))

        omegah = np.sqrt(.2)*np.ones(4)
        omegag = np.sqrt(.2)*np.ones(4)

        h_tilde = np.zeros((N, T))
        g_tilde = np.zeros((N, T))

        h = h0[:, None] + omegah[:, None]*h_tilde
        g = g0[:, None] + omegah[:, None]*g_tilde 

        omegah_hat = np.zeros(N)
        omegag_hat = np.zeros(N)

        Domegah = np.zeros(N)
        Domegag = np.zeros(N)

        ind_eps = np.zeros((N, T, len(r_p)))
        ind_dtau = np.zeros((N, T, len(r_p)))

        sigma_eps = np.concatenate((sigma_eps_common[None, :], sigma_eps_unique), axis=0)
        sigma_dtau = np.concatenate((sigma_dtau_common[None, :], sigma_dtau_unique), axis=0)

        ############################################################################
        ## Matrices for saving draws
        ############################################################################
        # -- Standard Deviations 
        sigma_eps_draws = np.empty((n_draws, N, T))
        sigma_dtau_draws = np.empty((n_draws, N, T))

        # -- Standard Deviations 
        tau_draws = np.empty((n_draws, N, T))
        dtau_draws = np.empty((n_draws, N, T))
        tau_f_draws = np.empty((n_draws, N, T))

        # --- Scale for outliers in eps
        scale_eps_draws = np.empty((n_draws, N, T))
        sigma_eps_total_draws = np.empty((n_draws, N, T))
        ps_draws = np.empty((n_draws, N))

        # Factor Loadings
        alpha_eps_draws = np.empty((n_draws, n_y, T))
        alpha_tau_draws = np.empty((n_draws, n_y, T))

        # g-values 
        g_eps_draws = np.empty((n_draws, N))
        g_dtau_draws = np.empty((n_draws, N))

        # Decomposition of series
        y_eps_common_draws = np.empty((n_draws, n_y, T))
        y_eps_unique_draws = np.empty((n_draws, n_y, T))
        y_tau_common_draws = np.empty((n_draws, n_y, T))
        y_tau_unique_draws = np.empty((n_draws, n_y, T))
        y_f_tau_common_draws = np.empty((n_draws, n_y, T))
        y_f_tau_unique_draws = np.empty((n_draws, n_y, T))

        # Decomposition of variance for each series
        var_y_eps_common_draws = np.empty((n_draws, n_y, T))
        var_y_eps_common_total_draws = np.empty((n_draws, n_y, T))
        var_y_dtau_common_draws = np.empty((n_draws, n_y, T))
        var_y_eps_unique_draws = np.empty((n_draws, n_y, T))
        var_y_eps_unique_total_draws = np.empty((n_draws, n_y, T))
        var_y_dtau_unique_draws = np.empty((n_draws, n_y, T))

        # Sigma dalpha
        sigma_dalpha_draws = np.empty((n_draws, 2*n_y))


        # PRINT: tqdm
        if self.verbose:
            iters = tqdm(range(n_draws))
        else:
            iters = range(n_draws)

        for idraw in iters:
            
            # SD of eps_unique and eps_common, which is stochastic volatility times scale in mixture distribution 
            sigma_eps_scl = sigma_eps*scale_eps

            # Step 1.a.1: draw tau, tau_f, dtau, and eps
            eps_common, tau_f, dtau, tau = ufm.sample_eps_tau_multivar(Y_scaled, alpha_eps, alpha_tau,
                                                                sigma_eps_scl, sigma_dtau)

            ### Step 1.a.2 : Draw Factor Loadings 
            # -- Draw alpha_eps and alpha_tau;
            alpha_eps, alpha_tau, dalpha = ufm.sample_alpha_tvp_multivar(Y_scaled, prior_var_alpha, sigma_dalpha, tau,
                                                                eps_common, sigma_eps_scl[1:])

            # Step 1.a.3: Draw Standard Deviations of Alpha TVPs;
            sigma_dalpha = ufm.sample_dalpha_sigma(dalpha, nu_prior_alpha, s2_prior_alpha)

            ### Step 1(b): Draw mixture indicators for log chi-squared(1)
            eps_unique = Y_scaled - alpha_eps*eps_common[None, :] - alpha_tau*tau[0][None, :] - tau[1:]
            eps = np.concatenate((eps_common[None, :], eps_unique), axis=0)
            eps_scaled = eps/scale_eps

            ln_e2 = np.log(eps_scaled**2 + 0.001)   # c = 0.001 factor from ksv, restud(1998), page 370)
            for i in range(N):
                h_tilde[i], h0[i], omegah[i], omegah_hat[i], Domegah[i], ind_eps[i], sigma_eps[i] = uf.SVRW(ln_e2[i], h_tilde[i], h0[i], omegah[i], a0, b0, Vomega)
                h[i] = h0[i] + omegah[i]*h_tilde[i]

            ln_e2 = np.log(dtau**2 + 0.001)   # c = 0.001 factor from ksv, restud(1998), page 370)
            for i in range(N):
                g_tilde[i], g0[i], omegag[i], omegag_hat[i], Domegag[i], ind_dtau[i], sigma_dtau[i] = uf.SVRW(ln_e2[i], g_tilde[i], g0[i], omegag[i], a0, b0, Vomega)
                g[i] = g0[i] + omegag[i]*g_tilde[i]

            # Step 3: Draw Scale of epsilon
            for i in range(N):
                scale_eps[i] = uf.sample_scale_eps(eps[i], sigma_eps[i], ind_eps[i], scl_eps_vec, prob_scl_eps_vec[i])

            # Step 4; Draw probability of outlier;
            for i in range(N):
                prob_scl_eps_vec[i] = uf.sample_ps(scale_eps[i], ps_prior, n_scl_eps);
                
            # Save draws
            sigma_eps_draws[idraw] = sigma_eps
            sigma_dtau_draws[idraw] = sigma_dtau
            scale_eps_draws[idraw] = scale_eps
            sigma_eps_total_draws[idraw] = sigma_eps*scale_eps
            alpha_eps_draws[idraw] = alpha_eps
            alpha_tau_draws[idraw] = alpha_tau
            g_eps_draws[idraw] = omegah
            g_dtau_draws[idraw] = omegag
            ps_draws[idraw] = prob_scl_eps_vec[:, 0]
            tau_draws[idraw] = tau
            dtau_draws[idraw] = dtau
            tau_f_draws[idraw] = tau_f
            sigma_dalpha_draws[idraw] = sigma_dalpha
            y_eps_common_draws[idraw] = alpha_eps*eps_common[None, :]
            y_eps_unique_draws[idraw] = eps_unique
            y_tau_common_draws[idraw] = alpha_tau*tau[0][None, :]
            y_tau_unique_draws[idraw] = tau[1:]
            y_f_tau_common_draws[idraw] = alpha_tau*tau_f[0][None, :]
            y_f_tau_unique_draws[idraw] = tau_f[1:]
            var_y_eps_common_draws[idraw] = (alpha_eps*sigma_eps[0][None, :])**2
            var_y_eps_common_total_draws[idraw] = (alpha_eps*(sigma_eps[0]*scale_eps[0])[None, :])**2
            var_y_dtau_common_draws[idraw] = (alpha_tau*sigma_dtau[0][None, :])**2
            var_y_eps_unique_draws[idraw] = sigma_eps[1:]**2
            var_y_eps_unique_total_draws[idraw] = (sigma_eps[1:]*scale_eps[1:])**2
            var_y_dtau_unique_draws[idraw] = sigma_dtau[1:]**2            

        # 저장
        self.sigma_eps_draws = sigma_eps_draws
        self.sigma_dtau_draws = sigma_dtau_draws
        self.scale_eps_draws = scale_eps_draws
        self.sigma_eps_total_draws = sigma_eps_total_draws
        self.alpha_eps_draws = alpha_eps_draws
        self.alpha_tau_draws = alpha_tau_draws
        self.g_eps_draws = g_eps_draws
        self.g_dtau_draws = g_dtau_draws
        self.ps_draws = ps_draws
        self.tau_draws = tau_draws
        self.dtau_draws = dtau_draws
        self.tau_f_draws = tau_f_draws
        self.sigma_dalpha_draws = sigma_dalpha_draws
        self.y_eps_common_draws = y_eps_common_draws
        self.y_eps_unique_draws = y_eps_unique_draws
        self.y_tau_common_draws = y_tau_common_draws
        self.y_tau_unique_draws = y_tau_unique_draws
        self.y_f_tau_common_draws = y_f_tau_common_draws
        self.y_f_tau_unique_draws = y_f_tau_unique_draws
        self.var_y_eps_common_draws = var_y_eps_common_draws
        self.var_y_eps_common_total_draws = var_y_eps_common_total_draws
        self.var_y_dtau_common_draws = var_y_dtau_common_draws
        self.var_y_eps_unique_draws = var_y_eps_unique_draws        
        self.var_y_eps_unique_total_draws = var_y_eps_unique_total_draws
        self.var_y_dtau_unique_draws = var_y_dtau_unique_draws
        

    
    def _fit_ucsv_svrw_multivar_cython(self, Y: np.ndarray, T: int, nper: int, n_draws: int, thinning: int, n_burnin: int,
                            prior_a0: int = 0, prior_b0: int = 10, prior_Vomega: int = 1):
        """UCSV SVRW 다변량 모델을 적합한다
        
        Args:
            Y(2d array): DATA
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
        n_y = Y.shape[0]           # number of observed variables
        N = n_y + 1                     # number of observed variables + 1 (for common factor)
        self.N = N
        T = Y.shape[1]             # number of observations
        nper = 4                        # number of oservations per year

        ### scale data
        sd_ddp_median = np.median(np.std(Y[:, 1:] - Y[:, :-1], axis=1))
        scale_y = sd_ddp_median/5
        Y_scaled = Y/scale_y


        ### LOOP
        self.used_draws = range(n_burnin, n_draws, thinning)  # take used_draws from range(n_draws)

        ############################################################################
        # 10-component mixture from Omori, Chib, Shephard, and Nakajima JOE (2007)
        r_p = np.array([0.00609, 0.04775, 0.13057, 0.20674, 0.22715, 0.18842, 0.12047, 0.05591, 0.01575, 0.00115])
        r_m = np.array([1.92677, 1.34744, 0.73504, 0.02266, -0.85173, -1.97278, -3.46788, -5.55246, -8.68384, -14.65000])
        r_v = np.array([0.11265, 0.17788, 0.26768, 0.40611, 0.62699, 0.98583, 1.57469, 2.54498, 4.16591, 7.33342])
        r_s = np.sqrt(r_v)
        ############################################################################

        ### scale mixture of epsilon component
        scl_eps_vec = np.arange(1, 11) # [1, 2, ... , 10]  # 10개인건 위에것과 연결되는듯
        ps_prior = np.array([1 - 1/(4*nper), 1/(4*nper)]) * nper * 10


        ############################################################################
        ### Parameters for prior for factor loadings -- note these
        ############################################################################
        ### depend on scaling (scale_y) introduced above, so they are in scaled y
        # units
        # ... inital values of factor loadings
        omega_tau = 10/scale_y
        omega_eps = 10/scale_y
        sigma_tau = 0.4/scale_y
        sigma_eps = 0.4/scale_y
        var_alpha_tau = (omega_tau**2)*np.ones((n_y, n_y)) + (sigma_tau**2)*np.eye(n_y)
        var_alpha_eps = (omega_eps**2)*np.ones((n_y, n_y)) + (sigma_eps**2)*np.eye(n_y)
        prior_var_alpha = np.zeros((2*n_y, 2*n_y))
        prior_var_alpha[:n_y, :n_y] = var_alpha_eps
        prior_var_alpha[n_y:, n_y:] = var_alpha_tau

        # Alpha TVP parameters -- use "Number of prior obs (nu) and prior squared (s2)" as parameters
        # .. as in Del Negro and Otrok;
        nu_prior_alpha = 0.1*T             
        s2_prior_alpha = (0.25/np.sqrt(T))**2/(scale_y**2);

        # Initial values for sigma_alpha
        var_dalpha = 1/np.random.gamma(nu_prior_alpha/2, 2/(nu_prior_alpha*s2_prior_alpha), 2*n_y)
        sigma_dalpha = np.sqrt(var_dalpha)


        ############################################################################
        # Initial Values of parameters -- I save alpha values for each date, so the
        # same program can be used when TVP is allowed
        alpha_eps = ufm.nsample2(T, n_y).T
        alpha_tau = ufm.nsample2(T, n_y).T
        sigma_eps_common = ufm.nsample(T)
        sigma_eps_unique = ufm.nsample2(T, n_y).T
        sigma_dtau_common = ufm.nsample(T)
        sigma_dtau_unique = ufm.nsample2(T, n_y).T
        scale_eps = np.ones((N, T))

        # Initial Value of ps
        n_scl_eps = len(scl_eps_vec)
        ps = ps_prior[0]/ps_prior.sum()
        prob_scl_eps_vec = ps*np.ones(n_scl_eps)
        prob_scl_eps_vec[1:] = (1 - ps)/(n_scl_eps-1)
        prob_scl_eps_vec = np.repeat(prob_scl_eps_vec[None, :], N, axis=0)


        ############################################################################
        ## Parameters for h and g
        ############################################################################
        a0 = prior_a0
        b0 = prior_b0
        Vomega = prior_Vomega

        h0 = np.log(Y.var(axis=1)) / 5; h0 = np.concatenate(([min(h0)], h0))                  
        g0 = np.log(Y.var(axis=1)) / 10; g0 = np.concatenate(([min(g0)], g0))
        tau0 = Y.mean(axis=1); tau0 = np.concatenate(([min(tau0)], tau0))

        omegah = np.sqrt(.2)*np.ones(4)
        omegag = np.sqrt(.2)*np.ones(4)

        h_tilde = np.zeros((N, T))
        g_tilde = np.zeros((N, T))

        h = h0[:, None] + omegah[:, None]*h_tilde
        g = g0[:, None] + omegah[:, None]*g_tilde 

        omegah_hat = np.zeros(N)
        omegag_hat = np.zeros(N)

        Domegah = np.zeros(N)
        Domegag = np.zeros(N)

        ind_eps = np.zeros((N, T, len(r_p)))
        ind_dtau = np.zeros((N, T, len(r_p)))

        sigma_eps = np.concatenate((sigma_eps_common[None, :], sigma_eps_unique), axis=0)
        sigma_dtau = np.concatenate((sigma_dtau_common[None, :], sigma_dtau_unique), axis=0)

        ############################################################################
        ## Matrices for saving draws
        ############################################################################
        # -- Standard Deviations 
        sigma_eps_draws = np.empty((n_draws, N, T))
        sigma_dtau_draws = np.empty((n_draws, N, T))

        # -- Standard Deviations 
        tau_draws = np.empty((n_draws, N, T))
        dtau_draws = np.empty((n_draws, N, T))
        tau_f_draws = np.empty((n_draws, N, T))

        # --- Scale for outliers in eps
        scale_eps_draws = np.empty((n_draws, N, T))
        sigma_eps_total_draws = np.empty((n_draws, N, T))
        ps_draws = np.empty((n_draws, N))

        # Factor Loadings
        alpha_eps_draws = np.empty((n_draws, n_y, T))
        alpha_tau_draws = np.empty((n_draws, n_y, T))

        # g-values 
        g_eps_draws = np.empty((n_draws, N))
        g_dtau_draws = np.empty((n_draws, N))

        # Decomposition of series
        y_eps_common_draws = np.empty((n_draws, n_y, T))
        y_eps_unique_draws = np.empty((n_draws, n_y, T))
        y_tau_common_draws = np.empty((n_draws, n_y, T))
        y_tau_unique_draws = np.empty((n_draws, n_y, T))
        y_f_tau_common_draws = np.empty((n_draws, n_y, T))
        y_f_tau_unique_draws = np.empty((n_draws, n_y, T))

        # Decomposition of variance for each series
        var_y_eps_common_draws = np.empty((n_draws, n_y, T))
        var_y_eps_common_total_draws = np.empty((n_draws, n_y, T))
        var_y_dtau_common_draws = np.empty((n_draws, n_y, T))
        var_y_eps_unique_draws = np.empty((n_draws, n_y, T))
        var_y_eps_unique_total_draws = np.empty((n_draws, n_y, T))
        var_y_dtau_unique_draws = np.empty((n_draws, n_y, T))

        # Sigma dalpha
        sigma_dalpha_draws = np.empty((n_draws, 2*n_y))


        # PRINT: tqdm
        if self.verbose:
            iters = tqdm(range(n_draws))
        else:
            iters = range(n_draws)

        for idraw in iters:
            
            # SD of eps_unique and eps_common, which is stochastic volatility times scale in mixture distribution 
            sigma_eps_scl = sigma_eps*scale_eps

            # Step 1.a.1: draw tau, tau_f, dtau, and eps
            # print("Cython1")
            eps_common, tau_f, dtau, tau = ufm.sample_eps_tau_multivar_cython(self.uf_cython_multivar, Y_scaled, alpha_eps, alpha_tau,
                                                                sigma_eps_scl, sigma_dtau)

            ### Step 1.a.2 : Draw Factor Loadings 
            # -- Draw alpha_eps and alpha_tau;
            # print("Cython2")
            alpha_eps, alpha_tau, dalpha = ufm.sample_alpha_tvp_multivar_cython(self.uf_cython_multivar, Y_scaled, prior_var_alpha, sigma_dalpha, tau,
                                                                eps_common, sigma_eps_scl[1:])

            # Step 1.a.3: Draw Standard Deviations of Alpha TVPs;
            sigma_dalpha = ufm.sample_dalpha_sigma(dalpha, nu_prior_alpha, s2_prior_alpha)

            ### Step 1(b): Draw mixture indicators for log chi-squared(1)
            eps_unique = Y_scaled - alpha_eps*eps_common[None, :] - alpha_tau*tau[0][None, :] - tau[1:]
            eps = np.concatenate((eps_common[None, :], eps_unique), axis=0)
            eps_scaled = eps/scale_eps

            ln_e2 = np.log(eps_scaled**2 + 0.001)   # c = 0.001 factor from ksv, restud(1998), page 370)
            for i in range(N):
                h_tilde[i], h0[i], omegah[i], omegah_hat[i], Domegah[i], ind_eps[i], sigma_eps[i] = uf.SVRW(ln_e2[i], h_tilde[i], h0[i], omegah[i], a0, b0, Vomega)
                h[i] = h0[i] + omegah[i]*h_tilde[i]

            ln_e2 = np.log(dtau**2 + 0.001)   # c = 0.001 factor from ksv, restud(1998), page 370)
            for i in range(N):
                g_tilde[i], g0[i], omegag[i], omegag_hat[i], Domegag[i], ind_dtau[i], sigma_dtau[i] = uf.SVRW(ln_e2[i], g_tilde[i], g0[i], omegag[i], a0, b0, Vomega)
                g[i] = g0[i] + omegag[i]*g_tilde[i]

            # Step 3: Draw Scale of epsilon
            for i in range(N):
                scale_eps[i] = uf.sample_scale_eps(eps[i], sigma_eps[i], ind_eps[i], scl_eps_vec, prob_scl_eps_vec[i])

            # Step 4; Draw probability of outlier;
            for i in range(N):
                prob_scl_eps_vec[i] = uf.sample_ps(scale_eps[i], ps_prior, n_scl_eps);
                
            # Save draws
            sigma_eps_draws[idraw] = sigma_eps
            sigma_dtau_draws[idraw] = sigma_dtau
            scale_eps_draws[idraw] = scale_eps
            sigma_eps_total_draws[idraw] = sigma_eps*scale_eps
            alpha_eps_draws[idraw] = alpha_eps
            alpha_tau_draws[idraw] = alpha_tau
            g_eps_draws[idraw] = omegah
            g_dtau_draws[idraw] = omegag
            ps_draws[idraw] = prob_scl_eps_vec[:, 0]
            tau_draws[idraw] = tau
            dtau_draws[idraw] = dtau
            tau_f_draws[idraw] = tau_f
            sigma_dalpha_draws[idraw] = sigma_dalpha
            y_eps_common_draws[idraw] = alpha_eps*eps_common[None, :]
            y_eps_unique_draws[idraw] = eps_unique
            y_tau_common_draws[idraw] = alpha_tau*tau[0][None, :]
            y_tau_unique_draws[idraw] = tau[1:]
            y_f_tau_common_draws[idraw] = alpha_tau*tau_f[0][None, :]
            y_f_tau_unique_draws[idraw] = tau_f[1:]
            var_y_eps_common_draws[idraw] = (alpha_eps*sigma_eps[0][None, :])**2
            var_y_eps_common_total_draws[idraw] = (alpha_eps*(sigma_eps[0]*scale_eps[0])[None, :])**2
            var_y_dtau_common_draws[idraw] = (alpha_tau*sigma_dtau[0][None, :])**2
            var_y_eps_unique_draws[idraw] = sigma_eps[1:]**2
            var_y_eps_unique_total_draws[idraw] = (sigma_eps[1:]*scale_eps[1:])**2
            var_y_dtau_unique_draws[idraw] = sigma_dtau[1:]**2            

        # 저장
        self.sigma_eps_draws = sigma_eps_draws
        self.sigma_dtau_draws = sigma_dtau_draws
        self.scale_eps_draws = scale_eps_draws
        self.sigma_eps_total_draws = sigma_eps_total_draws
        self.alpha_eps_draws = alpha_eps_draws
        self.alpha_tau_draws = alpha_tau_draws
        self.g_eps_draws = g_eps_draws
        self.g_dtau_draws = g_dtau_draws
        self.ps_draws = ps_draws
        self.tau_draws = tau_draws
        self.dtau_draws = dtau_draws
        self.tau_f_draws = tau_f_draws
        self.sigma_dalpha_draws = sigma_dalpha_draws
        self.y_eps_common_draws = y_eps_common_draws
        self.y_eps_unique_draws = y_eps_unique_draws
        self.y_tau_common_draws = y_tau_common_draws
        self.y_tau_unique_draws = y_tau_unique_draws
        self.y_f_tau_common_draws = y_f_tau_common_draws
        self.y_f_tau_unique_draws = y_f_tau_unique_draws
        self.var_y_eps_common_draws = var_y_eps_common_draws
        self.var_y_eps_common_total_draws = var_y_eps_common_total_draws
        self.var_y_dtau_common_draws = var_y_dtau_common_draws
        self.var_y_eps_unique_draws = var_y_eps_unique_draws        
        self.var_y_eps_unique_total_draws = var_y_eps_unique_total_draws
        self.var_y_dtau_unique_draws = var_y_dtau_unique_draws



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
        posterior_mean = {
            "g_eps": np.mean(self.g_eps_draws[self.used_draws], axis=0),
            "g_dtau": np.mean(self.g_dtau_draws[self.used_draws], axis=0), 
            "ps": np.mean(self.ps_draws[self.used_draws], axis=0),
        }

        posterior_variance = {
            "g_eps": np.var(self.g_eps_draws[self.used_draws], axis=0),
            "g_dtau": np.var(self.g_dtau_draws[self.used_draws], axis=0),
            "ps": np.var(self.ps_draws[self.used_draws], axis=0),
        }

        hpd_lower_bound = {
            "g_eps": np.percentile(self.g_eps_draws[self.used_draws], 2.5, axis=0),
            "g_dtau": np.percentile(self.g_dtau_draws[self.used_draws], 2.5, axis=0),
            "ps": np.percentile(self.ps_draws[self.used_draws], 2.5, axis=0),
        }

        hpd_upper_bound = {
            "g_eps": np.percentile(self.g_eps_draws[self.used_draws], 97.5, axis=0),
            "g_dtau": np.percentile(self.g_dtau_draws[self.used_draws], 97.5, axis=0),
            "ps": np.percentile(self.ps_draws[self.used_draws], 97.5, axis=0),
        }

        ESS = {
            "g_eps": [self._effective_sample_size(self.g_eps_draws[self.used_draws, i]) for i in range(self.N)],  # col별로 ESS 계산
            "g_dtau": [self._effective_sample_size(self.g_dtau_draws[self.used_draws, i]) for i in range(self.N)],
            "ps": [self._effective_sample_size(self.ps_draws[self.used_draws, i]) for i in range(self.N)],
        }

        # 저장
        self.summary_stats = {
            "posterior_mean": posterior_mean,
            "posterior_variance": posterior_variance,
            "hpd_lower_bound": hpd_lower_bound,
            "hpd_upper_bound": hpd_upper_bound,
            "ESS": ESS,
        }

