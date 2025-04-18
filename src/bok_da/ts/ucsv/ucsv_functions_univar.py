import sys
import numpy as np
from scipy import sparse
import scipy


def sample_tau_by_kalman(Q, R, x_init, P_init, y, rand1, rand2):
    """
    관측 데이터 y와 프로세스 노이즈 Q, 관측 노이즈 R을 사용하여
    Kalman 필터와 스무더를 통해 잠재 변수 tau_a와 tau_f를 샘플링

    Parameters
    ----------
    Q : numpy.array of shape (T,)
        프로세스 노이즈 공분산의 대각 요소들로 구성된 배열
    R : numpy.array of shape (T,)
        관측 노이즈 공분산의 대각 요소들로 구성된 배열
    x_init : float
        초기 상태 변수 값
    P_init : float
        초기 상태 공분산 값
    y : numpy.array of shape (T,)
        관측된 데이터의 배열
    rand1 : numpy.array of shape (T,)
        필터링 단계에서 사용되는 표준 정규 분포를 따르는 난수 배열
    rand2 : numpy.array of shape (T+1,)
        스무딩 단계에서 사용되는 표준 정규 분포를 따르는 난수 배열

    Returns
    -------
    tau_a : numpy.array of shape (T+1,)
        스무딩된 상태 변수의 배열
    tau_f : numpy.array of shape (T,)
        필터링된 상태 변수의 배열
    """
    
    T = y.shape[0] 

    x0 = x_init
    P0 = P_init

    P_u = np.zeros(T + 1, dtype=float, order="F")
    P_p = np.zeros(T + 1, dtype=float, order="F")
    x_u = np.zeros(T + 1, dtype=float, order="F")
    x_p = np.zeros(T + 1, dtype=float, order="F")
    x_u[0] = x0
    P_u[0] = P0

    tau_a = np.zeros(T + 1, dtype=float, order="F")
    tau_f = np.zeros(T, dtype=float, order="F")

    # Forward pass: Kalman filter
    for t in range(T):
        x1 = x0
        P1 = P0 + Q[t]
        Ht = P1 + R[t]
        K = P1 / Ht
        x0 = x1 + K * (y[t] - x1)
        P0 = P1 - K * P1
        x_u[t + 1] = x0
        P_u[t + 1] = P0
        x_p[t + 1] = x1
        P_p[t + 1] = P1
        # Generate random draw from filtered mean and variance 
        tau_f[t] = x0 + np.sqrt(P0) * rand1[t]

    # Backward pass: Kalman smoother
    xT = x0
    PT = P0
    x = xT + np.sqrt(PT) * rand2[T]
    tau_a[T] = x

    for t in range(T - 1, 0, -1):
        x1 = x_p[t + 1]
        P1 = P_p[t + 1]
        x0 = x_u[t]
        P0 = P_u[t]
        AS = P0 / P1
        xT = x0 + AS * (x - x1)
        PT = P0 - AS * P0
        x = xT + np.sqrt(PT) * rand2[t]
        tau_a[t] = x

    return tau_a, tau_f



def sample_tau(y, sigma_dtau, sigma_eps_scl):
    """
    관측된 데이터 y와 프로세스 노이즈의 표준편차 sigma_dtau,
    관측 노이즈의 표준편차 sigma_eps_scl을 기반으로 상태 변수 tau와
    그 변화량 dtau를 Kalman 필터와 스무더를 사용하여 샘플링

    Parameters
    ----------
    y : numpy.array of shape (T,)
        관측된 데이터의 1차원 배열
    sigma_dtau : float 또는 numpy.array of shape (T,)
        프로세스 노이즈의 표준편차로 스칼라 또는 배열일 수 있음
    sigma_eps_scl : float 또는 numpy.array of shape (T,)
        관측 노이즈의 표준편차로 스칼라 또는 배열일 수 있음

    Returns
    -------
    tau : numpy.array of shape (T,)
        스무딩된 상태 변수의 배열
    dtau : numpy.array of shape (T,)
        상태 변수의 변화량(tau의 차분) 배열
    tau_f : numpy.array of shape (T,)
        Kalman 필터링된 상태 변수의 배열
    """
    big = 1e6
    nobs = len(y)
    x0 = 0
    P0 = big
    rand1 = np.random.standard_normal(nobs)
    rand2 = np.random.standard_normal(nobs + 1)

    if np.isscalar(sigma_dtau):
        Q = np.full(nobs, sigma_dtau**2, dtype=float, order='F')
    else:
        Q = np.array(sigma_dtau**2, dtype=float, order='F')

    if np.isscalar(sigma_eps_scl):
        R = np.full(nobs, sigma_eps_scl**2, dtype=float, order='F')
    else:
        R = np.array(sigma_eps_scl**2, dtype=float, order='F')

    tau_a, tau_f = sample_tau_by_kalman(Q, R, x0, P0, y, rand1, rand2)
    dtau = tau_a[1:] - tau_a[:-1]
    tau = tau_a[1:]
    return tau, dtau, tau_f



def sparse_cholesky(A): # The input matrix A must be a sparse symmetric positive-definite.
    n = A.shape[0]
    LU = sparse.linalg.splu(A, permc_spec='NATURAL', diag_pivot_thresh=0) # sparse LU decomposition
    
    if ( LU.perm_r == np.arange(n) ).all() and ( LU.U.diagonal() > 0 ).all(): # check the matrix A is positive definite.
        return LU.L @ ( sparse.diags(LU.U.diagonal()**0.5) )
    else:
        sys.exit('The matrix is not positive definite')


def SVRW(x, z_tilde, z0, omega, a0, b0, Vomega):
    """
    Stock and Watson(2016), Chan(2018)의 Stochastic Volatility Random Walk(SVRW) 모델 구현 함수
    혼합 정규 분포 기반의 선형 상태 공간 모형에 기반하며, Gaussian mixture approximation 방법을 통해 샘플링된 상태 변수를 구함 

    Parameters
    ----------
    x : numpy.array of shape (T,)
        관측된 로그 변동성 데이터
    z_tilde : numpy.array of shape (T,)
              이전 시점에서 추정된 상태 변수       
    z0 : float
         초기값(상태 공간 모형의 첫 번째 시점의 상태 변수)
    omega : float
            상태 변수의 변동성을 나타내며, Gibbs 샘플링을 통해 업데이트됨
    a0 : float
         선형 회귀에서 상수항(z0)의 사전 분포의 평균
    b0 : float
         선형 회귀에서 상수항(z0)의 사전 분포의 분산
    Vomega : float
             선형 회귀에서 기울기(omega)의 사전 분포의 분산

    Returns
    -------
    z_tilde : numpy.array of shape (T,)
              Gibbs 샘플링을 통해 업데이트된 상태 변수
    z0 : float
         Gibbs 샘플링을 통해 업데이트된 초기값
    omega : float
            1/2 확률로 부호가 바뀐 상태 변수의 변동성(omega)
    omega_hat : float
                Gibbs 샘플링을 통해 업데이트된 상태 변수의 변동성(omega), 즉 omega의 사후 분포의 평균
    Domega : float
             Gibbs 샘플링을 통해 업데이트된 상태 변수의 변동성의 분산, 즉 omega의 사후 분포의 분산
    ind_e : numpy.array of shape (T,n)
            샘플링된 지표 변수로, Mixture 모형에서 각 상태가 선택된 인덱스를 나타내는 배열
    sigma_e : numpy.array of shape (T,)
              원 데이터(log 취하기 전)에서의 변동성 
    """
    # 10-component mixture from Omori, Chib, Shephard, and Nakajima JOE (2007)
    r_p = np.array([0.00609, 0.04775, 0.13057, 0.20674, 0.22715, 0.18842, 0.12047, 0.05591, 0.01575, 0.00115])
    r_m = np.array([1.92677, 1.34744, 0.73504, 0.02266, -0.85173, -1.97278, -3.46788, -5.55246, -8.68384, -14.65000])
    r_v = np.array([0.11265, 0.17788, 0.26768, 0.40611, 0.62699, 0.98583, 1.57469, 2.54498, 4.16591, 7.33342])
    r_s = np.sqrt(r_v)
    
    T = len(x)
    n = len(r_p)

    z = z0 + omega*z_tilde
    
    # Compute likelihood for each mixture and each time period
    # 필수적인 초기화 연산
    xrep = np.repeat(x[:, None], n, axis=1)
    zrep = np.repeat(z[:, None], n, axis=1)
    mrep = np.tile(r_m, (T, 1))
    srep = np.tile(r_s, (T, 1))
    prep = np.tile(r_p, (T, 1))

    # sample S from a n-point discrete distribution
    pxlike = prep*np.exp(-0.5*((xrep - zrep - mrep)/srep)**2)/srep
    p_post = pxlike / pxlike.sum(axis=1)[:, None]

    # If data are missing, posterior = prior (which is in prep); 
    p_post[np.isnan(p_post)] = prep[np.isnan(p_post)]
    
    # Draw Indicators from posterior
    # S 샘플링 부분 최적화
    U = np.random.uniform(0, 1, T)
    S = n - (U[:, None] <= p_post.cumsum(axis=1)).sum(axis=1)
    ind_e = sparse.csr_matrix((np.ones(T), (np.arange(T), S)), shape=(T, n)).toarray()
    
    
    # sample z_tilde
    H = sparse.diags([1, -1], [0, -1], shape=(T, T), format='csr')  # TEST: 살짝 빨라짐 TEST20241011 참고, 두 코드의 결과는 같음 (csc로 할떄는 워닝이 뜨는데 csr로 하면 워닝이 안 뜸) / 이부분은 확실히 빠르다
    d_s = r_m[S]
    iOs = sparse.diags(1/r_v[S])
    Kh = H.T @ H + (omega**2) *iOs
    z_tilde_hat = sparse.linalg.spsolve(Kh, omega*iOs @ (x - d_s - z0));
    z_tilde = z_tilde_hat + sparse.linalg.spsolve(sparse_cholesky(Kh).T, np.random.randn(T));
    

    # sample z0 and omegaz
    Xbeta = np.hstack((np.ones((T, 1)), z_tilde[:, None]))
    iVbeta = np.diag([1/b0, 1/Vomega]);    
    Kbeta = iVbeta + Xbeta.T @ (iOs.toarray()) @ (Xbeta);
    beta_hat = np.linalg.solve(Kbeta, (iVbeta @ ([a0, 0]) + Xbeta.T @ (iOs.toarray()) @ (x - d_s)));
    beta = beta_hat + np.linalg.solve(np.linalg.cholesky(Kbeta).T, np.random.randn(2));
    z0 = beta[0]; omega = beta[1];

    # randomly permute the signs h_tilde and omegah
    U = -1 + 2*(np.random.uniform() > .5);
    z_tilde *= U
    omega *= U
    
    # compute the mean and variance of the conditional density of omegah    
    Dbeta = np.linalg.solve(Kbeta, np.eye(2));
    omega_hat = beta_hat[1];
    Domega = Dbeta[1,1];
    
    sigma_e = np.exp(0.5*(z0 + omega*z_tilde))
    # omega = g_draw in draw_g function
    
    return z_tilde, z0, omega, omega_hat, Domega, ind_e, sigma_e



def sample_scale_eps(e, sigma, ind_e, scale_e_vec, prob_scale_e_vec):
    """

    gibbs sampling 과정에서 outlier 를 control 하는 s_t 의 값을 sampling 하는 함수입니다.
    gaussian mixture model 의 dirichlet - categorical distribution 의 conjugacy 사용합니다.

    Parameters
    ----------
    e: numpy.array of shape (T,)
       이전 iteration 을 통해서 얻은 trend prediction 과 실제 데이터 y 의 residual.
    sigma: numpy.array of shape (T,)
           SVRW function 을 통해서 sampling 된 volatility 변수 sigma.
    ind_e: numpy.array of shape (T,n)
           샘플링된 지표 변수로, Mixture 모형에서 각 상태가 선택된 인덱스를 나타내는 배열
    scale_e_vec: numpy.array of shape (10,)
                 s_t 가 가질 수 있는 값 할당해 놓은 벡터.
    prob_scale_e_vec: numpy.array of shape (10,)
                      s_t 변수가 가질 수 있는 값들의 확률을 할당해 놓은 벡터.


    Returns
    -------
    scale_e: numpy.array of shape (T,)
             accept 된 s_t 의 value
    
    """

    # 10-component mixture from Omori, Chib, Shephard, and Nakajima JOE (2007)
    r_p = np.array([0.00609, 0.04775, 0.13057, 0.20674, 0.22715, 0.18842, 0.12047, 0.05591, 0.01575, 0.00115])
    r_m = np.array([1.92677, 1.34744, 0.73504, 0.02266, -0.85173, -1.97278, -3.46788, -5.55246, -8.68384, -14.65000])
    r_v = np.array([0.11265, 0.17788, 0.26768, 0.40611, 0.62699, 0.98583, 1.57469, 2.54498, 4.16591, 7.33342])
    r_s = np.sqrt(r_v)

    n = len(scale_e_vec) # NOTE: s_t 가 가질 수 있는 값의 수
    T = len(e) # NOTE: 시계열상의 point 수

    # OCSN approximation 을 하기 위해서 s_t -> ln(s_t^2) 로 변환해 주는 과정
    scl_mean = np.log(scale_e_vec**2)

    # T 길이를 가진 시계열 data 에서 T 시점 별 gaussian mixture 의 mean, sd 값 계산
    mean_cs = ind_e.dot(r_m)
    sd_cs = ind_e.dot(r_s)

    # c = 0.001 factor from ksv, restud(1998), page 370) NOTE: 잔차에 보정값 두고 계산한 것
    c = 0.001
    lnres2 = np.log(e**2 + c)
    # ln(\epsilon_t^2)-ln(\sigma_{\epsilon, t}^2) 계산한 것.
    res_e = lnres2 - mean_cs - np.log(sigma**2)

    # s_t 의 posterior probability 값을 뱉는 부분
    res_e_mat = np.repeat(res_e[:, None], n, axis=1) - np.repeat(scl_mean[None, :], T, axis=0)
    tmp_scl = np.repeat((1/np.sqrt(sd_cs))[:, None], n, axis=1)
    tmp_exp = np.exp(-0.5*(res_e_mat/sd_cs[:, None])**2)
    # normal 정규화 완료된 값에 대해서 density 구하는데 sqrt{\pi} 어차피 normalizing 과정에서 cancel out 됨
    den_prob = tmp_scl*tmp_exp*prob_scale_e_vec[None, :] 
    den_marg = den_prob.sum(axis=1)
    # Posterior probability of different scale factors
    p_post = den_prob/den_marg[:, None]

    # Draw Scale Factors from Posterior
    # 0~1 사이의 난수 시점 T의 갯수만큼 추출
    U = np.random.uniform(0, 1, T)
    # posterior distribution 인 categorical distribution 에서 sampling
    bb = n - (U[:, None] <= p_post.cumsum(axis=1)).sum(axis=1)

    # accept 된 s_t 의 value
    scale_e = scale_e_vec[bb] 

    return scale_e



def sample_ps(scale_eps, ps_prior, n_scl_eps):
    """
    beta - binomial conjugacy 를 사용한 gibbs sampling 으로 
    prob_scl_eps_vec 를 sampling 하는 함수

    Parameters
    ----------
    scale_eps: numpy.array of shape (T,)
               sample_scale_eps 함수를 통해 sampling 한 s_t
    ps_prior: numpy.array of shape (2,)
              P 의 prior 분포의 parameter. beta distribution 이므로 shape, scale parameter 가짐.
    n_scl_eps: int
               s_t 가 가질 수 있는 값의 개수

    Returns
    -------
    prob_scl_eps_vec: numpy.array of shape (10,)
                      s_t 분포의 확률값

    """

    n1 = np.sum(scale_eps == 1)
    # s_t 가 outlier 값을 가진 횟수
    n2 = len(scale_eps) - n1
    # binomial, beta  conjugacy 로 계산한 posterior
    ps = np.random.beta(ps_prior[0] + n1, ps_prior[1] + n2, size = 1)
    prob_scl_eps_vec = ps*np.ones(n_scl_eps) # non-outlier 값을 가질 probability
    prob_scl_eps_vec[1:] = (1-ps)/(n_scl_eps-1) # outlier 값을 가질 probability 균일하게 나눠서 배치

    return prob_scl_eps_vec