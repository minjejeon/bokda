import numpy as np
from scipy.special import erfinv
from scipy import stats


def nsample(n):
    return np.sqrt(2) * erfinv(2 * np.random.rand(n) - 1)

def nsample2(n, m):
    return np.sqrt(2) * erfinv(2 * np.random.rand(m, n).T - 1)


def sample_dalpha_sigma(dalpha, nu_prior_alpha, s2_prior_alpha):

    n = dalpha.shape[0]
    T = dalpha.shape[1]

    SSR_mat = np.sum(dalpha**2, axis=1)
    SSR_prior = nu_prior_alpha*s2_prior_alpha

    a = (T + nu_prior_alpha)/2
    
    sigma_draw = np.nan*np.zeros(n)
    for i in range(n):
        b = 2/(SSR_mat[i] + SSR_prior)
        var_dalpha = 1/np.random.gamma(a, b)
        sigma_draw[i] = np.sqrt(var_dalpha)

    return sigma_draw


### sample_alpha_tvp ###

def sample_alpha_tvp_by_kalman(H, R, F, Q, x_init, P_init, y, rand2):
    
    """
    Kalman filter 및 smoother를 사용하여, state-space model에서 시간에 따라 변하는
    상태 변수 'alpha_t'를 샘플링 하는 함수.
    관측값 'y'가 주어졌을 때, 상태 변수 'alpha_t'의 사후분포로부터 샘플을 추출하기 위해
    forward pass(Kalman filter)와 backward pass(Kalman Smoother)를 수행.
    MUCSV 모델의 추정 과정에서 latent common factor인 'tau_{c,t}'와 epsilon_{c,t}에
    대한 factor loading 'alpha_tau'와 'alpha_epsilon'을 샘플링하는 데 사용.

    Parameters
    ----------
    H : numpy array of shape (n_y, N+1, T)
        각 시간 't'에서의 관측 행렬 'H_t'.
        - 'n_y': 관측 변수의 수.
        - 'N+1': 상태 변수의 수.
        - 'T': 시간 구간의 길이.
    R : numpy array of shape (n_y, n_y, T)
        각 시간 't'에서의 관측 오차 공분산 행렬 'R_t'.
    F : numpy array of shape (N+1, N+1)
        상태 전이 행렬 'F' (시간에 따라 일정하다고 가정).
    Q : numpy array of shape (N+1, N+1)
        상태 노이즈 공분산 행렬 `Q` (시간에 따라 일정하다고 가정).
    x_init : numpy array of shape (N+1,)
        초기 상태의 평균 벡터.
    P_init : numpy array of shape (N+1, N+1)
        초기 상태의 공분산 행렬.
    y : numpy array of shape (n_y, T)
        관측 데이터 행렬.
    rand2 : numpy array of shape (N+1, T+1)
        Backward pass에서 사용되는 표준 정규분포에서의 난수들.

    Returns
    -------
    x_draw : numpy array of shape (N+1, T+1)
        시간 't = 0'부터 'T'까지 샘플링된 상태 변수 'alpha_t'.
    """
    
    m = H.shape[0]  # number of measurements
    n = H.shape[1]  # number of states
    T = H.shape[2]  # number of observations
    d = n - m       # number of observations
    
    x0 = np.array(x_init, float, order="F")
    x1 = np.zeros(n, float, order="F")
    xT = np.zeros(n, float, order="F")
    P0 = np.array(P_init, float, order="F")
    P1 = np.zeros((n, n), float, order="F")
    PT = np.zeros((n, n), float, order="F")
    xd = np.zeros(n, float, order="F")          # random draw from N(x0, P0)
    PTm = np.zeros((m, m), float, order="F")
    
    x_u = np.zeros((n, T + 1), float, order="F")
    P_u = np.zeros((n, n, T + 1), float, order="F")
    x_u[:, 0] = x0
    P_u[:, :, 0] = P0   
    x_p = np.zeros((n, T + 1), float, order="F")
    P_p = np.zeros((n, n, T + 1), float, order="F")
    chol_P = np.zeros((n, n), float, order="F")
    chol_Pm = np.zeros((m, m), float, order="F")
    
    x_draw = np.zeros((n, T + 1), float, order="F")
    x_draw_f = np.zeros((n, T + 1), float, order="F")
    
    nu = np.zeros(m, float, order="F")
    S = np.zeros((m, m), float, order="F")
    invS = np.zeros((m, m), float, order="F")
    Im = np.array(np.eye(m), float, order="F")
    In = np.array(np.eye(n), float, order="F")
    K = np.zeros((n, m), float, order="F")

    FP0 = np.zeros((n, n), float, order="F")
    AS = np.zeros((n, n), float, order="F")
    Tnn = np.zeros((n, n), float, order="F")
    Tmm = np.zeros((m, m), float, order="F")
    Tnn2 = np.zeros((n, n), float, order="F")
    Tnm = np.zeros((n, m), float, order="F")
    Tmn = np.zeros((m, n), float, order="F")
    C = np.zeros((m,m), np.int32, order="F")

    Tn = np.zeros(n, float, order="F")
    Tm = np.zeros(m, float, order="F")
    
    sl = np.array(np.zeros((m, n)), float, order="F")
    sl[:, d:] = Im
    
    for t in range(T):
        
        # x1 = F[t] * x0    
        x1 = F @ x0 
        
        # P1 = F[t] * P0 * F[t]' + Q[t], Z.shape = F.dot(P0).shape
        P1 = F @ P0 @ F.T + Q
        
        # nu = y[t] - H[t].dot(x1)
        nu = y[:,t] - H[:,:,t] @ x1
        
        # S = H[t].dot(P1).dot(H[t].T) + R[t], size(Tmn) = size(H[t].dot(P1))
        S = H[:,:,t] @ P1 @ H[:,:,t].T + R[:,:,t] 
        
        # invS = np.linalg.pinv(S)
        invS = np.linalg.solve(S, Im)
        
        # K = P1.dot(H[t].T).dot(invS), size(Tnm) = size(P1.dot(H[t].T))
        K = P1 @ H[:,:,t].T @ invS       
        
        # x0 = x1 + K.dot(nu)
        x0 = x1 + K @ nu
        
        # P0 = (np.eye(n_states) - K.dot(H[t])).dot(P1) = P1 - K.dot(H).P1
        P0 = P1 - K @ H[:,:,t] @ P1
        
        # P0 = 0.5*(P0 + P0.T)
        # sys_mat(P0, .5, .5, In, P0)
        Tnn = P0
        for i in range(n):
            for j in range(n):
                P0[i, j] = (Tnn[i, j] + Tnn[j, i])*0.5
                
        # assign x0, x1, P0, P1 to x_u, x_p, P_u, P_p
        P_p[:,:,t+1] = P1
        P_u[:,:,t+1] = P0
        x_p[:,t+1] = x1
        x_u[:,t+1] = x0
      
    # xT <- x0, PT <- P0
    PT = P0
    xT = x0
    
    # x = xT + cholesky(P3).dot(randn(ns))
    chol_P = np.linalg.cholesky(PT)
    xd = xT + chol_P @ rand2[:,-1]
    x_draw[:,-1] = xd
    
    for t in range(T)[::-1]:
        x0 = x_u[:,t]
        x1 = x_p[:,t+1]
        P0 = P_u[:,:,t]
        P1 = P_p[:,:,t+1]

        # FP0 = F.dot(P0)
        # AS = solve(P1, FP0)
        # PT = P0 - AS.T.dot(FP0)
        FP0 = F @ P0
        AS = np.linalg.solve(P1,FP0)
        PT = P0 - AS.T @ FP0

        # PT = 0.5*(PT + PT.T)
        # sys_mat(PT, .5, .5, In, PT)
        Tnn = PT
        for i in range(n):
            for j in range(n):
                PT[i, j] = (Tnn[i, j] + Tnn[j, i])*0.5
                
        # xT = x0 + AS.T.dot(xd - x1)
        xd -= x1
        xT = x0 + AS.T @ xd

        
        # xd = xT + cholesky(PT).dot(rand2[:, t])
        chol_P = np.linalg.cholesky(PT)
        xd = xT + chol_P @ rand2[:,t]

        x_draw[:,t] = xd      
    
    return np.array(x_draw)


def sample_alpha_tvp_multivar(y, prior_var_alpha, sigma_dalpha, tau, eps_common, sigma_eps_unique):
    """
    MUCSV 의 estimation 과정에서 1.a.2 번째 sampling process 를 구현한 함수.
    latent common factor for trend inflation "tau_{c,t}"
    latent common transient component "epsilon_{c,t} 에 각각 해당하는 factor loading
    alpha_tau, alpha_epsilon 을 sampling 하는 함수.

    Parameters
    ----------
    y : numpy array of shape (n_y,T)
        입력 데이터 matrix. 
        n_y 개의 number of observed variable, 
        T 개의 number of observation 을 가지고 있다.
    prior_var_alpha : numpy array of shape (2*n_y, 2*n_y)
                      alpha_eps 와 alpha_tau 에 동일하게 적용되는 prior 의 covariance matrix.
    sigma_dalpha : numpy array of shape (2*n_y,)
                   sample_dalpha_sigma() function 을 통해 sampling 되는 factor loading alpha_tau, 
                   alpha_epsilon 의 deviation parameter 인 lambda_tau, lambda_epsilon 을 의미.
                   (1.a.3 순서에 sampling 되는 변수.)
    tau : numpy array of shape (N,T)
          common and unique trend component - 두 변수가 concanate 된 결과
    eps_common : numpy array of shape (T,)
                 common transient component
    sigma_eps_unique : numpy array of shape (n_y,Y)
                       unique transient component 의 volatility variable

    
    Returns
    -------
    alpha_eps : numpy array of shape (n_y,T)
                factor loading matrix for alpha_epsilon
    alpha_tau : numpy array of shape (n_y,T)
                factor loading matrix for alpha_tau
    dalpha : numpy array of shape (2*n_y,T)
             alpha_eps 와 alpha_tau 의 1차 차분값들로 이루어진 matrix
    """
    
    ### UNTITLED Summary of this function goes here
    #   Detailed explanation goes here
    # sigma_eps_unique = sigma_eps_unique_scl

    tau_unique = tau[1:]
    tau_common = tau[0]
    
    n_y = y.shape[0]
    nobs = y.shape[1]
    y = y - tau_unique    # Eliminates tau_unique from y ;

    # Brute Force Calculation
    ns = 2*n_y

    # First n_y elements of state are alpha_eps; second n_y elements of state are alpha_tau
    Q = np.diag(sigma_dalpha**2)     # Q_t = Q
    F = np.eye(ns)              # F_t = F
    H = np.concatenate((np.kron(eps_common[:, None], np.eye(3)), np.kron(tau_common[:, None], np.eye(3))), axis=1).reshape(-1, 3, 6)
    H = np.transpose(H, (1, 2, 0))
    R = np.transpose(np.eye(n_y) * sigma_eps_unique.T[:, None, :]**2, (1, 2, 0))
    
    # random draws from N(0, 1)
    rand1 = nsample2(ns, nobs)
    rand2 = nsample2(ns, nobs+1)
    
    # Set up KF to run
    # Initial conditions
    x0 = np.zeros(ns)
    P0 = prior_var_alpha
    
    
    H = np.array(H, float, order="F")
    R = np.array(R, float, order="F")
    F = np.array(F, float, order="F")
    Q = np.array(Q, float, order="F")
    x0 = np.array(x0, float, order="F")
    P0 = np.array(P0, float, order="F")
    y = np.array(y, float, order="F")
    rand1 = np.array(rand1, float, order="F")
    rand2 = np.array(rand2, float, order="F")

    x_draw = sample_alpha_tvp_by_kalman(H, R, F, Q, x0, P0, y, rand2)   

    dalpha_eps = x_draw[:n_y, 1:] - x_draw[:n_y, :-1]
    dalpha_tau = x_draw[n_y:, 1:] - x_draw[n_y:, :-1]
    alpha_eps = x_draw[:n_y, 1:]
    alpha_tau = x_draw[n_y:, 1:]
    dalpha = np.concatenate((dalpha_eps, dalpha_tau), axis=0)

    return alpha_eps, alpha_tau, dalpha



### sample_eps_tau ###

def sample_eps_tau_by_kalman(H, R, F, Q, x_init, P_init, y, rand1, rand2):
    
    """
    Kalman filter 및 smoother를 사용하여, state-space model에서 시간에 따라 변하는 
    factor loading alpha_tau, alpha_epsilon 에 corresponding 하는
    factor matrix인 tau 와 epsilon 을 sampling 하는 함수.
    관측값 'y'가 주어졌을 때, tau 와 epsilon의 사후분포로부터 샘플을 추출하기 위해
    forward pass(Kalman filter)와 backward pass(Kalman Smoother)를 수행.
    MUCSV 모델의 추정 과정에서 factor loading 'alpha_tau'와 'alpha_epsilon' 에 corresponding 하는
    latent common factor인 'tau_{c,t}, tau_{i,t}'와 epsilon_{c,t} 를 sampling 하는 함수.

    Parameters
    ----------
    H : numpy array of shape (n_y, N+1, T)
        각 시간 't'에서의 관측 행렬 'H_t'.
        - 'n_y': 관측 변수의 수.
        - 'N+1': 상태 변수의 수.
        - 'T': 시간 구간의 길이.
    R : numpy array of shape (n_y, n_y, T)
        각 시간 't'에서의 관측 오차 공분산 행렬 'R_t'.
    F : numpy array of shape (N+1, N+1)
        상태 전이 행렬 'F' (시간에 따라 일정하다고 가정).
    Q : numpy array of shape (N+1, N+1, T)
        상태 노이즈 공분산 행렬 `Q` (시간에 따라 일정하다고 가정).
    x_init : numpy array of shape (N+1,)
        초기 상태의 평균 벡터.
    P_init : numpy array of shape (N+1, N+1)
        초기 상태의 공분산 행렬.
    y : numpy array of shape (n_y, T)
        관측 데이터 행렬.
    rand1 : numpy array of shape (N+1, T)
            filtering 단계에서 사용되는 표준 정규분포에서의 난수들.
    rand2 : numpy array of shape (N+1, T+1)
            smoothing 단계에서 사용되는 표준 정규분포에서의 난수들.

    Returns
    -------
    x_draw_f : numpy array of shape (N+1, T+1)
               시간 t = T 부터 0 까지 샘플링된 'tau_f'. 
               kalman smoothing 이전에 filtering 과정에서 sampling 된 상태 변수.
    x_draw : numpy array of shape (N+1, T+1)
             시간 't = 0'부터 'T'까지 샘플링된 'tau_t'
    """

    m = H.shape[0]  # number of measurements
    n = H.shape[1]  # number of states
    T = H.shape[2]  # number of observations
    d = n - m       # number of observations

    x0 = np.array(x_init, float, order="F")
    x1 = np.zeros(n, float, order="F")
    xT = np.zeros(n, float, order="F")
    P0 = np.array(P_init, float, order="F")
    P1 = np.zeros((n, n), float, order="F")
    PT = np.zeros((n, n), float, order="F")
    xd = np.zeros(n, float, order="F")          # random draw from N(x0, P0)
    PTm = np.zeros((m, m), float, order="F")
    
    x_u = np.zeros((n, T + 1), float, order="F")
    P_u = np.zeros((n, n, T + 1), float, order="F")
    x_u[:,0] = x0
    P_u[:,:,0] = P0
    x_p = np.zeros((n, T + 1), float, order="F")
    P_p = np.zeros((n, n, T + 1), float, order="F")
    chol_P = np.zeros((n, n), float, order="F")
    chol_Pm = np.zeros((m, m), float, order="F")
    
    x_draw = np.zeros((n, T + 1), float, order="F")
    x_draw_f = np.zeros((n, T + 1), float, order="F")

    nu = np.zeros(m, float, order="F")
    S = np.zeros((m, m), float, order="F")
    invS = np.zeros((m, m), float, order="F")
    Im = np.array(np.eye(m), float, order="F")
    In = np.array(np.eye(n), float, order="F")
    K = np.zeros((n, m), float, order="F")

    FP0 = np.zeros((n, n), float, order="F")
    AS = np.zeros((n, n), float, order="F")
    Tnn = np.zeros((n, n), float, order="F")
    Tmm = np.zeros((m, m), float, order="F")
    Tnn2 = np.zeros((n, n), float, order="F")
    Tnm = np.zeros((n, m), float, order="F")
    Tmn = np.zeros((m, n), float, order="F")
    C = np.zeros((m,m), np.int32, order="F")

    Tn = np.zeros(n, float, order="F")
    Tm = np.zeros(m, float, order="F")
    
    sl = np.array(np.zeros((m, n)), float, order="F")
    sl[:,d:] = Im
    
    for t in range(T):
            
        x1 = F @ x0 # (nxn) x (nx1)
   
        P1 = F @ P0 @ F.T + Q[:,:,t] # (nxn) x (nxn) x (nxn) + (nxn) 
 
        nu = y[:,t] - H[:,:,t] @ x1 # (m,)
         
        S = H[:,:,t] @ P1 @ H[:,:,t].T + R[:,:,t] # (mxn) x (nxn) x (nxm) + (mxm)
        
        invS = np.linalg.solve(S, Im) # (mxm) x (mxm)

        K = P1 @ H[:,:,t].T @ invS # (nxn) x (nxm) x (mxm) 

        x0 = x1 + K @ nu # (nx1) + (nxm)x(mx1) 

        P0 = P1 - K @ H[:,:,t] @ P1
         
        Tnn = P0
        for i in range(n):
            for j in range(n):
                P0[i, j] = (Tnn[i, j] + Tnn[j, i])*0.5
                
        # assign x0, x1, P0, P1 to x_u, x_p, P_u, P_p
        P_p[:,:,t+1] = P1
        P_u[:,:,t+1] = P0
        x_p[:,t+1] = x1
        x_u[:,t+1] = x0

        # draw x ~ N(x0, P0) and save to x_draw
        # X = X0 + cholesky(P0).dot(randn(ns))
        chol_P = np.linalg.cholesky(P0)
        xd = chol_P @ rand1[:,t] + x0
        x_draw_f[:,t+1] = xd

    # xT <- x0, PT <- P0
    PT = P0
    xT = x0

    # x = xT + cholesky(P3).dot(randn(ns))
    chol_P = np.linalg.cholesky(PT)
    xd = chol_P @ rand2[:,-1] + xT
    x_draw[:,-1] = xd

    for t in range(T)[::-1]:
        x0 = x_u[:,t]
        x1 = x_p[:,t+1]
        P0 = P_u[:,:,t]
        P1 = P_p[:,:,t+1]

        FP0 = F @ P0
        AS = np.linalg.solve(P1, FP0)
        PT = P0 - AS.T @ FP0

        Tnn = PT
        for i in range(n):
            for j in range(n):
                PT[i, j] = (Tnn[i, j] + Tnn[j, i])*0.5
                
        xd -= x1
        xT = x0 + AS.T @ xd
        xd = xT

        
        if t > 0:
            chol_P = np.linalg.cholesky(PT)
            Tn = chol_P @ rand2[:,t]
            xd += Tn

        else:            
            PTm = sl @ PT @ sl.T
            chol_Pm = np.linalg.cholesky(PTm)
            Tm = chol_Pm @ rand2[d:,t]
            xd[d:] += Tm
            
        x_draw[:,t] = xd

    return np.array(x_draw_f), np.array(x_draw)


def sample_eps_tau_multivar(y, alpha_eps, alpha_tau, sigma_eps_scl, sigma_dtau):
    
    """
    MUCSV 의 estimation 과정에서 1.a.1 번째 sampling process 를 구현한 함수.
    factor matrix tau 와 epsilon 을 sampling 하는 함수.
    위의 함수와 비교해 보면, factor analysis 에서 factor loading 에 corresponding 하는 latent factor 를 sampling.
    그러나 위의 함수와 달리 latent common factor 과 함께 sector-specific factor도 함께 sampling.

    Parameters
    ----------
    y : numpy array of shape (n_y, T)
        입력 데이터 matrix. 
        n_y 개의 number of observed variable, 
        T 개의 number of observation 을 가지고 있다.
    alpha_eps : numpy array of shape (n_y,T)
                이전 시점에 sampling 된 factor loading for epsilon
    alpha_tau : numpy array of shape (n_y,T)
                이전 시점에 sampling 된 factor loading for tau
    sigma_eps_scl : numpy array of shape (n_y+1,T)
                    scaled 된 epsilon 의 volatility variable - sigma_epsilon
    sigma_dtau : numpy array of shape (n_y+1,T)
                 scaled 된 tau 의 volatility variable - sigma_tau
    
    Returns
    -------
    eps_common : numpy array of shape (T,)
                 latent factor for alpha_epsilon. epsilon_{c,t} 로 표기.
    tau_f : numpy array of shape (N,T)
            tua sampling 과정에서 kalman smoothing 과정을 통과하지 않은 결과. (backward pass 를 통과하지 않은 결과)
    dtau : numpy array of shape (N,T)
           tau 의 1차 차분값 vector
    tau : numpy array of shape (N,T)
          latent factor for alpha_tau and alpha_epsilon(common component) and sector specific component.
          (factor loading 에 corresponding 되지 않은 tau)
          tau_{c,t} + tau_{i,t} 변수 함께 sampling된 결과.
    """

    sigma_eps_common = sigma_eps_scl[0]
    sigma_eps_unique = sigma_eps_scl[1:]
    sigma_dtau_common = sigma_dtau[0]
    sigma_dtau_unique = sigma_dtau[1:]

    samll = 1e-6; big = 1e6
    n_y = y.shape[0]
    nobs = y.shape[1]

    # Set up State Vector
    # --- State Vector
    #     (1) eps(t)
    #     (2) tau(t)
    #     (3) tau_u(t)

    ns = 2 + n_y        # size of state
    F = np.zeros((ns, ns));     F[1:, 1:] = np.eye(ns-1)
    H = np.concatenate((alpha_eps, alpha_tau, np.eye(n_y).reshape(-1, 1).dot(np.ones((1, nobs)))), axis=0).reshape(n_y, ns, -1, order='F')
    Q = np.eye(ns) * np.concatenate((sigma_eps_common[None, :]**2, sigma_dtau_common[None, :]**2, sigma_dtau_unique**2), axis=0).T[:, None, :]
    Q = np.transpose(Q, (1, 2, 0))
    R = np.eye(n_y) * sigma_eps_unique.T[:, None, :]**2
    R = np.transpose(R, (1, 2, 0))
    
    rand1 = nsample2(ns, nobs)
    rand2 = nsample2(ns, nobs+1)

    # Set up KF to run
    # Initial conditions
    x0 = np.zeros(ns)               # x_0|0
    P0 = np.zeros((ns, ns))         # P_0|0
    P0[2:, 2:] = big*np.eye(n_y)    # Vague prior for tau_unique initial values 
    
    
    H = np.array(H, float, order="F")
    R = np.array(R, float, order="F")
    F = np.array(F, float, order="F")
    Q = np.array(Q, float, order="F")
    x0 = np.array(x0, float, order="F")
    P0 = np.array(P0, float, order="F")
    y = np.array(y, float, order="F")
    rand1 = np.array(rand1, float, order="F")
    rand2 = np.array(rand2, float, order="F")

    x_draw_f, x_draw = sample_eps_tau_by_kalman(H, R, F, Q, x0, P0, y, rand1, rand2)   

    ########### return x_draw, x_draw_f ##############
    eps_common = x_draw[0, 1:]
    tau_f = x_draw_f[1:, 1:]    
    dtau = x_draw[1:, 1:] - x_draw[1:, :-1]
    tau = x_draw[1:, 1:]
    
    return eps_common, tau_f, dtau, tau







### --- cython --- ###


def sample_alpha_tvp_multivar_cython(cython_module, y, prior_var_alpha, sigma_dalpha, tau, eps_common, sigma_eps_unique):
    """
    MUCSV 의 estimation 과정에서 1.a.2 번째 sampling process 를 구현한 함수.
    latent common factor for trend inflation "tau_{c,t}"
    latent common transient component "epsilon_{c,t} 에 각각 해당하는 factor loading
    alpha_tau, alpha_epsilon 을 sampling 하는 함수.

    Parameters
    ----------
    y : numpy array of shape (n_y,T)
        입력 데이터 matrix. 
        n_y 개의 number of observed variable, 
        T 개의 number of observation 을 가지고 있다.
    prior_var_alpha : numpy array of shape (2*n_y, 2*n_y)
                      alpha_eps 와 alpha_tau 에 동일하게 적용되는 prior 의 covariance matrix.
    sigma_dalpha : numpy array of shape (2*n_y,)
                   sample_dalpha_sigma() function 을 통해 sampling 되는 factor loading alpha_tau, 
                   alpha_epsilon 의 deviation parameter 인 lambda_tau, lambda_epsilon 을 의미.
                   (1.a.3 순서에 sampling 되는 변수.)
    tau : numpy array of shape (N,T)
          common and unique trend component - 두 변수가 concanate 된 결과
    eps_common : numpy array of shape (T,)
                 common transient component
    sigma_eps_unique : numpy array of shape (n_y,Y)
                       unique transient component 의 volatility variable

    
    Returns
    -------
    alpha_eps : numpy array of shape (n_y,T)
                factor loading matrix for alpha_epsilon
    alpha_tau : numpy array of shape (n_y,T)
                factor loading matrix for alpha_tau
    dalpha : numpy array of shape (2*n_y,T)
             alpha_eps 와 alpha_tau 의 1차 차분값들로 이루어진 matrix
    """
    
    ### UNTITLED Summary of this function goes here
    #   Detailed explanation goes here
    # sigma_eps_unique = sigma_eps_unique_scl

    tau_unique = tau[1:]
    tau_common = tau[0]
    
    n_y = y.shape[0]
    nobs = y.shape[1]
    y = y - tau_unique    # Eliminates tau_unique from y ;

    # Brute Force Calculation
    ns = 2*n_y

    # First n_y elements of state are alpha_eps; second n_y elements of state are alpha_tau
    Q = np.diag(sigma_dalpha**2)     # Q_t = Q
    F = np.eye(ns)              # F_t = F
    H = np.concatenate((np.kron(eps_common[:, None], np.eye(3)), np.kron(tau_common[:, None], np.eye(3))), axis=1).reshape(-1, 3, 6)
    H = np.transpose(H, (1, 2, 0))
    R = np.transpose(np.eye(n_y) * sigma_eps_unique.T[:, None, :]**2, (1, 2, 0))
    
    # random draws from N(0, 1)
    rand1 = nsample2(ns, nobs)
    rand2 = nsample2(ns, nobs+1)
    
    # Set up KF to run
    # Initial conditions
    x0 = np.zeros(ns)
    P0 = prior_var_alpha
    
    
    H = np.array(H, float, order="F")
    R = np.array(R, float, order="F")
    F = np.array(F, float, order="F")
    Q = np.array(Q, float, order="F")
    x0 = np.array(x0, float, order="F")
    P0 = np.array(P0, float, order="F")
    y = np.array(y, float, order="F")
    rand1 = np.array(rand1, float, order="F")
    rand2 = np.array(rand2, float, order="F")

    x_draw = cython_module.sample_alpha_tvp_by_kalman(H, R, F, Q, x0, P0, y, rand2)   

    dalpha_eps = x_draw[:n_y, 1:] - x_draw[:n_y, :-1]
    dalpha_tau = x_draw[n_y:, 1:] - x_draw[n_y:, :-1]
    alpha_eps = x_draw[:n_y, 1:]
    alpha_tau = x_draw[n_y:, 1:]
    dalpha = np.concatenate((dalpha_eps, dalpha_tau), axis=0)

    return alpha_eps, alpha_tau, dalpha



def sample_eps_tau_multivar_cython(cython_module, y, alpha_eps, alpha_tau, sigma_eps_scl, sigma_dtau):
    """
    MUCSV 의 estimation 과정에서 1.a.1 번째 sampling process 를 구현한 함수.
    factor matrix tau 와 epsilon 을 sampling 하는 함수.
    위의 함수와 비교해 보면, factor analysis 에서 factor loading 에 corresponding 하는 latent factor 를 sampling.
    그러나 위의 함수와 달리 latent common factor 과 함께 sector-specific factor도 함께 sampling.

    Parameters
    ----------
    y : numpy array of shape (n_y, T)
        입력 데이터 matrix. 
        n_y 개의 number of observed variable, 
        T 개의 number of observation 을 가지고 있다.
    alpha_eps : numpy array of shape (n_y,T)
                이전 시점에 sampling 된 factor loading for epsilon
    alpha_tau : numpy array of shape (n_y,T)
                이전 시점에 sampling 된 factor loading for tau
    sigma_eps_scl : numpy array of shape (n_y+1,T)
                    scaled 된 epsilon 의 volatility variable - sigma_epsilon
    sigma_dtau : numpy array of shape (n_y+1,T)
                 scaled 된 tau 의 volatility variable - sigma_tau
    
    Returns
    -------
    eps_common : numpy array of shape (T,)
                 latent factor for alpha_epsilon. epsilon_{c,t} 로 표기.
    tau_f : numpy array of shape (N,T)
            tua sampling 과정에서 kalman smoothing 과정을 통과하지 않은 결과. (backward pass 를 통과하지 않은 결과)
    dtau : numpy array of shape (N,T)
           tau 의 1차 차분값 vector
    tau : numpy array of shape (N,T)
          latent factor for alpha_tau and alpha_epsilon(common component) and sector specific component.
          (factor loading 에 corresponding 되지 않은 tau)
          tau_{c,t} + tau_{i,t} 변수 함께 sampling된 결과.
    """

    sigma_eps_common = sigma_eps_scl[0]
    sigma_eps_unique = sigma_eps_scl[1:]
    sigma_dtau_common = sigma_dtau[0]
    sigma_dtau_unique = sigma_dtau[1:]

    samll = 1e-6; big = 1e6
    n_y = y.shape[0]
    nobs = y.shape[1]

    # Set up State Vector
    # --- State Vector
    #     (1) eps(t)
    #     (2) tau(t)
    #     (3) tau_u(t)

    ns = 2 + n_y        # size of state
    F = np.zeros((ns, ns));     F[1:, 1:] = np.eye(ns-1)
    H = np.concatenate((alpha_eps, alpha_tau, np.eye(n_y).reshape(-1, 1).dot(np.ones((1, nobs)))), axis=0).reshape(n_y, ns, -1, order='F')
    Q = np.eye(ns) * np.concatenate((sigma_eps_common[None, :]**2, sigma_dtau_common[None, :]**2, sigma_dtau_unique**2), axis=0).T[:, None, :]
    Q = np.transpose(Q, (1, 2, 0))
    R = np.eye(n_y) * sigma_eps_unique.T[:, None, :]**2
    R = np.transpose(R, (1, 2, 0))
    
    rand1 = nsample2(ns, nobs)
    rand2 = nsample2(ns, nobs+1)

    # Set up KF to run
    # Initial conditions
    x0 = np.zeros(ns)               # x_0|0
    P0 = np.zeros((ns, ns))         # P_0|0
    P0[2:, 2:] = big*np.eye(n_y)    # Vague prior for tau_unique initial values 
    
    
    H = np.array(H, float, order="F")
    R = np.array(R, float, order="F")
    F = np.array(F, float, order="F")
    Q = np.array(Q, float, order="F")
    x0 = np.array(x0, float, order="F")
    P0 = np.array(P0, float, order="F")
    y = np.array(y, float, order="F")
    rand1 = np.array(rand1, float, order="F")
    rand2 = np.array(rand2, float, order="F")

    x_draw_f, x_draw = cython_module.sample_eps_tau_by_kalman(H, R, F, Q, x0, P0, y, rand1, rand2)   

    ########### return x_draw, x_draw_f ##############
    eps_common = x_draw[0, 1:]
    tau_f = x_draw_f[1:, 1:]    
    dtau = x_draw[1:, 1:] - x_draw[1:, :-1]
    tau = x_draw[1:, 1:]
    
    return eps_common, tau_f, dtau, tau


