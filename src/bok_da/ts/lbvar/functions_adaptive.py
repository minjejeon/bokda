import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from numpy.linalg import inv, solve
from scipy.linalg import solve_triangular, cholesky
from scipy.stats import uniform, norm, invgamma

from scipy.special import loggamma, digamma, polygamma
from scipy.optimize import bisect
from numpy.random import randn

from tqdm import tqdm

from .. import container_class



def EXOv_total_maker(Parameters, Raw_Data):
    """
    """
    n = Parameters.n 
    T = Parameters.T
    Trend = Parameters.Trend

    if Trend == 1:
        Raw_Data.EXOv = np.ones((T, 1))
    elif Trend == 2:
        Raw_Data.EXOv = np.column_stack((np.ones((T, 1)), np.arange(1, T + 1).reshape(-1, 1)))
    elif Trend == 3:
        Raw_Data.EXOv = np.column_stack((
            np.ones((T, 1)),
            np.arange(1, T + 1).reshape(-1, 1),
            (np.arange(1, T + 1) ** 2).reshape(-1, 1)
        ))

    TT = n - 4
    if Trend == 1:
        Raw_Data.EXOv_AR = np.ones((TT, 1))
    elif Trend == 2:
        Raw_Data.EXOv_AR = np.column_stack((np.ones((TT, 1)), np.arange(1, TT + 1).reshape(-1, 1)))
    elif Trend == 3:
        Raw_Data.EXOv_AR = np.column_stack((
            np.ones((TT, 1)),
            np.arange(1, TT + 1).reshape(-1, 1),
            (np.arange(1, TT + 1) ** 2).reshape(-1, 1)
        ))
    return Raw_Data


def LBVAR_variable_maker(Raw_Data, Parameters):
    """
    """
    T = Parameters.T
    nvar = Parameters.nvar
    p = Parameters.p
    k = Parameters.k
    n = Parameters.n


    Z = np.empty((T, k))
    for i in range(p) :
        Z[:, i * nvar : (i+1) * nvar] = Raw_Data.Set[p-(i+1):(n)-(i+1),:]
        # Raw_Data.Set.shape[0] = Paramters['n']
    Z = np.column_stack((Raw_Data.EXOv, Z))
    Y = Raw_Data.Set[p:,:]

    Raw_Data.Z = Z
    Raw_Data.Y = Y
    return Raw_Data


def Prior_Maker(Raw_Data, Parameters, Priors, hyperparameters):
    """
    """
    nvar = Parameters.nvar 
    n = Parameters.n
    num_of_par = Parameters.num_of_parameter
    c = Parameters.c
    p = Parameters.p
    
    kappa_3 = hyperparameters['kappa_3']
    kappa_4 = hyperparameters['kappa_4']

    ###############################################
    # Make Sigma_hat
    ###############################################

    Sigma_hat_dum = np.zeros((nvar, nvar))

    for i in range(0, Parameters.nvar) :
        Y_AR = Raw_Data.Set[:, i]
        X_AR = np.empty((n-4, 4))

        for j in range(0, 4) :
            X_AR[:, j] = Y_AR[4-(j+1):Y_AR.shape[0]-(j+1),:].reshape(-1)

        X_AR = np.column_stack((Raw_Data.EXOv_AR, X_AR))
        Y_AR = Y_AR[4:,:]
        
        Beta_AR = np.matmul(np.linalg.inv(np.matmul(X_AR.T, X_AR)), np.matmul(X_AR.T, Y_AR))
        Tem_sq_sum = np.power((Y_AR - np.matmul(X_AR,Beta_AR)),2)
        Sigma_hat_dum[i,i] = np.mean(Tem_sq_sum)
    
    Priors.Sigma_hat = Sigma_hat_dum

    ###############################################
    # Make C & Index_kappa 
    ###############################################
    C = np.array([np.zeros((num_of_par + i, 1)) for i in range(nvar)], dtype=object)

    Index_kappa_1 = np.zeros((nvar, p))
    Index_kappa_2 = np.zeros((nvar, num_of_par - c - p))
    
    for ii in range(nvar):              #### [0부터 시작]

       lag = 1
       jj = 0 
       count1 = 0
       count2 = 0

       for j in range(num_of_par + ii): #### [0부터 시작]

            if jj >=  nvar :            #### [0부터 시작]이니까 등호 추가
                jj = 0                  #### [0부터 시작]
                lag += 1
            
            if j < num_of_par:          #### [0부터 시작]이니까 등호 제거
                if j < c:               #### [0부터 시작]이니까 등호 제거
                    C[ii][j] = 2 * kappa_4 * Priors.Sigma_hat[ii, ii]

                elif j == (ii + c + (nvar * (lag - 1))):   
                    C[ii][j] = 2 * (1/lag**2)
                    Index_kappa_1[ii, count1] = j         #### 여기에 속하는 j들은 Index1
                    count1 += 1

                else :                                     
                    C[ii][j] = (2 / (lag**2)) * (Priors.Sigma_hat[ii, ii] / Priors.Sigma_hat[jj, jj])
                    Index_kappa_2[ii, count2] = j        #### 여기에 속하는 j들은 Index2
                    count2 += 1
            else:
                C[ii][j] = 2 * kappa_3 * (Priors.Sigma_hat[ii, ii] / Priors.Sigma_hat[jj, jj])
            
            if j == (c-1):              #### [0부터 시작]이니까 하나 빼주기
                jj = 0                  #### [0부터 시작]
            else:
                jj += 1  
    
    Priors.C = C
    Priors.Index_kappa_1 = Index_kappa_1.astype(int)
    Priors.Index_kappa_2 = Index_kappa_2.astype(int)

    ###############################################
    # Make M
    ###############################################
    M = np.array([np.zeros((num_of_par + i, 1)) for i in range(nvar)], dtype=object)

    for i in Parameters.RV_list:
        M[i][i+c] = 1

    Priors.M = M

    return Priors



######################################################

def psi(x, alpha, lambd):
    """Calculate the psi function."""
    return -alpha * (np.cosh(x) - 1) - lambd * (np.exp(x) - x - 1)

def psi_prime(x, alpha, lambd):
    """Calculate the derivative of the psi function."""
    return -alpha * np.sinh(x) - lambd * (np.exp(x) - 1)

def chi(x, s, sp, t, tp, theta, eta, iota, xi):
    """Calculate the value of chi function."""
    A = np.where((-sp <= x) & (x <= tp), 1, 0)
    B = np.where(x > tp, np.exp(-eta - iota * (x - t)), 0)
    C = np.where(x < -sp, np.exp(-theta + xi * (x + s)), 0)
    return A + B + C

def gig_generator_vec(p, a, b):
    """
    Generate samples from the Generalized Inverse Gaussian (GIG) distribution.
    Parameters:
        p (float): Shape parameter.
        a (float): Scale parameter a.
        b (np.ndarray): Scale parameter b (array).
    Returns:
        np.ndarray: Array of GIG samples.
    """

    lambd = -p if p < 0 else p
    omega = np.sqrt(a * b)
    alpha = np.sqrt(omega**2 + lambd**2) - lambd
    psi_1 = -psi(1, alpha, lambd)

    t = np.zeros_like(b)
    s = np.zeros_like(b)
    cond1 = (1 / 2 <= psi_1) & (psi_1 <= 2)
    cond2 = psi_1 > 2
    cond3 = psi_1 < 1 / 2
    t[cond1] = 1
    s[cond1] = 1
    t[cond2] = np.sqrt(2 / (alpha[cond2] + lambd))
    s[cond2] = np.sqrt(4 / (alpha[cond2] * np.cosh(1) + lambd))
    t[cond3] = np.log(4 / (alpha[cond3] + 2 * lambd))
    # s[cond3] = np.minimum(
    #     1 / lambd,
    #     np.log(1 + 1 / alpha[cond3] + np.sqrt((1 / alpha[cond3] ** 2) + 2 / alpha[cond3])),
    # )

    alpha_cond3 = alpha[cond3] + 1e-10
    s[cond3] = np.minimum(
        1 / lambd,
        np.log(1 + 1 / alpha_cond3 + np.sqrt((1 / alpha_cond3 ** 2) + 2 / alpha_cond3)),
    )  # NOTE: divide by zero 방지

    eta = -psi(t, alpha, lambd)
    iota = -psi_prime(t, alpha, lambd)
    theta = -psi(-s, alpha, lambd)
    xi = psi_prime(-s, alpha, lambd)
    p_param = 1 / xi
    r_param = 1 / iota
    tp = t - r_param * eta
    sp = s - p_param * theta
    q = tp + sp

    samples = np.full(len(b), np.nan)  
    cond = np.isnan(samples) 

    while np.any(cond): 
        u = np.random.rand(np.sum(cond))
        v = np.random.rand(np.sum(cond))
        w = np.random.rand(np.sum(cond))

        x_candidates = np.where(
            u < q[cond] / (p_param[cond] + q[cond] + r_param[cond]),
            -sp[cond] + q[cond] * v,
            np.where(
                u < (q[cond] + r_param[cond]) / (p_param[cond] + q[cond] + r_param[cond]),
                tp[cond] + r_param[cond] * np.log(1 / v),
                -sp[cond] - p_param[cond] * np.log(1 / v),
            ),
        )

        accept = (
            w * chi(x_candidates, s[cond], sp[cond], t[cond], tp[cond], theta[cond], eta[cond], iota[cond], xi[cond])
            <= np.exp(psi(x_candidates, alpha[cond], lambd))
        )

        samples[cond] = np.where(accept, x_candidates, samples[cond])

        cond = np.isnan(samples)

    samples = (lambd / omega + np.sqrt(1 + (lambd**2) / (omega**2))) * np.exp(samples)
    samples = np.where(p < 0, 1 / samples, samples) / np.sqrt(a / b)
    return samples



############################################
# H_drawing
############################################
def h_drawing(y, h, sigma_h, h0, Parameters):
    """
    Draw latent variable h using a mixture of normals.
    Parameters:
        y (np.ndarray): Observation vector.
        h (np.ndarray): Latent state vector.
        sigma_h (float): Variance of h.
        h0 (float): Initial state.
    Returns:
        np.ndarray: Updated latent state vector h.
    """

    T = Parameters.T ## [코멘트] : 인풋으로 안받고 이렇게 뺌. 
    
    # Constants for the mixture of normals
    mu_i = np.array([-10.12999, -3.97281, -8.56686, 2.77786, 0.61942, 1.79518, -1.08819])
    sigma2_i = np.array([5.79596, 2.61369, 5.17950, 0.16735, 0.64009, 0.34023, 1.26261])
    p_i = np.array([0.00730, 0.10556, 0.00002, 0.04395, 0.34001, 0.24566, 0.25750])
    sigma_i = np.sqrt(sigma2_i)

    # Initialize arrays
    S = np.empty(T, dtype=int)
    P = np.empty(7)

    # -----------------------------------------
    # Step 1: Sample mixture components
    # -----------------------------------------
    
    # Vectorized computation
    Y = y[:, None]  # Shape (T, 1)
    H = h[:, None]  # Shape (T, 1)
    D = mu_i - 1.2704  # Shape (7,)
    PDFs = norm.pdf(Y, loc=H + D, scale=sigma_i)  # Shape (T, 7)
    P = PDFs * p_i  # Shape (T, 7)
    P /= P.sum(axis=1, keepdims=True) 

    # Draw S using cumulative sums
    U = np.random.rand(T)  # Shape (T,)
    cumsum_P = np.cumsum(P, axis=1)  # Shape (T, 7)
    S = (cumsum_P > U[:, None]).argmax(axis=1)  # Vectorized searchsorted


    # -----------------------------------------    
    # Step 2: Draw h
    # -----------------------------------------

    # Construct matrix H_phi
    H_phi = np.eye(T) - np.eye(T, k=-1)
    # Compute alpha
    alpha = solve(H_phi, np.concatenate(([h0], np.zeros(T - 1))))

    # Construct diagonal matrices and terms
    d = mu_i[S] - 1.2704
    inv_sigma2_i_S = 1.0 / sigma2_i[S]
    Sigma_y_inv = np.diag(inv_sigma2_i_S)
    Sigma_h_inv = (1 / sigma_h) * np.eye(T)
    
    # Compute K_h and its Cholesky decomposition
    K_h = H_phi.T @ Sigma_h_inv @ H_phi + Sigma_y_inv
    Chol_K_h = cholesky(K_h, lower=True)
    # Compute mean of h (ĥ)
    h_hat = solve(K_h, H_phi.T @ Sigma_h_inv @ H_phi @ alpha + Sigma_y_inv @ (y - d))

    # Sample h using the Cholesky decomposition
    rand_vec = np.random.randn(T)
    h[:] = h_hat + solve_triangular(Chol_K_h, rand_vec, lower=True) # 여기서도 triangular solve? 쓰면 빨라지는건가?

    return h


############################################
# nu_sampling
############################################
def log_p(x, sum_kappa, sum_kappa_log,  Parameters, hyperparameters):

    nvar, p = Parameters.nvar, Parameters.p
    d1, d2 = hyperparameters['d_1'], hyperparameters['d_2']
    
    log_p = (nvar**2) * p * (x * np.log(x / 2) - loggamma(x)) + \
            (x - 1) * sum_kappa_log -(x / 2) * sum_kappa + \
            (d1 - 1) * np.log(x) -d2 * x

    return log_p

def dlog_p(x, sum_kappa, sum_kappa_log,  Parameters, hyperparameters):
    
    nvar, p = Parameters.nvar, Parameters.p
    d1, d2 = hyperparameters['d_1'], hyperparameters['d_2']
    
    dlog_p = (nvar**2) * p * (np.log(x / 2) + 1 - digamma(x))+ \
             sum_kappa_log -(1 / 2) * sum_kappa +\
             (d1 - 1) / x - d2
    return dlog_p

def d2log_p(x, Parameters, hyperparameters):

    nvar, p = Parameters.nvar, Parameters.p
    d1 = hyperparameters['d_1']

    # trigamma(x) = polygamma(1, x)
    d2log_p = (nvar**2) * p * ((1 / x) - polygamma(1, x)) -(d1 - 1) / (x**2)

    return d2log_p


# Main Function
def nu_p_sampling(psi_i, nu_p, Parameters, Prior, hyperparameters):

    nvar = Parameters.nvar
    Index_kappa_1 = Prior.Index_kappa_1
    Index_kappa_2 = Prior.Index_kappa_2    

    # Initialize sums
    sum_kappa = 0
    sum_kappa_log = 0

    # Compute sums
    for ii in range(nvar):  # Adjusted for Python's 0-based indexing
        psi = psi_i[ii]
        sum_kappa += np.sum(psi[Index_kappa_1[ii]]) + np.sum(psi[Index_kappa_2[ii]])
        sum_kappa_log += np.sum(np.log(psi[Index_kappa_1[ii]])) + np.sum(np.log(psi[Index_kappa_2[ii]]))

    # Find the argmin using bisection
    argmin = bisect(
        lambda x: dlog_p(x, sum_kappa, sum_kappa_log, Parameters, hyperparameters),
        0.001, 10
    )

    # Compute the Hessian
    hessian = -d2log_p(argmin, Parameters, hyperparameters)
    hessian = np.sqrt(1 / hessian)

    # Propose a new nu_p
    new_nu_p = nu_p + hessian * randn()

    # Metropolis-Hastings acceptance step
    alpha = min(log_p(new_nu_p, sum_kappa, sum_kappa_log, Parameters, hyperparameters) -
                log_p(nu_p, sum_kappa, sum_kappa_log, Parameters, hyperparameters), 0)
    u = np.log(uniform(0, 1).rvs())

    # Accept or reject
    if u < alpha:
        nu_p = new_nu_p

    return nu_p


############################################
# MCMC_MNG -> posterior_draw로 이름 변경
# gig_generator
############################################
def mcmc_mng(Raw_Data, Parameters, Prior, hyperparameters):
    """
    Perform MCMC sampling for the specified model.

    Parameters:
        raw_data (dict): Data for the model.
        parameters (dict): Model parameters.
        prior (dict): Prior distributions and parameters.
        hyperparameters (dict): Hyperparameters for the model.

    Returns:
        dict: Contains the sampled variables after burn-in.
    """
    # Draws = {}
    Draws = container_class.Container()

    # Initialize storage
    theta_i = [np.empty((Parameters.nvar * Parameters.p + Parameters.c + i,
                         Parameters.ndraws + Parameters.burnin)) for i in range(Parameters.nvar)]
    h_i = [np.empty((Parameters.T, Parameters.ndraws + Parameters.burnin)) for i in range(Parameters.nvar)]
    psi_i = [np.empty((Parameters.nvar * Parameters.p + Parameters.c + i,
                       Parameters.ndraws + Parameters.burnin)) for i in range(Parameters.nvar)]
    u = [np.empty((Parameters.T, Parameters.ndraws + Parameters.burnin)) for i in range(Parameters.nvar)]
    kappa_1 = np.empty(Parameters.ndraws + Parameters.burnin)
    kappa_2 = np.empty(Parameters.ndraws + Parameters.burnin)
    nu_p = np.empty(Parameters.ndraws + Parameters.burnin)
    h_0 = np.empty((Parameters.nvar, Parameters.ndraws + Parameters.burnin))
    sigma_h = np.empty((Parameters.nvar, Parameters.ndraws + Parameters.burnin))
    inv_vh = (1 / hyperparameters["v_hi"]) * np.eye(Parameters.nvar)

    # Initial values
    nu_p_current = 0.6
    kappa_1_current = 0.4
    kappa_2_current = 0.001
    h_0_current = np.log(np.diag(Prior.Sigma_hat))
    h_i_current = [np.full(Parameters.T, h_0_current[i]) for i in range(Parameters.nvar)]
    psi_i_current = [0.3 * np.ones(Parameters.nvar * Parameters.p + Parameters.c + i)
                     for i in range(Parameters.nvar)]
    sigma_h_current = np.full(Parameters.nvar, hyperparameters["S_hi"])
    a_h = hyperparameters["a_hi"] * np.ones(Parameters.nvar)    
    h_s = np.empty(Parameters.nvar)    
    theta_i_current = [np.empty(Parameters.nvar * Parameters.p + Parameters.c + i) for i in range(Parameters.nvar)]
    u_i_current = [np.empty(Parameters.T) for i in range(Parameters.nvar)]

    # MCMC iterations
    for d in tqdm(range(Parameters.burnin + Parameters.ndraws)):
        # Step 1: Draw theta
        for ii in range(Parameters.nvar):
            c_ii = deepcopy(Prior.C[ii])
            psi_ii = psi_i_current[ii]
            c_ii[Prior.Index_kappa_1[ii]] = (Prior.C[ii][Prior.Index_kappa_1[ii]]
                                                 * kappa_1_current * psi_ii[Prior.Index_kappa_1[ii]].reshape(-1, 1))
            c_ii[Prior.Index_kappa_2[ii]] = (Prior.C[ii][Prior.Index_kappa_2[ii]] 
                                                 * kappa_2_current * psi_ii[Prior.Index_kappa_2[ii]].reshape(-1, 1))
            v_ii_inv = np.diag(1/c_ii.reshape(-1))
            y_ii = Raw_Data.Y[:, ii]
            x_ii = np.hstack((Raw_Data.Z, -Raw_Data.Y[:, :ii]))
            omega_h_inv = np.diag(np.exp(-h_i_current[ii]))
            k_ii = v_ii_inv + x_ii.T @ omega_h_inv @ x_ii
            k_ii = (k_ii + k_ii.T) / 2
            chol = cholesky(k_ii, lower=True)
            tmp = solve_triangular(chol, v_ii_inv @ Prior.M[ii] + x_ii.T @ omega_h_inv @ y_ii, lower=True)
            theta_hat_ii = solve_triangular(chol.T, tmp, lower=False)
            tmp = solve_triangular(chol.T, np.random.randn(Parameters.nvar * Parameters.p + Parameters.c + ii), lower=False)
            theta_i_current[ii] = theta_hat_ii.reshape(-1) + tmp
            theta_i[ii][:, d] = theta_i_current[ii]
            u_i_current[ii] = y_ii.reshape(1,-1) - (x_ii @ theta_i_current[ii])
            u[ii][:, d] = u_i_current[ii]

        # Step 2: Draw psi
        for ii in range(Parameters.nvar):
            c_ii = deepcopy(Prior.C[ii])
            psi_ii = psi_i_current[ii]
            theta_ii = theta_i_current[ii]
            m_ii = Prior.M[ii]
            c_ii[Prior.Index_kappa_1[ii]] = (Prior.C[ii][Prior.Index_kappa_1[ii]] * kappa_1_current)
            c_ii[Prior.Index_kappa_2[ii]] = (Prior.C[ii][Prior.Index_kappa_2[ii]] * kappa_2_current)
            psi_ii = np.empty(Parameters.nvar * Parameters.p + Parameters.c + ii)
            term = (theta_ii - m_ii[:,0])**2 / c_ii[:,0]
            psi_ii = np.maximum(gig_generator_vec(nu_p_current - 0.5, nu_p_current, term), 1e-10)
            psi_i[ii][:, d] = psi_ii
            psi_i_current[ii] = psi_ii

        # Step 3: Draw kappa_1 and kappa_2
        C = np.array(Prior.C)
        M = np.array(Prior.M)
        Index_kappa_1 = Prior.Index_kappa_1
        Index_kappa_2 = Prior.Index_kappa_2
        nvar = Parameters.nvar
        p = Parameters.p
        c_11 = hyperparameters["c_11"]
        c_21 = hyperparameters["c_21"]
        psi_kappa_1 = np.array([psi_i_current[ii][Index_kappa_1[ii]] for ii in range(nvar)])
        psi_kappa_2 = np.array([psi_i_current[ii][Index_kappa_2[ii]] for ii in range(nvar)])
        C_kappa_1 = np.squeeze(np.array([C[ii][Index_kappa_1[ii]] for ii in range(nvar)])) * psi_kappa_1
        C_kappa_2 = np.squeeze(np.array([C[ii][Index_kappa_2[ii]] for ii in range(nvar)])) * psi_kappa_2
        theta_kappa_1 = np.array([theta_i_current[ii][Index_kappa_1[ii]] for ii in range(nvar)])
        theta_kappa_2 = np.array([theta_i_current[ii][Index_kappa_2[ii]] for ii in range(nvar)])
        M_kappa_1 = np.array([M[ii][Index_kappa_1[ii]] for ii in range(nvar)])
        M_kappa_2 = np.array([M[ii][Index_kappa_2[ii]] for ii in range(nvar)])
        b1 = np.sum((theta_kappa_1 - np.squeeze(M_kappa_1))**2 / C_kappa_1)
        b2 = np.sum((theta_kappa_2 - np.squeeze(M_kappa_2))**2 / C_kappa_2)
        kappa_1_current = max(gig_generator_vec(hyperparameters["c_11"] - (Parameters.nvar * Parameters.p) / 2, 2 * hyperparameters["c_21"], np.array([b1]))[0], 1e-10)
        kappa_2_current = max(gig_generator_vec(hyperparameters["c_12"] - ((Parameters.nvar - 1) * Parameters.nvar * Parameters.p) / 2, 2 * hyperparameters["c_22"], np.array([b2]))[0], 1e-10)
        kappa_1[d] = kappa_1_current
        kappa_2[d] = kappa_2_current

        # Step 4: Draw nu_p
        nu_p_current = nu_p_sampling(psi_i_current, nu_p_current, Parameters, Prior, hyperparameters)
        nu_p[d] = nu_p_current

        # Step 5: Draw h_i
        for ii in range(Parameters.nvar):
            tmp = np.log(np.array(u_i_current[ii])**2 + 10**(-4)).reshape(-1)
            h_i_current[ii] = h_drawing(tmp, h_i_current[ii], sigma_h_current[ii], h_0_current[ii], Parameters)
            h_i[ii][:, d] = h_i_current[ii]

        # Step 6: Draw h_0
        h_s = np.array([h_i_current[ii][0] for ii in range(Parameters.nvar)])
        # h_0인지 h_theta인지 모르겠네... -> 나중에 논문 확인해서 수정 (어차피 변수명이고, 저장되는 값도 아니라서 큰 상관은 없을 듯)
        kh_0 = np.diag(1/sigma_h_current) + inv_vh        
        kh_0 = (kh_0 + kh_0.T) / 2
        k_0_hat = solve(kh_0, inv_vh @ a_h + np.diag(1/sigma_h_current) @ h_s)

        h_0_current = k_0_hat + solve_triangular(cholesky(kh_0, lower=False), np.random.randn(Parameters.nvar), lower=False)
        h_0[:, d] = h_0_current

        # Step 7: Draw sigma_h
        for ii in range(Parameters.nvar):
            sum_h = np.sum((h_i_current[ii] - np.concatenate(([h_0_current[ii]], h_i_current[ii][:-1])))**2)
            sigma_h_current[ii] = invgamma.rvs(a = hyperparameters["nu_hi"] + Parameters.T / 2,
                                   scale=hyperparameters["S_hi"] + 0.5 * sum_h)
            sigma_h[ii, d] = sigma_h_current[ii]
    
    # Assign values to the dictionary keys one by one
    Draws.theta_i = [theta[:, Parameters.burnin:] for theta in theta_i]
    Draws.h_i = [h[:, Parameters.burnin:] for h in h_i]
    Draws.psi_i = [psi[:, Parameters.burnin:] for psi in psi_i]
    Draws.u = [u_[:, Parameters.burnin:] for u_ in u]
    Draws.kappa_1 = kappa_1[Parameters.burnin:]
    Draws.kappa_2 = kappa_2[Parameters.burnin:]
    Draws.nu_p = nu_p[Parameters.burnin:]
    Draws.h_0 = h_0[:, Parameters.burnin:]
    Draws.sigma_h = sigma_h[:, Parameters.burnin:]
    return Draws



def Reduced_Transform(Parameters, Draws):
    I = np.eye(Parameters.nvar)
    Draws.A_matrix = np.zeros((Parameters.nvar, Parameters.nvar, Parameters.ndraws))
    Draws.Bet = np.empty((Parameters.nvar, Parameters.k + Parameters.c, Parameters.ndraws))
    Draws.Sigma_struct = np.zeros((Parameters.nvar, Parameters.nvar, Parameters.ndraws, Parameters.Forecast_period))
    Draws.Sigma = np.zeros((Parameters.nvar, Parameters.nvar, Parameters.ndraws, Parameters.Forecast_period))
    for d in tqdm(range(Parameters.ndraws)):
        #######################################
        # Make A_matrix & Bet
        #######################################
        # A_mat_tmp is a lower triangular matrix
        A_mat_tmp = I
        Bet_tmp = np.zeros((Parameters.nvar, Parameters.k + Parameters.c))
        for i in range(Parameters.nvar):
            if i >= 1:
                A_mat_tmp[i, 0:i] = Draws.theta_i[i][(Parameters.k + Parameters.c):, d]
            Bet_tmp[i, :] = Draws.theta_i[i][0:Parameters.k + Parameters.c, d]
        Draws.A_matrix[:, :, d] = A_mat_tmp
        inv_A_mat_tmp = inv(A_mat_tmp)
        Draws.Bet[:, :, d] = inv_A_mat_tmp @ Bet_tmp
        #####################################
        # Make Sigma and Sigma_struct
        #####################################
        h = np.zeros(Parameters.nvar)
        for j in range(Parameters.Forecast_period):
            if j == 0:
                for q in range(Parameters.nvar):
                    h[q] = Draws.h_i[q][-1, d]
            h = h + np.sqrt(Draws.sigma_h[:, d]) * randn(Parameters.nvar) # element wise multiplication
            Draws.Sigma_struct[:, :, d, j] = np.diag(np.exp(h))
            Draws.Sigma[:, :, d, j] = inv_A_mat_tmp @ Draws.Sigma_struct[:, :, d, j] @ inv_A_mat_tmp.T
            Draws.Sigma[:, :, d, j] = (Draws.Sigma[:, :, d, j] + Draws.Sigma[:, :, d, j].T)/2
    return Draws




def Forecast_function(Raw_Data, Parameters, Draw, verbose=True):

    U_forecast = np.empty((Parameters.nvar, Parameters.Forecast_period, Parameters.ndraws))
    for d in range(0, Parameters.ndraws) :
        for i in range(0, Parameters.Forecast_period) : 
            U_forecast[:,i,d] = np.random.multivariate_normal(np.zeros(Parameters.nvar), Draw.Sigma[:,:,d, i]) ## 기존 함수에서 Draw.Sigma 부분만 변함
 
    Y_Forecast = np.empty((Parameters.nvar, Parameters.Forecast_period, Parameters.ndraws))
    X_Forecast_init = np.empty((Parameters.nvar, Parameters.p))
    
    if verbose:
        iters = tqdm(range(0, Parameters.ndraws))
    else:
        iters = range(0, Parameters.ndraws)
    # for d in tqdm(range(0, Parameters.ndraws)) :
    for d in iters:
        for p in range(0, Parameters.p) :
            X_Forecast_init[:,p] = Raw_Data.Y[-(p+1),:]
        
        Bet = Draw.Bet[:,:,d]

        for j in range(0, Parameters.Forecast_period) :
            if Parameters.Trend == 1 :
                X_Forecast = np.insert(X_Forecast_init.reshape(-1, order = 'F'), 0, 1)
            elif Parameters.Trend == 2 :
                X_Forecast = np.insert(X_Forecast_init.reshape(-1, order = 'F'), 0, Parameters.T+j)
                X_Forecast = np.insert(X_Forecast.reshape(-1, order = 'F'), 0, 1)
            elif Parameters.Trend == 3 :
                X_Forecast = np.insert(X_Forecast_init.reshape(-1, order = 'F'), 0, (Parameters.T+j)**2)
                X_Forecast = np.insert(X_Forecast.reshape(-1, order = 'F'), 0, Parameters.T+j)
                X_Forecast = np.insert(X_Forecast.reshape(-1, order = 'F'), 0, 1)
            Y = Bet @ X_Forecast + U_forecast[:,j,d]
            Y_Forecast[:,j,d] = Y
            X_Forecast_init = np.hstack((Y.reshape(-1,1), X_Forecast_init[:,0:Parameters.p-1]))

    Mean = np.empty((Parameters.nvar, Parameters.Forecast_period))
    UP = np.empty((Parameters.nvar, Parameters.Forecast_period))
    DOWN = np.empty((Parameters.nvar, Parameters.Forecast_period))
    for i in range(0, Parameters.nvar) :
        for j in range(0, Parameters.Forecast_period) :
            Mean[i,j] = np.mean(Y_Forecast[i,j,:])
            UP[i,j] = np.quantile(Y_Forecast[i,j,:], Parameters.pereb)
            DOWN[i,j] = np.quantile(Y_Forecast[i,j,:], 1 - Parameters.pereb)
            
    Forecast_Results = container_class.Container()
    Forecast_Results.Total = Y_Forecast
    Forecast_Results.Mean = Mean.T
    Forecast_Results.UP = UP.T
    Forecast_Results.DOWN = DOWN.T
    return Forecast_Results