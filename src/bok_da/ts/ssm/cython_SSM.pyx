import numpy as np
cimport numpy as np
from libc.math cimport pow, sqrt, log, pi


cpdef double clnpdfmvn(np.ndarray[double, ndim=2] x, 
                       np.ndarray[double, ndim=2] m, 
                       np.ndarray[double, ndim=2] C):
    cdef int d = x.shape[0]
    cdef double denom0 = pow(2 * pi, d / 2.0)
    cdef double denom, mahal, logp
    cdef np.ndarray[double, ndim=2] diff = x - m
    cdef np.ndarray[double, ndim=2] invC

    # Attempt to invert C, assuming it's not singular
    try:
        invC = np.linalg.inv(C)
    except np.linalg.LinAlgError:
        raise ValueError("Covariance matrix is singular or not square.")
    
    denom = denom0 * sqrt(abs(np.linalg.det(C)))
    mahal = np.dot(np.dot(diff.T, invC), diff)
    logp = -0.5 * mahal - log(denom)
    
    return logp


cpdef ckalman_filter_DFM(np.ndarray[double, ndim=2] Y, 
                         int K, 
                         np.ndarray[double, ndim=2] H, 
                         np.ndarray[double, ndim=2] F, 
                         np.ndarray[double, ndim=2] Mu, 
                         np.ndarray[double, ndim=2] Omega, 
                         np.ndarray[double, ndim=2] Sigma, 
                         np.ndarray[double, ndim=2] U_LL, 
                         np.ndarray[double, ndim=2] P_LL):
    cdef int T = Y.shape[0]
    cdef np.ndarray[double, ndim=1] lnLm = np.zeros(T)
    cdef np.ndarray[double, ndim=2] U_ttm = np.zeros((T, K))
    cdef np.ndarray[double, ndim=3] P_ttm = np.zeros((T, K, K))
    cdef np.ndarray[double, ndim=2] U_tLm = np.zeros((T, K))
    cdef np.ndarray[double, ndim=3] P_tLm = np.zeros((T, K, K))

    cdef np.ndarray[double, ndim=2] U_tL, P_tL, y_tL, f_tL, invf_tL, U_tt, P_tt
    cdef double lnp
    cdef int t

    for t in range(T):
        U_tL = Mu + np.dot(F, U_LL)
        P_tL = np.dot(np.dot(F, P_LL), F.T) + Omega

        y_tL = np.dot(H, U_tL)
        f_tL = np.dot(np.dot(H, P_tL), H.T) + Sigma

        y_t = Y[t, :].T.reshape(-1, 1)
        lnp = clnpdfmvn(y_t, y_tL, f_tL)
        lnLm[t] = lnp

        invf_tL = np.linalg.inv(f_tL)

        U_tt = U_tL + np.dot(np.dot(P_tL, H.T), np.dot(invf_tL, (y_t - y_tL)))
        P_tt = P_tL - np.dot(np.dot(np.dot(P_tL, H.T), invf_tL), np.dot(H, P_tL))
        P_tt = (P_tt + P_tt.T) / 2

        U_ttm[t, :] = U_tt.T
        P_ttm[t, :, :] = P_tt
        U_tLm[t, :] = U_tL.T
        P_tLm[t, :, :] = P_tL

        U_LL, P_LL = U_tt, P_tt

    return lnLm, U_ttm, P_ttm, U_tLm, P_tLm


cpdef cKalman_Smoother_DFM(np.ndarray[double, ndim=2] F, 
                           np.ndarray[double, ndim=2] Beta_ttm, 
                           np.ndarray[double, ndim=3] P_ttm, 
                           np.ndarray[double, ndim=2] Beta_tLm, 
                           np.ndarray[double, ndim=3] P_tLm):
    cdef int T = Beta_ttm.shape[0]
    cdef int K = Beta_ttm.shape[1]
    cdef np.ndarray[double, ndim=2] Beta_tTm = Beta_ttm.copy()
    cdef np.ndarray[double, ndim=2] P_tTm = np.zeros((T, K), dtype=np.float64)
    cdef np.ndarray[double, ndim=2] P_t1T = P_ttm[T-1, :, :]
    cdef int t = T - 2
    cdef np.ndarray[double, ndim=2] P_tt, P_t1t, weight, P_tT, invPt1t
    cdef np.ndarray[double, ndim=1] beta_tT

    P_tTm[T-1, :] = np.diag(P_ttm[T-1, :, :])

    while t >= 0:
        P_tt = P_ttm[t, :, :]
        P_t1t = P_tLm[t+1, :, :]
        invPt1t = np.linalg.inv(P_t1t)
        weight = P_tt @ F.T @ invPt1t
        beta_tT = Beta_ttm[t, :].T + weight @ (Beta_tTm[t+1, :].T - Beta_tLm[t+1, :].T)
        Beta_tTm[t, :] = beta_tT.T

        P_tT = P_tt + P_tt @ F.T @ invPt1t @ (P_t1T - P_t1t) @ invPt1t @ F @ P_tt
        P_tTm[t, :] = np.diag(P_tT).T

        P_t1T = P_tT
        t = t-1

    return Beta_tTm, P_tTm


cpdef cmake_R0(np.ndarray[double, ndim=2] mu, 
               np.ndarray[double, ndim=2] G, 
               np.ndarray[double, ndim=2] omega):
    cdef int k = G.shape[0]
    cdef np.ndarray[double, ndim=2] Mu = np.linalg.inv(np.eye(k) - G) @ mu
    cdef int k2 = k * k
    cdef np.ndarray[double, ndim=2] G2 = np.kron(G, G)
    cdef np.ndarray[double, ndim=2] eyeG2 = np.eye(k2) - G2
    cdef np.ndarray[double, ndim=2] omegavec = omega.reshape(k2, 1)
    
    cdef np.ndarray[double, ndim=2] R00 = np.linalg.inv(eyeG2) @ omegavec
    cdef np.ndarray[double, ndim=2] R0 = R00.reshape(k, k).T
    R0 = (R0 + R0.T) / 2

    return Mu, R0


cpdef clnpdfn(np.ndarray[double, ndim=2] x, 
              np.ndarray[double, ndim=2] mu, 
              np.ndarray[double, ndim=2] sig2vec):
    cdef int n = x.shape[0]
    cdef np.ndarray[double, ndim=2] c, e, e2, y

    if mu.shape[0] == 1:
        mu = np.ones(n, dtype=np.double) * mu
    if sig2vec.shape[0] == 1:
        sig2vec = np.ones(n, dtype=np.double) * sig2vec
    
    c = -0.5 * np.log(2 * np.pi * sig2vec)
    e = x - mu
    e2 = np.multiply(e, e)
    y = c - np.divide(0.5 * e2, sig2vec)
    
    return y


cpdef ckalman_filter_TVP(np.ndarray[double, ndim=2] Y, 
                         np.ndarray[double, ndim=2] X, 
                         np.ndarray[double, ndim=2] Z, 
                         int K, 
                         np.ndarray[double, ndim=2] F, 
                         np.ndarray[double, ndim=2] Mu, 
                         np.ndarray[double, ndim=2] gam, 
                         np.ndarray[double, ndim=2] Omega, 
                         np.ndarray[double, ndim=2] Sigma, 
                         np.ndarray[double, ndim=2] U_LL, 
                         np.ndarray[double, ndim=2] P_LL):
    cdef int T = Y.shape[0]
    cdef np.ndarray[double, ndim=1] lnLm = np.zeros(T)
    cdef np.ndarray[double, ndim=2] U_ttm = np.zeros((T, K))
    cdef np.ndarray[double, ndim=3] P_ttm = np.zeros((T, K, K))
    cdef np.ndarray[double, ndim=2] U_tLm = np.zeros((T, K))
    cdef np.ndarray[double, ndim=3] P_tLm = np.zeros((T, K, K))

    cdef np.ndarray[double, ndim=2] H, U_tL, P_tL, y_tL, f_tL, invf_tL, U_tt, P_tt
    cdef double lnp
    cdef int t

    for t in range(T):
        U_tL = Mu + np.dot(F, U_LL)
        P_tL = np.dot(np.dot(F, P_LL), F.T) + Omega

        H = X[t, :].reshape(1, -1)

        y_tL = np.dot(H, U_tL) + Z[t, :].reshape(1, -1) @ gam
        f_tL = np.dot(np.dot(H, P_tL), H.T) + Sigma

        y_t = Y[t, :].T.reshape(-1, 1)
        lnp = clnpdfn(y_t, y_tL, f_tL)
        lnLm[t] = lnp

        invf_tL = np.linalg.inv(f_tL)

        U_tt = U_tL + np.dot(np.dot(P_tL, H.T), np.dot(invf_tL, (y_t - y_tL)))
        P_tt = P_tL - np.dot(np.dot(np.dot(P_tL, H.T), invf_tL), np.dot(H, P_tL))
        P_tt = (P_tt + P_tt.T) / 2

        U_ttm[t, :] = U_tt.T
        P_ttm[t, :, :] = P_tt
        U_tLm[t, :] = U_tL.T
        P_tLm[t, :, :] = P_tL

        U_LL, P_LL = U_tt, P_tt

    return lnLm, U_ttm, P_ttm, U_tLm, P_tLm


cpdef cKalman_Smoother_TVP(np.ndarray[double, ndim=2] F, 
                           np.ndarray[double, ndim=2] Beta_ttm, 
                           np.ndarray[double, ndim=3] P_ttm, 
                           np.ndarray[double, ndim=2] Beta_tLm, 
                           np.ndarray[double, ndim=3] P_tLm):
    cdef int T = Beta_ttm.shape[0]
    cdef int K = Beta_ttm.shape[1]
    cdef np.ndarray[double, ndim=2] Beta_tTm = Beta_ttm.copy()
    cdef np.ndarray[double, ndim=2] P_tTm = np.zeros((T, K), dtype=np.float64)
    cdef np.ndarray[double, ndim=2] P_t1T = P_ttm[T-1, :, :]
    cdef int t = T - 2
    cdef np.ndarray[double, ndim=2] P_tt, P_t1t, weight, P_tT, invPt1t
    cdef np.ndarray[double, ndim=2] beta_tT

    P_tTm[T-1, :] = np.diag(P_ttm[T-1, :, :])

    while t >= 0:
        P_tt = P_ttm[t, :, :]
        P_t1t = P_tLm[t+1, :, :]
        invPt1t = np.linalg.inv(P_t1t)
        weight = P_tt @ F.T @ invPt1t
        beta_tT = Beta_ttm[t, :].T.reshape(-1, 1) + weight @ (Beta_tTm[t+1, :].T - Beta_tLm[t+1, :].T).reshape(-1, 1)
        Beta_tTm[t, :] = beta_tT.T

        P_tT = P_tt + P_tt @ F.T @ invPt1t @ (P_t1T - P_t1t) @ invPt1t @ F @ P_tt
        P_tTm[t, :] = np.diag(P_tT).T

        P_t1T = P_tT
        t = t-1

    return Beta_tTm, P_tTm


cpdef ckalman_filter_UC(np.ndarray[double, ndim=2] Y, 
                        int K, 
                        np.ndarray[double, ndim=2] F, 
                        np.ndarray[double, ndim=2] Mu, 
                        np.ndarray[double, ndim=2] Omega, 
                        np.ndarray[double, ndim=2] Sigma, 
                        np.ndarray[double, ndim=2] U_LL, 
                        np.ndarray[double, ndim=2] P_LL,
                        np.ndarray[double, ndim=2] H):
    cdef int T = Y.shape[0]
    cdef np.ndarray[double, ndim=1] lnLm = np.zeros(T)
    cdef np.ndarray[double, ndim=2] U_ttm = np.zeros((T, K))
    cdef np.ndarray[double, ndim=3] P_ttm = np.zeros((T, K, K))
    cdef np.ndarray[double, ndim=2] U_tLm = np.zeros((T, K))
    cdef np.ndarray[double, ndim=3] P_tLm = np.zeros((T, K, K))

    cdef np.ndarray[double, ndim=2] U_tL, P_tL, y_tL, f_tL, invf_tL, U_tt, P_tt
    cdef double lnp
    cdef int t

    for t in range(T):
        U_tL = Mu + np.dot(F, U_LL)
        P_tL = np.dot(np.dot(F, P_LL), F.T) + Omega

        y_tL = np.dot(H, U_tL).reshape(-1, 1)
        f_tL = np.dot(np.dot(H, P_tL), H.T) + Sigma

        y_t = np.array([Y[t, 0]], dtype=np.double).reshape(-1, 1)
        lnp = clnpdfn(y_t, y_tL, f_tL)
        lnLm[t] = lnp

        invf_tL = np.linalg.inv(f_tL)

        U_tt = U_tL + np.dot(np.dot(P_tL, H.T), np.dot(invf_tL, (y_t - y_tL)))
        P_tt = P_tL - np.dot(np.dot(np.dot(P_tL, H.T), invf_tL), np.dot(H, P_tL))
        P_tt = (P_tt + P_tt.T) / 2

        U_ttm[t, :] = U_tt.T
        P_ttm[t, :, :] = P_tt
        U_tLm[t, :] = U_tL.T
        P_tLm[t, :, :] = P_tL

        U_LL, P_LL = U_tt, P_tt

    return lnLm, U_ttm, P_ttm, U_tLm, P_tLm