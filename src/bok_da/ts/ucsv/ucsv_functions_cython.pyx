import numpy as np
cimport numpy as np
from cython cimport boundscheck, wraparound

def sample_tau_by_kalman(np.ndarray[double, ndim=1] Q, np.ndarray[double, ndim=1] R, double x_init, double P_init, np.ndarray[double, ndim=1] y, np.ndarray[double, ndim=1] rand1, np.ndarray[double, ndim=1] rand2):
    """
    관측 데이터 y와 프로세스 노이즈 Q, 관측 노이즈 R을 사용하여
    Kalman 필터와 스무더를 통해 잠재 변수 tau_a와 tau_f를 샘플링
    """
    
    cdef int T = y.shape[0] 

    cdef double x0 = x_init
    cdef double P0 = P_init

    cdef np.ndarray[double, ndim=1] P_u = np.zeros(T + 1, dtype=float, order="F")
    cdef np.ndarray[double, ndim=1] P_p = np.zeros(T + 1, dtype=float, order="F")
    cdef np.ndarray[double, ndim=1] x_u = np.zeros(T + 1, dtype=float, order="F")
    cdef np.ndarray[double, ndim=1] x_p = np.zeros(T + 1, dtype=float, order="F")
    x_u[0] = x0
    P_u[0] = P0

    cdef np.ndarray[double, ndim=1] tau_a = np.zeros(T + 1, dtype=float, order="F")
    cdef np.ndarray[double, ndim=1] tau_f = np.zeros(T, dtype=float, order="F")

    cdef int t
    cdef double x1, P1, Ht, K
    cdef double xT, PT, x, AS

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

# @boundscheck(False)
# @wraparound(False)
def sample_tau(np.ndarray[double, ndim=1] y, sigma_dtau, sigma_eps_scl):
    """
    관측된 데이터 y와 프로세스 노이즈의 표준편차 sigma_dtau,
    관측 노이즈의 표준편차 sigma_eps_scl을 기반으로 상태 변수 tau와
    그 변화량 dtau를 Kalman 필터와 스무더를 사용하여 샘플링
    """
    cdef double big = 1e6
    cdef int nobs = len(y)
    cdef double x0 = 0
    cdef double P0 = big
    cdef np.ndarray[double, ndim=1] rand1 = np.random.standard_normal(nobs)
    cdef np.ndarray[double, ndim=1] rand2 = np.random.standard_normal(nobs + 1)
    cdef np.ndarray[double, ndim=1] Q, R

    if np.isscalar(sigma_dtau):
        Q = np.full(nobs, sigma_dtau**2, dtype=float, order='F')
    else:
        Q = np.array(sigma_dtau**2, dtype=float, order='F')

    if np.isscalar(sigma_eps_scl):
        R = np.full(nobs, sigma_eps_scl**2, dtype=float, order='F')
    else:
        R = np.array(sigma_eps_scl**2, dtype=float, order='F')

    tau_a, tau_f = sample_tau_by_kalman(Q, R, x0, P0, y, rand1, rand2)
    cdef np.ndarray[double, ndim=1] dtau = tau_a[1:] - tau_a[:-1]
    cdef np.ndarray[double, ndim=1] tau = tau_a[1:]
    return tau, dtau, tau_f
