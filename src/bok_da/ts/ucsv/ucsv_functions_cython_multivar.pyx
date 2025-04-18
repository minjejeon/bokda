import numpy as np
cimport numpy as np
from cython cimport boundscheck, wraparound

def sample_eps_tau_by_kalman(np.ndarray[double, ndim=3] H, np.ndarray[double, ndim=3] R, np.ndarray[double, ndim=2] F, np.ndarray[double, ndim=3] Q, np.ndarray[double, ndim=1] x_init, np.ndarray[double, ndim=2] P_init, np.ndarray[double, ndim=2] y, np.ndarray[double, ndim=2] rand1, np.ndarray[double, ndim=2] rand2):
    """
    """

    cdef int m = H.shape[0]
    cdef int n = H.shape[1]
    cdef int T = H.shape[2]
    cdef int d = n - m

    cdef np.ndarray[double, ndim=1] x0 = np.array(x_init, dtype=float, order="F")
    cdef np.ndarray[double, ndim=1] x1 = np.zeros(n, dtype=float, order="F")
    cdef np.ndarray[double, ndim=1] xT = np.zeros(n, dtype=float, order="F")
    cdef np.ndarray[double, ndim=2] P0 = np.array(P_init, dtype=float, order="F")
    cdef np.ndarray[double, ndim=2] P1 = np.zeros((n, n), dtype=float, order="F")
    cdef np.ndarray[double, ndim=2] PT = np.zeros((n, n), dtype=float, order="F")
    cdef np.ndarray[double, ndim=1] xd = np.zeros(n, dtype=float, order="F")
    cdef np.ndarray[double, ndim=2] PTm = np.zeros((m, m), dtype=float, order="F")

    cdef np.ndarray[double, ndim=2] x_u = np.zeros((n, T + 1), dtype=float, order="F")
    cdef np.ndarray[double, ndim=3] P_u = np.zeros((n, n, T + 1), dtype=float, order="F")
    x_u[:,0] = x0
    P_u[:,:,0] = P0
    cdef np.ndarray[double, ndim=2] x_p = np.zeros((n, T + 1), dtype=float, order="F")
    cdef np.ndarray[double, ndim=3] P_p = np.zeros((n, n, T + 1), dtype=float, order="F")
    cdef np.ndarray[double, ndim=2] chol_P = np.zeros((n, n), dtype=float, order="F")
    cdef np.ndarray[double, ndim=2] chol_Pm = np.zeros((m, m), dtype=float, order="F")

    cdef np.ndarray[double, ndim=2] x_draw = np.zeros((n, T + 1), dtype=float, order="F")
    cdef np.ndarray[double, ndim=2] x_draw_f = np.zeros((n, T + 1), dtype=float, order="F")

    cdef np.ndarray[double, ndim=1] nu = np.zeros(m, dtype=float, order="F")
    cdef np.ndarray[double, ndim=2] S = np.zeros((m, m), dtype=float, order="F")
    cdef np.ndarray[double, ndim=2] invS = np.zeros((m, m), dtype=float, order="F")
    cdef np.ndarray[double, ndim=2] Im = np.eye(m, dtype=float, order="F")
    cdef np.ndarray[double, ndim=2] In = np.eye(n, dtype=float, order="F")
    cdef np.ndarray[double, ndim=2] K = np.zeros((n, m), dtype=float, order="F")

    cdef np.ndarray[double, ndim=2] FP0 = np.zeros((n, n), dtype=float, order="F")
    cdef np.ndarray[double, ndim=2] AS = np.zeros((n, n), dtype=float, order="F")
    cdef np.ndarray[double, ndim=2] Tnn = np.zeros((n, n), dtype=float, order="F")
    cdef np.ndarray[double, ndim=2] Tmm = np.zeros((m, m), dtype=float, order="F")
    cdef np.ndarray[double, ndim=2] Tnn2 = np.zeros((n, n), dtype=float, order="F")
    cdef np.ndarray[double, ndim=2] Tnm = np.zeros((n, m), dtype=float, order="F")
    cdef np.ndarray[double, ndim=2] Tmn = np.zeros((m, n), dtype=float, order="F")
    cdef np.ndarray[int, ndim=2] C = np.zeros((m, m), dtype=np.int32, order="F")

    cdef np.ndarray[double, ndim=1] Tn = np.zeros(n, dtype=float, order="F")
    cdef np.ndarray[double, ndim=1] Tm = np.zeros(m, dtype=float, order="F")

    cdef np.ndarray[double, ndim=2] sl = np.zeros((m, n), dtype=float, order="F")
    sl[:, d:] = Im

    cdef int t, i, j

    for t in range(T):
        x1 = np.dot(F, x0)
        P1 = np.dot(F, np.dot(P0, F.T)) + Q[:, :, t]
        nu = y[:, t] - np.dot(H[:, :, t], x1)
        S = np.dot(H[:, :, t], np.dot(P1, H[:, :, t].T)) + R[:, :, t]
        invS = np.linalg.inv(S)
        K = np.dot(P1, np.dot(H[:, :, t].T, invS))
        x0 = x1 + np.dot(K, nu)
        P0 = P1 - np.dot(K, np.dot(H[:, :, t], P1))
        P0 = 0.5 * (P0 + P0.T)
        P_p[:, :, t + 1] = P1
        P_u[:, :, t + 1] = P0
        x_p[:, t + 1] = x1
        x_u[:, t + 1] = x0
        chol_P = np.linalg.cholesky(P0)
        xd = np.dot(chol_P, rand1[:, t]) + x0
        x_draw_f[:, t + 1] = xd

    PT = P0
    xT = x0
    chol_P = np.linalg.cholesky(PT)
    xd = xT + np.dot(chol_P, rand2[:, -1])
    x_draw[:, -1] = xd

    for t in range(T - 1, -1, -1):
        x0 = x_u[:, t]
        x1 = x_p[:, t + 1]
        P0 = P_u[:, :, t]
        P1 = P_p[:, :, t + 1]
        FP0 = np.dot(F, P0)
        AS = np.linalg.solve(P1, FP0)
        PT = P0 - np.dot(AS.T, FP0)
        PT = 0.5 * (PT + PT.T)
        xd = xd - x1
        xT = x0 + np.dot(AS.T, xd)
        xd = xT

        if t > 0:
            chol_P = np.linalg.cholesky(PT)
            Tn = np.dot(chol_P, rand2[:, t])
            xd = xd + Tn
        else:
            PTm = np.dot(sl, np.dot(PT, sl.T))
            chol_Pm = np.linalg.cholesky(PTm)
            Tm = np.dot(chol_Pm, rand2[d:, t])
            xd[d:] = xd[d:] + Tm

        x_draw[:, t] = xd

    return np.array(x_draw_f), np.array(x_draw)


def sample_alpha_tvp_by_kalman(np.ndarray[double, ndim=3] H, np.ndarray[double, ndim=3] R, np.ndarray[double, ndim=2] F, np.ndarray[double, ndim=2] Q, np.ndarray[double, ndim=1] x_init, np.ndarray[double, ndim=2] P_init, np.ndarray[double, ndim=2] y, np.ndarray[double, ndim=2] rand2):
    """
    """

    cdef int m = H.shape[0]
    cdef int n = H.shape[1]
    cdef int T = H.shape[2]
    cdef int d = n - m

    cdef np.ndarray[double, ndim=1] x0 = np.array(x_init, dtype=float, order="F")
    cdef np.ndarray[double, ndim=1] x1 = np.zeros(n, dtype=float, order="F")
    cdef np.ndarray[double, ndim=1] xT = np.zeros(n, dtype=float, order="F")
    cdef np.ndarray[double, ndim=2] P0 = np.array(P_init, dtype=float, order="F")
    cdef np.ndarray[double, ndim=2] P1 = np.zeros((n, n), dtype=float, order="F")
    cdef np.ndarray[double, ndim=2] PT = np.zeros((n, n), dtype=float, order="F")
    cdef np.ndarray[double, ndim=1] xd = np.zeros(n, dtype=float, order="F")

    cdef np.ndarray[double, ndim=2] x_u = np.zeros((n, T + 1), dtype=float, order="F")
    cdef np.ndarray[double, ndim=3] P_u = np.zeros((n, n, T + 1), dtype=float, order="F")
    x_u[:, 0] = x0
    P_u[:, :, 0] = P0
    cdef np.ndarray[double, ndim=2] x_p = np.zeros((n, T + 1), dtype=float, order="F")
    cdef np.ndarray[double, ndim=3] P_p = np.zeros((n, n, T + 1), dtype=float, order="F")
    cdef np.ndarray[double, ndim=2] chol_P = np.zeros((n, n), dtype=float, order="F")

    cdef np.ndarray[double, ndim=2] x_draw = np.zeros((n, T + 1), dtype=float, order="F")

    cdef np.ndarray[double, ndim=1] nu = np.zeros(m, dtype=float, order="F")
    cdef np.ndarray[double, ndim=2] S = np.zeros((m, m), dtype=float, order="F")
    cdef np.ndarray[double, ndim=2] invS = np.zeros((m, m), dtype=float, order="F")
    cdef np.ndarray[double, ndim=2] Im = np.eye(m, dtype=float, order="F")
    cdef np.ndarray[double, ndim=2] K = np.zeros((n, m), dtype=float, order="F")

    cdef np.ndarray[double, ndim=2] FP0 = np.zeros((n, n), dtype=float, order="F")
    cdef np.ndarray[double, ndim=2] AS = np.zeros((n, n), dtype=float, order="F")
    cdef np.ndarray[double, ndim=2] PT_copy
    cdef int t

    for t in range(T):
        x1 = np.dot(F, x0)
        P1 = np.dot(F, np.dot(P0, F.T)) + Q
        nu = y[:, t] - np.dot(H[:, :, t], x1)
        S = np.dot(H[:, :, t], np.dot(P1, H[:, :, t].T)) + R[:, :, t]
        invS = np.linalg.inv(S)
        K = np.dot(P1, np.dot(H[:, :, t].T, invS))
        x0 = x1 + np.dot(K, nu)
        P0 = P1 - np.dot(K, np.dot(H[:, :, t], P1))
        P0 = 0.5 * (P0 + P0.T)
        P_p[:, :, t + 1] = P1
        P_u[:, :, t + 1] = P0
        x_p[:, t + 1] = x1
        x_u[:, t + 1] = x0

    PT = P0
    xT = x0
    chol_P = np.linalg.cholesky(PT)
    xd = xT + np.dot(chol_P, rand2[:, -1])
    x_draw[:, -1] = xd

    for t in range(T - 1, -1, -1):
        x0 = x_u[:, t]
        x1 = x_p[:, t + 1]
        P0 = P_u[:, :, t]
        P1 = P_p[:, :, t + 1]
        FP0 = np.dot(F, P0)
        AS = np.linalg.solve(P1, FP0)
        PT = P0 - np.dot(AS.T, FP0)
        PT = 0.5 * (PT + PT.T)
        xd = xd - x1
        xT = x0 + np.dot(AS.T, xd)
        chol_P = np.linalg.cholesky(PT)
        xd = xT + np.dot(chol_P, rand2[:, t])
        x_draw[:, t] = xd

    return np.array(x_draw)
