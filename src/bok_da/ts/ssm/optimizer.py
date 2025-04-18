import numpy as np

from ...utils.operator import cols, rows, inv, eye, reshape, ones, zeros, length, maxc, chol, diag, minc, eig, recserar, is_nan, is_None
from ...utils.rng import rand


def make_R0(mu: np.ndarray, 
            G: np.ndarray, 
            omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct Mu and R0 matrices
    """

    k = rows(G)
    Mu = inv(eye(k) - G) @ mu
    k2 = np.square(k)
    G2 = np.kron(G,G)
    eyeG2 = eye(k2) - G2
    omegavec = reshape(omega, k2, 1)
    
    R00 = inv(eyeG2) @ omegavec; #  vec(R00)
    R0 = reshape(R00, k, k).T
    R0 = (R0 + R0.T) / 2

    return Mu, R0


def Gradient(func: callable, 
             para: np.ndarray, 
             S: any) -> np.ndarray:
    """
    Gradient 계산하기

    Args:
        func (callable): gradiant를 계산하는 함수, para와 S의 함수
        para (np.ndarray): `func` 함수의 argument
        S (any): `func` 함수의 두 번째 argument

    Returns:
        grdd (np.ndarray): A column vector containing the gradient of `func` at `para`.
    """
    if cols(para) > 1:
        print('파라메터는 벡터여야 하며, 행렬이면 안됩니다')
        print(para)
    
    para_index = np.arange(length(para))
    x0 = para[para_index]  # extract the relevant components of para */

    F_val = func(para,S)
    f0 = F_val[0] if isinstance(F_val, tuple) else F_val    
    n = length(f0)  # MLE의 경우 일반적으로 n = 1
    k = length(x0)
    F_new = zeros(k, n)

    # Computation of stepsize (dh) for gradient */
    ax0 = abs(x0)
    dax0 = ones(k, 1)
    
    for j in range(0, k):
          if x0[j, 0] == 0:
            dax0[j, 0] = 1
          else:
            dax0[j, 0] = x0[j]/ax0[j]

          if is_nan(dax0[j,0]) == 1:
            dax0[j, 0] = 1

    ma = np.vstack((ax0.T, (1e-2)*ones(1,k)))
    maxma, tmp = maxc(ma)
    maxma = (1e-8)*maxma[0]
    dh = np.multiply(maxma, dax0)
    
    xdh = x0 + dh  # dh의 스케일이 대단히 작음 => 수치적 (small)오차 발생 
    dh = xdh - x0  # 이렇게 하면 dh의 오차가 줄어듦

    for i in range(k):
        para_new = para.copy()  ### 반드시 Deepcopy해야!!!!!!!
        para_new_i = para[para_index[i], 0] + dh[i,0]
        para_new[para_index[i], 0] = para_new_i

        F_val_new = func(para_new,S)
        f_new = F_val_new[0] if isinstance(F_val_new, tuple) else F_val_new
        
        F_new[i,:] = f_new.T
    
    Dh = np.kron(ones(1,n),dh)
    F0 = np.kron(ones(k,1),f0.T)

    grdd = np.divide((F_new - F0), Dh) # Gradient, k by 1

    return grdd


def FHESS(func, x, S):
    '''
    FHESS Computes numerical Hessian of a function
    
    * func :  a string name of a function
    * x :  a vector at which to evaluate the Hessian of the function
    * index = parameter index, 예: index = np.ndarray([0, 1])
      주의: index는 반드시 1차원 array여야 하며, 행렬이면 안됨!!
    * S = f의 두번째 input
    '''
    index = np.arange(length(x))
    xarg = x[index]
    k = length(xarg)
    F_val = func(x,S)
    f0 = F_val[0] if isinstance(F_val, tuple) else F_val
    
    eps = np.finfo(float).eps
    h = np.power(eps,1/3) * np.maximum(abs(xarg), 1e-2*ones(k, 1))
        
    xargh = x[index] + h
    h = xargh - x[index]

    e_diag = np.zeros(k)
    for j in range(k):
        e_diag[j] = h[j, 0]

    row_indices = np.arange(k)
    col_indices = np.arange(k)
    
    ee = np.zeros((max(row_indices) + 1, max(col_indices) + 1))
    for i, (row, col) in enumerate(zip(row_indices, col_indices)):
        ee[row, col] = e_diag[i]
    # ee = np.asarray(ee0.toarray())

    ## Gradient 계산하기
    grad = zeros(k, 1)
    for i in range(k):
        xee = x.copy()
        xee[index] = xarg + ee[:, i].reshape(-1, 1)
        
        grad_i = func(xee, S)
        grad[i, 0] = grad_i[0] if isinstance(grad_i, tuple) else grad_i

    ## Hessian 계산하기
    H = h @ h.T
    Hess0 = H.copy()

    for i in range(k):
       for j in range(i, k):
           xeeij  = x.copy()
           xargeeij = xarg + ee[:,i].reshape(-1, 1) + ee[:, j].reshape(-1, 1)
           xeeij[index] = xargeeij

           grad_ij = func(xeeij,S)
           grad_ij0 = grad_ij[0] if isinstance(grad_ij, tuple) else grad_ij

           Hess0[i,j] = (grad_ij0 - grad[i, 0] - grad[j, 0] + f0) / H[i, j]
           Hess0[j,i] = Hess0[i,j]
   
    Hess0 = 0.5 * (Hess0 + Hess0.T)
    Hess = np.nan_to_num(Hess0, copy=False)
    return Hess


def DO_CKR2(FUN, CONSTR, arg, S, maxiter=500, co=1, printi=0, verbose=False):
    """
    Modified deterministic optimizer (DO) to include partial argument vector
    MORE IMPORTANTLY, the initial Hessian and its inverse are 
    no longer needed. In case the Hessian is not p.d., a better
    approximation is calculated using cholmod within the program.

    Args:
        FUN: pointer to function to be maximized
        CONSTR: pointer to constraint procedure
        arg: full vector of argument
        S: second argument of FUN
        maxiter: max number of desired iterations
        co : = 1 if constrained optimizer, and 0 otherwise.
        printi: =1 to print intermediate results

    Returns:
        mu: maxima (full vector)
        fmax: maximized value of function
        g:  gradient at argmax
        V: Cov computed as inverse of negative Hessian (or a nearby p.d. matrix)
        Vinv: -Hessian
    """

    argnew = arg.copy()
    narg = length(arg)
    index = np.arange(narg)
    adj = 3

    if co == 1:
       valid = CONSTR(argnew,S)  #/* check if parameter constraints are valid for initial value */

       if valid == 0:
          print("초기값이 제약을 만족하지 않습니다. 제약함수 또는 초기값을 확인하세요")
          print("초기값")
          print(argnew)

    db = 1
    iter = 1
    s = 1

    while db > 1e-6 and iter <= maxiter and s >= 1e-6:
        G = - Gradient(FUN, argnew, S)
        H = - FHESS(FUN, argnew, S)
        H = H.real
        H = 0.5*(H + H.T)

        if rows(H) > 1:
            Hinv = inv(H)
            if is_None(H) == 1:
                Hc = chol(H)
                Hinvc = inv(Hc)
                Hinv = Hinvc*Hinvc.T
        else:
             H = np.maximum(abs(H), 1e-016)
             Hinv = 1/H

        db = - Hinv @ G  #the change in estimate 
        s = 1.0
        s_hat = s
        x00 = argnew.copy()
        
        FCD0 = FUN(x00, S)
        fcd0 = FCD0[0] if isinstance(FCD0, tuple) else FCD0
        x00[index] = x00[index] + s*db/adj

        if co==1 and CONSTR(x00,S)==0:
            while s > 0 and CONSTR(x00, S) == 0:
                x00 = argnew.copy()
                x00[index] = x00[index] + s*db/adj
                if s < 1e-6:
                    x00 = argnew.copy()
                    if verbose:
                        print('close to constraint (minimum step reached)')
                    s = 0
                s_hat = s # output of this 'while' looping
                s = s*0.8

            FCD1 = FUN(x00,S)
            fcd1 = FCD1[0] if isinstance(FCD1, tuple) else FCD1

            if fcd1 < fcd0:
                x00 = argnew.copy()
                s_hat = 0

        fcd1 = fcd0 - 1 # new value of function
        s = s_hat

        while fcd1 < fcd0:       
            x00 = argnew.copy()
            x00[index] = x00[index] + s*db/adj

            if co == 1:
                valid = CONSTR(x00, S)
            else:
                valid = 1

            if valid == 1 and s < 1e-12:
                s = 0     

            if valid == 0:
                fcd1 = fcd0 - 1
            else:
                FCD1 = FUN(x00,S)  #new value of function
                fcd1 = FCD1[0] if isinstance(FCD1, tuple) else FCD1
                
            s_hat = s
            s = s*0.8

        s = s_hat
        argnew[index] = argnew[index] + s*db/adj
    
        if printi == 1:
            print("=====================================")
            print("current DO iteration is ", iter)
            F_val = FUN(argnew,S)  #new value of function
            f_val = F_val[0] if isinstance(F_val, tuple) else F_val
            print("current function value is ", f_val)
            print("current step size is ", s_hat)
            print("-------------------------------------")
            print(" indices  argmax    gradient")
            print('-------------------------------------')
            Report_interm = np.hstack((np.asmatrix(np.round(index)).T, np.round(argnew[index], 3), np.round(G, 3)))
            print(Report_interm.round(3))
            print("-------------------------------------")

        db, tmp = maxc(abs(G))
        iter = iter + 1

    mu = argnew # maximum 
    Fmax = FUN(mu,S)  # value of function at maximum 
    fmax = Fmax[0] if isinstance(Fmax, tuple) else Fmax
    
    H = - FHESS(FUN, mu, S)
    G = Gradient(FUN, mu, S)
    H = H.real
    
    if rows(H) > 1:
        H = 0.5*(H + H.T)
        V = inv(H)
        if is_None(V) == 1:
           Vinvc = chol(H)
           Vinv = Vinvc.T*Vinvc
           Vc = inv(Vinvc)
           V = Vc*Vc.T
        else:
           Vinv = H
    else:  # in case H is a scalar
        Vinv = np.maximum(abs(H), 1e-016)
        V = 1/Vinv

    return mu, fmax, G, V, Vinv


def simulated_annealing(FUN, constr, theta0, S, tau, m, SF, co, eps, mr, n):
    """perform simulated annealing part of SA_Newton"""
    Lik0 = FUN(theta0, S)
    lik0 = Lik0[0] if isinstance(Lik0, tuple) else Lik0

    likg = lik0.copy()    # storage for global function max 
    argg = theta0.copy()  # storage for global maxima
    narg = len(theta0)

    acceptm = zeros(narg, n) # total number of MH acceptances

    for j in range(len(tau)):
        tauj = tau[j]
        reject = 0
        for i in range(int(m[j])):
            valid = False
            while not valid:
                ind = np.random.randint(0, narg)
                dh = np.random.randn() / SF[ind]
                thetap = theta0.copy()
                thetap[ind] += dh
                
                valid = constr(thetap, S) if co == 1 else True
                if not valid:
                    continue
                
                Likp = FUN(thetap, S)
                likp = Likp[0] if isinstance(Likp, tuple) else Likp
                dlik = likp - likg

                alpha = 1 if dlik > eps else np.exp(dlik/tauj)
                accept = rand(1, 1) < alpha
                if accept:
                    theta0 = thetap
                    lik0 = likp
                    acceptm[ind] += 1
                    reject = 0
                    if likp > likg:
                        argg, likg = thetap, likp
                else:
                    reject += 1
            if reject > mr:
                print("rejection 수가 너무 많습니다.")
        if j > 0: 
            if j == n-1:
                print(f"\r{j+1}/{n} cycle finished")
            else:
                print(f"\r{j+1}/{n} cycle finished", end="")
        else: 
            print(f"{j+1}/{n} cycle finished", end="")

    return argg, likg


def SA_Newton(FUN, constr, arg, S, verbose=False):
    a = 0.9
    IT = 0.1
    narg = rows(arg)
    index = np.arange(narg)
    b = np.minimum(narg*2, 100)
    cs = 5
    IM = narg*20
    SF = 100*ones(narg,1) # scale factor
    n = 10
    mr = 400
    eps = 1e-6
    maxiter = np.maximum(narg*10, 50)
    co = 1
    printi = 0
    if verbose:
        print('지금 우도함수 극대화 중입니다.')
        print('잠시만 기다려주세요.')

    theta0 = arg.copy()
    valid = constr(theta0, S) if co == 1 else True

    if not valid:
        print('초기값이 제약을 만족하지 않습니다')
        print(theta0)
        return

    if n == 0 and maxiter > 0: # when n=0, swith to deterministic optimizer 
        # deterministic maximizer 
        mu = argg.copy()
        mu, fmax, g, V, Vinv = DO_CKR2(FUN, constr, mu, S, verbose=verbose)

    if n > 1:
        tau = recserar(zeros(n,1), IT, a); # temperatures reduction schedule 
        m = recserar(b*ones(n,1), IM, np.array([1.0])); # stage lengths 
    elif n == 1:
        tau = IT # temperatures reduction schedule 
        m = IM

    argg, likg = simulated_annealing(FUN, constr, arg, S, tau, m, SF, co, eps, mr, n)

    if maxiter == 0:   # if maxiter = 0, skip the deterministic optimizer  
        mu = argg      # and calculate things that need to be output 
        fmax = likg
        g = Gradient(FUN, mu, S) # % gradient 
        H = -FHESS(FUN,mu,S) # variance-covariance 

        if rows(H)==1:  # in case H is a scalar 
            Vinv = np.maximum(abs(H), 1e-016)
            V = 1/Vinv
        else:
            H = H.real
            H = 0.5*(H + H.T)
            V = inv(H)
            if is_None(V) == 1:
                Vinvc = chol(H)
                Vinv = Vinvc.T*Vinvc
                Vc = inv(Vinvc)
                V = Vc*Vc.T
            else:
	            Vinv = H

    if n > 0 and maxiter > 0:
        mu,fmax,g,V,Vinv = DO_CKR2(FUN, constr, argg, S, verbose=verbose)
    
    if verbose:
        print('극대화 작업이 완료되었습니다.')

    if printi==1:
        print('==================================')
        print('final function value	is ', fmax)
        print('----------------------------------')
        print('indices  argmax ')
        print(np.hstack((np.asmatrix(index).T, argg[index])))
        print('==================================')

    return mu,fmax,g,V,Vinv


def Kalman_Smoother(F, Beta_ttm, P_ttm, Beta_tLm, P_tLm):
# Beta_tTm, P_tTm 둘다 T by K
    T = rows(Beta_ttm) 
    K = cols(Beta_ttm)
    Beta_tTm = Beta_ttm.copy()
    P_tTm = zeros(T, K)
    P_tTm[T-1, :] = diag(P_ttm[T-1, :, :]).T # % 대각만 저장하기
    P_t1T = P_ttm[T-1, :, :]; # 1기 이후 분산-공분산
    t = T-2
    while t >= 0:

        P_tt = P_ttm[t, :, :]
        P_t1t = P_tLm[t+1, :,:]

        weight = P_tt @ F.T @ inv(P_t1t)

        beta_tT = Beta_ttm[t, :].T + weight @ (Beta_tTm[t+1, :].T - Beta_tLm[t+1, :].T)
        print(beta_tT.T.shape)

        Beta_tTm[t, :] = beta_tT.T

        P_tT = P_tt + P_tt @ F.T*inv(P_t1t) @ (P_t1T - P_t1t) @ inv(P_t1t) @ F @ P_tt
        print(diag(P_tT).T.shape)
        P_tTm[t, :] = diag(P_tT).T  # % 대각행렬(분산)만 저장하기

        P_t1T = P_tT  # % 다음기에 사용될 것, K by K
        t = t - 1

    return Beta_tTm, P_tTm


def paramconst_SSM(para, Spec):
    
    sig2index = Spec['sig2index']
    
    para1 = trans_SSM(para, Spec)

    validm = ones(10, 1)
    p = Spec['Lag']
    if p > 0:
        phi = para1[(sig2index+1):]
    else:
        phi = zeros(1,1)

    if p>1:
        Phi1 = phi.T
        Phi2 = np.hstack((eye(p-1), zeros(p-1, 1)))
        Phi = np.vstack((Phi1, Phi2))
    else:
        Phi = phi

    tmp, D = eig(Phi)
    abseigval = np.abs(diag(D))

    maxabseigval, tmp = maxc(abseigval)
    validm[0] = maxabseigval < 0.9

    sig2m = para1[sig2index]
    msig2, tmp = minc(sig2m)
    validm[1] = msig2 > 0
    valid, tmp = minc(validm)
    return valid


def trans_SSM(para, Spec):
    para1 = para.copy()

    sig2index = Spec['sig2index']
    w2index = Spec['w2index']
    para1[sig2index] = np.exp(para[sig2index])
    para1[w2index] = np.exp(para[w2index])

    return para1