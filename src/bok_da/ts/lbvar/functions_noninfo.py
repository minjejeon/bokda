import numpy as np
from tqdm import tqdm
import scipy as sc

#### 관련 데이터 생성 ####

def EXOv_total_maker(Parameters, Raw_Data) :
    if Parameters.Trend == 1 :
        Raw_Data.EXOv_total = np.matrix(np.ones(Parameters.n)).T
    elif Parameters.Trend == 2 :
        Raw_Data.EXOv_total = np.matrix(np.ones(Parameters.n)).T
        Raw_Data.EXOv_total = np.column_stack((Raw_Data.EXOv_total,np.matrix(np.arange(1,Parameters.n+1)).T))
    elif Parameters.Trend == 3 :
        Raw_Data.EXOv_total = np.matrix(np.ones(Parameters.n)).T
        Raw_Data.EXOv_total = np.column_stack((Raw_Data.EXOv_total,np.matrix(np.arange(1,Parameters.n+1)).T))
        Raw_Data.EXOv_total = np.column_stack((Raw_Data.EXOv_total, np.matrix(np.arange(1,Parameters.n+1)**2).T))
    
    if Parameters.Trend == 1 :
        Raw_Data.EXOv = np.matrix(np.ones(Parameters.T)).T
    elif Parameters.Trend == 2 :
        Raw_Data.EXOv = np.matrix(np.ones(Parameters.T)).T
        Raw_Data.EXOv = np.column_stack((Raw_Data.EXOv,np.matrix(np.arange(1,Parameters.T+1)).T))
    elif Parameters.Trend == 3 :
        Raw_Data.EXOv = np.matrix(np.ones(Parameters.T)).T
        Raw_Data.EXOv = np.column_stack((Raw_Data.EXOv,np.matrix(np.arange(1,Parameters.T+1)).T))
        Raw_Data.EXOv = np.column_stack((Raw_Data.EXOv, np.matrix(np.arange(1,Parameters.T+1)**2).T))
    
    TT = Parameters.n - 4
    if Parameters.Trend == 1 :
        Raw_Data.EXOv_AR = np.matrix(np.ones(TT)).T
    elif Parameters.Trend == 2 :
        Raw_Data.EXOv_AR = np.matrix(np.ones(TT)).T
        Raw_Data.EXOv_AR = np.column_stack((Raw_Data.EXOv_AR,np.matrix(np.arange(1,TT+1)).T))
    elif Parameters.Trend == 3 :
        Raw_Data.EXOv_AR = np.matrix(np.ones(Parameters.T)).T
        Raw_Data.EXOv_AR = np.column_stack((Raw_Data.EXOv_AR,np.matrix(np.arange(1,TT+1)).T))
        Raw_Data.EXOv_AR = np.column_stack((Raw_Data.EXOv_AR, np.matrix(np.arange(1,TT+1)**2).T))
    return Raw_Data
        

def LBVAR_variable_maker(Raw_Data, Parameters) :
    Z = np.matrix(np.empty((Parameters.T, Parameters.k)))
    for i in range(0, Parameters.p) :
        Z[:, i*Parameters.nvar:Parameters.nvar+i*Parameters.nvar] = Raw_Data.Set[Parameters.p-(i+1):(Raw_Data.Set.shape[0])-(i+1),:]
    
    Z = np.column_stack((Raw_Data.EXOv, Z))
    
    Y = Raw_Data.Set[Parameters.p:,:]
    
    Raw_Data.Z = Z
    Raw_Data.Y = Y
    return Raw_Data

#### Minnesota Prior 생성 ####

def Noninfor_Posterior(Raw_Data, Parameters, Draw) :
    PI = np.linalg.inv(Raw_Data.Z.T @ Raw_Data.Z) @ Raw_Data.Z.T @ Raw_Data.Y
    u = Raw_Data.Y - Raw_Data.Z @ PI
    Omega = (u.T @ u) / (Parameters.T - Parameters.k)
    
    nu0 = 0
    N0 = np.zeros(((Raw_Data.Z.T @ Raw_Data.Z).shape[0], (Raw_Data.Z.T @ Raw_Data.Z).shape[1]))
    S0 = np.eye(Parameters.nvar)
    B0 = np.zeros((Parameters.num_of_parameter, Parameters.nvar))
    
    nu = nu0 + Parameters.T
    N = N0 + Raw_Data.Z.T @ Raw_Data.Z
    B = np.linalg.inv(N) @ (N0 @ B0 + Raw_Data.Z.T @ Raw_Data.Z @ PI)
    S = (nu0/nu) * S0 + (Parameters.T/nu) * Omega + (1/nu) * (PI - B0).T @ N0 @ np.linalg.inv(N) @ (Raw_Data.Z.T @ Raw_Data.Z) @ (PI - B0)
    inv_S = (np.linalg.inv(S) + np.linalg.inv(S).T)/2
    
    Sigma = np.empty((Parameters.nvar, Parameters.nvar, Parameters.ndraws))
    Bet_Prime = np.empty((Parameters.num_of_parameter, Parameters.nvar, Parameters.ndraws))
    Bet = np.empty((Parameters.nvar, Parameters.num_of_parameter, Parameters.ndraws))
    U_B = np.empty((Raw_Data.Y.shape[0], Raw_Data.Y.shape[1], Parameters.ndraws))
    
    Chol_N = np.linalg.cholesky(N)
    Sigma_candi = np.empty((Parameters.nvar, Parameters.nvar))
    
    for d in tqdm(range(0, Parameters.ndraws)) :
        Sigma_candi = sc.stats.wishart.rvs(df=nu, scale = inv_S/nu)
        Sigma_candi = np.linalg.solve(Sigma_candi, np.eye(Parameters.nvar))
        Sigma_candi = (Sigma_candi.T + Sigma_candi)/2
        Sigma[:,:,d] = Sigma_candi
        
        Chol_Sigma = np.linalg.cholesky(Sigma[:,:,d])
        Bet_Prime[:,:,d] = B + np.linalg.solve(Chol_N.T, np.random.normal(0, 1, (Parameters.k+Parameters.c, Parameters.nvar))) @ Chol_Sigma.T
        Bet[:,:,d] = Bet_Prime[:,:,d].T
        U_B[:,:,d] = Raw_Data.Y - Raw_Data.Z @ Bet_Prime[:,:,d]
    
    Draw.Sigma = Sigma
    Draw.Bet = Bet
    Draw.Bet_Prime = Bet_Prime
    Draw.U_B = U_B

    return Draw
    
    