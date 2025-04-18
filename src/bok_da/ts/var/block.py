import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from ...utils.operator import array, meanc, demeanc, chol, zeros, inv, rows, eye, matrix, vec, reshape
from ...utils.rng import rand

from .var import irf_estimate, randper
from .var import order_var


class BlockVarResult:
    def __init__(self, phi_hat, lag, **kwargs):
        self.phi_hat = phi_hat
        self.lag = lag
        self.k = phi_hat.shape[1]
        
        self.phi_hat = [phi_hat[i*self.k:(i+1)*self.k, :] for i in range(self.lag)]
        
        for key, value in kwargs.items():
            setattr(self, key, value)
            
    def get_description(self):
        desc = {
            "결과": ['phi_hat', 'sig_hat', 'g0_hat', 'f_mat', 'y', 'u_hat1', 'u_hat2', 'lag'],
            "설명": ['축약형 VAR(p) 모형의 계수 행렬 추정량', '축약형 VAR(p) 모형의 분산-공분산 행렬 추정량', 
                     'y20 계수의 OLS 추정량 (가정: 글로벌 변수들이 도메스틱 변수들을 계수 및 오차항을 통해 동시다발적으로 영향을 미친다)',
                     '축약형 VAR(p) 모형의 동반행렬 형태의 계수 행렬 추정량', '추정에 사용된 완전한 VAR 시스템',
                     '도메스틱 부분 시스템의 잔차항', '글로벌 부분 시스템의 잔차항', '추정에 사용된 시차']
        }

        df = pd.DataFrame(desc)
        df.set_index("결과", inplace=True)
        df.index.name = None
        #df = df.style.set_properties(**{'text-align': 'left'})
        #df = df.set_table_styles([{'selector': 'th', 'props': [('text-align', 'left')]}])

        return df

def var_block(y1, y2, lag=None, verbose: bool=False):
    '''블록 외생성 제약 하 다변량 VAR(p) 모형에 대한 OLS 추정
    
    Args:
        y1 : 도메스틱 부분-시스템, 글로벌(Y2)에 의존
        y2 : 글로벌 부분-시스템, 도메스틱(Y1)과 독립
        lag : VAR(p) 모형의 최적 시차 (default = None)
        
    Returns:
        phi_hat : 축약형 VAR(p) 모형 계수 행렬의 OLS 추정량
        sig_hat : 축약형 VAR(p) 모형 분산-공분산 행렬의 OLS 추정량
        g0_hat : y20 계수의 OLS 추정량 (가정: 글로벌 변수들이 도메스틱 변수들을 계수 및 오차항을 통해 동시다발적으로 영향을 미친다)
        f_mat : 축약형 VAR(p) 모형의 동반행렬 형태의 계수 행렬 추정량
        y : 추정에 사용된 완전한 VAR 시스템
        u_hat1 : 도메스틱 부분 시스템의 잔차항
        u_hat2 : 글로벌 부분 시스템의 잔차항
    '''
    p = lag
    Y1 = np.array(y1)
    if Y1.shape[1] < 2 :
        Y1 = Y1.reshape(-1,1)
        print("Y1은 단변수 (univariate) 입니다.")
        
    Y2 = np.array(y2)
    if Y2.shape[1] < 2:
        Y2 = Y2.reshape(-1,1)
        print("Y2는 단변수 (univariate) 입니다.")
    
    T1,K1 = Y1.shape
    T2,K2 = Y2.shape
    K = K1+K2
    
    if p == None:
        y = np.hstack((Y1,Y2))
        p_res = order_var(y, verbose=verbose)
        p = p_res.bic
    else:
        p = p
    
    # <Step 0>: Demeaning Variables
    Y1_ = meanc(Y1)
    Y2_ = meanc(Y2)

    y1 = demeanc(Y1)
    y2 = demeanc(Y2)
    y = np.hstack((y1,y2))
    T = rows(y)

    # <Step 1>: Determining LHS and RHS variables of each equation
    y10 = y1[p:,:] # p+1 to T
    y20 = y2[p:,:] # p+1 to T
    y0 = y[p:,:] # p+1 to T
    T1 = rows(y0)

    Y2_lag = []
    Y_lag = []
    for j in range(p):
        Y2_lag.append(y2[p-j-1:T-j-1,:])
        Y_lag.append(y[p-j-1:T-j-1,:])

    Y2_lag = np.hstack(Y2_lag)
    Y_lag = np.hstack(Y_lag)

    Y1_lag = np.hstack((y20, Y_lag))

    # <Step 2>: OLS estimator of each equation
    # (1) Domestic Equation
    XX1 = Y1_lag.T @ Y1_lag
    phi_hat1 = inv(XX1) @ (Y1_lag.T @ y10)
    u_hat1 = y10 - Y1_lag @ phi_hat1
    H_hat = (u_hat1.T @ u_hat1) / (T-p*(K+1))

    # (2) Global Equation
    XX2 = Y2_lag.T @ Y2_lag
    phi_hat2 = inv(XX2) @ (Y2_lag.T @ y20)
    u_hat2 = y20 - Y2_lag @ phi_hat2
    SIG22_hat = (u_hat2.T @ u_hat2) / (T-p*K2)

    # <Step 3>: Reconstructing the Results
    A1 = phi_hat1.T
    A22m = phi_hat2.T

    G0_hat = A1[:,0:K2]

    A11m = []
    Gi = []
    for i in range(p):
        A1_temp = A1[:,K2:]
        A11m.append(A1_temp[:,i*(K1+K2):(i+1)*(K1+K2)-K2])
        Gi.append(A1_temp[:,i*(K1+K2)+K1:(i+1)*K])

    A11m = np.hstack(A11m)
    Gi = np.hstack(Gi)

    A12m = []
    for i in range(p):
        A12m.append(Gi[:,i*(K2):(i+1)*K2] + G0_hat @ A22m[:,i*K2:(i+1)*K2])

    A12m = np.hstack(A12m)

    phi_hat_trans = []
    for i in range(p):
        temp = np.block([[A11m[:,i*K1:(i+1)*K1], A12m[:,i*K2:(i+1)*K2]],
                         [zeros(K2,K1), A22m[:,i*K2:(i+1)*K2]]])
        phi_hat_trans.append(temp)

    phi_hat_trans = np.hstack(phi_hat_trans)
    phi_hat = phi_hat_trans.T

    SIG12_hat = G0_hat @ SIG22_hat
    SIG11_hat = H_hat + SIG12_hat @ inv(SIG22_hat) @ SIG12_hat.T

    SIG_hat = np.block([[SIG11_hat, SIG12_hat], [SIG12_hat.T, SIG22_hat]])

    # <Step 4>: Companion Form
    if p > 1:
        F1 = phi_hat.T 
        F2 = np.hstack((eye((p-1)*K), zeros(K*(p-1),K)))
        F = np.vstack((F1, F2))  # p*k by p*k
    elif p==1:
        F = phi_hat.T
    
    #phi_hat = [phi_hat[i*K:(i+1)*K, :] for i in range(p)]
    
    return BlockVarResult(phi_hat=phi_hat, sig_hat=SIG_hat, g0_hat=G0_hat, f_mat=F, y=y, u_hat1=u_hat1, u_hat2=u_hat2, lag=p)


def B0invSolve_block(Y1,Y2,phi_hat,SIG_hat,restrict):
    '''단기 또는 장기 제약을 통한 블록 외생성 제약 하 구조 VAR(p) 모형의 식별
    
    Args:
        Y1 : 도메스틱 부분-시스템, 글로벌에 의존
        Y2 : 글로벌 부분-시스템, 도메스틱에 독립적
        phi_hat : 축약형 VAR(p) 모형 계수 행렬의 OLS 추정량
        SIG_hat : 축약형 VAR(p) 모형 분산-공분산 행렬의 OLS 추정량
        restrict : 장기 또는 단기 제약에 대한 옵션 ('short' 또는 'long'으로 입력)
    
    Return:
        B0inv : 구조형 VAR 모형의 식별에 사용되는 B0의 역행렬
    '''
    Y1 = np.array(Y1)
    if Y1.shape[1] < 2 :
        Y1 = Y1.reshape(-1,1)
        print("Y1은 단변수 (univariate) 입니다.")
        
    Y2 = np.array(Y2)
    if Y2.shape[1] < 2 :
        Y2 = Y2.reshape(-1,1)
        print("Y2는 단변수 (univariate) 입니다.")
    
    T1,K1 = Y1.shape
    T2,K2 = Y2.shape
    K = K1+K2
    
    SIG11_hat = SIG_hat[0:K1,0:K1]
    SIG12_hat = SIG_hat[0:K1,K1:K]
    SIG21_hat = SIG_hat[K1:K,0:K1]
    SIG22_hat = SIG_hat[K1:K,K1:K]
    
    # 1. Short-run Restriction
    if restrict == 'short':
        B0inv22 = chol(SIG22_hat).T
        B0inv12 = SIG12_hat @ inv(B0inv22.T)
        B0inv11_temp1 = SIG11_hat - B0inv12 @ (B0inv12.T)
        B0inv11_temp2 = 0.5*(B0inv11_temp1 + B0inv11_temp1.T)
        B0inv11 = chol(B0inv11_temp2).T
        B0inv = np.block([[B0inv11, B0inv12], [zeros(K2,K1), B0inv22]])
    
    # 2. Long-run Restriction
    elif restrict == 'long':
        Am = phi_hat.T
        K,n = Am.shape
        p = n/K
        p = int(p)
        A1 = eye(K)
        for j in range(p):
            A1 = A1 - Am[:,j*K:(j+1)*K]
        P1 = inv(A1)

        P1_11 = P1[0:K1,0:K1]
        P1_12 = P1[0:K1,K1:K]
        P1_21 = zeros(K2,K1)
        P1_22 = P1[K1:K,K1:K]

        temp = P1_22 @ SIG22_hat @ P1_22.T
        temp = 0.5*(temp+temp.T)
        L_22 = chol(temp).T

        L_1222 = P1_11@SIG12_hat@P1_22.T + P1_12@SIG22_hat@P1_22.T
        L_12 = L_1222 @ inv(L_22.T)

        L_1111_temp = P1_11@SIG11_hat@P1_11.T + P1_12@SIG12_hat.T@P1_11.T + P1_11@SIG12_hat@P1_12.T + P1_12@SIG22_hat@P1_12.T
        L_1111 = L_1111_temp - L_12 @ L_12.T
        L_1111 = 0.5*(L_1111+L_1111.T)
        L_11 = chol(L_1111).T

        L = np.block([[L_11, L_12], [zeros(K2,K1), L_22]])
        B0inv = A1 @ L
    else:
        print("경고: 올바르지 못한 restrict값 입니다.")
        print("restrict 변수는 단기제약의 경우 short, 장기제약의 경우 long이어야합니다.")
        
    return B0inv

class BlockVectorAutoRegression:
    def __init__(self, lag: int=None):
        self.lag = lag
        self.res = None
        
    def fit(self, df1, df2, irf: str=None, h: int=16, q: float=90, n: int=2000, verbose: bool=False):
        if irf is None:
            self.res = var_block(df1, df2, lag=self.lag)
        elif irf in ['short', 'long']:
            self.res = block_var_irf_bootstrap(df1, df2, lag=self.lag, h=h, irf=irf, q=q, n=n, verbose=verbose)
        return self.res
    
class BlockIrfResult:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
            
    def get_description(self):
        desc = {
            "결과": ['theta', 'cilv', 'cihv', 'cum_theta', 'cum_cilv', 'cum_cihv',
                     'phi_hat', 'sig_hat', 'g0_hat', 'f_mat', 'y', 'u_hat1', 'u_hat2', 'lag'],
            "설명": ['추정된 VAR(p) 모형의 충격반응함수', '충격반응함수 신뢰구간의 하한', 
                     '충격반응함수 신뢰구간의 상한', '추정된 VAR(p) 모형의 누적된 충격반응함수', '누적된 충격반응함수 신뢰구간의 하한',
                     '누적된 충격반응함수 신뢰구간의 상한',
                     '축약형 VAR(p) 모형의 계수 행렬 추정량', '축약형 VAR(p) 모형의 분산-공분산 행렬 추정량', 
                     'y20 계수의 OLS 추정량 (가정: 글로벌 변수들이 도메스틱 변수들을 계수 및 오차항을 통해 동시다발적으로 영향을 미친다)',
                     '축약형 VAR(p) 모형의 동반행렬 형태의 계수 행렬 추정량', '추정에 사용된 완전한 VAR 시스템',
                     '도메스틱 부분 시스템의 잔차항', '글로벌 부분 시스템의 잔차항', '시차(None이면 BIC를 기준으로 선택된 값, None이 아니면 입력된 값)']
        }

        df = pd.DataFrame(desc)
        df.set_index("결과", inplace=True)
        df.index.name = None
        #df = df.style.set_properties(**{'text-align': 'left'})
        #df = df.set_table_styles([{'selector': 'th', 'props': [('text-align', 'left')]}])

        return df
    
    def plot_irf(self, cum: bool=False, title: bool=True, title_fontsize: int=14):
        if cum:
            self._plot_irf(self.cum_theta, self.cum_cihv, self.cum_cilv, self.__restrict, cum, self.__h, title, title_fontsize)
        else:
            self._plot_irf(self.theta, self.cihv, self.cilv, self.__restrict, cum, self.__h, title, title_fontsize)
        
    def _plot_irf(self, theta, cihv, cilv, restrict, cum, h, title, title_fontsize):
        
        col_names = self.__col_names
        K = len(col_names)
        H = h
        
        Theta = theta
        CIHv = cihv
        CILv = cilv
        
        name = []
        for idx in range(K):
            for idx2 in range(K):
                name1 = f'{col_names[idx2]} to {col_names[idx]}'
                name.append(name1)

        name = np.array(name)
        time = np.matrix(np.arange(0,H+1))

        plt.figure(figsize=(K*3.3, K*3.3))  
        for i in range(1,K**2+1):
            plt.subplot(K,K,i)
            plt.plot(time.T, Theta[i-1,:].T, '-k', time.T, CIHv[i-1,:].T, '-.b', time.T, CILv[i-1,:].T, '-.b', time.T, zeros(H+1,1), '-r')
            plt.xlim([0,H+1])
            plt.title(name[i-1])
            
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        
        if title and cum:
            plt.suptitle('Cumulative IRF Estimation Results', fontsize=title_fontsize)
        elif title and not cum:
            plt.suptitle('IRF Estimation Results', fontsize=title_fontsize)

        plt.legend(['IRF', 'Bootstrap'])

def block_var_irf_bootstrap(y1, y2,
                            lag: int=None, h: int=16, 
                            irf: str='short', q: float=90, 
                            n: int=2000,
                            verbose: bool=False):
    '''블록 외생성 제약 하 VAR(p) 모형의 충격반응함수 및 부트스트래핑을 통한 충격반응함수의 신뢰구간 도출
    
    Args:
        y1 : 도메스틱 부분-시스템, 글로벌에 의존
        y2 : 글로벌 부분-시스템, 도메스틱과 독립적
        p : VAR(p) 모형의 최적 시차 (default = None)
        h : 충격반응함수 도출 시 고려하는 최대 예측시차
        irf: 사용하고자하는 식별 방법 ('short' = 단기제약, 'long' = 장기제약)
        qt : 충격반응함수의 신뢰구간 (confidence interval)
            입력하지 않을 시 default = 90 (90% 신뢰구간)
        n : 부트스트래핑 횟수
    
    Returns:
        Theta : VAR 모형의 충격반응함수
        CILv : 충격반응함수 신뢰구간의 하한
        CIHv : 충격반응함수 신뢰구간의 상한
        cumTheta : VAR 모형의 누적된 충격반응함수
        cumCILv : 누적된 충격반응함수 신뢰구간의 하한
        cumCIHv : 누적된 충격반응함수 신뢰구간의 상한
    
        Steps:
        <1 단계>: 축약형 VAR(p) 모형 추정 (demean 사용)
        <2 단계>: 단기, 또는 장기 제약으로 B0inv 도출
        <3 단계>: 잔차를 섞어 새로운 잔차를 랜덤 샘플링
        <4 단계>: 부트스트랩 샘플을 얻은 후 축약형 VAR(p) 모형을 다시 추정
        <5 단계>: 충격반응함수를 추정
        <6 단계>: <3 단계>부터 <5단계>까지 반복 한 이후 신뢰구간을 도출
    '''
    p = lag
    H = h
    restrict = irf
    qt = q
    col1_names = y1.columns
    col2_names = y2.columns
    col_names = col1_names.append(col2_names)
    
    Y1 = np.array(y1)
    if Y1.shape[1] < 2 :
        Y1 = Y1.reshape(-1,1)
        print("Y1은 단변수 (univariate) 입니다.")
        
    Y2 = np.array(y2)
    if Y2.shape[1] < 2 :
        Y2 = Y2.reshape(-1,1)
        print("Y2는 단변수 (univariate) 입니다.")
    
    T1,K1 = Y1.shape
    T2,K2 = Y2.shape
    K = K1+K2
    T = T1

    # <Step 1>: OLS estimator of each equation
    res = var_block(Y1, Y2, p, verbose=verbose)
    phi_hat = res.phi_hat
    SIG_hat = res.sig_hat
    G0_hat = res.g0_hat
    F = res.f_mat
    Y = res.y
    u_hat1 = res.u_hat1
    u_hat2 = res.u_hat2
    phat = res.lag
    
    
    psave = p
    p = phat
    u_hat = np.hstack((u_hat1, u_hat2))

    # <Step 2>: Get B0inv
    B0inv = B0invSolve_block(Y1,Y2,phi_hat,SIG_hat,restrict)

    # Initial IRF
    Theta = irf_estimate(F,p,H,B0inv)
    cumTheta = zeros(K**2,H+1)
    for c in range(K**2):
        cumtemp = np.cumsum(Theta[c,:])
        cumTheta[c,:] = cumtemp

    # Confidence Interval by Bootstrapping
    n_bootstrap = n
    
    IRFm = zeros(n,K**2*(H+1))
    cumIRFm = zeros(n,K**2*(H+1))
    
    for iter in range(n_bootstrap):
        # <Step 3>: Obtain a new set of errors
        indu = matrix(range(T-p)) # 0 from 93
        indu = indu.T
        indu = randper(indu)
        indu = indu.astype(int) # make matrix as an integer
    
        ustar = u_hat[indu,:]

        ystar = zeros(T,K)
        ind_ystar = np.fix(rand(1,1)*(T-p+1))+1
        ind_ystar = ind_ystar[0,0]
        ind_ystar = ind_ystar.astype(int)
        ystar[0:p-1,:] = Y[ind_ystar:ind_ystar+p-1,:]

        # <Step 4>: Obtain the Bootstrap Sample and estimate again
        for it in range(p,T):
            ystar[it,:] = ustar[it-p,:]
            ystar[it,0:K1] = ystar[it,0:K1] + ystar[it,K1:K] @ G0_hat.T
            for jt in range(p):
                ystar[it,:] = ystar[it,:] + ystar[it-jt-1,:] @ phi_hat[jt*K:(jt+1)*K,:]
                
        ystar1 = ystar[:,0:K1]
        ystar2 = ystar[:,K1:K]
        
        #phi_star, SIG_star, G0_star, F_star, Y_star, u_star1, u_star2, phat_star = OLS_VAR_block(ystar1, ystar2, p)
        res_star = var_block(ystar1, ystar2, p)
        phi_star = res_star.phi_hat
        SIG_star = res_star.sig_hat
        G0_star = res_star.g0_hat
        F_star = res_star.f_mat
        Y_star = res_star.y
        u_star1 = res_star.u_hat1
        u_star2 = res_star.u_hat2
        phat_star = res_star.lag
        
        u_star = np.hstack((u_star1,u_star2))
        B0inv_star = B0invSolve_block(ystar1, ystar2, phi_star, SIG_star, restrict)

        # <Step 5>: Compute the Impulse Response Function
        IRFboot = irf_estimate(F_star, p, H, B0inv_star)
        cumIRFboot = zeros(K**2,H+1)
        for c in range(K**2):
            cumIRFboot[c,:] = np.cumsum(IRFboot[c,:])
        
        IRFm[iter,:] = vec(IRFboot.T).T
        cumIRFm[iter,:] = vec(cumIRFboot.T).T

    # <Step 6>: Obtain Quantiles
    if qt is None:
        CIlow = np.quantile(np.array(IRFm), 0.05, axis=0) # must be an option (default = 5%)
        CIhigh = np.quantile(np.array(IRFm), 0.95, axis=0) # must be an option (default = 95%)
        cumCIlow = np.quantile(np.array(cumIRFm), 0.05, axis=0) # must be an option (default = 5%)
        cumCIhigh = np.quantile(np.array(cumIRFm), 0.95, axis=0) # must be an option (default = 95%)
    else:
        qtlow = ((100-qt)/2)*0.01
        qthigh = (100 - qtlow)*0.01
        CIlow = np.quantile(np.array(IRFm), qtlow, axis=0)
        CIhigh = np.quantile(np.array(IRFm), qthigh, axis=0)
        cumCIlow = np.quantile(np.array(cumIRFm), qtlow, axis=0)
        cumCIhigh = np.quantile(np.array(cumIRFm), qthigh, axis=0)
    
    CIlow_mat = matrix(CIlow)
    CIhigh_mat = matrix(CIhigh)
    cumCIlow_mat = matrix(cumCIlow)
    cumCIhigh_mat = matrix(cumCIhigh)

    CILv = reshape(CIlow_mat.T, H+1, K**2).T
    CIHv = reshape(CIhigh_mat.T, H+1, K**2).T
    cumCILv = reshape(cumCIlow_mat.T,H+1,K**2).T
    cumCIHv = reshape(cumCIhigh_mat.T,H+1,K**2).T
    
    if verbose:
        if psave == None:
            print(" ")
            print("BIC로 추정한 VAR 최적 시차 = ", phat)
        else:
            print(" ")
            print("사용자가 입력한 VAR 최적 시차 = ", phat)

        '''
        Plotting Results
        '''
        name = []
        for idx in range(1,K+1):
            for idx2 in range(1,K+1):
                warnings.filterwarnings("ignore")
                name1 = fr"$e_{idx2} \Rightarrow y_{idx}$"
                name.append(name1)

        name = np.array(name)
        time = np.matrix(np.arange(0,H+1))

        plt.figure(figsize=(K*3.3, K*3.3))  
        for i in range(1,K**2+1):
            plt.subplot(K,K,i)
            plt.plot(time.T, Theta[i-1,:].T, '-k', time.T, CIHv[i-1,:].T, '-.b', time.T, CILv[i-1,:].T, '-.b', time.T, zeros(H+1,1), '-r')
            plt.xlim([0,H+1])
            plt.title(name[i-1])

        plt.legend(['IRF', 'Bootstrap'])
        plt.show()

        plt.figure(figsize=(K*3.3, K*3.3))  
        for i in range(1,K**2+1):
            plt.subplot(K,K,i)
            plt.plot(time.T, cumTheta[i-1,:].T, '-k', time.T, cumCIHv[i-1,:].T, '-.b', time.T, cumCILv[i-1,:].T, '-.b', time.T, zeros(H+1,1), '-r')
            plt.xlim([0,H+1])
            plt.title(name[i-1])

        plt.legend(['cumIRF', 'Bootstrap'])
        plt.show()
    
    return BlockIrfResult(theta=Theta, cilv=CILv, cihv=CIHv, cum_theta=cumTheta, cum_cilv=cumCILv, cum_cihv=cumCIHv,
                          phi_hat=phi_hat, sig_hat=SIG_hat, g0_hat=G0_hat, f_mat=F, y=Y, u_hat1=u_hat1, u_hat2=u_hat2, lag=phat,
                          _BlockIrfResult__restrict=restrict, _BlockIrfResult__lag=lag, _BlockIrfResult__h=h,
                          _BlockIrfResult__col_names=col_names, _BlockIrfResult__phi_hat=phi_hat, _BlockIrfResult__sig_hat=SIG_hat,
                          _BlockIrfResult__f_mat=F, _BlockIrfResult__u_hat1=u_hat1, _BlockIrfResult__u_hat2=u_hat2)
