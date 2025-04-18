import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import seaborn as sns
import scipy.stats as sp
import warnings

from typing import Union

from ...utils.operator import array, inv, ones, rows, meanc, demeanc, zeros, diag, eye, rev, vec, chol, reshape, matrix, det, sumc, minc
from ...utils.pdf import cdfn
from ...utils.rng import rand


class ArResult:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
            
    def get_description(self):
        desc = {
            "결과": ['phi_hat', 'omega_hat', 'f_mat', 'y0', 'y_lag', 'y_pred'],
            "설명": ['AR(p) 모형 OLS 계수 추정량', 'AR(p) 모형 OLS 분산 추정량', 
                     '동반행렬(Companion Form) 형태의 계수 추정량',
                     '추정에 사용된 반응변수', ' 추정에 사용된 설명변수', '예측치']
        }

        df = pd.DataFrame(desc)
        df.set_index("결과", inplace=True)
        df.index.name = None

        return df
    
    def get_table(self):
        index = [f'AR_L{i+1}' for i in range(self.__lag)]
        cols = ["Coefficients", "S.E.", "t-value", "p-value"]
        data = np.hstack((self.phi_hat, self.__stde, self.__t_val, self.__p_val))

        result = pd.DataFrame(np.around(data,4), index=index, columns=cols)
        print("==================================================")
        print(result)
        print("==================================================")
        
        if self.__h is not None:
            index1 = [f'H={h+1}' for h in range(self.__h)]
        
            cols1 = ["Predicted Value for H"]
            data1 = np.around(self.y_pred, 4)

            result1 = pd.DataFrame(data1, index=index1, columns=cols1)
            print(result1)
            print("==================================================")
            
    def plot_fitted_resid(self, figsize=(10,5)):
        self._plot_fr(self.y0, self.__y_, self.__y_hat, self.__u_hat, figsize)
        
    def _plot_fr(self, Y0, Y_, y_hat, u_hat, figsize):
        # Plotting
        ymax = np.concatenate((Y0,y_hat)).max()
        ymin = np.concatenate((Y0,y_hat)).min()
        umax = max(u_hat)
        umin = min(u_hat)
        if ymin > 0:
            weight = 0.9
        else:
            weight = 1.1
        if ymin < 1:
            weight = -0.5

        fig, ax = plt.subplots(figsize=figsize)
        twin1 = ax.twinx()

        #ax.plot(np.arange(1,rows(Y0)+1), Y0, '-k', np.arange(1,rows(y_hat)+1), y_hat, '-b', linewidth=1.5)
        #twin1.plot(np.arange(1,rows(u_hat)+1), u_hat, '--r', linewidth=1.5, alpha = 0.8)

        ax.plot(self.__index[self.__lag:], Y0, '-k', self.__index[self.__lag:], y_hat, '-b', linewidth=1.5)
        twin1.plot(self.__index[self.__lag:], u_hat, '--r', linewidth=1.5, alpha = 0.8)

        #if weight > 0:
        #    ran = (ymin*weight, ymax*1.1)
        #else:
        #    ran = (ymin+weight, ymax*1.1)

        #ax.set(xlim=(1,rows(Y0)), ylim=ran)
        #twin1.set(ylim=(umin*1.1, umax*1.1))
        plt.grid(True)
            
        plt.title(f"Plot of Data({self.__var_name[0]}), Fitted Value and Residual")
        ax.legend([f'{self.__var_name[0]}', 'Fitted'],loc='upper left')
        twin1.legend(['Residual'], loc='upper right')
        ax.set_xmargin(0)
        twin1.set_xmargin(0)

        fig.tight_layout
        plt.show()
        
    def plot_forecasts(self, figsize=(10,5)):
        self._plot_f(self.y0, self.y_pred, self.__y_, figsize)
        
    def _plot_f(self, Y0, Y_predm, Y_, figsize):
        Y_H = np.vstack((Y0, Y_predm))
        yhmax = np.max(Y_H)
        yhmin = np.min(Y_H)
        if yhmin > 0:
            weight1 = 0.9
        else:
            weight1 = 1.1
        if yhmin < 1:
            weight1 = -0.5

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(self.__index[self.__lag:], Y0, '-k', linewidth=1.5)
        plt.title(f"Plot of Data({self.__var_name[0]}) and Predicted Value")
        plt.grid(True)

        last_date = self.__index[-1]
        forecasting_dates = pd.date_range(start=last_date + pd.DateOffset(days=1), periods=self.__h, freq=self.__index.freq)
        ax.plot(forecasting_dates, Y_predm, '--r', linewidth=1.5)
        
        ax.legend([f'{self.__var_name[0]}', 'Predicted'])
        ax.set_xmargin(0)
        plt.show()


def ar(y: Union[pd.Series, pd.DataFrame], 
       lag: int=1,
       h: int=None,
       verbose: bool=False):
    
    ''' AR(p) 모형에 대한 OLS 추정 함수
    
    Args:
        y : 추정하고자하는 변수, T by 1 벡터
        p : 추정하는 모형의 시차
        printi : 추정 결과 출력 여부, 입력하지 않은 경우 None
        H : 최대 예측 시차, 입력하지 않을 경우 None
        
    Returns:
        phi_hat : OLS 계수 추정량, p by 1 벡터
        Omega_hat : OLS 분산 추정량, 1 by 1 벡터
        F : 동반행렬 (Companion Form) 형태의 계수 추정량
        Y0 : 추정에 사용된 반응변수, T-p by 1 벡터
        y_lag : 추정에 사용된 설명변수, T-p by p 벡터
        Y_predm : 예측된 값, H by 1 벡터
        
    * 절편항을 고려하지 않기 위해 y에 대해 자동으로 demeaning 과정이 들어감.
    ** 이후 예측 시, demeaning과정에서 제거된 mean을 다시 더해줌.
    '''
    p = lag
    H = h
    
    if isinstance(y, pd.Series):
        y = y.to_frame()
        
    var_name = y.columns
    index = y.index
    
    y = np.array(y)
    
    if y.shape[1] >= 2:
        print("경고: 입력되는 X는 단변수어야합니다.")
    
    # 1. 반응변수 (LHS variable) 지정 및 demeaning
    Y_ = meanc(y)
    Y = demeanc(y)
    T1 = rows(Y)
    
    Y0 = Y[p:,:]
    T = rows(Y0)

    # 2. 절편을 포함한 설명변수 (RHS variable) 지정
    y_lag = zeros(T,p)
    for j in np.arange(p):
        y_lag[:,j] = Y[p-1-j:T1-j-1,:].reshape(1,-1)

    # 3. OLS estimator    
    XX = y_lag.T @ y_lag
    phi_hat = inv(XX) @ (y_lag.T @ Y0)

    # 4. Omega_hat
    y_hat = y_lag @ phi_hat
    u_hat = Y0 - y_hat
    sig2_hat = (u_hat.T @ u_hat) / (T-p)
    
    varbhat = diag(ones(p,1)*sig2_hat)*inv(XX)  # k by k, variance of bhat
    stde = np.sqrt(diag(varbhat)).reshape(-1,1) # k by 1, standard error
    b0 = zeros(p,1)  # null hypothesis
    t_val = np.divide((phi_hat - b0), stde) # k by 1, t values
    p_val = 2*(1-cdfn(np.abs(t_val))) # k by 1, t values
    
    # 5. Companion Form
    if p > 1:
        F1 = phi_hat.T 
        F2 = np.hstack((eye(p-1), zeros(p-1,1)))
        F = np.vstack((F1, F2))  # p by p
    elif p==1:
        F = phi_hat.T
       
    # 6. Prediction
    if H is None:
        Y_predm = None
    else:
        Y_predm = zeros(H,1)
        Y_Lag = rev(Y[-p:,:]) # 예측에 사용될 설명변수, 최근값이 위로 오도록 역순하기
        FF = F
        for h in range(0, H):
            Y_h = FF @ Y_Lag
            y_h = Y_h[0,0]
            Y_predm[h,:] = y_h
            Y_Lag = np.vstack((y_h, Y_Lag[0:p-1,:]))
                # Y(t-1)을 1기-얘측치로 대체
                # 대신 Y(t-P)는 제외하기
            FF = FF @ F
            
        Y_predm = Y_predm + ones(H,1)*Y_
    
    if verbose:
        index = [f'AR_L{i+1}' for i in range(p)]
        cols = ["Coefficients", "S.E.", "t-value", "p-value"]
        data = np.hstack((phi_hat, stde, t_val, p_val))

        result = pd.DataFrame(np.around(data,4), index=index, columns=cols)
        print(result)
        print("==================================================================")
        
        if H is None:
            index1 = ["H = None"]
        else:
            index1 = [f'H={h+1}' for h in range(H)]
        
        #cols1 = ["Predicted Value for H = " + str(H) + ' is']
        cols1 = ["Predicted Value for H = " + ' is']
        if H is None:
            data1 = []
        else:
            data1 = np.around(Y_predm,4)
            
        result1 = pd.DataFrame(data1, index=index1, columns=cols1)
        print(result1)
    
    Y0 = Y0 + ones(rows(Y0),1) * Y_
    y_hat = y_hat + ones(rows(y_hat),1) * Y_
    if verbose:
        # Plotting
        ymax = np.concatenate((Y0,y_hat)).max()
        ymin = np.concatenate((Y0,y_hat)).min()
        umax = max(u_hat)
        umin = min(u_hat)
        if ymin > 0:
            weight = 0.9
        else:
            weight = 1.1
        if ymin < 1:
            weight = -0.5

        fig, ax = plt.subplots(figsize=[10,5])
        twin1 = ax.twinx()

        ax.plot(np.arange(1,rows(Y0)+1), Y0, '-k', np.arange(1,rows(y_hat)+1), y_hat, '-b', linewidth=1.5)
        twin1.plot(np.arange(1,rows(u_hat)+1), u_hat, '--r', linewidth=1.5, alpha = 0.8)

        if weight > 0:
            ran = (ymin*weight, ymax*1.1)
        else:
            ran = (ymin+weight, ymax*1.1)

        ax.set(xlim=(1,rows(Y0)), ylim=ran)
        twin1.set(ylim=(umin*1.1, umax*1.1))
        plt.grid(True)
        plt.title("Plot of Data, Fitted Value and Residual")
        ax.legend(['Data', 'Fitted'],loc='upper left')
        twin1.legend(['Residual'], loc='upper right')

        fig.tight_layout
        plt.show()

        if H is not None:
            Y_H = np.vstack((Y0, Y_predm))
            yhmax = np.max(Y_H)
            yhmin = np.min(Y_H)
            if yhmin > 0:
                weight1 = 0.9
            else:
                weight1 = 1.1
            if yhmin < 1:
                weight1 = -0.5

            plt.figure(figsize=[10,5])
            plt.plot(np.arange(1,rows(Y0)+1), Y0, '-k', linewidth=1.5)
            plt.title("Plot of Data and Predicted Value")
            plt.xlim([1,rows(Y_H)])
            if weight1 > 0:
                plt.ylim([yhmin*weight1, yhmax*1.1])
            else:
                plt.ylim([yhmin+weight1, yhmax*1.1])
            plt.grid(True)

            plt.plot(np.arange(rows(Y0)+1,rows(Y_H)+1), Y_predm, '--r', linewidth=1.5)

            plt.legend(['Data', 'Predicted'])
            plt.show()
    
    return ArResult(phi_hat=phi_hat, sig2_hat=sig2_hat, f_mat=F, y0=Y0, y_lag=y_lag, y_pred=Y_predm,
                    _ArResult__y_hat=y_hat, _ArResult__y_=Y_, _ArResult__u_hat=u_hat, _ArResult__stde=stde,
                    _ArResult__t_val=t_val, _ArResult__p_val=p_val, _ArResult__lag=p, _ArResult__h=H,
                    _ArResult__var_name=var_name, _ArResult__index=index)