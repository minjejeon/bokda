import math
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches


def generate_verbose_iter_range(verbose, start, end, desc=""):
    if verbose:
        return tqdm(range(start, end), desc=desc)
    else:
        return range(start, end)


def Recursive_IRF(Parameters, Draw, verbose = True) :

    #### 1 : IRF Calculation

    if verbose:
        print("> Impulse Response Calculating...")

    wimpu = np.empty((Draw.Bet[:, Parameters.c:, 0].shape[1], Draw.Bet[:,Parameters.c:,0].shape[1], Parameters.nstep))
    Imp = np.empty((Parameters.nvar, Parameters.nvar, Parameters.nstep, Parameters.ndraws))
    Cat_matrix = np.column_stack((np.eye(Parameters.nvar*Parameters.p - Parameters.nvar), np.zeros((Parameters.nvar*Parameters.p - Parameters.nvar, Parameters.nvar))))
    Draw.BB = np.zeros((Draw.Bet[:, Parameters.c:, 0].shape[0] + Cat_matrix.shape[0], Draw.Bet[:, Parameters.c:, 0].shape[1], Parameters.ndraws))

    iters = generate_verbose_iter_range(verbose, 0, Parameters.ndraws, desc="1/2")
    for d in iters:
        BB = Draw.Bet[:, Parameters.c:, d]
        Draw.BB[:,:,d] = np.vstack((BB, Cat_matrix))

        for j in range(0, Parameters.nstep) :
            if j == 0 :
                wimpu[:,:,j] = np.eye(Draw.Bet[:, Parameters.c:, 0].shape[1])
            elif j > 0 :
                wimpu[:,:,j] = np.linalg.matrix_power(Draw.BB[:,:,d], j)
            Imp[:,:,j,d] = wimpu[:Parameters.nvar, :Parameters.nvar, j] @ np.linalg.cholesky(Draw.Sigma[:,:,d])
    
    Draw.Imp = Imp

    Impulse = np.zeros((Parameters.nstep, Parameters.nvar, Parameters.ndraws, Parameters.nvar))
    Impres = np.zeros((Parameters.nstep, Parameters.nvar, 3, Parameters.nvar))
    #Impres : 첫 번째열 -> IRF 추정 horizon, 두 번째 열  -> 충격반응의 대상, 세 번째 열 -> 0:median, 1,2:quantile, 네 번째 열 -> 충격의 원천

    iters = generate_verbose_iter_range(verbose, 0, Parameters.nvar, desc="2/2")
    for i in iters:
        for d in range(0, Parameters.ndraws) :
            for j in range(0, Parameters.nstep) :
                Impulse[j,:,d,i] = Draw.Imp[:,i,j,d]
    
        for o in range(0, Parameters.nstep) :
                for j in range(0, Parameters.nvar) :
                    Impres[o,j,1,i] = np.quantile(Impulse[o,j,:,i], 1 - Parameters.pereb)
                    Impres[o,j,0,i] = np.median(Impulse[o,j,:,i])
                    Impres[o,j,2,i] = np.quantile(Impulse[o,j,:,i], Parameters.pereb)

    Draw.Impulse = Impulse
    Draw.Impres = Impres

    if verbose:
        print("> Forecast error variance decomposition Calculating...")

    ##### 2 : Forecast error variance decomposition 

    FEVD0 = Draw.Imp**2
    FEVD1 = np.zeros((Parameters.nvar, Parameters.nvar, Parameters.nstep, Parameters.ndraws))
    iters = generate_verbose_iter_range(verbose, 0, Parameters.ndraws, desc="1/4")
    for d in iters:
        for i in range(0, Parameters.nvar) :
            for j in range(0, Parameters.nvar) :
                FEVD1[i,j,:,d] = np.cumsum(FEVD0[i,j,:,d])

    FEVD2 = np.zeros((Parameters.nvar, Parameters.nstep, Parameters.ndraws))
    iters = generate_verbose_iter_range(verbose, 0, Parameters.ndraws, desc="2/4")
    for d in iters:
        for i in range(0, Parameters.nvar) :
            for j in range(0, Parameters.nstep) :
                FEVD2[i,j,d] = np.cumsum(FEVD1[i,:,j,d])[Parameters.nvar-1]

    FEVD3 = np.zeros((Parameters.nvar, Parameters.nvar, Parameters.nstep, Parameters.ndraws))
    iters = generate_verbose_iter_range(verbose, 0, Parameters.ndraws, desc="3/4")
    for d in iters:
        for i in range(0, Parameters.nvar) :
            for j in range(0, Parameters.nstep) :
                FEVD3[i,:,j,d] = (FEVD1[i,:,j,d]/FEVD2[i,j,d])*100

    FEVD = np.zeros((Parameters.nvar, Parameters.nvar, Parameters.nstep))
    for i in range(0, Parameters.nvar) :
        for j in range(0, Parameters.nvar) :
            for k in range(0, Parameters.nstep) :
                FEVD[i,j,k] = np.quantile(FEVD3[i,j,k,:], 0.5)

    Draw.FEVD = FEVD

    ##### 3. Shock Series calculation

    Draw.Shock = np.zeros((Parameters.nvar, Parameters.T, Parameters.ndraws))
    iters = generate_verbose_iter_range(verbose, 0, Parameters.ndraws, desc="4/4")
    for d in iters:
        Draw.Shock[:,:,d] = np.linalg.inv(np.linalg.cholesky(Draw.Sigma[:,:,d])) @ Draw.U_B[:,:,d].T
    
    Draw.Shock_inf = np.zeros((Parameters.nvar, Parameters.T, 3))
    for i in range(0, Parameters.nvar) :
        for j in range(0, Parameters.T) :
            Draw.Shock_inf[i,j,0] = np.median(Draw.Shock[i,j,:])
            Draw.Shock_inf[i,j,1] = np.quantile(Draw.Shock[i,j,:], Parameters.pereb)
            Draw.Shock_inf[i,j,2] = np.quantile(Draw.Shock[i,j,:], 1-Parameters.pereb)
    
    ##### 4. Historical Decomposition

    if verbose:
        print("> Historical Decomposition Calculating...")

    wimpu = np.empty((Draw.Bet[:, Parameters.c:, 0].shape[1], Draw.Bet[:,Parameters.c:,0].shape[1], Parameters.T))
    Imp = np.empty((Parameters.nvar, Parameters.nvar, Parameters.T, Parameters.ndraws))
    Cat_matrix = np.column_stack((np.eye(Parameters.nvar*Parameters.p - Parameters.nvar), np.zeros((Parameters.nvar*Parameters.p - Parameters.nvar, Parameters.nvar))))
    BB_stack = np.zeros((Draw.Bet[:, Parameters.c:, 0].shape[0] + Cat_matrix.shape[0], Draw.Bet[:, Parameters.c:, 0].shape[1], Parameters.ndraws))

    iters = generate_verbose_iter_range(verbose, 0, Parameters.ndraws, desc="1/3")
    for d in iters:
        BB = Draw.Bet[:, Parameters.c:, d]
        BB_stack[:,:,d] = np.vstack((BB, Cat_matrix))

        for j in range(0, Parameters.T) :
            if j == 0 :
                wimpu[:,:,j] = np.eye(Draw.Bet[:, Parameters.c:, 0].shape[1])
            elif j > 0 :
                wimpu[:,:,j] = np.linalg.matrix_power(BB_stack[:,:,d], j)
            Imp[:,:,j,d] = wimpu[:Parameters.nvar, :Parameters.nvar, j] @ np.linalg.cholesky(Draw.Sigma[:,:,d])

    Impulse = np.zeros((Parameters.T, Parameters.nvar, Parameters.ndraws, Parameters.nvar))

    iters = generate_verbose_iter_range(verbose, 0, Parameters.nvar, desc="2/3")
    for i in iters:
    # for i in tqdm(range(0, Parameters.nvar)) :
        for d in range(0, Parameters.ndraws) :
            for j in range(0, Parameters.T) :
                Impulse[j,:,d,i] = Imp[:,i,j,d]

    Draw.HD_shock = np.zeros((Parameters.T+1, Parameters.nvar,Parameters.nvar, Parameters.ndraws))

    iters = generate_verbose_iter_range(verbose, 0, Parameters.ndraws, desc="3/3")
    for d in iters:
        for o in range(0, Parameters.nvar) : 
            for j in range(0, Parameters.nvar) :
                for i in range(0, Parameters.T+1) :
                    Draw.HD_shock[i,j,o,d] = np.dot(Impulse[:,o,d,j][0:i], Draw.Shock[j,:,d][0:i][::-1])

    Draw.med_HD_shock = np.zeros((Parameters.T+1, Parameters.nvar, Parameters.nvar))
    for o in range(0, Parameters.nvar) :
        for j in range(0, Parameters.nvar) :
            for i in range(1, Parameters.T+1) :
                Draw.med_HD_shock[i,j,o] = np.median(Draw.HD_shock[i,j,o,:])

    Draw.med_HD_shock = Draw.med_HD_shock[1:,:,:]
    return Draw


            

### plot ###


def plot_shock_series_calculation(columns, Parameters, Draw):
    """
    """
    fig, axs = plt.subplots(Parameters.nvar, 1, figsize=(8, 2*Parameters.nvar))
    # fig, axs = plt.subplots(Parameters.nvar)

    fig.suptitle("", fontsize=14, y=0.95)
    for j, ax in enumerate(axs.flatten()[:len(columns)]):
        # 중간값 플로팅 (검정색 실선)
        ax.plot(np.arange(0,Parameters.T), Draw.Shock_inf[j, :, 0], color='red', linewidth=1)
        ax.fill_between(np.arange(0,Parameters.T), Draw.Shock_inf[j, :, 1], Draw.Shock_inf[j, :, 2], color='blue', alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-')
        
        # 각 subplot의 제목 설정
        ax.set_title(columns[j], fontsize=10)
        
        # 축 스타일 설정
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.grid(False)
        ax.set_xlim(left=0)

    fig.tight_layout()
    plt.show()




def plot_impulse_response(columns, Parameters, Draw):
    """
    """
    counts = len(columns)
    plot_rows = math.ceil(counts/3)
    plot_cols = 3

    for i, title in enumerate(columns):
        fig, axs = plt.subplots(plot_rows, plot_cols, figsize=(7, 2*plot_rows))
        # fig, axs = plt.subplots(plot_rows, plot_cols)
        fig.suptitle(title, fontsize=14, y=1)

        # 각 큰 subplot 안에 j 개수만큼 작은 플롯 추가
        for j, ax in enumerate(axs.flatten()[:len(columns)]):
            # 중간값 플로팅 (검정색 실선)
            ax.plot(np.arange(0,Parameters.nstep), Draw.Impres[:, j, 0, i], color='black', linewidth=2)
            ax.fill_between(np.arange(0,Parameters.nstep), Draw.Impres[:, j, 1, i], Draw.Impres[:, j, 2, i], color='lightblue', alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='-')
            
            # 각 subplot의 제목 설정
            ax.set_title(columns[j], fontsize=10)
            # 축 스타일 설정
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
            ax.grid(False)
            ax.set_xlim(left=0)
        # 빈 subplot 제거
        for k in range(len(columns), len(axs.flatten())):
            fig.delaxes(axs.flatten()[k])
            fig.subplots_adjust(hspace=0.5, wspace=0.3)
    fig.tight_layout()
    plt.show()
        
    

def plot_FEVD(columns, Parameters, Draw):
    """
    """
    nstep = Parameters.nstep
    fig, axs = plt.subplots(Parameters.nvar,1,  figsize=(8, 2*Parameters.nvar))

    for i in range(0, Parameters.nvar) :
        axs[i].stackplot((range(0, nstep)), Draw.FEVD[i,:,:], alpha=0.7)
        axs[i].set_title(columns[i])

    fig.legend(
        labels=columns,  # 범례 항목
        loc="lower center",  # 하단 중앙에 위치
        bbox_to_anchor=(0.5, -0.1),  # 그림 바깥에 배치
        ncol=3,
        frameon=False  # 범례 테두리 제거
    )

    # 공통 설정

    # 간격 조정
    fig.tight_layout()
    plt.show()



def plot_historical_decomposition(columns, Parameters, Draw):
    """
    """
    fig, ax = plt.subplots(Parameters.nvar, 1, figsize=(8, 3*Parameters.nvar))

    colors = cm.tab10(range(Parameters.nvar))
    for j in range(0, Parameters.nvar) :

        axes = ax[j]
        Total_Positive_values = np.zeros(Parameters.T)
        Total_Negative_values = np.zeros(Parameters.T)

        for i in range(0, Parameters.nvar):
            # Positive values
            Positive_values = np.maximum(Draw.med_HD_shock[:, i, j], 0)
            axes.bar(
                np.arange(Parameters.T), Positive_values, bottom=Total_Positive_values, width=1.0, alpha=0.95, label=columns[i],edgecolor="gray", color=colors[i]
            )
            Total_Positive_values += Positive_values

            # Negative values
            Negative_values = np.minimum(Draw.med_HD_shock[:, i, j], 0)
            axes.bar(
                np.arange(Parameters.T), Negative_values, bottom=Total_Negative_values, width=1.0, alpha=0.95, edgecolor="gray", color=colors[i])
            Total_Negative_values += Negative_values

        axes.axhline(0, color="black", linewidth=0.8)

        # 전체 합계 라인 그래프
        axes.plot(
            np.arange(Parameters.T),
            np.sum(Draw.med_HD_shock[:, :, j], axis=1),
            color="black",
            linewidth=2
        )

        # 그래프 레이블 및 범례 설정
        axes.set_title(columns[j], fontsize=12)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
    
    # Create custom legend handles
    handles = [mpatches.Patch(color=colors[i], label=columns[i]) for i in range(Parameters.nvar)]

    # Add legend to the figure
    fig.legend(handles=handles, loc='lower center', 
                ncol=3,
                bbox_to_anchor=(0.5, -0.05),
               frameon=False  # 범례 테두리 제거
               )
    # 그래프 출력
    plt.tight_layout()
    plt.show()