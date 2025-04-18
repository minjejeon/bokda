# LBVAR

> **LBVAR 모델 개요**

---

### 개요

- Large Bayesian Vector Autoregression(LBVAR) 모델을 구현합니다. 
- 어떤 사전분포를 사용하는지에 따라 4개의 클래스로 구분됩니다. 사전분포에 대한 자세한 내용은 Chan et al. (2020), Chan (2022), Chan (2021)을 참고하시기 바랍니다. 
- 일부 클래스에서는 하이퍼파라미터 최적화 옵션을 제공하여 모델의 성능을 향상시킬 수 있습니다.
- 모델 적합 후 예측(forecast)을 수행하거나 변수들 간의 구조적 관계를 분석(Structure Analysis)할 수 있는 기능을 제공합니다. 
- 신용구간의 경우 사용자가 선택할 수 있으며, 분위수를 기준으로 합니다. (68\% 밴드 : 0.16, 90\% 밴드 : 0.05, 95\% 밴드 : 0.025, 99\% 밴드 : 0.005)

| **Class**              | **설명**              | **하이퍼파라미터 최적화** | **forecast**  | **Structure Analysis**|
| ---------------------  | ----------------------------------------------------------------------| - | - | - |
| `LBVAR_Symmetry`       | 대칭적 사전 분포(Symmetric Prior)를 사용                                  | O | O | O |
| `LBVAR_Asymmetry`      | 비대칭적 사전 분포(Asymmetric Prior)를 사용                                | O | O | X |
| `LBVAR_Adaptive`       | 유의하지 않은 계수 값들을 0으로 수축시키는 Adaptive Hierarchical Priors를 사용   | X | O | X |
| `LBVAR_Noninformative` | 비정보적 사전분포(Noninformative Prior)를 사용함.                           | X | X | O |


---

### 하이퍼파라미터 최적화

본 패키지에서 Hyperparameter를 결정하는 방법을 크게 세 가지 형태로 제공합니다. 

1. 연구자 본인의 선택
2. Marginal Likelihood를 극대화하는 Hyperparameter 선택
3. Hyperparameter 자체에 대하여 MH 알고리즘을 이용한 선택

#### 하이퍼파라미터 최적화에 대한 가이드

- LBVAR의 경우 추정 전 선택되어지는 Hyperparameter가 모델 추정에 있어서 매우 결정적인 영향을 지니는 것으로 알려져 있습니다. 2번, 3번 방법을 사용하는 경우의 예측력이 1번의 경우보다 높을 수 있다는 것이 알려져있으므로, 예측력을 높이기 위해서는 데이터를 활용하여 hyperparameter를 선택하는 2번, 3번 방법을 추천합니다. (Giannone et al. (2015))


- 하지만 추정해야할 계수의 수가 표본의 수보다 많아지는 경우 역행렬 계산이 이루어지지 않기 때문에 2번, 3번 방법이 작동하지 않을 수 있습니다. 더 나아가 모형의 추정 자체가 이루어지지 않을 수 있습니다. 이러한 경우 모형의 추정을 위해 Hyperparameter를 보다 크게 수축하는 형태로 변형을 주어 계산이 이루어질 수 있도록 조정해 주어야 합니다.

    - Symmetry의 hyperparamter는 패키지 내에서 `np.array([kappa_1, kappa_2, kappa_3, kappa_4, kappa_5])`로 입력됩니다. 

    - Symmetry의 경우 우선 $\kappa_1$을 조금씩 줄여 조정을 가합니다. 만약 $\kappa_1$가 지나칠정도로 작아질 경우 $\kappa_2$에 조정을 가하여 균형을 맞추어 추정을 맞춥니다.

    - Asymmetry의 hyperparameter는 패키지 내에서 `np.array([kappa_1, kappa_2, kappa_3])`로 입력됩니다. 
    
    - Asymmetry의 경우 자기 자신 외의 변수의 수축정도와 관련있는 $\kappa_2$를 조금씩 줄여가며 조정을 가합니다. 일반적으로 예측의 경우 자기자신이 가장 많은 정보가 담겨있는 것으로 알려저 있기 때문입니다. 

- Asymmetry와 Symmetry 모두 가능하다면 변수를 적절히 맞추어 Hyperparameter 최적화 옵션을 사용하기를 권장드립니다.

#### 하이퍼파라미터 최적화 예시

예를 들어, 한국은행 Ecos에서 구한 129개 변수(표본수 96개, 시차 4, 전부 LV_list로 포함)를 사용하는 경우

- Symmetry

    - 기본 Hyperparameter인 `[1, 1, 100, 1, 1]`을 사용할 경우 모형 추정 자체가 이루어지지 않습니다. 
    - `[0.001, 1, 100, 1, 1]`로 매우 크게 $\kappa_1$을 수축시킬 경우 추정이 가능하게 됩니다.
    - 특히 LV_list에 담기는 많을 수록 더 크게 수축시켜야 하는 특징을 지닙니다.

- Asymmetry

    - 기본값인 [0.05, 0.005, 100]을 사용하는 경우에 문제없이 추정됩니다. 
    - 하지만 만약 계산이 작동하지 않는 경우 $\kappa_2$를 조금씩 줄여가며 조정을 가합니다.

---


### Structure VAR Model

- **Structure VAR Model**은 시계열분석에 있어서 특정 변수의 외생적 변화가 자기자신 혹은 여타 변수들에 미치는 영향을 분석하기 위한 방법론입니다.

- 축약형(Reduced Form) 모형 추정에 있어서 대칭적 사전분포(Symmetric Prior)혹은 비정보적 사전분포(Noninformative Prior)를 선택할 수 있습니다. 

- 아래 네가지 방법론을 적용하고, 그에 맞는 그래프를 그릴 수 있는 기능을 제공합니다. 

- Structure Model의 변환에는 Recursive Ordering 방법을 적용하였습니다.


| **방법론**              | **설명**              |
| ---------------------- | ------------------- |
| 충격데이터(Shock Series)  | 추정된 잔차(residuals)를 통해 도출된 값으로, 모델이 설명하지 못하는 부분을 확인|
| 충격반응함수(Impulse Response Function)                | 특정 변수의 외생적 변화가 각종 변수에 미치는 동태적 움직임을 분석      | 
| 예측오차 분산분해(Forecast error variance Decomposition)| 변수의 변화에 어떤 충격들이 주요한 영향을 미쳤는지를 확인            | 
| 역사 분해(Historical Decomposition)                    | 변수의 특정 시기에 어떤 충격들이 어떤 영향들을 미치는지를 확인       | 

---

### Reference

Bańbura, Marta and Giannone, Domenico and Reichlin, Lucrezia. 2010. "Large Bayesian vector auto regressions." Journal of Applied Econometrics

Chan, Joshua C. C. 2021. "Minnesota-type adaptive hierarchical priors for large Bayesian VARs." International Journal of Forecasting

Chan, Joshua C.C. 2022. "Asymmetric conjugate priors for large Bayesian VARs." Quantitative Economics

Chan, Joshua C. C. and Jacobi, Liana and Zhu, Dan. 2020. "Efficient selection of hyperparameters in large Bayesian VARs using automatic differentiation." Journal of Forecasting

Giannone, Domenico and Lenza, Michele and Primiceri, Giorgio E. 2015. "Prior Selection for Vector Autoregressions" The Review of Economics and Statistics

Robert B. Litterman. 1986. "Forecasting with Bayesian Vector Autoregression-Five Years of Experience." Journal of Business & Economic Statistics

Thomas Doan, Robert Litterman and Christopher Sims. 1984. "Forecasting and conditional projection using realistic prior distributions." Econometric Reviews
