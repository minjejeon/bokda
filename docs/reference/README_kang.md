# Functions list
* double stars (bold) function: directly used by user
* double hashes (H2): folder name
* triple hashes (H3): file name

## basic functions
### matrix and data operator
- rows
- cols
- meanc
- demeanc
- stdc
- diag
- sumc
- zeros
- ones
- eye
- det
- inv
- vec
- reshape
- maxc
- minc
- length
- chol
- rev
- eig
- recserar
- MA
- is_array
- is_invertible
- is_nan
- is_None
- is_symmetric
- is_pos_def
- array
- **matrix**
- standdc
- lndet1

### random number generator
- rand
- randn
- randbeta
- randitg
- randig
- randwishart
- randiw
- randb

### density (pdf)
- lnpdfn
- lnpdfn1
- lnpdfmvn
- lnpdfmvn1
- cdfn


### OLS and time-series data
- **plot_HD**
- **logdiff**
- **detrend**
- **autocov**
- **autocor**


## Vector Auto-Regression Model
### VAR
- **order_VAR**
- **OLS_ARp**
- **OLS_VAR**
- B0invSolve
- irf_estimate
- randper
- **SVAR**
- **VAR_GenIRF**
- **VarDecomp**
- **HistDecomp**
- **Long_Var**

### VECM
- **VECM_MLE**
- **recover_VAR**
- **VECM_IRF**

### VAR-Block
- **OLS_VAR_block**
- B0invSolve_block
- **SVAR_block**

### VAR-test
- WN_test
- GCtest
- MIC
- UR_ADF_GLS
- read_table_VECM
- COINT_test


## State Space Model
### cython function
- clnpdfmvn
- ckalman_filter_DFM
- cKalman_Smoother_DFM
- cmake_R0
- clnpdfn
- ckalman_filter_TVP
- cKalman_Smoother_TVP

### optimizer
- make_R0
- Gradient
- FHESS
- DO_CKR2
- simulated_annealing
- SA_Newton
- Kalman_Smoother
- paramconst_SSM
- trans_SSM

### Dynamic Factor Model
- trans_DFM
- paramconst_DFM
- kalman_filter_DFM
- lnlik_DFM
- **SSM_DFM_model**

### Unobserved Component Model
- lnlik_UC
- lnlik_UC_w_drift
- lnlik_UC_wo_drift
- lnlik_UC_w_TV_drift
- Kalman_Smoother_UC
- **SSM_UC_model**
- **SSM_UC_model2**

### Time-Varying Parameter Model
- OLS_TVP
- trans_TVP
- lnlik_TVP
- paramconst_TVP
- Kalman_Smoother_TVP
- **SSM_TVP_model**


## Bayesian Linear Regression
- **Bayes_gen_hyper**
- Bayes_sample_beta
- Bayes_sample_sig2
- Bayes_sample_gam
- Bayes_sample_b_
- **Bayes_result_table**
- Bayes_var
- Bayes_R2
- **Bayes_Gibbs_VS**
- **Bayes_Gibbs_LIN**
- **Bayes_histogram**
- **Bayes_scatter**