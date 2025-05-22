//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// Gen_Gamma.cpp
//
// Code generation for function 'Gen_Gamma'
//

// Include files
#include "Gen_Gamma.h"
#include "Gen_Gamma_data.h"
#include "gammaln.h"
#include "rand.h"
#include "rt_nonfinite.h"
#include "coder_array.h"
#include "mwmathutil.h"

// Variable Definitions
static emlrtRSInfo
    emlrtRSI{
        12,          // lineNo
        "Gen_Gamma", // fcnName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Gamma."
        "m" // pathName
    };

static emlrtRSInfo
    b_emlrtRSI{
        13,          // lineNo
        "Gen_Gamma", // fcnName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Gamma."
        "m" // pathName
    };

static emlrtRSInfo
    c_emlrtRSI{
        15,          // lineNo
        "Gen_Gamma", // fcnName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Gamma."
        "m" // pathName
    };

static emlrtRSInfo d_emlrtRSI{
    8,         // lineNo
    "lnpdfig", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\lnpdfig.m" // pathName
};

static emlrtRSInfo e_emlrtRSI{
    9,         // lineNo
    "lnpdfig", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\lnpdfig.m" // pathName
};

static emlrtBCInfo
    emlrtBCI{
        -1,           // iFirst
        -1,           // iLast
        11,           // lineNo
        27,           // colNo
        "diag_Sigma", // aName
        "Gen_Gamma",  // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Gamma."
        "m", // pName
        0    // checkKind
    };

static emlrtBCInfo
    b_emlrtBCI{
        -1,          // iFirst
        -1,          // iLast
        15,          // lineNo
        9,           // colNo
        "gamma",     // aName
        "Gen_Gamma", // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Gamma."
        "m", // pName
        0    // checkKind
    };

static emlrtRTEInfo
    c_emlrtRTEI{
        8,           // lineNo
        9,           // colNo
        "Gen_Gamma", // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Gamma."
        "m" // pName
    };

// Function Definitions
void Gen_Gamma(const emlrtStack *sp, coder::array<real_T, 1U> &diag_Sigma,
               real_T q, real_T a_10, real_T a_00, real_T c_10, real_T c_00,
               coder::array<real_T, 1U> &b_gamma)
{
  emlrtStack b_st;
  emlrtStack st;
  real_T d;
  real_T d1;
  int32_T i;
  int32_T loop_ub;
  st.prev = sp;
  st.tls = sp->tls;
  b_st.prev = &st;
  b_st.tls = st.tls;
  loop_ub = diag_Sigma.size(0);
  for (i = 0; i < loop_ub; i++) {
    diag_Sigma[i] = 1000.0 * diag_Sigma[i];
  }
  c_10 *= 1000.0;
  c_00 *= 1000.0;
  b_gamma.set_size(&c_emlrtRTEI, sp, diag_Sigma.size(0));
  i = diag_Sigma.size(0);
  for (loop_ub = 0; loop_ub < i; loop_ub++) {
    real_T alpha;
    real_T b_alpha;
    real_T b_beta;
    real_T beta;
    real_T density_1_tmp;
    if (loop_ub + 1 > diag_Sigma.size(0)) {
      emlrtDynamicBoundsCheckR2012b(loop_ub + 1, 1, diag_Sigma.size(0),
                                    &emlrtBCI, (emlrtCTX)sp);
    }
    st.site = &emlrtRSI;
    alpha = a_10 / 2.0;
    beta = c_10 / 2.0;
    //  to compute the log inverted gamma density on a grid
    //  beta is also a vector
    //  alpha = shape parameter
    //  beta = scale parameter
    //  mean = beta / (alpha - 1)
    b_st.site = &d_emlrtRSI;
    if (beta < 0.0) {
      emlrtErrorWithMessageIdR2018a(
          &b_st, &emlrtRTEI, "Coder:toolbox:ElFunDomainError",
          "Coder:toolbox:ElFunDomainError", 3, 4, 3, "log");
    }
    d = alpha;
    b_st.site = &d_emlrtRSI;
    coder::gammaln(&b_st, &d);
    b_st.site = &e_emlrtRSI;
    if (diag_Sigma[loop_ub] < 0.0) {
      emlrtErrorWithMessageIdR2018a(
          &b_st, &emlrtRTEI, "Coder:toolbox:ElFunDomainError",
          "Coder:toolbox:ElFunDomainError", 3, 4, 3, "log");
    }
    density_1_tmp = muDoubleScalarLog(diag_Sigma[loop_ub]);
    st.site = &b_emlrtRSI;
    b_alpha = a_00 / 2.0;
    b_beta = c_00 / 2.0;
    //  to compute the log inverted gamma density on a grid
    //  beta is also a vector
    //  alpha = shape parameter
    //  beta = scale parameter
    //  mean = beta / (alpha - 1)
    b_st.site = &d_emlrtRSI;
    if (b_beta < 0.0) {
      emlrtErrorWithMessageIdR2018a(
          &b_st, &emlrtRTEI, "Coder:toolbox:ElFunDomainError",
          "Coder:toolbox:ElFunDomainError", 3, 4, 3, "log");
    }
    d1 = b_alpha;
    b_st.site = &d_emlrtRSI;
    coder::gammaln(&b_st, &d1);
    b_st.site = &e_emlrtRSI;
    if (diag_Sigma[loop_ub] < 0.0) {
      emlrtErrorWithMessageIdR2018a(
          &b_st, &emlrtRTEI, "Coder:toolbox:ElFunDomainError",
          "Coder:toolbox:ElFunDomainError", 3, 4, 3, "log");
    }
    if (loop_ub + 1 > b_gamma.size(0)) {
      emlrtDynamicBoundsCheckR2012b(loop_ub + 1, 1, b_gamma.size(0),
                                    &b_emlrtBCI, (emlrtCTX)sp);
    }
    alpha = muDoubleScalarExp(((alpha * muDoubleScalarLog(beta) - d) -
                               (alpha + 1.0) * density_1_tmp) -
                              beta / diag_Sigma[loop_ub]) *
            q;
    st.site = &c_emlrtRSI;
    b_gamma[loop_ub] =
        (coder::b_rand() <
         alpha / (alpha + muDoubleScalarExp(
                              ((b_alpha * muDoubleScalarLog(b_beta) - d1) -
                               (b_alpha + 1.0) * density_1_tmp) -
                              b_beta / diag_Sigma[loop_ub]) *
                              (1.0 - q)));
    if (*emlrtBreakCheckR2012bFlagVar != 0) {
      emlrtBreakCheckR2012b((emlrtCTX)sp);
    }
  }
}

// End of code generation (Gen_Gamma.cpp)
