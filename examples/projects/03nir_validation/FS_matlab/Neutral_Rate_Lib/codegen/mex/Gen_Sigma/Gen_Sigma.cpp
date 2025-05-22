//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// Gen_Sigma.cpp
//
// Code generation for function 'Gen_Sigma'
//

// Include files
#include "Gen_Sigma.h"
#include "Gen_Sigma_data.h"
#include "rand.h"
#include "randn.h"
#include "rt_nonfinite.h"
#include "warning.h"
#include "blas.h"
#include "coder_array.h"
#include "mwmathutil.h"
#include <cstddef>

// Variable Definitions
static emlrtRSInfo
    emlrtRSI{
        11,          // lineNo
        "Gen_Sigma", // fcnName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Sigma."
        "m" // pathName
    };

static emlrtRSInfo
    b_emlrtRSI{
        14,          // lineNo
        "Gen_Sigma", // fcnName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Sigma."
        "m" // pathName
    };

static emlrtRSInfo
    c_emlrtRSI{
        60,                  // lineNo
        "eml_mtimes_helper", // fcnName
        "C:\\Program "
        "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\ops\\eml_mtimes_"
        "helper.m" // pathName
    };

static emlrtRSInfo d_emlrtRSI{
    12,       // lineNo
    "randig", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\randig.m" // pathName
};

static emlrtRSInfo e_emlrtRSI{
    18,        // lineNo
    "randgam", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\randgam.m" // pathName
};

static emlrtRSInfo f_emlrtRSI{
    11,       // lineNo
    "gamrnd", // fcnName
    "C:\\Program Files\\MATLAB\\R2021a\\toolbox\\stats\\eml\\gamrnd.m" // pathName
};

static emlrtRSInfo g_emlrtRSI{
    1,     // lineNo
    "rnd", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\rnd.p" // pathName
};

static emlrtRSInfo h_emlrtRSI{
    1,        // lineNo
    "gamrnd", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+"
    "internal\\private\\gamrnd.p" // pathName
};

static emlrtRSInfo i_emlrtRSI{
    1,       // lineNo
    "randg", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+"
    "internal\\private\\randg.p" // pathName
};

static emlrtRSInfo j_emlrtRSI{
    44,       // lineNo
    "mpower", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\ops\\mpower.m" // pathName
};

static emlrtRSInfo
    k_emlrtRSI{
        71,      // lineNo
        "power", // fcnName
        "C:\\Program "
        "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\ops\\power.m" // pathName
    };

static emlrtBCInfo
    emlrtBCI{
        -1,          // iFirst
        -1,          // iLast
        10,          // lineNo
        26,          // colNo
        "Mu",        // aName
        "Gen_Sigma", // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Sigma."
        "m", // pName
        0    // checkKind
    };

static emlrtBCInfo
    b_emlrtBCI{
        -1,          // iFirst
        -1,          // iLast
        10,          // lineNo
        36,          // colNo
        "Mu",        // aName
        "Gen_Sigma", // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Sigma."
        "m", // pName
        0    // checkKind
    };

static emlrtECInfo
    emlrtECI{
        -1,          // nDims
        10,          // lineNo
        16,          // colNo
        "Gen_Sigma", // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Sigma."
        "m" // pName
    };

static emlrtBCInfo
    c_emlrtBCI{
        -1,          // iFirst
        -1,          // iLast
        12,          // lineNo
        26,          // colNo
        "gamma",     // aName
        "Gen_Sigma", // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Sigma."
        "m", // pName
        0    // checkKind
    };

static emlrtRTEInfo
    emlrtRTEI{
        83,         // lineNo
        5,          // colNo
        "fltpower", // fName
        "C:\\Program "
        "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\ops\\power.m" // pName
    };

static emlrtRTEInfo
    b_emlrtRTEI{
        14,    // lineNo
        9,     // colNo
        "log", // fName
        "C:\\Program "
        "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\elfun\\log.m" // pName
    };

static emlrtBCInfo
    d_emlrtBCI{
        -1,           // iFirst
        -1,           // iLast
        15,           // lineNo
        9,            // colNo
        "diag_Sigma", // aName
        "Gen_Sigma",  // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Sigma."
        "m", // pName
        0    // checkKind
    };

static emlrtRTEInfo
    c_emlrtRTEI{
        4,           // lineNo
        1,           // colNo
        "Gen_Sigma", // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Sigma."
        "m" // pName
    };

static emlrtRTEInfo
    d_emlrtRTEI{
        6,           // lineNo
        14,          // colNo
        "Gen_Sigma", // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Sigma."
        "m" // pName
    };

static emlrtRTEInfo
    e_emlrtRTEI{
        10,          // lineNo
        9,           // colNo
        "Gen_Sigma", // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Sigma."
        "m" // pName
    };

// Function Definitions
void Gen_Sigma(const emlrtStack *sp, coder::array<real_T, 2U> &Mu,
               const coder::array<real_T, 1U> &b_gamma, real_T a_10,
               real_T a_00, real_T c_10, real_T c_00,
               coder::array<real_T, 1U> &diag_Sigma)
{
  ptrdiff_t incx_t;
  ptrdiff_t incy_t;
  ptrdiff_t n_t;
  coder::array<real_T, 1U> ehat;
  emlrtStack b_st;
  emlrtStack c_st;
  emlrtStack d_st;
  emlrtStack e_st;
  emlrtStack f_st;
  emlrtStack g_st;
  emlrtStack h_st;
  emlrtStack i_st;
  emlrtStack j_st;
  emlrtStack st;
  real_T ur[2];
  int32_T b_Mu;
  int32_T b_loop_ub;
  int32_T i;
  int32_T i1;
  int32_T i2;
  int32_T i3;
  int32_T iter;
  int32_T loop_ub;
  st.prev = sp;
  st.tls = sp->tls;
  b_st.prev = &st;
  b_st.tls = st.tls;
  c_st.prev = &b_st;
  c_st.tls = b_st.tls;
  d_st.prev = &c_st;
  d_st.tls = c_st.tls;
  e_st.prev = &d_st;
  e_st.tls = d_st.tls;
  f_st.prev = &e_st;
  f_st.tls = e_st.tls;
  g_st.prev = &f_st;
  g_st.tls = f_st.tls;
  h_st.prev = &g_st;
  h_st.tls = g_st.tls;
  i_st.prev = &h_st;
  i_st.tls = h_st.tls;
  j_st.prev = &i_st;
  j_st.tls = i_st.tls;
  emlrtHeapReferenceStackEnterFcnR2012b((emlrtCTX)sp);
  if (1 > Mu.size(0)) {
    loop_ub = 0;
  } else {
    loop_ub = Mu.size(0);
  }
  b_Mu = Mu.size(1) - 1;
  for (i = 0; i <= b_Mu; i++) {
    for (iter = 0; iter < loop_ub; iter++) {
      Mu[iter + loop_ub * i] = Mu[iter + Mu.size(0) * i];
    }
  }
  Mu.set_size(&c_emlrtRTEI, sp, loop_ub, b_Mu + 1);
  //  번인 제거
  diag_Sigma.set_size(&d_emlrtRTEI, sp, b_Mu + 1);
  i = b_Mu + 1;
  if (0 <= b_Mu) {
    if (2 > loop_ub) {
      i1 = -1;
      i2 = -1;
      i3 = 0;
    } else {
      i1 = 0;
      i2 = loop_ub - 1;
      if (loop_ub - 1 > loop_ub) {
        emlrtDynamicBoundsCheckR2012b(loop_ub - 1, 1, loop_ub, &b_emlrtBCI,
                                      (emlrtCTX)sp);
      }
      i3 = loop_ub - 1;
    }
    b_loop_ub = i2 - i1;
  }
  for (int32_T m{0}; m < i; m++) {
    real_T b;
    real_T ehat2;
    real_T x;
    if (m + 1 > b_Mu + 1) {
      emlrtDynamicBoundsCheckR2012b(m + 1, 1, b_Mu + 1, &emlrtBCI,
                                    (emlrtCTX)sp);
    }
    iter = i2 - i1;
    if (iter != i3) {
      emlrtSizeEqCheck1DR2012b(iter, i3, &emlrtECI, (emlrtCTX)sp);
    }
    ehat.set_size(&e_emlrtRTEI, sp, i2 - i1);
    for (iter = 0; iter < b_loop_ub; iter++) {
      ehat[iter] =
          Mu[((i1 + iter) + Mu.size(0) * m) + 1] - Mu[iter + Mu.size(0) * m];
    }
    //  잔차항, T-1 by 1
    st.site = &emlrtRSI;
    b_st.site = &c_emlrtRSI;
    if (ehat.size(0) < 1) {
      ehat2 = 0.0;
    } else {
      n_t = (ptrdiff_t)ehat.size(0);
      incx_t = (ptrdiff_t)1;
      incy_t = (ptrdiff_t)1;
      ehat2 =
          ddot(&n_t, &(ehat.data())[0], &incx_t, &(ehat.data())[0], &incy_t);
    }
    //  1 by 1
    if (m + 1 > b_gamma.size(0)) {
      emlrtDynamicBoundsCheckR2012b(m + 1, 1, b_gamma.size(0), &c_emlrtBCI,
                                    (emlrtCTX)sp);
    }
    st.site = &b_emlrtRSI;
    x = ((a_10 * b_gamma[m] + a_00 * (1.0 - b_gamma[m])) +
         static_cast<real_T>(loop_ub)) /
        2.0;
    //  Note that
    //  Suppose that x = randig(alpha,beta,1,1)
    //  E(x) = beta/(alpha-1)
    //  Var(x) = beta^2/[(alpha-2)*(alpha-1)^2]
    b_st.site = &d_emlrtRSI;
    //  Note that
    //  Suppose that x = randgam(alpha,beta,1,1)
    //  E(x) = alpha/beta
    //  Var(x) = alpha/(beta^2)
    //  Notice that in matlab alpha = a and beta = 1/b
    b = 1.0 / (1000.0 *
               ((c_10 * b_gamma[m] + c_00 * (1.0 - b_gamma[m])) + ehat2) / 2.0);
    c_st.site = &e_emlrtRSI;
    d_st.site = &f_emlrtRSI;
    e_st.site = &g_emlrtRSI;
    f_st.site = &h_emlrtRSI;
    g_st.site = &i_emlrtRSI;
    if (x <= 0.0) {
      if (x == 0.0) {
        x = 0.0;
      } else {
        x = rtNaN;
      }
    } else if ((!muDoubleScalarIsInf(x)) && (!muDoubleScalarIsNaN(x))) {
      real_T d;
      real_T p;
      real_T u;
      real_T v;
      if (x >= 1.0) {
        d = x - 0.33333333333333331;
        h_st.site = &i_emlrtRSI;
        u = coder::b_rand();
        p = 1.0;
      } else {
        d = (x + 1.0) - 0.33333333333333331;
        h_st.site = &i_emlrtRSI;
        coder::b_rand(ur);
        u = ur[0];
        if (x < 7.4567656047833286E-20) {
          p = 0.0;
        } else {
          ehat2 = 1.0 / x;
          h_st.site = &i_emlrtRSI;
          i_st.site = &j_emlrtRSI;
          j_st.site = &k_emlrtRSI;
          p = muDoubleScalarPower(ur[1], ehat2);
          if ((ur[1] < 0.0) && (!muDoubleScalarIsNaN(ehat2)) &&
              (muDoubleScalarFloor(ehat2) != ehat2)) {
            emlrtErrorWithMessageIdR2018a(&j_st, &emlrtRTEI,
                                          "Coder:toolbox:power_domainError",
                                          "Coder:toolbox:power_domainError", 0);
          }
        }
      }
      h_st.site = &i_emlrtRSI;
      ehat2 = 1.0 / muDoubleScalarSqrt(9.0 * d);
      iter = 0;
      x = 0.0;
      int32_T exitg1;
      do {
        exitg1 = 0;
        for (v = -1.0; v <= 0.0; v = ehat2 * x + 1.0) {
          h_st.site = &i_emlrtRSI;
          x = coder::randn();
        }
        v *= v * v;
        x *= x;
        if (u < 1.0 - 0.0331 * x * x) {
          exitg1 = 1;
        } else {
          h_st.site = &i_emlrtRSI;
          if (u < 0.0) {
            emlrtErrorWithMessageIdR2018a(
                &h_st, &b_emlrtRTEI, "Coder:toolbox:ElFunDomainError",
                "Coder:toolbox:ElFunDomainError", 3, 4, 3, "log");
          }
          h_st.site = &i_emlrtRSI;
          if (muDoubleScalarLog(u) <
              0.5 * x + d * ((1.0 - v) + muDoubleScalarLog(v))) {
            exitg1 = 1;
          } else {
            iter++;
            if (iter > 1000000) {
              h_st.site = &i_emlrtRSI;
              coder::internal::warning(&h_st);
              exitg1 = 1;
            } else {
              h_st.site = &i_emlrtRSI;
              u = coder::b_rand();
            }
          }
        }
      } while (exitg1 == 0);
      x = d * v * p;
    }
    ehat2 = b * x;
    if (b < 0.0) {
      ehat2 = rtNaN;
    }
    if (m + 1 > diag_Sigma.size(0)) {
      emlrtDynamicBoundsCheckR2012b(m + 1, 1, diag_Sigma.size(0), &d_emlrtBCI,
                                    (emlrtCTX)sp);
    }
    diag_Sigma[m] = 1.0 / ehat2 / 1000.0;
    if (*emlrtBreakCheckR2012bFlagVar != 0) {
      emlrtBreakCheckR2012b((emlrtCTX)sp);
    }
  }
  emlrtHeapReferenceStackLeaveFcnR2012b((emlrtCTX)sp);
}

// End of code generation (Gen_Sigma.cpp)
