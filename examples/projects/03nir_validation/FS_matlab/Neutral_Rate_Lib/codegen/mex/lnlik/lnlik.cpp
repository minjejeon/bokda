//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// lnlik.cpp
//
// Code generation for function 'lnlik'
//

// Include files
#include "lnlik.h"
#include "invpd.h"
#include "lnlik_NonLinear.h"
#include "lnlik_data.h"
#include "lnpdfmvn1.h"
#include "mtimes.h"
#include "rt_nonfinite.h"
#include "coder_array.h"
#include "mwmathutil.h"

// Variable Definitions
static emlrtRSInfo emlrtRSI{
    5,       // lineNo
    "lnlik", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik.m" // pathName
};

static emlrtRSInfo b_emlrtRSI{
    7,       // lineNo
    "lnlik", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik.m" // pathName
};

static emlrtRSInfo
    rd_emlrtRSI{
        5,              // lineNo
        "lnlik_Linear", // fcnName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
        "Linear.m" // pathName
    };

static emlrtRSInfo
    sd_emlrtRSI{
        13,             // lineNo
        "lnlik_Linear", // fcnName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
        "Linear.m" // pathName
    };

static emlrtRSInfo
    td_emlrtRSI{
        16,             // lineNo
        "lnlik_Linear", // fcnName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
        "Linear.m" // pathName
    };

static emlrtRSInfo
    ud_emlrtRSI{
        19,             // lineNo
        "lnlik_Linear", // fcnName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
        "Linear.m" // pathName
    };

static emlrtECInfo
    emlrtECI{
        2,              // nDims
        17,             // lineNo
        20,             // colNo
        "lnlik_Linear", // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
        "Linear.m" // pName
    };

static emlrtECInfo
    b_emlrtECI{
        2,              // nDims
        15,             // lineNo
        17,             // colNo
        "lnlik_Linear", // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
        "Linear.m" // pName
    };

static emlrtBCInfo emlrtBCI{
    -1,             // iFirst
    -1,             // iLast
    12,             // lineNo
    21,             // colNo
    "YLm",          // aName
    "lnlik_Linear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_Linear."
    "m", // pName
    0    // checkKind
};

static emlrtDCInfo emlrtDCI{
    12,             // lineNo
    21,             // colNo
    "lnlik_Linear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_Linear."
    "m", // pName
    1    // checkKind
};

static emlrtBCInfo b_emlrtBCI{
    -1,             // iFirst
    -1,             // iLast
    11,             // lineNo
    14,             // colNo
    "Y0",           // aName
    "lnlik_Linear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_Linear."
    "m", // pName
    0    // checkKind
};

static emlrtDCInfo b_emlrtDCI{
    11,             // lineNo
    14,             // colNo
    "lnlik_Linear", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_Linear."
    "m", // pName
    1    // checkKind
};

static emlrtRTEInfo
    emlrtRTEI{
        10,             // lineNo
        9,              // colNo
        "lnlik_Linear", // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
        "Linear.m" // pName
    };

static emlrtRTEInfo
    s_emlrtRTEI{
        11,             // lineNo
        5,              // colNo
        "lnlik_Linear", // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
        "Linear.m" // pName
    };

static emlrtRTEInfo
    t_emlrtRTEI{
        12,             // lineNo
        5,              // colNo
        "lnlik_Linear", // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik_"
        "Linear.m" // pName
    };

static emlrtRTEInfo v_emlrtRTEI{
    5,       // lineNo
    27,      // colNo
    "lnlik", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnlik.m" // pName
};

// Function Definitions
real_T lnlik(const emlrtStack *sp, const coder::array<real_T, 2U> &Mu,
             const coder::array<real_T, 2U> &Y0,
             const coder::array<real_T, 3U> &YLm,
             const coder::array<real_T, 1U> &beta,
             const coder::array<real_T, 2U> &Phi,
             const coder::array<real_T, 2U> &Omega,
             const coder::array<real_T, 1U> &diag_Sigma,
             const coder::array<real_T, 1U> &b_gamma, real_T is_Nonlinear)
{
  coder::array<real_T, 2U> r;
  coder::array<real_T, 2U> x_t;
  coder::array<real_T, 1U> y_t;
  coder::array<real_T, 1U> y_tL;
  emlrtStack b_st;
  emlrtStack c_st;
  emlrtStack st;
  real_T lnL;
  int32_T b_Omega[2];
  int32_T iv[2];
  st.prev = sp;
  st.tls = sp->tls;
  b_st.prev = &st;
  b_st.tls = st.tls;
  c_st.prev = &b_st;
  c_st.tls = b_st.tls;
  emlrtHeapReferenceStackEnterFcnR2012b((emlrtCTX)sp);
  //  우도함수 계산하기
  if (is_Nonlinear == 1.0) {
    int32_T loop_ub;
    x_t.set_size(&v_emlrtRTEI, sp, Mu.size(0), Mu.size(1));
    loop_ub = Mu.size(0) * Mu.size(1) - 1;
    for (int32_T i{0}; i <= loop_ub; i++) {
      x_t[i] = Mu[i];
    }
    st.site = &emlrtRSI;
    lnL = lnlik_NonLinear(&st, x_t, Y0, YLm, beta, Phi, Omega, diag_Sigma,
                          b_gamma);
  } else {
    real_T P;
    real_T T;
    int32_T i;
    st.site = &b_emlrtRSI;
    //  우도함수 계산하기
    b_st.site = &rd_emlrtRSI;
    P = static_cast<real_T>(YLm.size(1)) /
        (static_cast<real_T>(YLm.size(0)) * static_cast<real_T>(YLm.size(0)));
    T = static_cast<real_T>(YLm.size(2)) + P;
    lnL = 0.0;
    i = static_cast<int32_T>(T + (1.0 - (P + 1.0)));
    emlrtForLoopVectorCheckR2021a(P + 1.0, 1.0, T, mxDOUBLE_CLASS, i,
                                  &emlrtRTEI, &st);
    for (int32_T t{0}; t < i; t++) {
      real_T d;
      int32_T b_loop_ub;
      int32_T i1;
      int32_T i2;
      int32_T loop_ub;
      T = ((P + 1.0) + static_cast<real_T>(t)) - P;
      d = static_cast<int32_T>(muDoubleScalarFloor(T));
      if (T != d) {
        emlrtIntegerCheckR2012b(T, &b_emlrtDCI, &st);
      }
      if ((static_cast<int32_T>(T) < 1) ||
          (static_cast<int32_T>(T) > Y0.size(0))) {
        emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(T), 1, Y0.size(0),
                                      &b_emlrtBCI, &st);
      }
      loop_ub = Y0.size(1);
      y_t.set_size(&s_emlrtRTEI, &st, Y0.size(1));
      for (i1 = 0; i1 < loop_ub; i1++) {
        y_t[i1] = Y0[(static_cast<int32_T>(T) + Y0.size(0) * i1) - 1];
      }
      if (T != d) {
        emlrtIntegerCheckR2012b(T, &emlrtDCI, &st);
      }
      if ((static_cast<int32_T>(T) < 1) ||
          (static_cast<int32_T>(T) > YLm.size(2))) {
        emlrtDynamicBoundsCheckR2012b(static_cast<int32_T>(T), 1, YLm.size(2),
                                      &emlrtBCI, &st);
      }
      loop_ub = YLm.size(0);
      b_loop_ub = YLm.size(1);
      x_t.set_size(&t_emlrtRTEI, &st, YLm.size(0), YLm.size(1));
      for (i1 = 0; i1 < b_loop_ub; i1++) {
        for (i2 = 0; i2 < loop_ub; i2++) {
          x_t[i2 + x_t.size(0) * i1] =
              YLm[(i2 + YLm.size(0) * i1) +
                  YLm.size(0) * YLm.size(1) * (static_cast<int32_T>(T) - 1)];
        }
      }
      b_st.site = &sd_emlrtRSI;
      c_st.site = &eb_emlrtRSI;
      if (beta.size(0) != YLm.size(1)) {
        if (((YLm.size(0) == 1) && (YLm.size(1) == 1)) || (beta.size(0) == 1)) {
          emlrtErrorWithMessageIdR2018a(
              &c_st, &c_emlrtRTEI,
              "Coder:toolbox:mtimes_noDynamicScalarExpansion",
              "Coder:toolbox:mtimes_noDynamicScalarExpansion", 0);
        } else {
          emlrtErrorWithMessageIdR2018a(&c_st, &b_emlrtRTEI, "MATLAB:innerdim",
                                        "MATLAB:innerdim", 0);
        }
      }
      c_st.site = &db_emlrtRSI;
      coder::internal::blas::mtimes(&c_st, x_t, beta, y_tL);
      b_Omega[0] = Omega.size(1);
      b_Omega[1] = Omega.size(0);
      iv[0] = (*(int32_T(*)[2])((coder::array<real_T, 2U> *)&Omega)->size())[0];
      iv[1] = (*(int32_T(*)[2])((coder::array<real_T, 2U> *)&Omega)->size())[1];
      emlrtSizeEqCheckNDR2012b(&iv[0], &b_Omega[0], &b_emlrtECI, &st);
      loop_ub = Omega.size(1);
      r.set_size(&u_emlrtRTEI, &st, Omega.size(0), Omega.size(1));
      for (i1 = 0; i1 < loop_ub; i1++) {
        b_loop_ub = Omega.size(0);
        for (i2 = 0; i2 < b_loop_ub; i2++) {
          r[i2 + r.size(0) * i1] = 0.5 * (Omega[i2 + Omega.size(0) * i1] +
                                          Omega[i1 + Omega.size(0) * i2]);
        }
      }
      b_st.site = &td_emlrtRSI;
      invpd(&b_st, r, x_t);
      b_Omega[0] = x_t.size(1);
      b_Omega[1] = x_t.size(0);
      iv[0] = (*(int32_T(*)[2])x_t.size())[0];
      iv[1] = (*(int32_T(*)[2])x_t.size())[1];
      emlrtSizeEqCheckNDR2012b(&iv[0], &b_Omega[0], &emlrtECI, &st);
      r.set_size(&u_emlrtRTEI, &st, x_t.size(0), x_t.size(1));
      loop_ub = x_t.size(1);
      for (i1 = 0; i1 < loop_ub; i1++) {
        b_loop_ub = x_t.size(0);
        for (i2 = 0; i2 < b_loop_ub; i2++) {
          r[i2 + r.size(0) * i1] =
              0.5 * (x_t[i2 + x_t.size(0) * i1] + x_t[i1 + x_t.size(0) * i2]);
        }
      }
      b_st.site = &ud_emlrtRSI;
      lnL += lnpdfmvn1(&b_st, y_t, y_tL, r);
      if (*emlrtBreakCheckR2012bFlagVar != 0) {
        emlrtBreakCheckR2012b(&st);
      }
    }
  }
  emlrtHeapReferenceStackLeaveFcnR2012b((emlrtCTX)sp);
  return lnL;
}

// End of code generation (lnlik.cpp)
