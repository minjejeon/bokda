//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// Gen_Omega.cpp
//
// Code generation for function 'Gen_Omega'
//

// Include files
#include "Gen_Omega.h"
#include "Gen_Omega_data.h"
#include "cholmod.h"
#include "invpd.h"
#include "mtimes.h"
#include "randn.h"
#include "rt_nonfinite.h"
#include "blas.h"
#include "coder_array.h"
#include <cstddef>

// Variable Definitions
static emlrtRSInfo
    emlrtRSI{
        11,          // lineNo
        "Gen_Omega", // fcnName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Omega."
        "m" // pathName
    };

static emlrtRSInfo
    b_emlrtRSI{
        15,          // lineNo
        "Gen_Omega", // fcnName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Omega."
        "m" // pathName
    };

static emlrtRSInfo
    c_emlrtRSI{
        16,          // lineNo
        "Gen_Omega", // fcnName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Omega."
        "m" // pathName
    };

static emlrtRSInfo
    d_emlrtRSI{
        17,          // lineNo
        "Gen_Omega", // fcnName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Omega."
        "m" // pathName
    };

static emlrtRSInfo
    e_emlrtRSI{
        19,          // lineNo
        "Gen_Omega", // fcnName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Omega."
        "m" // pathName
    };

static emlrtRSInfo
    tc_emlrtRSI{
        6,             // lineNo
        "randwishart", // fcnName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_"
        "library\\randwishart.m" // pathName
    };

static emlrtRSInfo
    uc_emlrtRSI{
        7,             // lineNo
        "randwishart", // fcnName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_"
        "library\\randwishart.m" // pathName
    };

static emlrtRSInfo
    vc_emlrtRSI{
        8,             // lineNo
        "randwishart", // fcnName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_"
        "library\\randwishart.m" // pathName
    };

static emlrtECInfo
    emlrtECI{
        2,           // nDims
        18,          // lineNo
        18,          // colNo
        "Gen_Omega", // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Omega."
        "m" // pName
    };

static emlrtECInfo
    b_emlrtECI{
        2,           // nDims
        15,          // lineNo
        14,          // colNo
        "Gen_Omega", // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Omega."
        "m" // pName
    };

static emlrtECInfo
    c_emlrtECI{
        2,           // nDims
        12,          // lineNo
        13,          // colNo
        "Gen_Omega", // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Omega."
        "m" // pName
    };

static emlrtECInfo
    d_emlrtECI{
        -1,          // nDims
        11,          // lineNo
        12,          // colNo
        "Gen_Omega", // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Omega."
        "m" // pName
    };

static emlrtBCInfo
    emlrtBCI{
        -1,          // iFirst
        -1,          // iLast
        11,          // lineNo
        14,          // colNo
        "Y",         // aName
        "Gen_Omega", // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Omega."
        "m", // pName
        0    // checkKind
    };

static emlrtBCInfo
    b_emlrtBCI{
        -1,          // iFirst
        -1,          // iLast
        10,          // lineNo
        16,          // colNo
        "X",         // aName
        "Gen_Omega", // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Omega."
        "m", // pName
        0    // checkKind
    };

static emlrtRTEInfo
    l_emlrtRTEI{
        7,           // lineNo
        1,           // colNo
        "Gen_Omega", // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Omega."
        "m" // pName
    };

static emlrtRTEInfo
    m_emlrtRTEI{
        11,          // lineNo
        12,          // colNo
        "Gen_Omega", // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Omega."
        "m" // pName
    };

static emlrtRTEInfo
    n_emlrtRTEI{
        10,          // lineNo
        10,          // colNo
        "Gen_Omega", // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Omega."
        "m" // pName
    };

static emlrtRTEInfo
    o_emlrtRTEI{
        12,          // lineNo
        21,          // colNo
        "Gen_Omega", // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Omega."
        "m" // pName
    };

static emlrtRTEInfo
    p_emlrtRTEI{
        15,          // lineNo
        14,          // colNo
        "Gen_Omega", // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Omega."
        "m" // pName
    };

static emlrtRTEInfo
    r_emlrtRTEI{
        17,          // lineNo
        1,           // colNo
        "Gen_Omega", // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Omega."
        "m" // pName
    };

static emlrtRTEInfo
    s_emlrtRTEI{
        8,             // lineNo
        5,             // colNo
        "randwishart", // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_"
        "library\\randwishart.m" // pName
    };

static emlrtRTEInfo
    t_emlrtRTEI{
        8,             // lineNo
        7,             // colNo
        "randwishart", // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_"
        "library\\randwishart.m" // pName
    };

static emlrtRTEInfo
    u_emlrtRTEI{
        77,                  // lineNo
        13,                  // colNo
        "eml_mtimes_helper", // fName
        "C:\\Program "
        "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\ops\\eml_mtimes_"
        "helper.m" // pName
    };

static emlrtRTEInfo
    v_emlrtRTEI{
        18,          // lineNo
        1,           // colNo
        "Gen_Omega", // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\Gen_Omega."
        "m" // pName
    };

// Function Definitions
void Gen_Omega(const emlrtStack *sp, const coder::array<real_T, 2U> &Y,
               const coder::array<real_T, 3U> &X,
               const coder::array<real_T, 1U> &beta, real_T nu,
               const coder::array<real_T, 2U> &R0,
               coder::array<real_T, 2U> &Omega,
               coder::array<real_T, 2U> &Omega_inv)
{
  ptrdiff_t k_t;
  ptrdiff_t lda_t;
  ptrdiff_t ldb_t;
  ptrdiff_t ldc_t;
  ptrdiff_t m_t;
  ptrdiff_t n_t;
  coder::array<real_T, 2U> b_X;
  coder::array<real_T, 2U> ehat2;
  coder::array<real_T, 2U> r;
  coder::array<real_T, 1U> ehat;
  coder::array<real_T, 1U> r1;
  emlrtStack b_st;
  emlrtStack c_st;
  emlrtStack d_st;
  emlrtStack e_st;
  emlrtStack st;
  real_T alpha1;
  real_T beta1;
  int32_T b_Omega_inv[2];
  int32_T iv[2];
  int32_T b_loop_ub;
  int32_T i;
  int32_T i1;
  int32_T loop_ub;
  char_T TRANSA1;
  char_T TRANSB1;
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
  emlrtHeapReferenceStackEnterFcnR2012b((emlrtCTX)sp);
  //  Omega 샘플링 %%%%%%%%%%%%%%%%%%%%%%%%%%
  //  number of columns
  ehat2.set_size(&l_emlrtRTEI, sp, Y.size(1), Y.size(1));
  loop_ub = Y.size(1) * Y.size(1);
  for (i = 0; i < loop_ub; i++) {
    ehat2[i] = 0.0;
  }
  //  잔차항의 제곱의 합
  i = Y.size(0);
  for (int32_T t{0}; t < i; t++) {
    int32_T i2;
    if (t + 1 > X.size(2)) {
      emlrtDynamicBoundsCheckR2012b(t + 1, 1, X.size(2), &b_emlrtBCI,
                                    (emlrtCTX)sp);
    }
    if (t + 1 > Y.size(0)) {
      emlrtDynamicBoundsCheckR2012b(t + 1, 1, Y.size(0), &emlrtBCI,
                                    (emlrtCTX)sp);
    }
    loop_ub = Y.size(1);
    ehat.set_size(&m_emlrtRTEI, sp, Y.size(1));
    for (i1 = 0; i1 < loop_ub; i1++) {
      ehat[i1] = Y[t + Y.size(0) * i1];
    }
    st.site = &emlrtRSI;
    b_st.site = &g_emlrtRSI;
    loop_ub = X.size(1);
    if (beta.size(0) != X.size(1)) {
      if (((X.size(0) == 1) && (X.size(1) == 1)) || (beta.size(0) == 1)) {
        emlrtErrorWithMessageIdR2018a(
            &b_st, &emlrtRTEI, "Coder:toolbox:mtimes_noDynamicScalarExpansion",
            "Coder:toolbox:mtimes_noDynamicScalarExpansion", 0);
      } else {
        emlrtErrorWithMessageIdR2018a(&b_st, &b_emlrtRTEI, "MATLAB:innerdim",
                                      "MATLAB:innerdim", 0);
      }
    }
    b_loop_ub = X.size(0);
    b_X.set_size(&n_emlrtRTEI, &st, X.size(0), X.size(1));
    for (i1 = 0; i1 < loop_ub; i1++) {
      for (i2 = 0; i2 < b_loop_ub; i2++) {
        b_X[i2 + b_X.size(0) * i1] =
            X[(i2 + X.size(0) * i1) + X.size(0) * X.size(1) * t];
      }
    }
    b_st.site = &f_emlrtRSI;
    coder::internal::blas::mtimes(&b_st, b_X, beta, r1);
    if (ehat.size(0) != r1.size(0)) {
      emlrtSizeEqCheck1DR2012b(ehat.size(0), r1.size(0), &d_emlrtECI,
                               (emlrtCTX)sp);
    }
    loop_ub = ehat.size(0);
    for (i1 = 0; i1 < loop_ub; i1++) {
      ehat[i1] = ehat[i1] - r1[i1];
    }
    //  잔차항
    r.set_size(&o_emlrtRTEI, sp, ehat.size(0), ehat.size(0));
    loop_ub = ehat.size(0);
    for (i1 = 0; i1 < loop_ub; i1++) {
      b_loop_ub = ehat.size(0);
      for (i2 = 0; i2 < b_loop_ub; i2++) {
        r[i2 + r.size(0) * i1] = ehat[i2] * ehat[i1];
      }
    }
    iv[0] = (*(int32_T(*)[2])ehat2.size())[0];
    iv[1] = (*(int32_T(*)[2])ehat2.size())[1];
    b_Omega_inv[0] = (*(int32_T(*)[2])r.size())[0];
    b_Omega_inv[1] = (*(int32_T(*)[2])r.size())[1];
    emlrtSizeEqCheckNDR2012b(&iv[0], &b_Omega_inv[0], &c_emlrtECI,
                             (emlrtCTX)sp);
    loop_ub = ehat2.size(0) * ehat2.size(1);
    for (i1 = 0; i1 < loop_ub; i1++) {
      ehat2[i1] = ehat2[i1] + r[i1];
    }
    //  k by k
    if (*emlrtBreakCheckR2012bFlagVar != 0) {
      emlrtBreakCheckR2012b((emlrtCTX)sp);
    }
  }
  st.site = &b_emlrtRSI;
  invpd(&st, R0, r);
  iv[0] = (*(int32_T(*)[2])ehat2.size())[0];
  iv[1] = (*(int32_T(*)[2])ehat2.size())[1];
  b_Omega_inv[0] = (*(int32_T(*)[2])r.size())[0];
  b_Omega_inv[1] = (*(int32_T(*)[2])r.size())[1];
  emlrtSizeEqCheckNDR2012b(&iv[0], &b_Omega_inv[0], &b_emlrtECI, (emlrtCTX)sp);
  b_X.set_size(&p_emlrtRTEI, sp, ehat2.size(0), ehat2.size(1));
  loop_ub = ehat2.size(0) * ehat2.size(1);
  for (i = 0; i < loop_ub; i++) {
    b_X[i] = ehat2[i] + r[i];
  }
  st.site = &c_emlrtRSI;
  invpd(&st, b_X, ehat2);
  st.site = &d_emlrtRSI;
  //  sampling Wishart dist
  //  E(V) = Omega*nu
  b_st.site = &tc_emlrtRSI;
  cholmod(&b_st, ehat2, r);
  b_st.site = &uc_emlrtRSI;
  c_st.site = &uc_emlrtRSI;
  coder::randn(&c_st, static_cast<real_T>(ehat2.size(0)),
               static_cast<real_T>(Y.size(0)) + nu, b_X);
  c_st.site = &g_emlrtRSI;
  if (r.size(0) != b_X.size(0)) {
    if (((r.size(0) == 1) && (r.size(1) == 1)) ||
        ((b_X.size(0) == 1) && (b_X.size(1) == 1))) {
      emlrtErrorWithMessageIdR2018a(
          &c_st, &emlrtRTEI, "Coder:toolbox:mtimes_noDynamicScalarExpansion",
          "Coder:toolbox:mtimes_noDynamicScalarExpansion", 0);
    } else {
      emlrtErrorWithMessageIdR2018a(&c_st, &b_emlrtRTEI, "MATLAB:innerdim",
                                    "MATLAB:innerdim", 0);
    }
  }
  c_st.site = &f_emlrtRSI;
  if ((r.size(0) == 0) || (r.size(1) == 0) || (b_X.size(0) == 0) ||
      (b_X.size(1) == 0)) {
    Omega_inv.set_size(&r_emlrtRTEI, &c_st, r.size(1), b_X.size(1));
    loop_ub = r.size(1) * b_X.size(1);
    for (i = 0; i < loop_ub; i++) {
      Omega_inv[i] = 0.0;
    }
  } else {
    d_st.site = &h_emlrtRSI;
    e_st.site = &j_emlrtRSI;
    TRANSB1 = 'N';
    TRANSA1 = 'T';
    alpha1 = 1.0;
    beta1 = 0.0;
    m_t = (ptrdiff_t)r.size(1);
    n_t = (ptrdiff_t)b_X.size(1);
    k_t = (ptrdiff_t)r.size(0);
    lda_t = (ptrdiff_t)r.size(0);
    ldb_t = (ptrdiff_t)b_X.size(0);
    ldc_t = (ptrdiff_t)r.size(1);
    Omega_inv.set_size(&q_emlrtRTEI, &e_st, r.size(1), b_X.size(1));
    dgemm(&TRANSA1, &TRANSB1, &m_t, &n_t, &k_t, &alpha1, &(r.data())[0], &lda_t,
          &(b_X.data())[0], &ldb_t, &beta1, &(Omega_inv.data())[0], &ldc_t);
  }
  //  k by nu
  b_st.site = &vc_emlrtRSI;
  c_st.site = &g_emlrtRSI;
  ehat2.set_size(&s_emlrtRTEI, &b_st, Omega_inv.size(0), Omega_inv.size(1));
  loop_ub = Omega_inv.size(0) * Omega_inv.size(1) - 1;
  for (i = 0; i <= loop_ub; i++) {
    ehat2[i] = Omega_inv[i];
  }
  b_X.set_size(&t_emlrtRTEI, &b_st, Omega_inv.size(0), Omega_inv.size(1));
  loop_ub = Omega_inv.size(0) * Omega_inv.size(1) - 1;
  for (i = 0; i <= loop_ub; i++) {
    b_X[i] = Omega_inv[i];
  }
  c_st.site = &f_emlrtRSI;
  coder::internal::blas::mtimes(&c_st, ehat2, b_X, Omega_inv);
  //  k by k
  b_Omega_inv[0] = Omega_inv.size(1);
  b_Omega_inv[1] = Omega_inv.size(0);
  iv[0] = (*(int32_T(*)[2])Omega_inv.size())[0];
  iv[1] = (*(int32_T(*)[2])Omega_inv.size())[1];
  emlrtSizeEqCheckNDR2012b(&iv[0], &b_Omega_inv[0], &emlrtECI, (emlrtCTX)sp);
  r.set_size(&u_emlrtRTEI, sp, Omega_inv.size(0), Omega_inv.size(1));
  loop_ub = Omega_inv.size(1);
  for (i = 0; i < loop_ub; i++) {
    b_loop_ub = Omega_inv.size(0);
    for (i1 = 0; i1 < b_loop_ub; i1++) {
      r[i1 + r.size(0) * i] = 0.5 * (Omega_inv[i1 + Omega_inv.size(0) * i] +
                                     Omega_inv[i + Omega_inv.size(0) * i1]);
    }
  }
  Omega_inv.set_size(&v_emlrtRTEI, sp, r.size(0), r.size(1));
  loop_ub = r.size(0) * r.size(1);
  for (i = 0; i < loop_ub; i++) {
    Omega_inv[i] = r[i];
  }
  st.site = &e_emlrtRSI;
  invpd(&st, Omega_inv, Omega);
  emlrtHeapReferenceStackLeaveFcnR2012b((emlrtCTX)sp);
}

// End of code generation (Gen_Omega.cpp)
