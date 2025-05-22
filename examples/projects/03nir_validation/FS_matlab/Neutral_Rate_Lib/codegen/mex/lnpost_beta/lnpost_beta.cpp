//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// lnpost_beta.cpp
//
// Code generation for function 'lnpost_beta'
//

// Include files
#include "lnpost_beta.h"
#include "cholmod.h"
#include "diag.h"
#include "eml_int_forloop_overflow_check.h"
#include "eml_mtimes_helper.h"
#include "invpd.h"
#include "lnpost_beta_data.h"
#include "mtimes.h"
#include "rt_nonfinite.h"
#include "sum.h"
#include "blas.h"
#include "coder_array.h"
#include "mwmathutil.h"
#include <cstddef>

// Variable Definitions
static emlrtRSInfo
    emlrtRSI{
        7,             // lineNo
        "lnpost_beta", // fcnName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnpost_"
        "beta.m" // pathName
    };

static emlrtRSInfo
    b_emlrtRSI{
        8,             // lineNo
        "lnpost_beta", // fcnName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnpost_"
        "beta.m" // pathName
    };

static emlrtRSInfo
    c_emlrtRSI{
        12,            // lineNo
        "lnpost_beta", // fcnName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnpost_"
        "beta.m" // pathName
    };

static emlrtRSInfo
    d_emlrtRSI{
        13,            // lineNo
        "lnpost_beta", // fcnName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnpost_"
        "beta.m" // pathName
    };

static emlrtRSInfo
    e_emlrtRSI{
        16,            // lineNo
        "lnpost_beta", // fcnName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnpost_"
        "beta.m" // pathName
    };

static emlrtRSInfo
    f_emlrtRSI{
        19,            // lineNo
        "lnpost_beta", // fcnName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnpost_"
        "beta.m" // pathName
    };

static emlrtRSInfo
    g_emlrtRSI{
        21,            // lineNo
        "lnpost_beta", // fcnName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnpost_"
        "beta.m" // pathName
    };

static emlrtRSInfo
    h_emlrtRSI{
        22,            // lineNo
        "lnpost_beta", // fcnName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnpost_"
        "beta.m" // pathName
    };

static emlrtRSInfo
    i_emlrtRSI{
        27,            // lineNo
        "lnpost_beta", // fcnName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnpost_"
        "beta.m" // pathName
    };

static emlrtRSInfo
    xc_emlrtRSI{
        3,           // lineNo
        "lnpdfmvn1", // fcnName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\lnpdfmvn1."
        "m" // pathName
    };

static emlrtRSInfo
    yc_emlrtRSI{
        4,           // lineNo
        "lnpdfmvn1", // fcnName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\lnpdfmvn1."
        "m" // pathName
    };

static emlrtRSInfo
    ad_emlrtRSI{
        5,           // lineNo
        "lnpdfmvn1", // fcnName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\lnpdfmvn1."
        "m" // pathName
    };

static emlrtRSInfo bd_emlrtRSI{
    2,        // lineNo
    "lndet1", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\lndet1.m" // pathName
};

static emlrtRSInfo
    cd_emlrtRSI{
        17,    // lineNo
        "log", // fcnName
        "C:\\Program "
        "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\elfun\\log.m" // pathName
    };

static emlrtRSInfo dd_emlrtRSI{
    33,                           // lineNo
    "applyScalarFunctionInPlace", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+"
    "internal\\applyScalarFunctionInPlace.m" // pathName
};

static emlrtRSInfo ed_emlrtRSI{
    4,      // lineNo
    "sumc", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\sumc.m" // pathName
};

static emlrtRTEInfo
    emlrtRTEI{
        14,    // lineNo
        9,     // colNo
        "log", // fName
        "C:\\Program "
        "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\elfun\\log.m" // pName
    };

static emlrtECInfo emlrtECI{
    -1,        // nDims
    6,         // lineNo
    12,        // colNo
    "lnpdfn1", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\lnpdfn1.m" // pName
};

static emlrtECInfo
    b_emlrtECI{
        -1,          // nDims
        4,           // lineNo
        8,           // colNo
        "lnpdfmvn1", // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\lnpdfmvn1."
        "m" // pName
    };

static emlrtECInfo
    c_emlrtECI{
        -1,            // nDims
        21,            // lineNo
        5,             // colNo
        "lnpost_beta", // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnpost_"
        "beta.m" // pName
    };

static emlrtECInfo
    d_emlrtECI{
        2,             // nDims
        20,            // lineNo
        11,            // colNo
        "lnpost_beta", // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnpost_"
        "beta.m" // pName
    };

static emlrtECInfo
    e_emlrtECI{
        2,             // nDims
        18,            // lineNo
        15,            // colNo
        "lnpost_beta", // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnpost_"
        "beta.m" // pName
    };

static emlrtECInfo
    f_emlrtECI{
        2,             // nDims
        17,            // lineNo
        10,            // colNo
        "lnpost_beta", // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnpost_"
        "beta.m" // pName
    };

static emlrtECInfo
    g_emlrtECI{
        -1,            // nDims
        13,            // lineNo
        10,            // colNo
        "lnpost_beta", // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnpost_"
        "beta.m" // pName
    };

static emlrtBCInfo
    emlrtBCI{
        -1,            // iFirst
        -1,            // iLast
        13,            // lineNo
        32,            // colNo
        "Y0",          // aName
        "lnpost_beta", // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnpost_"
        "beta.m", // pName
        0         // checkKind
    };

static emlrtECInfo
    h_emlrtECI{
        2,             // nDims
        12,            // lineNo
        10,            // colNo
        "lnpost_beta", // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnpost_"
        "beta.m" // pName
    };

static emlrtBCInfo
    b_emlrtBCI{
        -1,            // iFirst
        -1,            // iLast
        11,            // lineNo
        16,            // colNo
        "X",           // aName
        "lnpost_beta", // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnpost_"
        "beta.m", // pName
        0         // checkKind
    };

static emlrtDCInfo
    emlrtDCI{
        7,             // lineNo
        12,            // colNo
        "lnpost_beta", // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnpost_"
        "beta.m", // pName
        1         // checkKind
    };

static emlrtDCInfo
    b_emlrtDCI{
        7,             // lineNo
        12,            // colNo
        "lnpost_beta", // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnpost_"
        "beta.m", // pName
        4         // checkKind
    };

static emlrtDCInfo
    c_emlrtDCI{
        8,             // lineNo
        1,             // colNo
        "lnpost_beta", // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnpost_"
        "beta.m", // pName
        1         // checkKind
    };

static emlrtDCInfo
    d_emlrtDCI{
        8,             // lineNo
        1,             // colNo
        "lnpost_beta", // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnpost_"
        "beta.m", // pName
        4         // checkKind
    };

static emlrtBCInfo
    c_emlrtBCI{
        -1,            // iFirst
        -1,            // iLast
        24,            // lineNo
        6,             // colNo
        "BA",          // aName
        "lnpost_beta", // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnpost_"
        "beta.m", // pName
        0         // checkKind
    };

static emlrtBCInfo
    d_emlrtBCI{
        -1,            // iFirst
        -1,            // iLast
        25,            // lineNo
        17,            // colNo
        "B1_inv",      // aName
        "lnpost_beta", // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnpost_"
        "beta.m", // pName
        0         // checkKind
    };

static emlrtBCInfo
    e_emlrtBCI{
        -1,            // iFirst
        -1,            // iLast
        25,            // lineNo
        40,            // colNo
        "B1_inv",      // aName
        "lnpost_beta", // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnpost_"
        "beta.m", // pName
        0         // checkKind
    };

static emlrtBCInfo
    f_emlrtBCI{
        -1,            // iFirst
        -1,            // iLast
        27,            // lineNo
        32,            // colNo
        "beta_st",     // aName
        "lnpost_beta", // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnpost_"
        "beta.m", // pName
        0         // checkKind
    };

static emlrtRTEInfo
    m_emlrtRTEI{
        7,             // lineNo
        1,             // colNo
        "lnpost_beta", // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnpost_"
        "beta.m" // pName
    };

static emlrtRTEInfo
    n_emlrtRTEI{
        8,             // lineNo
        1,             // colNo
        "lnpost_beta", // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnpost_"
        "beta.m" // pName
    };

static emlrtRTEInfo
    o_emlrtRTEI{
        11,            // lineNo
        5,             // colNo
        "lnpost_beta", // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnpost_"
        "beta.m" // pName
    };

static emlrtRTEInfo
    p_emlrtRTEI{
        11,            // lineNo
        10,            // colNo
        "lnpost_beta", // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnpost_"
        "beta.m" // pName
    };

static emlrtRTEInfo
    r_emlrtRTEI{
        12,            // lineNo
        15,            // colNo
        "lnpost_beta", // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnpost_"
        "beta.m" // pName
    };

static emlrtRTEInfo
    s_emlrtRTEI{
        13,            // lineNo
        29,            // colNo
        "lnpost_beta", // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnpost_"
        "beta.m" // pName
    };

static emlrtRTEInfo
    t_emlrtRTEI{
        13,            // lineNo
        15,            // colNo
        "lnpost_beta", // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnpost_"
        "beta.m" // pName
    };

static emlrtRTEInfo
    u_emlrtRTEI{
        17,            // lineNo
        1,             // colNo
        "lnpost_beta", // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnpost_"
        "beta.m" // pName
    };

static emlrtRTEInfo
    v_emlrtRTEI{
        77,                  // lineNo
        13,                  // colNo
        "eml_mtimes_helper", // fName
        "C:\\Program "
        "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\ops\\eml_mtimes_"
        "helper.m" // pName
    };

static emlrtRTEInfo
    w_emlrtRTEI{
        18,            // lineNo
        1,             // colNo
        "lnpost_beta", // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnpost_"
        "beta.m" // pName
    };

static emlrtRTEInfo
    x_emlrtRTEI{
        20,            // lineNo
        1,             // colNo
        "lnpost_beta", // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnpost_"
        "beta.m" // pName
    };

static emlrtRTEInfo
    y_emlrtRTEI{
        1,             // lineNo
        23,            // colNo
        "lnpost_beta", // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnpost_"
        "beta.m" // pName
    };

static emlrtRTEInfo
    ab_emlrtRTEI{
        25,            // lineNo
        10,            // colNo
        "lnpost_beta", // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnpost_"
        "beta.m" // pName
    };

static emlrtRTEInfo
    bb_emlrtRTEI{
        25,            // lineNo
        1,             // colNo
        "lnpost_beta", // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\Model_Lib\\lnpost_"
        "beta.m" // pName
    };

static emlrtRTEInfo
    cb_emlrtRTEI{
        4,           // lineNo
        8,           // colNo
        "lnpdfmvn1", // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\lnpdfmvn1."
        "m" // pName
    };

static emlrtRTEInfo db_emlrtRTEI{
    6,         // lineNo
    12,        // colNo
    "lnpdfn1", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\lnpdfn1.m" // pName
};

// Function Definitions
real_T lnpost_beta(const emlrtStack *sp, const coder::array<real_T, 2U> &Y0,
                   const coder::array<real_T, 3U> &YLm,
                   const coder::array<real_T, 1U> &beta_st, real_T p,
                   const coder::array<real_T, 1U> &b_,
                   const coder::array<real_T, 2U> &var_,
                   const coder::array<real_T, 2U> &Omega_inv,
                   const coder::array<boolean_T, 1U> &NonZeroRestriction)
{
  ptrdiff_t k_t;
  ptrdiff_t lda_t;
  ptrdiff_t ldb_t;
  ptrdiff_t ldc_t;
  ptrdiff_t m_t;
  ptrdiff_t n_t;
  coder::array<real_T, 2U> B1;
  coder::array<real_T, 2U> XX;
  coder::array<real_T, 2U> Xt;
  coder::array<real_T, 2U> b_YLm;
  coder::array<real_T, 2U> r;
  coder::array<real_T, 1U> BA;
  coder::array<real_T, 1U> XY;
  coder::array<real_T, 1U> e;
  coder::array<int32_T, 1U> r1;
  coder::array<int32_T, 1U> r2;
  emlrtStack b_st;
  emlrtStack c_st;
  emlrtStack d_st;
  emlrtStack e_st;
  emlrtStack f_st;
  emlrtStack g_st;
  emlrtStack st;
  real_T alpha1;
  real_T beta1;
  real_T lnpdf_beta;
  int32_T b_XX[2];
  int32_T iv[2];
  int32_T b_i;
  int32_T i;
  int32_T i1;
  int32_T loop_ub;
  int32_T nx;
  int32_T t;
  char_T TRANSA1;
  char_T TRANSB1;
  boolean_T b_p;
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
  emlrtHeapReferenceStackEnterFcnR2012b((emlrtCTX)sp);
  //  = T-p
  //  number of columns
  //  설명변수, 3차원
  st.site = &emlrtRSI;
  b_st.site = &j_emlrtRSI;
  c_st.site = &k_emlrtRSI;
  alpha1 =
      p * (static_cast<real_T>(Y0.size(1)) * static_cast<real_T>(Y0.size(1)));
  beta1 =
      p * (static_cast<real_T>(Y0.size(1)) * static_cast<real_T>(Y0.size(1)));
  if (!(beta1 >= 0.0)) {
    emlrtNonNegativeCheckR2012b(beta1, &b_emlrtDCI, (emlrtCTX)sp);
  }
  if (beta1 != static_cast<int32_T>(muDoubleScalarFloor(beta1))) {
    emlrtIntegerCheckR2012b(beta1, &emlrtDCI, (emlrtCTX)sp);
  }
  XX.set_size(&m_emlrtRTEI, sp, static_cast<int32_T>(alpha1),
              static_cast<int32_T>(alpha1));
  nx = static_cast<int32_T>(alpha1) * static_cast<int32_T>(alpha1);
  for (i = 0; i < nx; i++) {
    XX[i] = 0.0;
  }
  st.site = &b_emlrtRSI;
  b_st.site = &j_emlrtRSI;
  c_st.site = &k_emlrtRSI;
  alpha1 =
      p * (static_cast<real_T>(Y0.size(1)) * static_cast<real_T>(Y0.size(1)));
  if (!(alpha1 >= 0.0)) {
    emlrtNonNegativeCheckR2012b(alpha1, &d_emlrtDCI, (emlrtCTX)sp);
  }
  beta1 = static_cast<int32_T>(muDoubleScalarFloor(alpha1));
  if (alpha1 != beta1) {
    emlrtIntegerCheckR2012b(alpha1, &c_emlrtDCI, (emlrtCTX)sp);
  }
  XY.set_size(&n_emlrtRTEI, sp, static_cast<int32_T>(alpha1));
  if (alpha1 != beta1) {
    emlrtIntegerCheckR2012b(alpha1, &c_emlrtDCI, (emlrtCTX)sp);
  }
  nx = static_cast<int32_T>(alpha1);
  for (i = 0; i < nx; i++) {
    XY[i] = 0.0;
  }
  i = Y0.size(0);
  for (t = 0; t < i; t++) {
    if (t + 1 > YLm.size(2)) {
      emlrtDynamicBoundsCheckR2012b(t + 1, 1, YLm.size(2), &b_emlrtBCI,
                                    (emlrtCTX)sp);
    }
    nx = YLm.size(0);
    loop_ub = YLm.size(1);
    Xt.set_size(&o_emlrtRTEI, sp, YLm.size(0), YLm.size(1));
    for (i1 = 0; i1 < loop_ub; i1++) {
      for (b_i = 0; b_i < nx; b_i++) {
        Xt[b_i + Xt.size(0) * i1] =
            YLm[(b_i + YLm.size(0) * i1) + YLm.size(0) * YLm.size(1) * t];
      }
    }
    st.site = &c_emlrtRSI;
    nx = YLm.size(0);
    loop_ub = YLm.size(1);
    b_YLm.set_size(&p_emlrtRTEI, &st, YLm.size(0), YLm.size(1));
    for (i1 = 0; i1 < loop_ub; i1++) {
      for (b_i = 0; b_i < nx; b_i++) {
        b_YLm[b_i + b_YLm.size(0) * i1] =
            YLm[(b_i + YLm.size(0) * i1) + YLm.size(0) * YLm.size(1) * t];
      }
    }
    b_st.site = &m_emlrtRSI;
    coder::dynamic_size_checks(&b_st, b_YLm, Omega_inv, YLm.size(0),
                               Omega_inv.size(0));
    nx = YLm.size(0);
    loop_ub = YLm.size(1);
    b_YLm.set_size(&p_emlrtRTEI, &st, YLm.size(0), YLm.size(1));
    for (i1 = 0; i1 < loop_ub; i1++) {
      for (b_i = 0; b_i < nx; b_i++) {
        b_YLm[b_i + b_YLm.size(0) * i1] =
            YLm[(b_i + YLm.size(0) * i1) + YLm.size(0) * YLm.size(1) * t];
      }
    }
    b_st.site = &l_emlrtRSI;
    coder::internal::blas::mtimes(&b_st, b_YLm, Omega_inv, B1);
    st.site = &c_emlrtRSI;
    nx = YLm.size(0);
    loop_ub = YLm.size(1);
    b_YLm.set_size(&p_emlrtRTEI, &st, YLm.size(0), YLm.size(1));
    for (i1 = 0; i1 < loop_ub; i1++) {
      for (b_i = 0; b_i < nx; b_i++) {
        b_YLm[b_i + b_YLm.size(0) * i1] =
            YLm[(b_i + YLm.size(0) * i1) + YLm.size(0) * YLm.size(1) * t];
      }
    }
    b_st.site = &m_emlrtRSI;
    coder::dynamic_size_checks(&b_st, B1, b_YLm, B1.size(1), YLm.size(0));
    b_st.site = &l_emlrtRSI;
    if ((B1.size(0) == 0) || (B1.size(1) == 0) || (YLm.size(0) == 0) ||
        (YLm.size(1) == 0)) {
      b_YLm.set_size(&r_emlrtRTEI, &b_st, B1.size(0), YLm.size(1));
      nx = B1.size(0) * YLm.size(1);
      for (i1 = 0; i1 < nx; i1++) {
        b_YLm[i1] = 0.0;
      }
    } else {
      c_st.site = &n_emlrtRSI;
      d_st.site = &p_emlrtRSI;
      TRANSB1 = 'N';
      TRANSA1 = 'N';
      alpha1 = 1.0;
      beta1 = 0.0;
      m_t = (ptrdiff_t)B1.size(0);
      n_t = (ptrdiff_t)YLm.size(1);
      k_t = (ptrdiff_t)B1.size(1);
      lda_t = (ptrdiff_t)B1.size(0);
      ldb_t = (ptrdiff_t)YLm.size(0);
      ldc_t = (ptrdiff_t)B1.size(0);
      b_YLm.set_size(&q_emlrtRTEI, &d_st, B1.size(0), YLm.size(1));
      dgemm(&TRANSA1, &TRANSB1, &m_t, &n_t, &k_t, &alpha1, &(B1.data())[0],
            &lda_t, &(Xt.data())[0], &ldb_t, &beta1, &(b_YLm.data())[0],
            &ldc_t);
    }
    iv[0] = (*(int32_T(*)[2])XX.size())[0];
    iv[1] = (*(int32_T(*)[2])XX.size())[1];
    b_XX[0] = (*(int32_T(*)[2])b_YLm.size())[0];
    b_XX[1] = (*(int32_T(*)[2])b_YLm.size())[1];
    emlrtSizeEqCheckNDR2012b(&iv[0], &b_XX[0], &h_emlrtECI, (emlrtCTX)sp);
    nx = XX.size(0) * XX.size(1);
    for (i1 = 0; i1 < nx; i1++) {
      XX[i1] = XX[i1] + b_YLm[i1];
    }
    if (t + 1 > Y0.size(0)) {
      emlrtDynamicBoundsCheckR2012b(t + 1, 1, Y0.size(0), &emlrtBCI,
                                    (emlrtCTX)sp);
    }
    nx = Y0.size(1);
    r.set_size(&s_emlrtRTEI, sp, 1, Y0.size(1));
    for (i1 = 0; i1 < nx; i1++) {
      r[i1] = Y0[t + Y0.size(0) * i1];
    }
    st.site = &d_emlrtRSI;
    nx = YLm.size(0);
    loop_ub = YLm.size(1);
    b_YLm.set_size(&p_emlrtRTEI, &st, YLm.size(0), YLm.size(1));
    for (i1 = 0; i1 < loop_ub; i1++) {
      for (b_i = 0; b_i < nx; b_i++) {
        b_YLm[b_i + b_YLm.size(0) * i1] =
            YLm[(b_i + YLm.size(0) * i1) + YLm.size(0) * YLm.size(1) * t];
      }
    }
    b_st.site = &m_emlrtRSI;
    coder::dynamic_size_checks(&b_st, b_YLm, Omega_inv, YLm.size(0),
                               Omega_inv.size(0));
    nx = YLm.size(0);
    loop_ub = YLm.size(1);
    b_YLm.set_size(&p_emlrtRTEI, &st, YLm.size(0), YLm.size(1));
    for (i1 = 0; i1 < loop_ub; i1++) {
      for (b_i = 0; b_i < nx; b_i++) {
        b_YLm[b_i + b_YLm.size(0) * i1] =
            YLm[(b_i + YLm.size(0) * i1) + YLm.size(0) * YLm.size(1) * t];
      }
    }
    b_st.site = &l_emlrtRSI;
    coder::internal::blas::mtimes(&b_st, b_YLm, Omega_inv, B1);
    st.site = &d_emlrtRSI;
    b_st.site = &m_emlrtRSI;
    if (Y0.size(1) != B1.size(1)) {
      if (((B1.size(0) == 1) && (B1.size(1) == 1)) || (Y0.size(1) == 1)) {
        emlrtErrorWithMessageIdR2018a(
            &b_st, &b_emlrtRTEI,
            "Coder:toolbox:mtimes_noDynamicScalarExpansion",
            "Coder:toolbox:mtimes_noDynamicScalarExpansion", 0);
      } else {
        emlrtErrorWithMessageIdR2018a(&b_st, &c_emlrtRTEI, "MATLAB:innerdim",
                                      "MATLAB:innerdim", 0);
      }
    }
    b_st.site = &l_emlrtRSI;
    if ((B1.size(0) == 0) || (B1.size(1) == 0) || (Y0.size(1) == 0)) {
      BA.set_size(&t_emlrtRTEI, &b_st, B1.size(0));
      nx = B1.size(0);
      for (i1 = 0; i1 < nx; i1++) {
        BA[i1] = 0.0;
      }
    } else {
      c_st.site = &n_emlrtRSI;
      d_st.site = &p_emlrtRSI;
      TRANSB1 = 'T';
      TRANSA1 = 'N';
      alpha1 = 1.0;
      beta1 = 0.0;
      m_t = (ptrdiff_t)B1.size(0);
      n_t = (ptrdiff_t)1;
      k_t = (ptrdiff_t)B1.size(1);
      lda_t = (ptrdiff_t)B1.size(0);
      ldb_t = (ptrdiff_t)1;
      ldc_t = (ptrdiff_t)B1.size(0);
      BA.set_size(&q_emlrtRTEI, &d_st, B1.size(0));
      dgemm(&TRANSA1, &TRANSB1, &m_t, &n_t, &k_t, &alpha1, &(B1.data())[0],
            &lda_t, &r[0], &ldb_t, &beta1, &(BA.data())[0], &ldc_t);
    }
    nx = XY.size(0);
    if (XY.size(0) != BA.size(0)) {
      emlrtSizeEqCheck1DR2012b(XY.size(0), BA.size(0), &g_emlrtECI,
                               (emlrtCTX)sp);
    }
    for (i1 = 0; i1 < nx; i1++) {
      XY[i1] = XY[i1] + BA[i1];
    }
    if (*emlrtBreakCheckR2012bFlagVar != 0) {
      emlrtBreakCheckR2012b((emlrtCTX)sp);
    }
  }
  st.site = &e_emlrtRSI;
  invpd(&st, var_, Xt);
  iv[0] = (*(int32_T(*)[2])Xt.size())[0];
  iv[1] = (*(int32_T(*)[2])Xt.size())[1];
  b_XX[0] = (*(int32_T(*)[2])XX.size())[0];
  b_XX[1] = (*(int32_T(*)[2])XX.size())[1];
  emlrtSizeEqCheckNDR2012b(&iv[0], &b_XX[0], &f_emlrtECI, (emlrtCTX)sp);
  nx = Xt.size(0) * Xt.size(1);
  XX.set_size(&u_emlrtRTEI, sp, Xt.size(0), Xt.size(1));
  for (i = 0; i < nx; i++) {
    XX[i] = Xt[i] + XX[i];
  }
  b_XX[0] = XX.size(1);
  b_XX[1] = XX.size(0);
  iv[0] = (*(int32_T(*)[2])XX.size())[0];
  iv[1] = (*(int32_T(*)[2])XX.size())[1];
  emlrtSizeEqCheckNDR2012b(&iv[0], &b_XX[0], &e_emlrtECI, (emlrtCTX)sp);
  b_YLm.set_size(&v_emlrtRTEI, sp, XX.size(0), XX.size(1));
  nx = XX.size(1);
  for (i = 0; i < nx; i++) {
    loop_ub = XX.size(0);
    for (i1 = 0; i1 < loop_ub; i1++) {
      b_YLm[i1 + b_YLm.size(0) * i] =
          0.5 * (XX[i1 + XX.size(0) * i] + XX[i + XX.size(0) * i1]);
    }
  }
  XX.set_size(&w_emlrtRTEI, sp, b_YLm.size(0), b_YLm.size(1));
  nx = b_YLm.size(0) * b_YLm.size(1);
  for (i = 0; i < nx; i++) {
    XX[i] = b_YLm[i];
  }
  st.site = &f_emlrtRSI;
  invpd(&st, XX, B1);
  b_XX[0] = B1.size(1);
  b_XX[1] = B1.size(0);
  iv[0] = (*(int32_T(*)[2])B1.size())[0];
  iv[1] = (*(int32_T(*)[2])B1.size())[1];
  emlrtSizeEqCheckNDR2012b(&iv[0], &b_XX[0], &d_emlrtECI, (emlrtCTX)sp);
  b_YLm.set_size(&v_emlrtRTEI, sp, B1.size(0), B1.size(1));
  nx = B1.size(1);
  for (i = 0; i < nx; i++) {
    loop_ub = B1.size(0);
    for (i1 = 0; i1 < loop_ub; i1++) {
      b_YLm[i1 + b_YLm.size(0) * i] =
          0.5 * (B1[i1 + B1.size(0) * i] + B1[i + B1.size(0) * i1]);
    }
  }
  B1.set_size(&x_emlrtRTEI, sp, b_YLm.size(0), b_YLm.size(1));
  nx = b_YLm.size(0) * b_YLm.size(1);
  for (i = 0; i < nx; i++) {
    B1[i] = b_YLm[i];
  }
  st.site = &g_emlrtRSI;
  b_st.site = &m_emlrtRSI;
  coder::b_dynamic_size_checks(&b_st, Xt, b_, Xt.size(1), b_.size(0));
  b_st.site = &l_emlrtRSI;
  coder::internal::blas::mtimes(&b_st, Xt, b_, BA);
  if (XY.size(0) != BA.size(0)) {
    emlrtSizeEqCheck1DR2012b(XY.size(0), BA.size(0), &c_emlrtECI, (emlrtCTX)sp);
  }
  nx = XY.size(0);
  for (i = 0; i < nx; i++) {
    XY[i] = XY[i] + BA[i];
  }
  //  b_ = B0
  st.site = &h_emlrtRSI;
  b_st.site = &m_emlrtRSI;
  coder::b_dynamic_size_checks(&b_st, B1, XY, B1.size(1), XY.size(0));
  b_st.site = &l_emlrtRSI;
  coder::internal::blas::mtimes(&b_st, B1, XY, BA);
  //  full conditional mean
  loop_ub = NonZeroRestriction.size(0) - 1;
  t = 0;
  for (b_i = 0; b_i <= loop_ub; b_i++) {
    if (NonZeroRestriction[b_i]) {
      t++;
    }
  }
  nx = 0;
  for (b_i = 0; b_i <= loop_ub; b_i++) {
    if (NonZeroRestriction[b_i]) {
      if (b_i + 1 > BA.size(0)) {
        emlrtDynamicBoundsCheckR2012b(b_i + 1, 1, BA.size(0), &c_emlrtBCI,
                                      (emlrtCTX)sp);
      }
      BA[nx] = BA[b_i];
      nx++;
    }
  }
  BA.set_size(&y_emlrtRTEI, sp, t);
  loop_ub = NonZeroRestriction.size(0) - 1;
  nx = 0;
  for (b_i = 0; b_i <= loop_ub; b_i++) {
    if (NonZeroRestriction[b_i]) {
      nx++;
    }
  }
  r1.set_size(&y_emlrtRTEI, sp, nx);
  nx = 0;
  for (b_i = 0; b_i <= loop_ub; b_i++) {
    if (NonZeroRestriction[b_i]) {
      r1[nx] = b_i + 1;
      nx++;
    }
  }
  b_YLm.set_size(&ab_emlrtRTEI, sp, r1.size(0), r1.size(0));
  nx = r1.size(0);
  for (i = 0; i < nx; i++) {
    loop_ub = r1.size(0);
    for (i1 = 0; i1 < loop_ub; i1++) {
      if (r1[i1] > XX.size(0)) {
        emlrtDynamicBoundsCheckR2012b(r1[i1], 1, XX.size(0), &d_emlrtBCI,
                                      (emlrtCTX)sp);
      }
      if (r1[i] > XX.size(1)) {
        emlrtDynamicBoundsCheckR2012b(r1[i], 1, XX.size(1), &e_emlrtBCI,
                                      (emlrtCTX)sp);
      }
      b_YLm[i1 + b_YLm.size(0) * i] =
          XX[(r1[i1] + XX.size(0) * (r1[i] - 1)) - 1];
    }
  }
  XX.set_size(&bb_emlrtRTEI, sp, b_YLm.size(0), b_YLm.size(1));
  nx = b_YLm.size(0) * b_YLm.size(1);
  for (i = 0; i < nx; i++) {
    XX[i] = b_YLm[i];
  }
  loop_ub = NonZeroRestriction.size(0) - 1;
  nx = 0;
  for (b_i = 0; b_i <= loop_ub; b_i++) {
    if (NonZeroRestriction[b_i]) {
      nx++;
    }
  }
  r2.set_size(&y_emlrtRTEI, sp, nx);
  nx = 0;
  for (b_i = 0; b_i <= loop_ub; b_i++) {
    if (NonZeroRestriction[b_i]) {
      r2[nx] = b_i + 1;
      nx++;
    }
  }
  st.site = &i_emlrtRSI;
  nx = r2.size(0);
  for (i = 0; i < nx; i++) {
    if (r2[i] > beta_st.size(0)) {
      emlrtDynamicBoundsCheckR2012b(r2[i], 1, beta_st.size(0), &f_emlrtBCI,
                                    &st);
    }
  }
  //  uses precision instead of var $/
  b_st.site = &xc_emlrtRSI;
  cholmod(&b_st, XX, Xt);
  //  the matrix that makes the y uncorrelated $/
  if (r2.size(0) != t) {
    emlrtSizeEqCheck1DR2012b(r2.size(0), t, &b_emlrtECI, &st);
  }
  b_st.site = &yc_emlrtRSI;
  XY.set_size(&cb_emlrtRTEI, &b_st, r2.size(0));
  nx = r2.size(0);
  for (i = 0; i < nx; i++) {
    XY[i] = beta_st[r2[i] - 1] - BA[i];
  }
  c_st.site = &m_emlrtRSI;
  coder::b_dynamic_size_checks(&c_st, Xt, XY, Xt.size(1), XY.size(0));
  c_st.site = &l_emlrtRSI;
  coder::internal::blas::mtimes(&c_st, Xt, XY, e);
  //  standard normals: k times m matrix $/
  b_st.site = &ad_emlrtRSI;
  c_st.site = &bd_emlrtRSI;
  d_st.site = &bd_emlrtRSI;
  e_st.site = &bd_emlrtRSI;
  coder::diag(&e_st, Xt, XY);
  b_p = false;
  i = XY.size(0);
  for (loop_ub = 0; loop_ub < i; loop_ub++) {
    if (b_p || (XY[loop_ub] < 0.0)) {
      b_p = true;
    }
  }
  if (b_p) {
    emlrtErrorWithMessageIdR2018a(
        &d_st, &emlrtRTEI, "Coder:toolbox:ElFunDomainError",
        "Coder:toolbox:ElFunDomainError", 3, 4, 3, "log");
  }
  e_st.site = &cd_emlrtRSI;
  nx = XY.size(0);
  f_st.site = &dd_emlrtRSI;
  if ((1 <= XY.size(0)) && (XY.size(0) > 2147483646)) {
    g_st.site = &o_emlrtRSI;
    coder::check_forloop_overflow_error(&g_st);
  }
  for (loop_ub = 0; loop_ub < nx; loop_ub++) {
    XY[loop_ub] = muDoubleScalarLog(XY[loop_ub]);
  }
  //  gauss function
  d_st.site = &ed_emlrtRSI;
  alpha1 = coder::sum(&d_st, XY);
  b_st.site = &ad_emlrtRSI;
  //  log pdf of standard normal $/
  BA.set_size(&db_emlrtRTEI, &b_st, e.size(0));
  nx = e.size(0);
  for (i = 0; i < nx; i++) {
    BA[i] = 0.5 * e[i];
  }
  if (BA.size(0) != e.size(0)) {
    emlrtSizeEqCheck1DR2012b(BA.size(0), e.size(0), &emlrtECI, &b_st);
  }
  b_st.site = &ad_emlrtRSI;
  //  gauss function
  nx = BA.size(0);
  for (i = 0; i < nx; i++) {
    BA[i] = -0.91893853320467267 - BA[i] * e[i];
  }
  c_st.site = &ed_emlrtRSI;
  beta1 = coder::sum(&c_st, BA);
  lnpdf_beta = 0.5 * (2.0 * alpha1) + beta1;
  //  the log of the density $/
  //  beta sampling 하기
  emlrtHeapReferenceStackLeaveFcnR2012b((emlrtCTX)sp);
  return lnpdf_beta;
}

// End of code generation (lnpost_beta.cpp)
