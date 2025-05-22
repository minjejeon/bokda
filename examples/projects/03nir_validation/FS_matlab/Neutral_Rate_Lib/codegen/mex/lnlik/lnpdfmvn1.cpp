//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// lnpdfmvn1.cpp
//
// Code generation for function 'lnpdfmvn1'
//

// Include files
#include "lnpdfmvn1.h"
#include "cholmod.h"
#include "diag.h"
#include "eml_int_forloop_overflow_check.h"
#include "lnlik_data.h"
#include "mtimes.h"
#include "rt_nonfinite.h"
#include "sum.h"
#include "coder_array.h"
#include "mwmathutil.h"

// Variable Definitions
static emlrtRSInfo
    kd_emlrtRSI{
        3,           // lineNo
        "lnpdfmvn1", // fcnName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\lnpdfmvn1."
        "m" // pathName
    };

static emlrtRSInfo
    ld_emlrtRSI{
        4,           // lineNo
        "lnpdfmvn1", // fcnName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\lnpdfmvn1."
        "m" // pathName
    };

static emlrtRSInfo
    md_emlrtRSI{
        5,           // lineNo
        "lnpdfmvn1", // fcnName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\lnpdfmvn1."
        "m" // pathName
    };

static emlrtRSInfo nd_emlrtRSI{
    2,        // lineNo
    "lndet1", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\lndet1.m" // pathName
};

static emlrtRSInfo
    od_emlrtRSI{
        17,    // lineNo
        "log", // fcnName
        "C:\\Program "
        "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\elfun\\log.m" // pathName
    };

static emlrtRSInfo pd_emlrtRSI{
    33,                           // lineNo
    "applyScalarFunctionInPlace", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+"
    "internal\\applyScalarFunctionInPlace.m" // pathName
};

static emlrtRSInfo qd_emlrtRSI{
    4,      // lineNo
    "sumc", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\sumc.m" // pathName
};

static emlrtECInfo
    x_emlrtECI{
        -1,          // nDims
        4,           // lineNo
        8,           // colNo
        "lnpdfmvn1", // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\lnpdfmvn1."
        "m" // pName
    };

static emlrtECInfo y_emlrtECI{
    -1,        // nDims
    6,         // lineNo
    12,        // colNo
    "lnpdfn1", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\lnpdfn1.m" // pName
};

static emlrtRTEInfo
    r_emlrtRTEI{
        14,    // lineNo
        9,     // colNo
        "log", // fName
        "C:\\Program "
        "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\elfun\\log.m" // pName
    };

static emlrtRTEInfo
    sc_emlrtRTEI{
        4,           // lineNo
        8,           // colNo
        "lnpdfmvn1", // fName
        "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\lnpdfmvn1."
        "m" // pName
    };

static emlrtRTEInfo tc_emlrtRTEI{
    6,         // lineNo
    12,        // colNo
    "lnpdfn1", // fName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\lnpdfn1.m" // pName
};

// Function Definitions
real_T lnpdfmvn1(const emlrtStack *sp, const coder::array<real_T, 1U> &y,
                 const coder::array<real_T, 1U> &mu,
                 const coder::array<real_T, 2U> &P)
{
  coder::array<real_T, 2U> C;
  coder::array<real_T, 1U> b;
  coder::array<real_T, 1U> e;
  emlrtStack b_st;
  emlrtStack c_st;
  emlrtStack d_st;
  emlrtStack e_st;
  emlrtStack f_st;
  emlrtStack st;
  real_T retf;
  real_T z;
  int32_T i;
  int32_T k;
  int32_T nx;
  boolean_T p;
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
  emlrtHeapReferenceStackEnterFcnR2012b((emlrtCTX)sp);
  //  uses precision instead of var $/
  st.site = &kd_emlrtRSI;
  cholmod(&st, P, C);
  //  the matrix that makes the y uncorrelated $/
  if (y.size(0) != mu.size(0)) {
    emlrtSizeEqCheck1DR2012b(y.size(0), mu.size(0), &x_emlrtECI, (emlrtCTX)sp);
  }
  st.site = &ld_emlrtRSI;
  b.set_size(&sc_emlrtRTEI, &st, y.size(0));
  nx = y.size(0);
  for (i = 0; i < nx; i++) {
    b[i] = y[i] - mu[i];
  }
  b_st.site = &eb_emlrtRSI;
  if (b.size(0) != C.size(1)) {
    if (((C.size(0) == 1) && (C.size(1) == 1)) || (b.size(0) == 1)) {
      emlrtErrorWithMessageIdR2018a(
          &b_st, &c_emlrtRTEI, "Coder:toolbox:mtimes_noDynamicScalarExpansion",
          "Coder:toolbox:mtimes_noDynamicScalarExpansion", 0);
    } else {
      emlrtErrorWithMessageIdR2018a(&b_st, &b_emlrtRTEI, "MATLAB:innerdim",
                                    "MATLAB:innerdim", 0);
    }
  }
  b_st.site = &db_emlrtRSI;
  coder::internal::blas::mtimes(&b_st, C, b, e);
  //  standard normals: k times m matrix $/
  st.site = &md_emlrtRSI;
  b_st.site = &nd_emlrtRSI;
  c_st.site = &nd_emlrtRSI;
  d_st.site = &nd_emlrtRSI;
  coder::diag(&d_st, C, b);
  p = false;
  i = b.size(0);
  for (k = 0; k < i; k++) {
    if (p || (b[k] < 0.0)) {
      p = true;
    }
  }
  if (p) {
    emlrtErrorWithMessageIdR2018a(
        &c_st, &r_emlrtRTEI, "Coder:toolbox:ElFunDomainError",
        "Coder:toolbox:ElFunDomainError", 3, 4, 3, "log");
  }
  d_st.site = &od_emlrtRSI;
  nx = b.size(0);
  e_st.site = &pd_emlrtRSI;
  if ((1 <= b.size(0)) && (b.size(0) > 2147483646)) {
    f_st.site = &y_emlrtRSI;
    coder::check_forloop_overflow_error(&f_st);
  }
  for (k = 0; k < nx; k++) {
    b[k] = muDoubleScalarLog(b[k]);
  }
  //  gauss function
  c_st.site = &qd_emlrtRSI;
  retf = coder::sum(&c_st, b);
  st.site = &md_emlrtRSI;
  //  log pdf of standard normal $/
  b.set_size(&tc_emlrtRTEI, &st, e.size(0));
  nx = e.size(0);
  for (i = 0; i < nx; i++) {
    b[i] = 0.5 * e[i];
  }
  if (b.size(0) != e.size(0)) {
    emlrtSizeEqCheck1DR2012b(b.size(0), e.size(0), &y_emlrtECI, &st);
  }
  st.site = &md_emlrtRSI;
  //  gauss function
  nx = b.size(0);
  for (i = 0; i < nx; i++) {
    b[i] = -0.91893853320467267 - b[i] * e[i];
  }
  real_T b_retf;
  b_st.site = &qd_emlrtRSI;
  b_retf = coder::sum(&b_st, b);
  z = 0.5 * (2.0 * retf) + b_retf;
  //  the log of the density $/
  emlrtHeapReferenceStackLeaveFcnR2012b((emlrtCTX)sp);
  return z;
}

// End of code generation (lnpdfmvn1.cpp)
