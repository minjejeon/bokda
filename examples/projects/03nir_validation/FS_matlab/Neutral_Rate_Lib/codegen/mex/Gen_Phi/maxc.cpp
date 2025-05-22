//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// maxc.cpp
//
// Code generation for function 'maxc'
//

// Include files
#include "maxc.h"
#include "Gen_Phi_data.h"
#include "eml_int_forloop_overflow_check.h"
#include "rt_nonfinite.h"
#include "coder_array.h"
#include "mwmathutil.h"

// Variable Definitions
static emlrtRSInfo ie_emlrtRSI{
    4,      // lineNo
    "maxc", // fcnName
    "D:\\Dropbox\\Policy_Works\\NL_LBVAR_2023\\Codes\\M_library\\maxc.m" // pathName
};

static emlrtRSInfo je_emlrtRSI{
    17,    // lineNo
    "max", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\datafun\\max.m" // pathName
};

static emlrtRSInfo
    ke_emlrtRSI{
        38,         // lineNo
        "minOrMax", // fcnName
        "C:\\Program "
        "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\minOrMax."
        "m" // pathName
    };

static emlrtRSInfo
    le_emlrtRSI{
        77,        // lineNo
        "maximum", // fcnName
        "C:\\Program "
        "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\minOrMax."
        "m" // pathName
    };

static emlrtRSInfo me_emlrtRSI{
    161,             // lineNo
    "unaryMinOrMax", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+"
    "internal\\unaryMinOrMax.m" // pathName
};

// Function Definitions
real_T maxc(const emlrtStack *sp, const coder::array<real_T, 1U> &x)
{
  emlrtStack b_st;
  emlrtStack c_st;
  emlrtStack d_st;
  emlrtStack e_st;
  emlrtStack f_st;
  emlrtStack g_st;
  emlrtStack h_st;
  emlrtStack st;
  real_T mx;
  int32_T last;
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
  //  function [mx,ind] = maxc(x)
  st.site = &ie_emlrtRSI;
  b_st.site = &je_emlrtRSI;
  c_st.site = &ke_emlrtRSI;
  d_st.site = &le_emlrtRSI;
  if (x.size(0) < 1) {
    emlrtErrorWithMessageIdR2018a(&d_st, &j_emlrtRTEI,
                                  "Coder:toolbox:eml_min_or_max_varDimZero",
                                  "Coder:toolbox:eml_min_or_max_varDimZero", 0);
  }
  e_st.site = &me_emlrtRSI;
  last = x.size(0);
  if (x.size(0) <= 2) {
    if (x.size(0) == 1) {
      mx = x[0];
    } else if ((x[0] < x[1]) ||
               (muDoubleScalarIsNaN(x[0]) && (!muDoubleScalarIsNaN(x[1])))) {
      mx = x[1];
    } else {
      mx = x[0];
    }
  } else {
    int32_T idx;
    int32_T k;
    f_st.site = &nc_emlrtRSI;
    if (!muDoubleScalarIsNaN(x[0])) {
      idx = 1;
    } else {
      boolean_T exitg1;
      idx = 0;
      g_st.site = &oc_emlrtRSI;
      if (x.size(0) > 2147483646) {
        h_st.site = &u_emlrtRSI;
        coder::check_forloop_overflow_error(&h_st);
      }
      k = 2;
      exitg1 = false;
      while ((!exitg1) && (k <= last)) {
        if (!muDoubleScalarIsNaN(x[k - 1])) {
          idx = k;
          exitg1 = true;
        } else {
          k++;
        }
      }
    }
    if (idx == 0) {
      mx = x[0];
    } else {
      int32_T a;
      f_st.site = &mc_emlrtRSI;
      mx = x[idx - 1];
      a = idx + 1;
      g_st.site = &pc_emlrtRSI;
      if ((idx + 1 <= x.size(0)) && (x.size(0) > 2147483646)) {
        h_st.site = &u_emlrtRSI;
        coder::check_forloop_overflow_error(&h_st);
      }
      for (k = a; k <= last; k++) {
        real_T d;
        d = x[k - 1];
        if (mx < d) {
          mx = d;
        }
      }
    }
  }
  return mx;
}

// End of code generation (maxc.cpp)
