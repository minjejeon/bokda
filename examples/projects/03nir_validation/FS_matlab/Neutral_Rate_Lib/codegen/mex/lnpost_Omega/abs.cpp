//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// abs.cpp
//
// Code generation for function 'abs'
//

// Include files
#include "abs.h"
#include "eml_int_forloop_overflow_check.h"
#include "lnpost_Omega_data.h"
#include "rt_nonfinite.h"
#include "coder_array.h"
#include "mwmathutil.h"

// Variable Definitions
static emlrtRSInfo
    db_emlrtRSI{
        18,    // lineNo
        "abs", // fcnName
        "C:\\Program "
        "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\elfun\\abs.m" // pathName
    };

static emlrtRSInfo eb_emlrtRSI{
    74,                    // lineNo
    "applyScalarFunction", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+"
    "internal\\applyScalarFunction.m" // pathName
};

static emlrtRTEInfo fc_emlrtRTEI{
    30,                    // lineNo
    21,                    // colNo
    "applyScalarFunction", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+"
    "internal\\applyScalarFunction.m" // pName
};

// Function Definitions
namespace coder {
void b_abs(const emlrtStack *sp, const ::coder::array<real_T, 1U> &x,
           ::coder::array<real_T, 1U> &y)
{
  emlrtStack b_st;
  emlrtStack c_st;
  emlrtStack st;
  int32_T nx;
  st.prev = sp;
  st.tls = sp->tls;
  st.site = &db_emlrtRSI;
  b_st.prev = &st;
  b_st.tls = st.tls;
  c_st.prev = &b_st;
  c_st.tls = b_st.tls;
  nx = x.size(0);
  y.set_size(&fc_emlrtRTEI, &st, x.size(0));
  b_st.site = &eb_emlrtRSI;
  if ((1 <= x.size(0)) && (x.size(0) > 2147483646)) {
    c_st.site = &i_emlrtRSI;
    check_forloop_overflow_error(&c_st);
  }
  for (int32_T k{0}; k < nx; k++) {
    y[k] = muDoubleScalarAbs(x[k]);
  }
}

void b_abs(const emlrtStack *sp, const ::coder::array<real_T, 2U> &x,
           ::coder::array<real_T, 2U> &y)
{
  emlrtStack b_st;
  emlrtStack c_st;
  emlrtStack st;
  int32_T nx;
  st.prev = sp;
  st.tls = sp->tls;
  st.site = &db_emlrtRSI;
  b_st.prev = &st;
  b_st.tls = st.tls;
  c_st.prev = &b_st;
  c_st.tls = b_st.tls;
  nx = x.size(0) * x.size(1);
  y.set_size(&fc_emlrtRTEI, &st, x.size(0), x.size(1));
  b_st.site = &eb_emlrtRSI;
  if ((1 <= nx) && (nx > 2147483646)) {
    c_st.site = &i_emlrtRSI;
    check_forloop_overflow_error(&c_st);
  }
  for (int32_T k{0}; k < nx; k++) {
    y[k] = muDoubleScalarAbs(x[k]);
  }
}

} // namespace coder

// End of code generation (abs.cpp)
