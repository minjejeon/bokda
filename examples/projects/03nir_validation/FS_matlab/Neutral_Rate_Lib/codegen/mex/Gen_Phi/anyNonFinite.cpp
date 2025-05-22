//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// anyNonFinite.cpp
//
// Code generation for function 'anyNonFinite'
//

// Include files
#include "anyNonFinite.h"
#include "Gen_Phi_data.h"
#include "eml_int_forloop_overflow_check.h"
#include "rt_nonfinite.h"
#include "coder_array.h"
#include "mwmathutil.h"

// Variable Definitions
static emlrtRSInfo ld_emlrtRSI{
    29,             // lineNo
    "anyNonFinite", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+internal\\anyNonFinite."
    "m" // pathName
};

static emlrtRSInfo
    md_emlrtRSI{
        44,          // lineNo
        "vAllOrAny", // fcnName
        "C:\\Program "
        "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+"
        "internal\\vAllOrAny.m" // pathName
    };

static emlrtRSInfo
    nd_emlrtRSI{
        103,                  // lineNo
        "flatVectorAllOrAny", // fcnName
        "C:\\Program "
        "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+"
        "internal\\vAllOrAny.m" // pathName
    };

// Function Definitions
namespace coder {
namespace internal {
boolean_T anyNonFinite(const emlrtStack *sp,
                       const ::coder::array<real_T, 2U> &x)
{
  emlrtStack b_st;
  emlrtStack c_st;
  emlrtStack d_st;
  emlrtStack st;
  int32_T nx;
  boolean_T p;
  st.prev = sp;
  st.tls = sp->tls;
  st.site = &ld_emlrtRSI;
  b_st.prev = &st;
  b_st.tls = st.tls;
  c_st.prev = &b_st;
  c_st.tls = b_st.tls;
  d_st.prev = &c_st;
  d_st.tls = c_st.tls;
  b_st.site = &md_emlrtRSI;
  nx = x.size(0) * x.size(1);
  p = true;
  c_st.site = &nd_emlrtRSI;
  if ((1 <= nx) && (nx > 2147483646)) {
    d_st.site = &u_emlrtRSI;
    check_forloop_overflow_error(&d_st);
  }
  for (int32_T k{0}; k < nx; k++) {
    if ((!p) || (muDoubleScalarIsInf(x[k]) || muDoubleScalarIsNaN(x[k]))) {
      p = false;
    }
  }
  return !p;
}

} // namespace internal
} // namespace coder

// End of code generation (anyNonFinite.cpp)
