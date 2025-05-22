//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// diag.cpp
//
// Code generation for function 'diag'
//

// Include files
#include "diag.h"
#include "eml_int_forloop_overflow_check.h"
#include "lnpost_Omega_data.h"
#include "rt_nonfinite.h"
#include "coder_array.h"
#include "mwmathutil.h"

// Variable Definitions
static emlrtRSInfo ec_emlrtRSI{
    90,     // lineNo
    "diag", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\elmat\\diag.m" // pathName
};

static emlrtRTEInfo n_emlrtRTEI{
    102,    // lineNo
    19,     // colNo
    "diag", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\elmat\\diag.m" // pName
};

static emlrtRTEInfo ic_emlrtRTEI{
    109,    // lineNo
    24,     // colNo
    "diag", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\elmat\\diag.m" // pName
};

static emlrtRTEInfo jc_emlrtRTEI{
    100,    // lineNo
    5,      // colNo
    "diag", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\elmat\\diag.m" // pName
};

static emlrtRTEInfo kc_emlrtRTEI{
    82,     // lineNo
    5,      // colNo
    "diag", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\elmat\\diag.m" // pName
};

// Function Definitions
namespace coder {
void diag(const emlrtStack *sp, const ::coder::array<real_T, 1U> &v,
          ::coder::array<real_T, 2U> &d)
{
  emlrtStack b_st;
  emlrtStack st;
  int32_T loop_ub;
  int32_T nv;
  st.prev = sp;
  st.tls = sp->tls;
  b_st.prev = &st;
  b_st.tls = st.tls;
  nv = v.size(0);
  d.set_size(&kc_emlrtRTEI, sp, v.size(0), v.size(0));
  loop_ub = v.size(0) * v.size(0);
  for (int32_T i{0}; i < loop_ub; i++) {
    d[i] = 0.0;
  }
  st.site = &ec_emlrtRSI;
  if ((1 <= v.size(0)) && (v.size(0) > 2147483646)) {
    b_st.site = &i_emlrtRSI;
    check_forloop_overflow_error(&b_st);
  }
  for (loop_ub = 0; loop_ub < nv; loop_ub++) {
    d[loop_ub + d.size(0) * loop_ub] = v[loop_ub];
  }
}

void diag(const emlrtStack *sp, const ::coder::array<real_T, 2U> &v,
          ::coder::array<real_T, 1U> &d)
{
  if ((v.size(0) == 1) && (v.size(1) == 1)) {
    d.set_size(&jc_emlrtRTEI, sp, 1);
    d[0] = v[0];
  } else {
    int32_T m;
    int32_T n;
    if ((v.size(0) == 1) || (v.size(1) == 1)) {
      emlrtErrorWithMessageIdR2018a(
          sp, &n_emlrtRTEI, "Coder:toolbox:diag_varsizedMatrixVector",
          "Coder:toolbox:diag_varsizedMatrixVector", 0);
    }
    m = v.size(0);
    n = v.size(1);
    if (0 < v.size(1)) {
      m = muIntScalarMin_sint32(m, n);
    } else {
      m = 0;
    }
    d.set_size(&ic_emlrtRTEI, sp, m);
    m--;
    for (n = 0; n <= m; n++) {
      d[n] = v[n + v.size(0) * n];
    }
  }
}

} // namespace coder

// End of code generation (diag.cpp)
