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
#include "Minnesota_prior_data.h"
#include "eml_int_forloop_overflow_check.h"
#include "rt_nonfinite.h"
#include "coder_array.h"

// Variable Definitions
static emlrtRSInfo pc_emlrtRSI{
    90,     // lineNo
    "diag", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\elmat\\diag.m" // pathName
};

static emlrtRTEInfo ob_emlrtRTEI{
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
  d.set_size(&ob_emlrtRTEI, sp, v.size(0), v.size(0));
  loop_ub = v.size(0) * v.size(0);
  for (int32_T i{0}; i < loop_ub; i++) {
    d[i] = 0.0;
  }
  st.site = &pc_emlrtRSI;
  if ((1 <= v.size(0)) && (v.size(0) > 2147483646)) {
    b_st.site = &v_emlrtRSI;
    check_forloop_overflow_error(&b_st);
  }
  for (loop_ub = 0; loop_ub < nv; loop_ub++) {
    d[loop_ub + d.size(0) * loop_ub] = v[loop_ub];
  }
}

} // namespace coder

// End of code generation (diag.cpp)
