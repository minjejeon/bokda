//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// eye.cpp
//
// Code generation for function 'eye'
//

// Include files
#include "eye.h"
#include "Gen_Mu_data.h"
#include "eml_int_forloop_overflow_check.h"
#include "rt_nonfinite.h"
#include "coder_array.h"
#include "mwmathutil.h"

// Variable Definitions
static emlrtRSInfo
    y_emlrtRSI{
        50,    // lineNo
        "eye", // fcnName
        "C:\\Program "
        "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\elmat\\eye.m" // pathName
    };

static emlrtRSInfo
    ab_emlrtRSI{
        96,    // lineNo
        "eye", // fcnName
        "C:\\Program "
        "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\elmat\\eye.m" // pathName
    };

static emlrtRSInfo bb_emlrtRSI{
    21,                           // lineNo
    "checkAndSaturateExpandSize", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+"
    "internal\\checkAndSaturateExpandSize.m" // pathName
};

static emlrtRTEInfo emlrtRTEI{
    58,                   // lineNo
    23,                   // colNo
    "assertValidSizeArg", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+"
    "internal\\assertValidSizeArg.m" // pName
};

static emlrtRTEInfo b_emlrtRTEI{
    64,                   // lineNo
    15,                   // colNo
    "assertValidSizeArg", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\eml\\+coder\\+"
    "internal\\assertValidSizeArg.m" // pName
};

static emlrtRTEInfo
    v_emlrtRTEI{
        94,    // lineNo
        5,     // colNo
        "eye", // fName
        "C:\\Program "
        "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\elmat\\eye.m" // pName
    };

// Function Definitions
namespace coder {
void eye(const emlrtStack *sp, real_T varargin_1,
         ::coder::array<real_T, 2U> &b_I)
{
  emlrtStack b_st;
  emlrtStack st;
  real_T b_varargin_1;
  int32_T loop_ub;
  int32_T m_tmp_tmp;
  st.prev = sp;
  st.tls = sp->tls;
  st.site = &y_emlrtRSI;
  b_st.prev = &st;
  b_st.tls = st.tls;
  b_st.site = &bb_emlrtRSI;
  if ((varargin_1 != muDoubleScalarFloor(varargin_1)) ||
      muDoubleScalarIsInf(varargin_1) || (varargin_1 > 2.147483647E+9)) {
    emlrtErrorWithMessageIdR2018a(
        &b_st, &emlrtRTEI, "Coder:MATLAB:NonIntegerInput",
        "Coder:MATLAB:NonIntegerInput", 4, 12, MIN_int32_T, 12, MAX_int32_T);
  }
  if (varargin_1 <= 0.0) {
    b_varargin_1 = 0.0;
  } else {
    b_varargin_1 = varargin_1;
  }
  if (!(b_varargin_1 <= 2.147483647E+9)) {
    emlrtErrorWithMessageIdR2018a(&b_st, &b_emlrtRTEI, "Coder:MATLAB:pmaxsize",
                                  "Coder:MATLAB:pmaxsize", 0);
  }
  m_tmp_tmp = static_cast<int32_T>(varargin_1);
  b_I.set_size(&v_emlrtRTEI, sp, m_tmp_tmp, m_tmp_tmp);
  loop_ub = static_cast<int32_T>(varargin_1) * static_cast<int32_T>(varargin_1);
  for (int32_T i{0}; i < loop_ub; i++) {
    b_I[i] = 0.0;
  }
  if (static_cast<int32_T>(varargin_1) > 0) {
    st.site = &ab_emlrtRSI;
    if ((1 <= static_cast<int32_T>(varargin_1)) &&
        (static_cast<int32_T>(varargin_1) > 2147483646)) {
      b_st.site = &cb_emlrtRSI;
      check_forloop_overflow_error(&b_st);
    }
    for (loop_ub = 0; loop_ub < m_tmp_tmp; loop_ub++) {
      b_I[loop_ub + b_I.size(0) * loop_ub] = 1.0;
    }
  }
}

} // namespace coder

// End of code generation (eye.cpp)
