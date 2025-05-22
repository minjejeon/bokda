//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// error.cpp
//
// Code generation for function 'error'
//

// Include files
#include "error.h"
#include "rt_nonfinite.h"

// Variable Definitions
static emlrtMCInfo b_emlrtMCI{
    27,      // lineNo
    5,       // colNo
    "error", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\lang\\error.m" // pName
};

static emlrtRSInfo rd_emlrtRSI{
    27,      // lineNo
    "error", // fcnName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\lang\\error.m" // pathName
};

// Function Declarations
static void error(const emlrtStack *sp, const mxArray *b,
                  emlrtMCInfo *location);

// Function Definitions
static void error(const emlrtStack *sp, const mxArray *b, emlrtMCInfo *location)
{
  const mxArray *pArray;
  pArray = b;
  emlrtCallMATLABR2012b((emlrtCTX)sp, 0, nullptr, 1, &pArray,
                        (const char_T *)"error", true, location);
}

namespace coder {
void b_error(const emlrtStack *sp)
{
  static const int32_T iv[2]{1, 18};
  static const char_T varargin_1[18]{'A', ' ', 'i', 's', ' ', 'n',
                                     'o', 't', ' ', 's', 'y', 'm',
                                     'm', 'e', 't', 'r', 'i', 'c'};
  emlrtStack st;
  const mxArray *m;
  const mxArray *y;
  st.prev = sp;
  st.tls = sp->tls;
  y = nullptr;
  m = emlrtCreateCharArray(2, &iv[0]);
  emlrtInitCharArrayR2013a((emlrtCTX)sp, 18, m, &varargin_1[0]);
  emlrtAssign(&y, m);
  st.site = &rd_emlrtRSI;
  error(&st, y, &b_emlrtMCI);
}

} // namespace coder

// End of code generation (error.cpp)
