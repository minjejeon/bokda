//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// randn.cpp
//
// Code generation for function 'randn'
//

// Include files
#include "randn.h"
#include "rt_nonfinite.h"
#include "coder_array.h"

// Variable Definitions
static emlrtDCInfo j_emlrtDCI{
    110,     // lineNo
    30,      // colNo
    "randn", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\randfun\\randn.m", // pName
    4 // checkKind
};

static emlrtRTEInfo qc_emlrtRTEI{
    110,     // lineNo
    24,      // colNo
    "randn", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\randfun\\randn.m" // pName
};

// Function Definitions
namespace coder {
void randn(const emlrtStack *sp, real_T varargin_1,
           ::coder::array<real_T, 1U> &r)
{
  if (!(varargin_1 >= 0.0)) {
    emlrtNonNegativeCheckR2012b(varargin_1, &j_emlrtDCI, (emlrtCTX)sp);
  }
  r.set_size(&qc_emlrtRTEI, sp, static_cast<int32_T>(varargin_1));
  if (static_cast<int32_T>(varargin_1) != 0) {
    emlrtRandn(&(r.data())[0], static_cast<int32_T>(varargin_1));
  }
}

} // namespace coder

// End of code generation (randn.cpp)
