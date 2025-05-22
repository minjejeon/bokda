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
static emlrtDCInfo b_emlrtDCI{
    110,     // lineNo
    30,      // colNo
    "randn", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\randfun\\randn.m", // pName
    4 // checkKind
};

static emlrtRTEInfo ec_emlrtRTEI{
    110,     // lineNo
    24,      // colNo
    "randn", // fName
    "C:\\Program "
    "Files\\MATLAB\\R2021a\\toolbox\\eml\\lib\\matlab\\randfun\\randn.m" // pName
};

// Function Definitions
namespace coder {
void randn(const emlrtStack *sp, real_T varargin_1, real_T varargin_2,
           ::coder::array<real_T, 2U> &r)
{
  r.set_size(&ec_emlrtRTEI, sp, static_cast<int32_T>(varargin_1), r.size(1));
  if (!(varargin_2 >= 0.0)) {
    emlrtNonNegativeCheckR2012b(varargin_2, &b_emlrtDCI, (emlrtCTX)sp);
  }
  r.set_size(&ec_emlrtRTEI, sp, r.size(0), static_cast<int32_T>(varargin_2));
  if ((static_cast<int32_T>(varargin_1) != 0) &&
      (static_cast<int32_T>(varargin_2) != 0)) {
    emlrtRandn(&(r.data())[0], static_cast<int32_T>(varargin_1) *
                                   static_cast<int32_T>(varargin_2));
  }
}

} // namespace coder

// End of code generation (randn.cpp)
