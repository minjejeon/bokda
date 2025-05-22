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
static emlrtRTEInfo vb_emlrtRTEI{
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
  r.set_size(&vb_emlrtRTEI, sp, static_cast<int32_T>(varargin_1));
  if (static_cast<int32_T>(varargin_1) != 0) {
    emlrtRandn(&(r.data())[0], static_cast<int32_T>(varargin_1));
  }
}

} // namespace coder

// End of code generation (randn.cpp)
