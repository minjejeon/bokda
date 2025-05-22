//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// assertValidSizeArg.cpp
//
// Code generation for function 'assertValidSizeArg'
//

// Include files
#include "assertValidSizeArg.h"
#include "Minnesota_prior_data.h"
#include "rt_nonfinite.h"
#include "mwmathutil.h"

// Function Definitions
namespace coder {
namespace internal {
void assertValidSizeArg(const emlrtStack *sp, real_T varargin_1)
{
  if ((varargin_1 != varargin_1) || muDoubleScalarIsInf(varargin_1) ||
      (varargin_1 > 2.147483647E+9)) {
    emlrtErrorWithMessageIdR2018a(
        sp, &j_emlrtRTEI, "Coder:MATLAB:NonIntegerInput",
        "Coder:MATLAB:NonIntegerInput", 4, 12, MIN_int32_T, 12, MAX_int32_T);
  }
}

} // namespace internal
} // namespace coder

// End of code generation (assertValidSizeArg.cpp)
