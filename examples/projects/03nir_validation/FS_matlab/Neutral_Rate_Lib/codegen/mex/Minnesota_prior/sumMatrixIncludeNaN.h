//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// sumMatrixIncludeNaN.h
//
// Code generation for function 'sumMatrixIncludeNaN'
//

#pragma once

// Include files
#include "rtwtypes.h"
#include "coder_array.h"
#include "emlrt.h"
#include "mex.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Function Declarations
namespace coder {
real_T sumColumnB(const emlrtStack *sp, const ::coder::array<real_T, 2U> &x,
                  int32_T col, int32_T vlen, int32_T vstart);

real_T sumColumnB(const emlrtStack *sp, const ::coder::array<real_T, 2U> &x,
                  int32_T col, int32_T vlen);

real_T sumColumnB4(const ::coder::array<real_T, 2U> &x, int32_T col,
                   int32_T vstart);

} // namespace coder

// End of code generation (sumMatrixIncludeNaN.h)
