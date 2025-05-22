//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// sum.h
//
// Code generation for function 'sum'
//

#pragma once

// Include files
#include "rtwtypes.h"
#include "coder_array.h"
#include "emlrt.h"
#include "mex.h"
#include "omp.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Function Declarations
namespace coder {
real_T sum(const emlrtStack *sp, const ::coder::array<real_T, 2U> &x);

void sum(const emlrtStack *sp, const ::coder::array<real_T, 2U> &x,
         ::coder::array<real_T, 2U> &y);

} // namespace coder

// End of code generation (sum.h)
