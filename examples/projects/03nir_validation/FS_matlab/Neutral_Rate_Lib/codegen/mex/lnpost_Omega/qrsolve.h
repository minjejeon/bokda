//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// qrsolve.h
//
// Code generation for function 'qrsolve'
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
namespace internal {
void qrsolve(const emlrtStack *sp, const ::coder::array<real_T, 2U> &A,
             const ::coder::array<real_T, 2U> &B,
             ::coder::array<real_T, 2U> &Y);

}
} // namespace coder

// End of code generation (qrsolve.h)
