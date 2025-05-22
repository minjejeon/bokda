//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// applyScalarFunctionInPlace.h
//
// Code generation for function 'applyScalarFunctionInPlace'
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
void applyScalarFunctionInPlace(const emlrtStack *sp,
                                ::coder::array<real_T, 2U> &x);

}
} // namespace coder

// End of code generation (applyScalarFunctionInPlace.h)
