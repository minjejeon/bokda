//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// chol.h
//
// Code generation for function 'chol'
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
void cholesky(const emlrtStack *sp, ::coder::array<real_T, 2U> &A);

}

// End of code generation (chol.h)
