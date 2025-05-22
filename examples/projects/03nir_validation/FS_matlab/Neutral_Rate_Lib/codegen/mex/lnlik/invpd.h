//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// invpd.h
//
// Code generation for function 'invpd'
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
void invpd(const emlrtStack *sp, const coder::array<real_T, 2U> &A,
           coder::array<real_T, 2U> &Ainv);

// End of code generation (invpd.h)
