//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// cholmod.h
//
// Code generation for function 'cholmod'
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
void cholmod(const emlrtStack *sp, const coder::array<real_T, 2U> &A,
             coder::array<real_T, 2U> &R);

void cholmod(const emlrtStack *sp, const coder::array<real_T, 2U> &A,
             coder::array<real_T, 2U> &R, real_T *err);

// End of code generation (cholmod.h)
