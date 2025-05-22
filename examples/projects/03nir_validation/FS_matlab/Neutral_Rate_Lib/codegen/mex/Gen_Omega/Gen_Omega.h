//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// Gen_Omega.h
//
// Code generation for function 'Gen_Omega'
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
void Gen_Omega(const emlrtStack *sp, const coder::array<real_T, 2U> &Y,
               const coder::array<real_T, 3U> &X,
               const coder::array<real_T, 1U> &beta, real_T nu,
               const coder::array<real_T, 2U> &R0,
               coder::array<real_T, 2U> &Omega,
               coder::array<real_T, 2U> &Omega_inv);

// End of code generation (Gen_Omega.h)
