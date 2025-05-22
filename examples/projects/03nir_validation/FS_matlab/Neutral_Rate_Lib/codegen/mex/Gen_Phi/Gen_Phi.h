//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// Gen_Phi.h
//
// Code generation for function 'Gen_Phi'
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
void Gen_Phi(const emlrtStack *sp, const coder::array<real_T, 2U> &Y0,
             const coder::array<real_T, 3U> &YLm,
             const coder::array<real_T, 2U> &Phi0, real_T p,
             const coder::array<real_T, 1U> &b_,
             const coder::array<real_T, 2U> &var_,
             const coder::array<real_T, 2U> &Omega_inv,
             coder::array<real_T, 2U> &Phi, coder::array<real_T, 2U> &Fm,
             coder::array<real_T, 1U> &beta, real_T *is_reject);

// End of code generation (Gen_Phi.h)
