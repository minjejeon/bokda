//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// lnpost_beta.h
//
// Code generation for function 'lnpost_beta'
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
real_T lnpost_beta(const emlrtStack *sp, const coder::array<real_T, 2U> &Y0,
                   const coder::array<real_T, 3U> &YLm,
                   const coder::array<real_T, 1U> &beta_st, real_T p,
                   const coder::array<real_T, 1U> &b_,
                   const coder::array<real_T, 2U> &var_,
                   const coder::array<real_T, 2U> &Omega_inv,
                   const coder::array<boolean_T, 1U> &NonZeroRestriction);

// End of code generation (lnpost_beta.h)
