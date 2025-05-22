//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// Minnesota_prior.h
//
// Code generation for function 'Minnesota_prior'
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
void Minnesota_prior(const emlrtStack *sp, coder::array<real_T, 2U> &y,
                     real_T p, real_T lambda_1, real_T lambda_2,
                     coder::array<real_T, 2U> &beta_0,
                     coder::array<real_T, 2U> &B_0,
                     coder::array<real_T, 2U> &Omega_hat);

// End of code generation (Minnesota_prior.h)
