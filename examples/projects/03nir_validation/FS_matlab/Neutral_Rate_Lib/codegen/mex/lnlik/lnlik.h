//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// lnlik.h
//
// Code generation for function 'lnlik'
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
real_T lnlik(const emlrtStack *sp, const coder::array<real_T, 2U> &Mu,
             const coder::array<real_T, 2U> &Y0,
             const coder::array<real_T, 3U> &YLm,
             const coder::array<real_T, 1U> &beta,
             const coder::array<real_T, 2U> &Phi,
             const coder::array<real_T, 2U> &Omega,
             const coder::array<real_T, 1U> &diag_Sigma,
             const coder::array<real_T, 1U> &b_gamma, real_T is_Nonlinear);

// End of code generation (lnlik.h)
