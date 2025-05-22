//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// Gen_Sigma.h
//
// Code generation for function 'Gen_Sigma'
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
void Gen_Sigma(const emlrtStack *sp, coder::array<real_T, 2U> &Mu,
               const coder::array<real_T, 1U> &b_gamma, real_T a_10,
               real_T a_00, real_T c_10, real_T c_00,
               coder::array<real_T, 1U> &diag_Sigma);

// End of code generation (Gen_Sigma.h)
