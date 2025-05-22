//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// Gen_Gamma.h
//
// Code generation for function 'Gen_Gamma'
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
void Gen_Gamma(const emlrtStack *sp, coder::array<real_T, 1U> &diag_Sigma,
               real_T q, real_T a_10, real_T a_00, real_T c_10, real_T c_00,
               coder::array<real_T, 1U> &b_gamma);

// End of code generation (Gen_Gamma.h)
