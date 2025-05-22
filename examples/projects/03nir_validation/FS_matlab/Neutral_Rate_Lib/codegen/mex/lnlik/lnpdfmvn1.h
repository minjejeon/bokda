//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// lnpdfmvn1.h
//
// Code generation for function 'lnpdfmvn1'
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
real_T lnpdfmvn1(const emlrtStack *sp, const coder::array<real_T, 1U> &y,
                 const coder::array<real_T, 1U> &mu,
                 const coder::array<real_T, 2U> &P);

// End of code generation (lnpdfmvn1.h)
