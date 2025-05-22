//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// makeYX.h
//
// Code generation for function 'makeYX'
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
void makeYX(const emlrtStack *sp, const coder::array<real_T, 2U> &Y, real_T p,
            coder::array<real_T, 2U> &Y0, coder::array<real_T, 3U> &YLm);

// End of code generation (makeYX.h)
