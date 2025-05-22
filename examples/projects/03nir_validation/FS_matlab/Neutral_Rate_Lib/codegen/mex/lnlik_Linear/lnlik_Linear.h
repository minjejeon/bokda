//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// lnlik_Linear.h
//
// Code generation for function 'lnlik_Linear'
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
real_T lnlik_Linear(const emlrtStack *sp, const coder::array<real_T, 2U> &Y0, const coder::array<real_T, 3U> &YLm, const coder::array<real_T, 1U> &beta, const coder::array<real_T, 2U> &Omega);



// End of code generation (lnlik_Linear.h) 
