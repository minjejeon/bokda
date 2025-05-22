//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// Gen_Um.h
//
// Code generation for function 'Gen_Um'
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
void Gen_Um(const emlrtStack *sp, const coder::array<real_T, 2U> &Ym,
            coder::array<real_T, 2U> &Um, const coder::array<real_T, 1U> &rho,
            const coder::array<real_T, 2U> &Phi,
            const coder::array<real_T, 2U> &Sigma,
            const coder::array<real_T, 2U> &Omega,
            coder::array<real_T, 2U> &Uttm);

// End of code generation (Gen_Um.h)
