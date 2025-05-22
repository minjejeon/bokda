/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * lnpdfn.h
 *
 * Code generation for function 'lnpdfn'
 *
 */

#pragma once

/* Include files */
#include "Gen_C_types.h"
#include "rtwtypes.h"
#include "emlrt.h"
#include "mex.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Function Declarations */
void lnpdfn(const emlrtStack *sp, const emxArray_real_T *x,
            const emxArray_real_T *mu, const emxArray_real_T *sig2vec,
            emxArray_real_T *retf);

/* End of code generation (lnpdfn.h) */
