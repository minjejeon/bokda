/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * Gen_C.h
 *
 * Code generation for function 'Gen_C'
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
void Gen_C(const emlrtStack *sp, const emxArray_real_T *Um,
           const emxArray_real_T *B, const emxArray_real_T *Cbar0,
           const emxArray_real_T *omega, const emxArray_real_T *sigma,
           const emxArray_real_T *C_, const emxArray_real_T *V_,
           emxArray_real_T *Cbar, real_T *accept);

/* End of code generation (Gen_C.h) */
