/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * Gen_C_internal_types.h
 *
 * Code generation for function 'Gen_C'
 *
 */

#pragma once

/* Include files */
#include "Gen_C_types.h"
#include "rtwtypes.h"
#include "emlrt.h"

/* Type Definitions */
#ifndef typedef_rtDesignRangeCheckInfo
#define typedef_rtDesignRangeCheckInfo
typedef struct {
  int32_T lineNo;
  int32_T colNo;
  const char_T *fName;
  const char_T *pName;
} rtDesignRangeCheckInfo;
#endif /* typedef_rtDesignRangeCheckInfo */

/* End of code generation (Gen_C_internal_types.h) */
