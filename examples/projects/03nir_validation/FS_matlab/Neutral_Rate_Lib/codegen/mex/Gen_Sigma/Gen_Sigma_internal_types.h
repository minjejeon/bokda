//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// Gen_Sigma_internal_types.h
//
// Code generation for function 'Gen_Sigma'
//

#pragma once

// Include files
#include "Gen_Sigma_types.h"
#include "rtwtypes.h"
#include "emlrt.h"

// Type Definitions
struct rtDesignRangeCheckInfo {
  int32_T lineNo;
  int32_T colNo;
  const char_T *fName;
  const char_T *pName;
};

struct rtDoubleCheckInfo {
  int32_T lineNo;
  int32_T colNo;
  const char_T *fName;
  const char_T *pName;
  int32_T checkKind;
};

// End of code generation (Gen_Sigma_internal_types.h)
