//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// eml_mtimes_helper.cpp
//
// Code generation for function 'eml_mtimes_helper'
//

// Include files
#include "eml_mtimes_helper.h"
#include "lnpost_Omega_data.h"
#include "rt_nonfinite.h"
#include "coder_array.h"

// Function Definitions
namespace coder {
void b_dynamic_size_checks(const emlrtStack *sp,
                           const ::coder::array<real_T, 2U> &a,
                           const ::coder::array<real_T, 1U> &b,
                           int32_T innerDimA, int32_T innerDimB)
{
  if (innerDimA != innerDimB) {
    if ((a.size(1) == 1) || (b.size(0) == 1)) {
      emlrtErrorWithMessageIdR2018a(
          sp, &emlrtRTEI, "Coder:toolbox:mtimes_noDynamicScalarExpansion",
          "Coder:toolbox:mtimes_noDynamicScalarExpansion", 0);
    } else {
      emlrtErrorWithMessageIdR2018a(sp, &b_emlrtRTEI, "MATLAB:innerdim",
                                    "MATLAB:innerdim", 0);
    }
  }
}

void dynamic_size_checks(const emlrtStack *sp,
                         const ::coder::array<real_T, 2U> &a,
                         const ::coder::array<real_T, 1U> &b, int32_T innerDimA,
                         int32_T innerDimB)
{
  if (innerDimA != innerDimB) {
    if (((a.size(0) == 1) && (a.size(1) == 1)) || (b.size(0) == 1)) {
      emlrtErrorWithMessageIdR2018a(
          sp, &emlrtRTEI, "Coder:toolbox:mtimes_noDynamicScalarExpansion",
          "Coder:toolbox:mtimes_noDynamicScalarExpansion", 0);
    } else {
      emlrtErrorWithMessageIdR2018a(sp, &b_emlrtRTEI, "MATLAB:innerdim",
                                    "MATLAB:innerdim", 0);
    }
  }
}

} // namespace coder

// End of code generation (eml_mtimes_helper.cpp)
