//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// lnlik_Linear_initialize.cpp
//
// Code generation for function 'lnlik_Linear_initialize'
//

// Include files
#include "lnlik_Linear_initialize.h"
#include "_coder_lnlik_Linear_mex.h"
#include "lnlik_Linear_data.h"
#include "rt_nonfinite.h"

// Function Definitions
void lnlik_Linear_initialize()
{
  emlrtStack st{
      nullptr, // site
      nullptr, // tls
      nullptr  // prev
  };
  mex_InitInfAndNan();
  mexFunctionCreateRootTLS();
  emlrtBreakCheckR2012bFlagVar = emlrtGetBreakCheckFlagAddressR2012b();
  st.tls = emlrtRootTLSGlobal;
  emlrtClearAllocCountR2012b(&st, false, 0U, nullptr);
  emlrtEnterRtStackR2012b(&st);
  emlrtFirstTimeR2012b(emlrtRootTLSGlobal);
}

// End of code generation (lnlik_Linear_initialize.cpp)
