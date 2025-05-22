//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
//
// makeYX_initialize.cpp
//
// Code generation for function 'makeYX_initialize'
//

// Include files
#include "makeYX_initialize.h"
#include "_coder_makeYX_mex.h"
#include "makeYX_data.h"
#include "rt_nonfinite.h"

// Function Definitions
void makeYX_initialize()
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

// End of code generation (makeYX_initialize.cpp)
